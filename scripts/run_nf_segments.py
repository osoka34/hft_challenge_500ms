#!/usr/bin/env python3
"""
NeuralForecast segment lab for the 500 ms challenge.

Training scheme:
  1. Day panel (y in bp + causal hist features)
  2. Drop NaNs once on the union of feature packs (fair ablations)
  3. Cut into ~N contiguous segments → each = unique_id
  4. Inside each: train / val / test = 70 / 15 / 15, trimmed to common lengths
  5. Fit one small NHITS across all segments
  6. Score Pearson on boundary + walk-forward test points
  7. Linear imb Pearson on the same test points (sanity)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Lightning prints "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: …" on every Trainer
# init; walk-forward calls predict thousands of times and floods the terminal.
for _name in (
    "pytorch_lightning",
    "lightning",
    "lightning.pytorch",
    "lightning_fabric",
):
    logging.getLogger(_name).setLevel(logging.ERROR)


@contextmanager
def _quiet_lightning():
    """Temporarily silence Lightning INFO during predict loops."""
    loggers = [
        logging.getLogger(n)
        for n in (
            "pytorch_lightning",
            "lightning",
            "lightning.pytorch",
            "lightning_fabric",
        )
    ]
    prev = [lg.level for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for lg, level in zip(loggers, prev):
            lg.setLevel(level)

from full_eda_io import pearson  # noqa: E402
from nf_features import (  # noqa: E402
    FEATURE_SETS,
    columns_for_feature_sets,
    cut_into_segments,
    load_day_panel,
    temporal_masks,
)

OUT_DIR = ROOT / "outputs" / "nf_segments"
MODEL_DIR = OUT_DIR / "models"
Y_SCALE = 1e4  # train/eval in bp; Pearson is scale-invariant, MSE is not


def model_artifact_path(
    base: Path,
    *,
    instrument: str,
    day_tag: str,
    horizon: str,
    feature_set: str,
    loss_name: str,
) -> Path:
    """Directory for one fitted NeuralForecast artifact (nf.save / nf.load)."""
    safe = f"{instrument}__{day_tag}__{horizon}__{feature_set}__{loss_name}"
    return base / safe


def save_nf(nf, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dataset is huge and not needed for later predict(df=...); skip it.
    nf.save(path=str(path), save_dataset=False, overwrite=True)
    print(f"  saved model → {path}", flush=True)
    return path


def load_nf(path: Path):
    """Load a previously saved NeuralForecast; import custom losses first."""
    from neuralforecast import NeuralForecast

    import nf_losses  # noqa: F401 — register PearsonCorrLoss for unpickle/JSON

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"  loading model ← {path}", flush=True)
    return NeuralForecast.load(path=str(path), verbose=False)


def _reindex_ds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["ds"] = np.arange(len(out), dtype=np.int64)
    return out


def _prepare_segment_frame(df: pd.DataFrame, exog: list[str]) -> pd.DataFrame:
    cols = ["unique_id", "ds", "y", *exog]
    extra = [c for c in ("imb",) if c in df.columns and c not in cols]
    out = df[cols + extra].copy()
    out["y"] = out["y"].astype(np.float64)
    for c in list(exog) + extra:
        out[c] = out[c].astype(np.float64)
    out = out.replace([np.inf, -np.inf], np.nan)
    bad = int(out[["y", *exog]].isna().any(axis=1).sum())
    if bad:
        raise ValueError(
            f"{out['unique_id'].iloc[0]}: {bad} NaN rows after pre-clean; "
            "dropna on the panel before cut_into_segments"
        )
    return _reindex_ds(out.sort_values("ds"))


def split_segments(
    segs: pd.DataFrame,
    exog: list[str],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], int]:
    """70/15/15 per segment, then trim all series to common lengths for NF val_size."""
    pieces: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for uid, g in segs.groupby("unique_id", sort=True):
        g = _prepare_segment_frame(g, exog)
        try:
            m_tr, m_va, m_te = temporal_masks(
                len(g), train_frac=train_frac, val_frac=val_frac
            )
        except ValueError:
            continue
        pieces.append((uid, g.loc[m_tr], g.loc[m_va], g.loc[m_te]))
    if not pieces:
        raise RuntimeError("no segments survived the temporal split")

    n_train = min(len(tr) for _, tr, _, _ in pieces)
    n_val = min(len(va) for _, _, va, _ in pieces)
    n_test = min(len(te) for _, _, _, te in pieces)
    if min(n_train, n_val, n_test) < 32:
        raise RuntimeError(
            f"common split too short after trim: train={n_train} val={n_val} test={n_test}"
        )

    trains, fits, tests, kept = [], [], [], []
    for uid, tr, va, te in pieces:
        tr = _reindex_ds(tr.iloc[:n_train])
        va = va.iloc[:n_val]
        fit = _reindex_ds(pd.concat([tr, va], ignore_index=True))
        te = te.iloc[:n_test].copy()
        te["ds"] = np.arange(len(fit), len(fit) + len(te), dtype=np.int64)
        trains.append(tr)
        fits.append(fit)
        tests.append(te)
        kept.append(uid)
    return (
        pd.concat(trains, ignore_index=True),
        pd.concat(fits, ignore_index=True),
        pd.concat(tests, ignore_index=True),
        kept,
        n_val,
    )


def make_nf(
    feature_set: str,
    *,
    input_size: int,
    max_steps: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    seed: int,
    loss_name: str = "mse",
):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    from nf_losses import build_loss

    exog = FEATURE_SETS[feature_set]
    loss = build_loss(loss_name)
    kwargs = dict(
        h=1,
        input_size=input_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        early_stop_patience_steps=5,
        val_check_steps=max(25, max_steps // 8),
        batch_size=64,
        windows_batch_size=512,  # need a fat batch for stable Pearson over windows
        n_blocks=[1, 1, 1],
        mlp_units=[[64, 64], [64, 64], [64, 64]],
        n_pool_kernel_size=[2, 2, 1],
        n_freq_downsample=[4, 2, 1],
        scaler_type="robust",
        start_padding_enabled=True,
        random_seed=seed,
        alias=f"NHITS_{feature_set}_{loss_name}",
        loss=loss,
        valid_loss=loss,  # early-stop on the same objective as train
        dropout_prob_theta=dropout,
        enable_progress_bar=False,
        logger=False,
        # Prefer GPU when the active torch build supports this card (1080 Ti /
        # sm_61 needs an older cu118 wheel, e.g. .venv-gpu with torch 2.7.1+cu118).
        accelerator="gpu" if __import__("torch").cuda.is_available() else "cpu",
        devices=1,
        optimizer_kwargs={"weight_decay": weight_decay},
    )
    if exog:
        kwargs["hist_exog_list"] = list(exog)
        kwargs["exclude_insample_y"] = True
    return NeuralForecast(models=[NHITS(**kwargs)], freq=1), exog


def _nf_cols(exog: list[str]) -> list[str]:
    return ["unique_id", "ds", "y", *exog]


def _append_row(hist: pd.DataFrame, row: pd.Series, exog: list[str]) -> pd.DataFrame:
    rec = {c: row[c] for c in _nf_cols(exog)}
    if "imb" in hist.columns and "imb" in row.index:
        rec["imb"] = row["imb"]
    out = pd.concat([hist, pd.DataFrame([rec])], ignore_index=True)
    out["unique_id"] = out["unique_id"].astype(str)
    out["y"] = out["y"].astype(np.float64)
    for c in exog:
        out[c] = out[c].astype(np.float64)
    return _reindex_ds(out)


def boundary_predict(
    nf, fit_df: pd.DataFrame, test_df: pd.DataFrame, exog: list[str]
) -> pd.DataFrame:
    with _quiet_lightning():
        yhat = nf.predict(df=fit_df[_nf_cols(exog)].copy())
    pred_col = [c for c in yhat.columns if c not in ("unique_id", "ds")][0]
    yhat = yhat.rename(columns={pred_col: "y_hat"})
    y0 = (
        test_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id", as_index=False)
        .first()
    )
    keep = ["unique_id", "y"] + (["imb"] if "imb" in y0.columns else [])
    y0 = y0[keep].rename(columns={"y": "y_true"})
    return yhat.merge(y0, on="unique_id", how="inner")


def walkforward_predict(
    nf,
    fit_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    input_size: int,
    stride: int,
    exog: list[str],
    max_per_segment: int = 12,
    max_fail_rate: float = 0.1,
) -> pd.DataFrame:
    """Strided walk-forward with **one** batched ``nf.predict``.

    Hist for each cutoff is built from observed rows only (no model needed),
    so we materialize every window as its own ``unique_id`` and call Lightning
    once — reuses the fitted model / cached ``_pred_trainer`` instead of
    thousands of Trainer inits.
    """
    keep_hist = _nf_cols(exog) + (
        ["imb"] if "imb" in fit_df.columns and "imb" not in exog else []
    )
    win_len = max(input_size + 8, 16)
    window_frames: list[pd.DataFrame] = []
    meta: list[dict] = []

    for uid, te in test_df.groupby("unique_id", sort=True):
        hist = fit_df[fit_df["unique_id"] == uid][keep_hist].copy()
        hist = _reindex_ds(hist)
        te = te.sort_values("ds").reset_index(drop=True)
        n_pred = 0
        for i in range(len(te)):
            row = te.iloc[i]
            do_pred = (i % stride == 0) and (n_pred < max_per_segment)
            if do_pred:
                wid = f"{uid}__wf{i}"
                hist_win = hist.tail(win_len)[_nf_cols(exog)].copy()
                hist_win = _reindex_ds(hist_win)
                hist_win["unique_id"] = wid
                window_frames.append(hist_win)
                meta.append(
                    {
                        "wid": wid,
                        "unique_id": uid,
                        "y_true": float(row["y"]),
                        "imb": float(row["imb"]) if "imb" in row.index else np.nan,
                        "ds": int(row["ds"]),
                        "i": int(i),
                    }
                )
                n_pred += 1
            hist = _append_row(hist, row, exog)

    n_attempt = len(meta)
    n_fail = 0
    if not meta:
        out = pd.DataFrame()
        out.attrs["n_fail"] = 0
        out.attrs["n_attempt"] = 0
        return out

    batch = pd.concat(window_frames, ignore_index=True)
    print(f"  walkforward batched predict n={n_attempt} windows", flush=True)
    try:
        with _quiet_lightning():
            yhat = nf.predict(df=batch)
        col = [c for c in yhat.columns if c not in ("unique_id", "ds")][0]
        pred_map = {
            str(u): float(v)
            for u, v in zip(yhat["unique_id"].astype(str), yhat[col], strict=False)
        }
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"batched walk-forward predict failed: {type(e).__name__}: {e}") from e

    rows: list[dict] = []
    for m in meta:
        if m["wid"] not in pred_map or not np.isfinite(pred_map[m["wid"]]):
            n_fail += 1
            if n_fail <= 5:
                print(f"  walkforward miss {m['wid']}", flush=True)
            continue
        rows.append(
            {
                "unique_id": m["unique_id"],
                "y_hat": pred_map[m["wid"]],
                "y_true": m["y_true"],
                "imb": m["imb"],
                "ds": m["ds"],
                "i": m["i"],
            }
        )

    if n_attempt and n_fail / n_attempt > max_fail_rate:
        raise RuntimeError(
            f"walk-forward fail rate {n_fail}/{n_attempt} "
            f"> {max_fail_rate:.0%}; refusing biased score"
        )
    out = pd.DataFrame(rows)
    out.attrs["n_fail"] = n_fail
    out.attrs["n_attempt"] = n_attempt
    return out


def ols_imb_score(df: pd.DataFrame) -> dict:
    """Pearson(imb, y) on the scored rows (linear sanity check; not in-sample OLS fit)."""
    if df.empty or "imb" not in df.columns:
        return {"pearson_imb": None, "n": 0}
    x = df["imb"].to_numpy(dtype=np.float64)
    y = df["y_true"].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    return {"pearson_imb": pearson(x, y) if len(x) >= 5 else None, "n": int(len(x))}


def run_feature_set(
    segs: pd.DataFrame,
    feature_set: str,
    *,
    input_size: int,
    max_steps: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    seed: int,
    walkforward: bool,
    wf_stride: int,
    loss_name: str = "mse",
    model_path: Path | None = None,
    load_model: bool = False,
    save_model: bool = False,
) -> dict:
    exog = FEATURE_SETS[feature_set]
    _train_df, fit_df, test_df, kept, val_size = split_segments(segs, exog)

    print(
        f"[{feature_set}] loss={loss_name} series={len(kept)} fit_rows={len(fit_df)} "
        f"val_size={val_size} exog={exog}",
        flush=True,
    )
    t0 = time.time()
    loaded = False
    if load_model and model_path is not None and model_path.exists():
        with _quiet_lightning():
            nf = load_nf(model_path)
        loaded = True
        fit_sec = time.time() - t0
    else:
        nf, exog = make_nf(
            feature_set,
            input_size=input_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            dropout=dropout,
            weight_decay=weight_decay,
            seed=seed,
            loss_name=loss_name,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _quiet_lightning():
                nf.fit(df=fit_df[_nf_cols(exog)], val_size=val_size)
        fit_sec = time.time() - t0
        if save_model and model_path is not None:
            save_nf(nf, model_path)

    border = boundary_predict(nf, fit_df, test_df, exog)
    out: dict = {
        "feature_set": feature_set,
        "loss": loss_name,
        "exog": exog,
        "n_series": len(kept),
        "fit_seconds": round(float(fit_sec), 2),
        "model_loaded": loaded,
        "model_path": str(model_path) if model_path else None,
        "y_scale": Y_SCALE,
        "val_size": val_size,
        "boundary": {
            "n": int(len(border)),
            "pearson": pearson(border["y_hat"].to_numpy(), border["y_true"].to_numpy()),
            "pred_std": float(border["y_hat"].std()) if len(border) else None,
            "y_std": float(border["y_true"].std()) if len(border) else None,
            "baseline": ols_imb_score(border),
        },
    }

    if walkforward:
        t1 = time.time()
        wf = walkforward_predict(
            nf,
            fit_df,
            test_df,
            input_size=input_size,
            stride=wf_stride,
            exog=exog,
        )
        out["walkforward"] = {
            "n": int(len(wf)),
            "stride": wf_stride,
            "n_fail": int(wf.attrs.get("n_fail", 0)),
            "n_attempt": int(wf.attrs.get("n_attempt", 0)),
            "seconds": round(time.time() - t1, 2),
            "pearson": pearson(wf["y_hat"].to_numpy(), wf["y_true"].to_numpy())
            if len(wf)
            else None,
            "pred_std": float(wf["y_hat"].std()) if len(wf) else None,
            "y_std": float(wf["y_true"].std()) if len(wf) else None,
            "baseline": ols_imb_score(wf),
        }
        out["walkforward_rows"] = wf.to_dict(orient="records")
        print(
            f"  walkforward n={out['walkforward']['n']} "
            f"fail={out['walkforward']['n_fail']}/{out['walkforward']['n_attempt']} "
            f"pearson={out['walkforward']['pearson']} "
            f"baseline_imb={out['walkforward']['baseline'].get('pearson_imb')}",
            flush=True,
        )

    out["boundary_rows"] = border.to_dict(orient="records")
    return out


def _load_panels(instrument: str, days: list[str], horizon: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for day in days:
        try:
            parts.append(load_day_panel(instrument, day, horizon=horizon))
            print(f"  loaded {day} rows={len(parts[-1])}", flush=True)
        except Exception as exc:  # noqa: BLE001 — skip bad/missing days
            print(f"  skip {day}: {exc}", flush=True)
    if not parts:
        raise SystemExit("no days loaded")
    return pd.concat(parts, ignore_index=True)


def _cut_multi_day(panel: pd.DataFrame, n_segments: int) -> pd.DataFrame:
    """Cut each calendar day separately so segments never cross midnight."""
    chunks = []
    for (_, day), g in panel.groupby(["instrument", "day"], sort=True):
        chunks.append(cut_into_segments(g.copy(), n_segments=n_segments))
        print(f"  cut {day} → {chunks[-1]['unique_id'].nunique()} segs", flush=True)
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", default="BTCUSDT_PERP")
    p.add_argument("--day", default="2025-11-03", help="single day (ignored if --days set)")
    p.add_argument(
        "--days",
        nargs="+",
        default=None,
        help="one or more UTC days to pool as segments (overrides --day)",
    )
    p.add_argument("--horizon", default="y_500ms", choices=["y_500ms", "y_2s", "y_8s"])
    p.add_argument(
        "--feature-sets",
        nargs="+",
        default=["y_only", "imb", "core"],
        choices=sorted(FEATURE_SETS),
    )
    p.add_argument(
        "--n-segments",
        type=int,
        default=100,
        help="target contiguous segments per day before max-segments cap",
    )
    p.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="cap total unique_ids across all days (0 = all)",
    )
    p.add_argument("--input-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--loss",
        default="mse",
        choices=["mse", "pearson"],
        help="train/valid loss: mse (default) or pearson (=1-corr, SCORE-aligned)",
    )
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--wf-stride", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-models",
        action="store_true",
        help=f"save fitted NF models under --model-dir (default {MODEL_DIR})",
    )
    p.add_argument(
        "--load-models",
        action="store_true",
        help="skip fit if a matching artifact exists under --model-dir",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help="directory for nf.save / nf.load artifacts",
    )
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    days = list(args.days) if args.days else [args.day]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"loading {args.instrument} days={len(days)} {args.horizon}: {days}",
        flush=True,
    )
    panel = _load_panels(args.instrument, days, args.horizon)
    leaders = (
        sorted(panel["leader"].dropna().astype(str).unique())
        if "leader" in panel.columns
        else []
    )
    print(f"leaders={leaders}", flush=True)
    # Drop packs that need leader columns when no leader book exists.
    leader_cols = {
        "leader_ret_1_bp",
        "leader_imb",
        "leader_ofi",
        "leader_mid",
        "gap_dev",
        "div_150s",
        "div_300s",
        "kf_innov",
        "kf_level",
        "kf_vel",
        "kf_innov_cv",
        "kf_level_cv",
        "leader_ret_200ms_bp",
        "leader_ret_300ms_bp",
        "leader_tfi_500",
        "leader_tfi_s500",
    }
    active_sets = []
    for fs in args.feature_sets:
        need_leader = bool(leader_cols.intersection(FEATURE_SETS[fs]))
        if need_leader and (
            not leaders or panel["leader_ret_1_bp"].isna().all()
        ):
            print(f"skip {fs}: no leader features", flush=True)
            continue
        active_sets.append(fs)
    if not active_sets:
        raise SystemExit("no feature sets left to run")
    args.feature_sets = active_sets

    panel = panel.copy()
    panel["y"] = panel["y"].astype(np.float64) * Y_SCALE
    # NF h=1 predicts the *next* row. Store y[t]=y_comp[t-1] so a forecast from
    # cutoff t (features at t) targets y_comp[t] on the following row.
    # Shift must not cross day boundaries when pooling multiple days.
    panel = panel.sort_values(["day", "ds"]).reset_index(drop=True)
    panel["y_comp"] = panel["y"]
    panel["y"] = panel.groupby("day", sort=False)["y_comp"].shift(1)
    need = columns_for_feature_sets(args.feature_sets)
    before = len(panel)
    panel = panel.dropna(subset=need).reset_index(drop=True)
    print(f"pre-clean dropna {before}→{len(panel)} on {need}", flush=True)

    segs = _cut_multi_day(panel, n_segments=args.n_segments)
    if args.max_segments and args.max_segments > 0:
        keep = sorted(segs["unique_id"].unique())[: args.max_segments]
        segs = segs[segs["unique_id"].isin(keep)].copy()
    print(
        f"segments={segs['unique_id'].nunique()} rows={len(segs)} "
        f"days={segs['day'].nunique()} y_scale={Y_SCALE}",
        flush=True,
    )

    summaries = []
    t0 = time.time()
    day_tag = days[0] if len(days) == 1 else f"{days[0]}_to_{days[-1]}_n{len(days)}"
    model_root = Path(args.model_dir)
    if args.save_models or args.load_models:
        model_root.mkdir(parents=True, exist_ok=True)
        print(
            f"models dir={model_root} save={args.save_models} load={args.load_models}",
            flush=True,
        )
    for fs in args.feature_sets:
        mpath = model_artifact_path(
            model_root,
            instrument=args.instrument,
            day_tag=day_tag,
            horizon=args.horizon,
            feature_set=fs,
            loss_name=args.loss,
        )
        r = run_feature_set(
            segs,
            fs,
            input_size=args.input_size,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            seed=args.seed,
            walkforward=args.walkforward,
            wf_stride=args.wf_stride,
            loss_name=args.loss,
            model_path=mpath,
            load_model=args.load_models,
            save_model=args.save_models,
        )
        summaries.append(
            {
                k: v
                for k, v in r.items()
                if k not in ("boundary_rows", "walkforward_rows")
            }
        )
        tag = f"{args.instrument}_{day_tag}_{args.horizon}_{fs}"
        (OUT_DIR / f"{tag}.json").write_text(json.dumps(r, indent=2, default=float))
        b = r["boundary"]
        wf = r.get("walkforward") or {}
        print(
            f"→ {fs}: boundary={b['pearson']:.4f} (n={b['n']}, "
            f"base_imb={b['baseline'].get('pearson_imb')}) "
            f"wf={wf.get('pearson')} (n={wf.get('n')}) fit={r['fit_seconds']}s",
            flush=True,
        )

    payload = {
        "instrument": args.instrument,
        "days": days,
        "day": day_tag,
        "horizon": args.horizon,
        "n_segments": int(segs["unique_id"].nunique()),
        "n_days": int(segs["day"].nunique()),
        "input_size": args.input_size,
        "max_steps": args.max_steps,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "y_scale": Y_SCALE,
        "seconds": round(time.time() - t0, 1),
        "results": summaries,
    }
    out_path = Path(args.out) if args.out else OUT_DIR / (
        f"{args.instrument}_{day_tag}_{args.horizon}_summary.json"
    )
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps({"summary": summaries, "wrote": str(out_path)}, indent=2, default=float))


if __name__ == "__main__":
    main()
