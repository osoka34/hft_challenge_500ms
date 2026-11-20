"""Albers-style linear models on high-level features (arXiv:2108.09750).

Paper path: engineer features (absolute TFI, softsign, mean-divergence, past
returns, leader) → linear model. We score daily Pearson vs clipped y (SCORE).

Modes:
  --mode day     : within-day 70/30 ridge (quick probe)
  --mode wf      : expanding multi-day walk-forward (train past days → test next)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from full_eda_io import load_target_clip, pearson  # noqa: E402
from nf_features import FEATURE_SETS, load_day_panel  # noqa: E402

UNIVARIATE = [
    "imb",
    "tfi_100",
    "tfi_500",
    "tfi_1000",
    "tfi_2000",
    "tfi_s100",
    "tfi_s500",
    "tfi_s1000",
    "tfi_s2000",
    "pret_500_bp",
    "pret_1000_bp",
    "pret_2000_bp",
    "gap_dev",
    "div_150s",
    "div_300s",
    "kf_innov",
    "kf_level",
    "kf_vel",
    "leader_ret_1_bp",
    "leader_tfi_s500",
]

DEFAULT_PACKS = [
    "imb",
    "hl_kf",
    "hl_kf_imb3",
    "hl_kf_imb3_short",
    "hl_kf_imb3_micro",
    "alb_kf",
    "alb_kf_imb3",
]


def _pack_cols(name: str) -> list[str]:
    if name == "imb":
        return ["imb"]
    return list(FEATURE_SETS[name])


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Centered ridge. Returns (beta, mu_x, mu_y) or None if underdetermined."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 5:
        return None
    mu_x = X.mean(axis=0)
    mu_y = float(y.mean())
    Xc = X - mu_x
    yc = y - mu_y
    xtx = Xc.T @ Xc
    if ridge > 0:
        xtx = xtx + ridge * np.eye(xtx.shape[0])
    try:
        beta = np.linalg.solve(xtx, Xc.T @ yc)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx, Xc.T @ yc, rcond=None)[0]
    return beta, mu_x, mu_y


def _predict(X: np.ndarray, beta: np.ndarray, mu_x: np.ndarray, mu_y: float) -> np.ndarray:
    Xc = np.where(np.isfinite(X), X - mu_x, 0.0)
    return Xc @ beta + mu_y


def _clip_y(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(y, lo, hi)


def _day_score(
    yhat: np.ndarray,
    y: np.ndarray,
    *,
    lo: float,
    hi: float,
) -> dict:
    yc = _clip_y(y, lo, hi)
    ok = np.isfinite(yhat) & np.isfinite(yc)
    return {
        "pearson": float(pearson(yhat[ok], yc[ok])) if ok.sum() >= 5 else None,
        "n": int(ok.sum()),
    }


def _load_days(instrument: str, days: list[str], cols: list[str]) -> dict[str, pd.DataFrame]:
    need = sorted(set(cols) | {"y", "imb", "ts"})
    out: dict[str, pd.DataFrame] = {}
    for d in days:
        print(f"  load {d}", flush=True)
        p = load_day_panel(instrument, d)
        missing = [c for c in need if c not in p.columns]
        if missing:
            raise KeyError(f"{d}: missing {missing}")
        out[d] = p.sort_values("ts").reset_index(drop=True)
    return out


def run_within_day(
    panels: dict[str, pd.DataFrame],
    packs: list[str],
    *,
    ridge: float,
    clip: dict[tuple[str, str], tuple[float, float]],
    instrument: str,
) -> dict:
    results = {}
    for day, panel in panels.items():
        lo, hi = clip.get((instrument, "y_500ms"), (-np.inf, np.inf))
        uni = {}
        for c in UNIVARIATE:
            if c not in panel.columns:
                uni[c] = None
                continue
            ok = panel[[c, "y"]].dropna()
            yc = _clip_y(ok["y"].to_numpy(), lo, hi)
            uni[c] = float(pearson(ok[c].to_numpy(), yc)) if len(ok) >= 5 else None
        pack_scores = {}
        n = len(panel)
        cut = int(n * 0.70)
        tr, te = panel.iloc[:cut], panel.iloc[cut:]
        for name in packs:
            cols = _pack_cols(name)
            fit = _fit_ridge(tr[cols].to_numpy(np.float64), tr["y"].to_numpy(np.float64), ridge=ridge)
            if fit is None:
                pack_scores[name] = {"pearson": None, "n": 0}
                continue
            beta, mu_x, mu_y = fit
            yhat = _predict(te[cols].to_numpy(np.float64), beta, mu_x, mu_y)
            pack_scores[name] = _day_score(yhat, te["y"].to_numpy(np.float64), lo=lo, hi=hi)
            pack_scores[name]["cols"] = cols
        # imb baseline on same test rows
        imb_ok = te[["imb", "y"]].dropna()
        pack_scores["baseline_imb_ic"] = {
            "pearson": float(
                pearson(
                    imb_ok["imb"].to_numpy(),
                    _clip_y(imb_ok["y"].to_numpy(), lo, hi),
                )
            )
            if len(imb_ok) >= 5
            else None,
            "n": int(len(imb_ok)),
        }
        results[day] = {"univariate": uni, "packs": pack_scores}
    return results


def run_walkforward(
    panels: dict[str, pd.DataFrame],
    packs: list[str],
    *,
    ridge: float,
    clip: dict[tuple[str, str], tuple[float, float]],
    instrument: str,
    min_train_days: int = 3,
) -> dict:
    days = list(panels.keys())
    lo, hi = clip.get((instrument, "y_500ms"), (-np.inf, np.inf))
    per_day: dict[str, dict] = {}
    for i, day in enumerate(days):
        if i < min_train_days:
            continue
        train_days = days[:i]
        te = panels[day]
        print(f"  wf test={day} train={train_days[0]}..{train_days[-1]} ({len(train_days)}d)", flush=True)
        day_out = {}
        for name in packs:
            cols = _pack_cols(name)
            tr = pd.concat([panels[d][cols + ["y"]] for d in train_days], ignore_index=True)
            fit = _fit_ridge(tr[cols].to_numpy(np.float64), tr["y"].to_numpy(np.float64), ridge=ridge)
            if fit is None:
                day_out[name] = {"pearson": None, "n": 0}
                continue
            beta, mu_x, mu_y = fit
            yhat = _predict(te[cols].to_numpy(np.float64), beta, mu_x, mu_y)
            sc = _day_score(yhat, te["y"].to_numpy(np.float64), lo=lo, hi=hi)
            sc["cols"] = cols
            sc["coef"] = {c: float(b) for c, b in zip(cols, beta)}
            day_out[name] = sc
        imb_ok = te[["imb", "y"]].dropna()
        day_out["baseline_imb_ic"] = {
            "pearson": float(
                pearson(
                    imb_ok["imb"].to_numpy(),
                    _clip_y(imb_ok["y"].to_numpy(), lo, hi),
                )
            )
            if len(imb_ok) >= 5
            else None,
            "n": int(len(imb_ok)),
        }
        per_day[day] = day_out

    # aggregate mean daily SCORE
    summary = {}
    for name in list(packs) + ["baseline_imb_ic"]:
        vals = [per_day[d][name]["pearson"] for d in per_day if per_day[d].get(name, {}).get("pearson") is not None]
        summary[name] = {
            "mean_daily_pearson": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
            "n_days": len(vals),
        }
    return {"per_day": per_day, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="BTCUSDT_PERP")
    ap.add_argument("--day", default=None, help="single day (day mode)")
    ap.add_argument(
        "--days",
        nargs="+",
        default=None,
        help="explicit day list; default for wf = 2026-01-12..01-25",
    )
    ap.add_argument("--mode", choices=["day", "wf"], default="wf")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--packs", nargs="+", default=DEFAULT_PACKS)
    ap.add_argument("--min-train-days", type=int, default=3)
    ap.add_argument(
        "--out",
        default=None,
        help="JSON path (default outputs/albers_linear/...)",
    )
    args = ap.parse_args()

    if args.days:
        days = args.days
    elif args.mode == "day":
        days = [args.day or "2026-01-20"]
    else:
        days = [f"2026-01-{d:02d}" for d in range(12, 26)]

    all_cols: list[str] = []
    for p in args.packs:
        all_cols.extend(_pack_cols(p))
    if args.mode == "day":
        all_cols.extend(UNIVARIATE)

    print(f"mode={args.mode} instrument={args.instrument} days={days[0]}..{days[-1]} n={len(days)}", flush=True)
    panels = _load_days(args.instrument, days, all_cols)
    clip = load_target_clip()

    if args.mode == "day":
        body = run_within_day(panels, args.packs, ridge=args.ridge, clip=clip, instrument=args.instrument)
        out = {
            "mode": "day",
            "instrument": args.instrument,
            "ridge": args.ridge,
            "days": days,
            "results": body,
        }
    else:
        body = run_walkforward(
            panels,
            args.packs,
            ridge=args.ridge,
            clip=clip,
            instrument=args.instrument,
            min_train_days=args.min_train_days,
        )
        out = {
            "mode": "wf",
            "instrument": args.instrument,
            "ridge": args.ridge,
            "min_train_days": args.min_train_days,
            "days": days,
            **body,
        }

    out_path = Path(
        args.out
        or (
            ROOT
            / "outputs"
            / "albers_linear"
            / f"{args.instrument}_{args.mode}_{days[0]}_{days[-1]}_ridge{args.ridge:g}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out.get("summary") or out, indent=2)[:4000])
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
