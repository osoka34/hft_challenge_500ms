"""Forward stepwise pack build by expanding WF Pearson Δ."""

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
from run_paper_seq import (  # noqa: E402
    apply_gould_fold,
    apply_micro_adj_fold,
    apply_stoikov_g1_fold,
    fit,
    predict,
    verdict,
    wf_scores,
)
from screen_features_uni import DAYS_DEFAULT, _ensure_paper_feats  # noqa: E402


def _materialize_fold(panels: dict, days: list[str], feat: str) -> None:
    """Same expanding map as probe_pack_add: day 0 from itself, later days from the past only."""
    if feat not in ("stoikov_g1", "micro_adj", "gould_ey"):
        return

    def _apply(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
        if feat == "stoikov_g1":
            return apply_stoikov_g1_fold(tr, te)
        if feat == "micro_adj":
            return apply_micro_adj_fold(tr, te)
        return apply_gould_fold(tr, te)

    for i, d in enumerate(days):
        if i == 0:
            panels[d][feat] = _apply(panels[d], panels[d])
            continue
        tr = pd.concat([panels[x] for x in days[:i]], ignore_index=True)
        panels[d][feat] = _apply(tr, panels[d])


def _max_abs_corr_with_pack(
    panels: dict[str, pd.DataFrame],
    days: list[str],
    pack: list[str],
    feat: str,
) -> float:
    """Pooled |Pearson| of feat vs each pack column (duplicate sniff)."""
    xs = []
    packs = {c: [] for c in pack}
    for d in days:
        df = panels[d]
        if feat not in df.columns:
            continue
        xs.append(df[feat].to_numpy(float))
        for c in pack:
            packs[c].append(df[c].to_numpy(float) if c in df.columns else np.full(len(df), np.nan))
    if not xs:
        return float("nan")
    x = np.concatenate(xs)
    best = 0.0
    for c, chunks in packs.items():
        y = np.concatenate(chunks)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100:
            continue
        r = abs(float(pearson(x[m], y[m])))
        if np.isfinite(r) and r > best:
            best = r
    return best


def _resid_ic_mean(
    panels: dict[str, pd.DataFrame],
    pack: list[str],
    feat: str,
    inst: str,
    days: list[str],
    horizon: str,
    min_train: int = 3,
) -> float:
    """Mean daily Pearson of feat vs OOS residual of the pack (incremental sniff)."""
    lo, hi = load_target_clip().get((inst, horizon), (-np.inf, np.inf))
    scores = []
    need = pack + [feat, "y"]
    for i, d in enumerate(days):
        if i < min_train:
            continue
        for c in need:
            if c not in panels[d].columns:
                return float("nan")
        tr = pd.concat([panels[x][need] for x in days[:i]], ignore_index=True)
        te = panels[d]
        fitted = fit(tr[pack].to_numpy(float), tr["y"].to_numpy(float))
        if fitted is None:
            continue
        b, mx, my = fitted
        yhat = predict(te[pack].to_numpy(float), b, mx, my)
        yc = np.clip(te["y"].to_numpy(float), lo, hi)
        resid = yc - yhat
        x = te[feat].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(resid)
        if ok.sum() < 100:
            continue
        scores.append(float(pearson(x[ok], resid[ok])))
    return float(np.mean(scores)) if scores else float("nan")


def _score_candidate(
    panels: dict[str, pd.DataFrame],
    pack: list[str],
    feat: str,
    inst: str,
    days: list[str],
    horizon: str,
    base_mean: float,
) -> dict:
    cols = pack + [feat]
    sc, err = wf_scores(panels, cols, inst, days, horizon=horizon)
    if err:
        return {
            "feat": feat,
            "error": err,
            "delta": float("nan"),
            "wf_mean_add": float("nan"),
            "verdict": "NOT OK",
            "resid_ic": float("nan"),
            "max_abs_corr_pack": float("nan"),
        }
    mean_add = float(np.mean(sc))
    delta = mean_add - base_mean
    return {
        "feat": feat,
        "error": None,
        "wf_mean_add": mean_add,
        "delta": delta,
        "verdict": verdict(mean_add, base_mean),
        "resid_ic": _resid_ic_mean(panels, pack, feat, inst, days, horizon),
        "max_abs_corr_pack": _max_abs_corr_with_pack(panels, days, pack, feat),
    }


def forward_add(
    panels: dict[str, pd.DataFrame],
    *,
    pack: list[str],
    cands: list[str],
    inst: str,
    days: list[str],
    horizon: str,
    min_delta: float,
) -> dict:
    pack = list(pack)
    remaining = [c for c in cands if c not in pack]
    steps: list[dict] = []
    rejected: list[dict] = []

    sc0, err0 = wf_scores(panels, pack, inst, days, horizon=horizon)
    if err0:
        raise RuntimeError(f"base pack failed: {err0}")
    base_mean = float(np.mean(sc0))

    round_i = 0
    while remaining:
        round_i += 1
        print(f"round {round_i}: pack={pack}  try {remaining}", flush=True)
        trials = [
            _score_candidate(panels, pack, feat, inst, days, horizon, base_mean)
            for feat in remaining
        ]
        trials.sort(
            key=lambda t: (-(t["delta"] if np.isfinite(t["delta"]) else -1e9), t["feat"])
        )
        best = trials[0]
        step = {"round": round_i, "base_mean": base_mean, "trials": trials, "chosen": None}
        if not np.isfinite(best["delta"]) or best["delta"] < min_delta:
            step["stop"] = f"best Δ={best.get('delta')} < min_delta={min_delta}"
            steps.append(step)
            rejected.extend(trials)
            break
        # Accept best; others stay for next round (re-scored vs new pack)
        pack = pack + [best["feat"]]
        remaining = [c for c in remaining if c != best["feat"]]
        base_mean = float(best["wf_mean_add"])
        step["chosen"] = best["feat"]
        step["new_pack"] = list(pack)
        step["new_mean"] = base_mean
        steps.append(step)
        print(
            f"  accept {best['feat']} Δ={best['delta']:+.4f} "
            f"resid_ic={best['resid_ic']:+.4f} corr_pack={best['max_abs_corr_pack']:.2f}",
            flush=True,
        )
    else:
        # exhausted candidates by accepting all that cleared
        pass

    accepted = [s["chosen"] for s in steps if s.get("chosen")]
    return {
        "instrument": inst,
        "horizon": horizon,
        "days": days,
        "min_delta": min_delta,
        "final_pack": pack,
        "accepted": accepted,
        "wf_mean_final": base_mean,
        "steps": steps,
        "gate": f"accept if Δ ≥ {min_delta} (OK threshold is 0.003; WEAK is 0.001)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Greedy WF pack build: each round try every --cand on top of the current pack; "
            "accept the best if Δ ≥ --min-delta (default 0.003 = OK gate)."
        )
    )
    ap.add_argument(
        "--cand",
        action="append",
        required=True,
        help="Candidate column (repeatable). Only these enter the forward search.",
    )
    ap.add_argument(
        "--base-pack",
        default="hl_resid",
        help="FEATURE_SETS name to start from (default hl_resid)",
    )
    ap.add_argument("--instrument", default="BTCUSDT_PERP")
    ap.add_argument("--days", nargs="+", default=DAYS_DEFAULT)
    ap.add_argument("--horizon", default="y_500ms", choices=["y_500ms", "y_2s", "y_8s"])
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.003,
        help="Minimum WF Pearson Δ to accept into the pack (default 0.003 = OK gate)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.base_pack not in FEATURE_SETS:
        raise SystemExit(f"unknown base pack: {args.base_pack}")

    cands = list(dict.fromkeys(args.cand))
    days = args.days
    inst = args.instrument
    print(f"load {inst} horizon={args.horizon}…", flush=True)
    panels = {d: load_day_panel(inst, d, horizon=args.horizon) for d in days}
    _ensure_paper_feats(panels, inst, days, cands)
    for feat in cands:
        _materialize_fold(panels, days, feat)

    sample = panels[days[min(5, len(days) - 1)]]
    missing = [c for c in cands if c not in sample.columns]
    if missing:
        raise SystemExit(f"columns not on panel: {missing}")

    start = [c for c in FEATURE_SETS[args.base_pack] if c in sample.columns]
    out = forward_add(
        panels,
        pack=start,
        cands=cands,
        inst=inst,
        days=days,
        horizon=args.horizon,
        min_delta=args.min_delta,
    )
    out["base_pack_name"] = args.base_pack
    out["start_pack"] = start
    out["candidates"] = cands

    # Flatten last-round rejects for readability
    last = out["steps"][-1] if out["steps"] else None
    if last and last.get("chosen") is None:
        out["stopped_trials"] = last["trials"]
    out["dropped"] = [c for c in cands if c not in out["accepted"]]

    tag = "_".join(cands[:3]) + (f"_plus{len(cands) - 3}" if len(cands) > 3 else "")
    path = Path(args.out or ROOT / "outputs" / "paper_seq" / f"forward_add_{inst}_{tag}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    md_path = ROOT / "reports" / f"FORWARD_ADD_{inst}.md"
    lines = [
        f"# Forward add — {inst} · {args.horizon}",
        "",
        f"Base: `{args.base_pack}` → {start}",
        f"Candidates: {cands}",
        f"min_delta: {args.min_delta}",
        "",
        f"Accepted: **{out['accepted'] or '(none)'}**",
        f"Final pack ({out['wf_mean_final']:.4f}): `{out['final_pack']}`",
        f"Dropped: {out['dropped']}",
        "",
    ]
    for s in out["steps"]:
        lines.append(f"## Round {s['round']} (base wf_mean={s['base_mean']:.4f})")
        lines.append("")
        lines.append("| feat | Δ | resid_ic | |corr|_pack | verdict |")
        lines.append("|------|---|----------|------------|---------|")
        for t in s["trials"]:
            dlt = t["delta"]
            dlt_s = f"{dlt:+.4f}" if isinstance(dlt, float) and np.isfinite(dlt) else "err"
            ric = t["resid_ic"]
            ric_s = f"{ric:+.4f}" if isinstance(ric, float) and np.isfinite(ric) else "—"
            corr = t["max_abs_corr_pack"]
            corr_s = f"{corr:.2f}" if isinstance(corr, float) and np.isfinite(corr) else "—"
            lines.append(
                f"| `{t['feat']}` | {dlt_s} | {ric_s} | {corr_s} | {t['verdict']} |"
            )
        lines.append("")
        if s.get("chosen"):
            lines.append(f"Accepted `{s['chosen']}` → wf_mean {s['new_mean']:.4f}")
        else:
            lines.append(f"Stop: {s.get('stop')}")
        lines.append("")
    lines.append(f"JSON: `{path}`")
    md_path.write_text("\n".join(lines) + "\n")

    print(json.dumps({k: out[k] for k in ("accepted", "dropped", "final_pack", "wf_mean_final")}, indent=2))
    print(f"wrote {path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
