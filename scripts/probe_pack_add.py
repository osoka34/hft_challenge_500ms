"""Probe whether one admitted column adds WF Pearson on top of hl_resid.

Usage:
  python scripts/probe_pack_add.py --feat <column>
  python scripts/probe_pack_add.py --feat <column> --horizon y_2s
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

from nf_features import FEATURE_SETS, load_day_panel  # noqa: E402
from run_paper_seq import (  # noqa: E402
    apply_gould_fold,
    apply_micro_adj_fold,
    apply_stoikov_g1_fold,
    verdict,
    wf_scores,
)
from screen_features_uni import DAYS_DEFAULT, _ensure_paper_feats  # noqa: E402

BASE_PACK = FEATURE_SETS["hl_resid"]


def _materialize_fold(panels: dict, days: list[str], feat: str) -> None:
    """Write fold-calibrated columns onto each day using expanding past days.

    Day 0 is mapped from itself so early train rows are finite when later OOS
    folds concat days[:i]; day i≥1 uses only days[:i] as train (no peek).
    """
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat", required=True, help="Column to add on top of hl_resid")
    ap.add_argument("--instrument", default="BTCUSDT_PERP")
    ap.add_argument("--days", nargs="+", default=DAYS_DEFAULT)
    ap.add_argument("--horizon", default="y_500ms", choices=["y_500ms", "y_2s", "y_8s"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feat = args.feat
    days = args.days
    inst = args.instrument
    print(f"load {inst} horizon={args.horizon}…", flush=True)
    panels = {d: load_day_panel(inst, d, horizon=args.horizon) for d in days}
    _ensure_paper_feats(panels, inst, days, [feat])
    _materialize_fold(panels, days, feat)

    if feat not in panels[days[min(5, len(days) - 1)]].columns:
        raise SystemExit(f"missing column on panel: {feat}")

    base_cols = [c for c in BASE_PACK if c in panels[days[5]].columns]
    add_cols = base_cols + [feat]

    sc_base, err_b = wf_scores(panels, base_cols, inst, days, horizon=args.horizon)
    if err_b:
        raise SystemExit(f"base pack failed: {err_b}")
    sc_add, err_a = wf_scores(panels, add_cols, inst, days, horizon=args.horizon)
    if err_a:
        raise SystemExit(f"add pack failed: {err_a}")

    mean_base = float(np.mean(sc_base))
    mean_add = float(np.mean(sc_add))
    delta = mean_add - mean_base
    ver = verdict(mean_add, mean_base)
    out = {
        "instrument": inst,
        "horizon": args.horizon,
        "days": days,
        "feat": feat,
        "base_pack": base_cols,
        "wf_mean_base": mean_base,
        "wf_mean_add": mean_add,
        "delta": delta,
        "verdict": ver,
        "gate": "OK if delta≥0.003, WEAK if ≥0.001, else NOT OK",
    }
    path = Path(
        args.out
        or ROOT / "outputs" / "paper_seq" / f"pack_add_{inst}_{feat}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    print(f"Δ={delta:+.4f} → {ver}")


if __name__ == "__main__":
    main()
