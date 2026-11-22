"""Univariate feature screen (causal WF).

For each feature x: expanding train → OLS y~x on past days → Pearson on next day
(clipped y). Also report pooled Spearman/Pearson IC on OOS residuals-free raw y.

Use this BEFORE group packs. A feature is SIGNIFICANT if mean WF Pearson ≥
threshold (default 0.05) and beat noise (optional sign stability).
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

from full_eda_io import load_target_clip, pearson, spearman  # noqa: E402
from nf_features import load_day_panel  # noqa: E402
from run_paper_seq import (  # noqa: E402
    add_albers_sweep_proxy,
    add_boltzmann_gap,
    add_cross_btc_leader,
    add_lucchese_vol_side,
    add_lucchese_vol_side_far,
    add_cont_ofi_depth,
    add_stoikov_micro,
    add_imb_x_lead,
    add_phi_cnt,
    add_q_depl,
    add_liq_side,
    add_rel_liq,
    add_sweep_px,
    add_sign_age,
    add_size_gap,
    add_usdc_prem,
    add_sq_micro,
    add_vb_tfi,
    add_vpin,
    add_vwap_mid,
    apply_gould_fold,
    apply_micro_adj_fold,
    apply_stoikov_g1_fold,
)

DAYS_DEFAULT = [f"2026-01-{d:02d}" for d in range(12, 26)]

# Candidates: existing panel + paper-derived (filled below)
CORE_CANDS = [
    "imb",
    "imb3",
    "micro_gap",
    "spread_bp",
    "ret_1_bp",
    "ret_4_bp",
    "pret_500_bp",
    "pret_1000_bp",
    "depth_imb_near",
    "depth_imb_mid",
    "ofi_best",
    "d_imb",
    "ofi_near",
    "gap_dev",
    "div_150s",
    "div_300s",
    "kf_innov",
    "kf_level",
    "kf_vel",
    "leader_ret_1_bp",
    "leader_ret_200ms_bp",
    "leader_ret_300ms_bp",
    "leader_imb",
    "leader_ofi",
    "tfi_100",
    "tfi_500",
    "tfi_1000",
    "tfi_s100",
    "tfi_s500",
    "tfi_s1000",
    "leader_tfi_100",
    "leader_tfi_s100",
    "leader_tfi_s500",
]

PAPER_CANDS = [
    "stoikov_g1",
    "micro_adj",
    "gould_ey",
    "ofi_raw",
    "ofi_depth",
    "ofi_h2_d",
    "ofi_h4_d",
    "ofi_h8_d",
    "bid_of",
    "ask_of",
    "sweep_imb",
    "sweep_imba",
    "sweep_imbb",
    "vol_side_imb",
    "vol_side_imb_far",
    "boltz_gap",
    "vpin",
    "vb_tfi",
    "btc_ret_1_bp",
    "btc_imb",
    "btc_tfi_s100",
    "btc_kf_innov",
    "vwap_mid_bp",
    "phi_cnt",
    "sq_micro_bp",
    "q_depl",
    "rel_liq",
    "liq_side",
    "sweep_px",
    "sign_age",
    "size_gap",
    "usdc_prem_bp",
    "imb_x_lead",
]


def fit_uni(x: np.ndarray, y: np.ndarray):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 30:
        return None
    mx, my = float(x.mean()), float(y.mean())
    xc, yc = x - mx, y - my
    den = float(xc @ xc)
    if den <= 0:
        return None
    b = float(xc @ yc) / den
    return b, mx, my


def pred_uni(x, b, mx, my):
    return np.where(np.isfinite(x), (x - mx) * b + my, np.nan)


def _resolve_cands(panels: dict[str, pd.DataFrame], days: list[str], feats: list[str] | None) -> list[str]:
    if feats:
        return list(dict.fromkeys(feats))
    cands = []
    sample = panels[days[min(5, len(days) - 1)]]
    fold_cands = ("stoikov_g1", "micro_adj", "gould_ey")
    for c in CORE_CANDS + PAPER_CANDS:
        if c in fold_cands:
            cands.append(c)
        elif c in sample.columns:
            cands.append(c)
    return list(dict.fromkeys(cands))


def _ensure_paper_feats(panels: dict[str, pd.DataFrame], inst: str, days: list[str], cands: list[str]) -> None:
    """Load only the paper builders required by the candidate list."""
    need_stoikov = any(
        c in ("stoikov_g1", "micro_adj") or c.startswith("stoikov") for c in cands
    )
    need_ofi = any(
        c.startswith(("ofi_", "bid_of", "ask_of")) or c in ("ofi_raw", "ofi_depth") for c in cands
    )
    need_sweep = any("sweep" in c and not c.startswith("sweep_px") for c in cands)
    need_vol = any(c == "vol_side_imb" for c in cands)
    need_vol_far = any(c == "vol_side_imb_far" or c.endswith("_far") and "vol_side" in c for c in cands)
    need_boltz = any(c == "boltz_gap" or c.startswith("boltz") for c in cands)
    need_vpin = any(c == "vpin" or c.startswith("vpin") for c in cands)
    need_vb_tfi = any(c == "vb_tfi" or c.startswith("vb_tfi") for c in cands)
    need_vwap = any(c == "vwap_mid_bp" or c.startswith("vwap_mid") for c in cands)
    need_phi = any(c == "phi_cnt" or c.startswith("phi_cnt") for c in cands)
    need_sq = any(c == "sq_micro_bp" or c.startswith("sq_micro") for c in cands)
    need_qdepl = any(c == "q_depl" or c.startswith("q_depl") for c in cands)
    need_rel = any(c == "rel_liq" or c.startswith("rel_liq") for c in cands)
    need_liq_side = any(c == "liq_side" or c.startswith("liq_side") for c in cands)
    need_sweep_px = any(c == "sweep_px" or c.startswith("sweep_px") for c in cands)
    need_sign_age = any(c == "sign_age" or c.startswith("sign_age") for c in cands)
    need_size_gap = any(c == "size_gap" or c.startswith("size_gap") for c in cands)
    need_usdc = any(c == "usdc_prem_bp" or c.startswith("usdc_prem") for c in cands)
    need_imb_x_lead = any(c == "imb_x_lead" for c in cands)
    need_btc_x = any(c.startswith("btc_") for c in cands)
    if need_stoikov:
        add_stoikov_micro(panels, days)
    if need_ofi:
        add_cont_ofi_depth(panels, inst)
    if need_sweep:
        add_albers_sweep_proxy(panels, inst)
    if need_vol:
        add_lucchese_vol_side(panels, inst)
    if need_vol_far:
        add_lucchese_vol_side_far(panels, inst)
    if need_boltz:
        add_boltzmann_gap(panels, inst)
    if need_vpin:
        add_vpin(panels, inst)
    if need_vb_tfi:
        add_vb_tfi(panels, inst)
    if need_vwap:
        add_vwap_mid(panels, inst)
    if need_phi:
        add_phi_cnt(panels, inst)
    if need_sq:
        add_sq_micro(panels, inst)
    if need_qdepl:
        add_q_depl(panels, inst)
    if need_rel:
        add_rel_liq(panels, inst)
    if need_liq_side:
        add_liq_side(panels, inst)
    if need_sweep_px:
        add_sweep_px(panels, inst)
    if need_sign_age:
        add_sign_age(panels, inst)
    if need_size_gap:
        add_size_gap(panels, inst)
    if need_usdc:
        add_usdc_prem(panels, inst)
    if need_imb_x_lead:
        add_imb_x_lead(panels)
    if need_btc_x:
        if inst == "BTCUSDT_PERP":
            raise ValueError("btc_* cross features require a non-BTC instrument")
        print(f"cross BTC leader → {inst}…", flush=True)
        btc = {d: load_day_panel("BTCUSDT_PERP", d, horizon="y_500ms") for d in days}
        add_cross_btc_leader(panels, btc)


def screen(
    inst: str,
    days: list[str],
    thr: float,
    *,
    feats: list[str] | None = None,
    horizon: str = "y_500ms",
) -> dict:
    clip = load_target_clip()
    lo, hi = clip.get((inst, horizon), (-np.inf, np.inf))
    print(f"load {inst} horizon={horizon}…", flush=True)
    panels = {d: load_day_panel(inst, d, horizon=horizon) for d in days}

    # When screening the full lists, keep prior behaviour: all paper builders.
    if feats is None:
        print("paper features…", flush=True)
        add_stoikov_micro(panels, days)
        add_cont_ofi_depth(panels, inst)
        add_albers_sweep_proxy(panels, inst)
        cands = _resolve_cands(panels, days, None)
    else:
        cands = _resolve_cands(panels, days, feats)
        print(f"paper features for {cands}…", flush=True)
        _ensure_paper_feats(panels, inst, days, cands)

    sample = panels[days[min(5, len(days) - 1)]]
    fold_cands = ("stoikov_g1", "micro_adj", "gould_ey")
    missing = [c for c in cands if c not in fold_cands and c not in sample.columns]
    if missing:
        raise KeyError(f"columns not on panel after builders: {missing}")

    rows = []
    for feat in cands:
        print(f"  screen {feat}", flush=True)
        day_scores = []
        oos_x, oos_y = [], []
        for i, d in enumerate(days):
            if i < 3:
                continue
            tr = pd.concat([panels[x] for x in days[:i]], ignore_index=True)
            te = panels[d]
            # fold-calibrated features
            if feat == "stoikov_g1":
                x_tr = apply_stoikov_g1_fold(tr, tr)
                x_te = apply_stoikov_g1_fold(tr, te)
                y_tr = tr["y"].to_numpy(float)
            elif feat == "micro_adj":
                x_tr = apply_micro_adj_fold(tr, tr)
                x_te = apply_micro_adj_fold(tr, te)
                y_tr = tr["y"].to_numpy(float)
            elif feat == "gould_ey":
                x_tr = apply_gould_fold(tr, tr)
                x_te = apply_gould_fold(tr, te)
                y_tr = tr["y"].to_numpy(float)
            else:
                x_tr = tr[feat].to_numpy(float)
                x_te = te[feat].to_numpy(float)
                y_tr = tr["y"].to_numpy(float)

            fitted = fit_uni(x_tr, y_tr)
            if fitted is None:
                day_scores.append(float("nan"))
                continue
            b, mx, my = fitted
            yhat = pred_uni(x_te, b, mx, my)
            yc = np.clip(te["y"].to_numpy(float), lo, hi)
            ok = np.isfinite(yhat) & np.isfinite(yc)
            day_scores.append(float(pearson(yhat[ok], yc[ok])) if ok.sum() >= 100 else float("nan"))
            oos_x.append(x_te)
            oos_y.append(yc)

        sc = np.array(day_scores, float)
        sc_ok = sc[np.isfinite(sc)]
        xx = np.concatenate(oos_x) if oos_x else np.array([])
        yy = np.concatenate(oos_y) if oos_y else np.array([])
        m = np.isfinite(xx) & np.isfinite(yy)
        ic_p = float(pearson(xx[m], yy[m])) if m.sum() > 100 else float("nan")
        ic_s = float(spearman(xx[m], yy[m])) if m.sum() > 100 else float("nan")
        mean = float(np.mean(sc_ok)) if len(sc_ok) else float("nan")
        # sign stability: fraction of days with same sign as mean
        if len(sc_ok):
            stab = float(np.mean(np.sign(sc_ok) == np.sign(mean))) if mean != 0 else float("nan")
        else:
            stab = float("nan")
        sig = bool(np.isfinite(mean) and abs(mean) >= thr and stab >= 0.7)
        rows.append(
            {
                "feature": feat,
                "wf_mean": mean,
                "wf_std": float(np.std(sc_ok)) if len(sc_ok) else None,
                "wf_days": [None if not np.isfinite(x) else float(x) for x in sc],
                "oos_pearson": ic_p,
                "oos_spearman": ic_s,
                "sign_stability": stab,
                "significant": sig,
                "family": (
                    "stoikov"
                    if "stoikov" in feat or feat == "micro_adj"
                    else "gould"
                    if "gould" in feat
                    else "cont_ofi"
                    if feat.startswith(("ofi_", "bid_of", "ask_of"))
                    else "albers_sweep"
                    if "sweep" in feat
                    else "lucchese_vol"
                    if "vol_side" in feat
                    else "boltzmann"
                    if "boltz" in feat
                    else "vpin"
                    if feat == "vpin"
                    else "vb_tfi"
                    if "vb_tfi" in feat
                    else "vwap_mid"
                    if "vwap_mid" in feat
                    else "phi_cnt"
                    if "phi_cnt" in feat
                    else "sq_micro"
                    if "sq_micro" in feat
                    else "q_depl"
                    if feat == "q_depl" or feat.startswith("q_depl")
                    else "rel_liq"
                    if feat == "rel_liq" or feat.startswith("rel_liq")
                    else "liq_side"
                    if feat == "liq_side" or feat.startswith("liq_side")
                    else "sweep_px"
                    if feat == "sweep_px" or feat.startswith("sweep_px")
                    else "sign_age"
                    if feat == "sign_age" or feat.startswith("sign_age")
                    else "size_gap"
                    if feat == "size_gap" or feat.startswith("size_gap")
                    else "usdc_prem"
                    if feat == "usdc_prem_bp" or feat.startswith("usdc_prem")
                    else "leader"
                    if feat == "imb_x_lead"
                    else "cross_btc"
                    if feat.startswith("btc_")
                    else "tfi"
                    if "tfi" in feat
                    else "leader"
                    if "leader" in feat
                    else "gap_kf"
                    if feat.startswith(("gap", "div", "kf_"))
                    else "book"
                ),
            }
        )

    rows.sort(key=lambda r: (-(abs(r["wf_mean"]) if np.isfinite(r["wf_mean"]) else -1), r["feature"]))
    return {
        "instrument": inst,
        "horizon": horizon,
        "days": days,
        "threshold": thr,
        "feats": feats,
        "n_features": len(rows),
        "n_significant": sum(1 for r in rows if r["significant"]),
        "results": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="BTCUSDT_PERP")
    ap.add_argument("--days", nargs="+", default=DAYS_DEFAULT)
    ap.add_argument("--threshold", type=float, default=0.05, help="|WF mean Pearson| for SIGNIFICANT")
    ap.add_argument(
        "--feat",
        action="append",
        default=None,
        help="Screen only this column (repeatable). Default: all CORE_CANDS + PAPER_CANDS present on panel.",
    )
    ap.add_argument("--horizon", default="y_500ms", choices=["y_500ms", "y_2s", "y_8s"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = screen(
        args.instrument,
        args.days,
        args.threshold,
        feats=args.feat,
        horizon=args.horizon,
    )
    if args.feat and len(args.feat) == 1:
        default_json = ROOT / "outputs" / "paper_seq" / f"uni_screen_{args.instrument}_{args.feat[0]}.json"
        default_md = ROOT / "reports" / f"FEATURE_UNI_SCREEN_{args.feat[0]}.md"
    else:
        default_json = ROOT / "outputs" / "paper_seq" / f"uni_screen_{args.instrument}.json"
        default_md = ROOT / "reports" / "FEATURE_UNI_SCREEN.md"
    path = Path(args.out or default_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    # markdown table — single-feat runs write a side file so the full table is not wiped
    md_path = default_md
    lines = [
        f"# Univariate feature screen — {args.instrument}",
        "",
        f"Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson (`{args.horizon}`), "
        f"days {args.days[0]}..{args.days[-1]}.",
        f"SIGNIFICANT if `|wf_mean| ≥ {args.threshold}` and sign_stability ≥ 0.7.",
        f"Scope: {'feats ' + ', '.join(args.feat) if args.feat else 'full CORE+PAPER'}.",
        "",
        f"Significant: **{out['n_significant']}** / {out['n_features']}",
        "",
        "| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |",
        "|---------|---------|-------------|--------------|------|-----|--------|",
    ]
    for r in out["results"]:
        lines.append(
            f"| `{r['feature']}` | {r['wf_mean']:.4f} | {r['oos_pearson']:.4f} | "
            f"{r['oos_spearman']:.4f} | {r['sign_stability']:.2f} | "
            f"{'YES' if r['significant'] else 'no'} | {r['family']} |"
        )
    lines.append("")
    lines.append("## Significant only (candidates for later group tests)")
    lines.append("")
    for r in out["results"]:
        if r["significant"]:
            lines.append(f"- `{r['feature']}` — WF {r['wf_mean']:.4f}, spearman {r['oos_spearman']:.4f}")
    lines.append("")
    lines.append(f"JSON: `{path}`")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(f"wrote {md_path}")
    print(f"significant {out['n_significant']}/{out['n_features']}")
    for r in out["results"]:
        if r["significant"]:
            print(f"  YES {r['feature']:24s} wf={r['wf_mean']:+.4f}")


if __name__ == "__main__":
    main()
