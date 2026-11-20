"""Sequential paper-idea integration: build features → WF → append experiment log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from full_eda_io import load_target_clip, pearson, read_cols, mid_from_px  # noqa: E402
from nf_features import FEATURE_SETS, load_day_panel, cont_ofi  # noqa: E402

LOG = ROOT / "reports" / "HFT_PAPER_EXPERIMENTS.md"
OUT_DIR = ROOT / "outputs" / "paper_seq"
DAYS_DEFAULT = [f"2026-01-{d:02d}" for d in range(12, 26)]
BASE_PACK = FEATURE_SETS["hl_resid"]


def fit(X: np.ndarray, y: np.ndarray):
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 5:
        return None
    mx, my = X.mean(0), float(y.mean())
    b = np.linalg.lstsq(X - mx, y - my, rcond=None)[0]
    return b, mx, my


def predict(X, b, mx, my):
    return np.where(np.isfinite(X), X - mx, 0.0) @ b + my


def wf_scores(
    panels: dict[str, pd.DataFrame],
    cols: list[str],
    inst: str,
    days: list[str],
    min_train=3,
    horizon: str = "y_500ms",
):
    lo, hi = load_target_clip().get((inst, horizon), (-np.inf, np.inf))
    scores = []
    for i, d in enumerate(days):
        if i < min_train:
            continue
        need = cols + ["y"]
        for c in cols:
            if c not in panels[d].columns:
                return None, f"missing {c}"
        tr = pd.concat([panels[x][need] for x in days[:i]], ignore_index=True)
        te = panels[d]
        fitted = fit(tr[cols].to_numpy(float), tr["y"].to_numpy(float))
        if fitted is None:
            return None, "fit fail"
        b, mx, my = fitted
        yhat = predict(te[cols].to_numpy(float), b, mx, my)
        yc = np.clip(te["y"].to_numpy(float), lo, hi)
        ok = np.isfinite(yhat) & np.isfinite(yc)
        scores.append(float(pearson(yhat[ok], yc[ok])))
    return scores, None


def verdict(mean: float, baseline: float) -> str:
    d = mean - baseline
    if d >= 0.003:
        return "OK"
    if d >= 0.001:
        return "WEAK"
    return "NOT OK"


def append_log(block: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("# Paper integration log\n\n")
    with LOG.open("a") as f:
        f.write("\n" + block.strip() + "\n")


# ----- Paper feature builders (mutate panels in place) -----


def add_stoikov_micro(panels: dict[str, pd.DataFrame], days: list[str], n_bins: int = 10) -> None:
    """Stoikov G1: E[Δmid at next mid-change | I-bin], calibrated causally per WF day later.

    Here we precompute I and next mid-change Δ; g lookup applied inside WF with train-only bins.
    Stores: imb_I in [0,1], next_dmid_bp (NaN if no future change in day — only for calibration).
    """
    for d, df in panels.items():
        imb = df["imb"].to_numpy(float)
        # I = Qb/(Qb+Qa) = (1+imb)/2
        I = (1.0 + imb) * 0.5
        mid = df["mid"].to_numpy(float)
        n = len(mid)
        next_dmid = np.full(n, np.nan)
        # next mid-change looking forward within day (for calibration only; feature uses train g)
        last = n - 1
        for t in range(n - 2, -1, -1):
            if not np.isfinite(mid[t]) or not np.isfinite(mid[t + 1]):
                next_dmid[t] = next_dmid[t + 1] if t + 1 < n else np.nan
                continue
            if mid[t + 1] != mid[t]:
                next_dmid[t] = (mid[t + 1] / mid[t] - 1.0) * 1e4  # bp
            else:
                next_dmid[t] = next_dmid[t + 1]
        df["stoikov_I"] = I
        df["stoikov_next_dmid_bp"] = next_dmid
        # placeholder; filled per WF fold
        df["stoikov_g1"] = np.nan


def apply_stoikov_g1_fold(tr: pd.DataFrame, te: pd.DataFrame, n_bins: int = 10) -> np.ndarray:
    """Calibrate g1 on train mid-change rows; map te I → g1."""
    I_tr = tr["stoikov_I"].to_numpy(float)
    y_tr = tr["stoikov_next_dmid_bp"].to_numpy(float)
    ok = np.isfinite(I_tr) & np.isfinite(y_tr)
    # only rows that are about to see a mid change soon — use all with finite next_dmid
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    g = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = ok & (I_tr >= edges[b]) & (I_tr < edges[b + 1] if b < n_bins - 1 else I_tr <= edges[b + 1])
        if m.sum() >= 50:
            g[b] = float(np.mean(y_tr[m]))
    # fill empty bins by interpolation
    idx = np.where(np.isfinite(g))[0]
    if len(idx) == 0:
        return np.full(len(te), np.nan)
    g = np.interp(np.arange(n_bins), idx, g[idx])
    I_te = te["stoikov_I"].to_numpy(float)
    bins = np.clip(np.digitize(I_te, edges) - 1, 0, n_bins - 1)
    out = g[bins]
    out[~np.isfinite(I_te)] = np.nan
    return out


def add_cont_ofi_depth(panels: dict[str, pd.DataFrame], instrument: str) -> None:
    """Cont: raw OFI, OFI/depth, multi-horizon cum OFI, separate bid/ask OF (Kolm-style)."""
    for d, df in panels.items():
        book = read_cols(
            "book",
            instrument,
            d,
            columns=["ts", "bid_px", "ask_px", "bid_qty", "ask_qty"],
        ).sort_values("ts")
        bp = book["bid_px"].to_numpy()
        ap = book["ask_px"].to_numpy()
        bq = book["bid_qty"].to_numpy(float)
        aq = book["ask_qty"].to_numpy(float)
        ofi = cont_ofi(bp, ap, bq, aq)
        depth = bq + aq
        ofi_depth = np.full(len(ofi), np.nan)
        np.divide(ofi, depth, out=ofi_depth, where=depth > 0)
        # separate bid/ask OF pieces (Cont decomposition)
        n = len(bp)
        bid_of = np.full(n, np.nan)
        ask_of = np.full(n, np.nan)
        if n >= 2:
            bp0, bp1 = bp[:-1], bp[1:]
            ap0, ap1 = ap[:-1], ap[1:]
            bq0, bq1 = bq[:-1], bq[1:]
            aq0, aq1 = aq[:-1], aq[1:]
            bid_of[1:] = np.where(bp1 > bp0, bq1, np.where(bp1 < bp0, -bq0, bq1 - bq0))
            ask_of[1:] = np.where(ap1 < ap0, -aq1, np.where(ap1 > ap0, aq0, aq0 - aq1))
        # cum horizons on 500ms grid: 1,2,4,8 steps
        c = np.nancumsum(np.nan_to_num(ofi, nan=0.0))
        feat = pd.DataFrame(
            {
                "ts": book["ts"].to_numpy(np.int64),
                "ofi_raw": ofi,
                "ofi_depth": ofi_depth,
                "bid_of": bid_of,
                "ask_of": ask_of,
            }
        )
        for h, name in [(2, "ofi_h2"), (4, "ofi_h4"), (8, "ofi_h8")]:
            rolled = c.copy()
            rolled[h:] = c[h:] - c[:-h]
            rolled[:h] = np.nan
            feat[name] = rolled
        # normalize cum by contemporaneous depth
        for name in ("ofi_h2", "ofi_h4", "ofi_h8"):
            v = feat[name].to_numpy(float)
            nd = np.full(len(v), np.nan)
            np.divide(v, depth, out=nd, where=depth > 0)
            feat[name + "_d"] = nd
        out = pd.merge_asof(
            df[["ts"]].sort_values("ts"),
            feat.sort_values("ts"),
            on="ts",
            direction="backward",
        )
        for c in feat.columns:
            if c != "ts":
                df[c] = out[c].to_numpy()


def add_gould_local_imb(panels: dict[str, pd.DataFrame], days: list[str]) -> None:
    """Gould-style: store imb; fold-wise isotonic/bin E[y|imb] as feature (train-only)."""
    for df in panels.values():
        df["gould_ey"] = np.nan  # filled per fold


def apply_gould_fold(tr: pd.DataFrame, te: pd.DataFrame, n_bins: int = 20) -> np.ndarray:
    x = tr["imb"].to_numpy(float)
    y = tr["y"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    edges = np.quantile(x[ok], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.full(len(te), np.nan)
    # digitize
    g = []
    centers = []
    for b in range(len(edges) - 1):
        m = ok & (x >= edges[b]) & (x < edges[b + 1] if b < len(edges) - 2 else x <= edges[b + 1])
        if m.sum() >= 30:
            g.append(float(np.mean(y[m])))
            centers.append(0.5 * (edges[b] + edges[b + 1]))
    if len(g) < 2:
        return np.full(len(te), np.nan)
    # map te via interpolation on bin centers (monotone-ish nonparametric)
    return np.interp(te["imb"].to_numpy(float), centers, g, left=g[0], right=g[-1])


def add_albers_sweep_proxy(panels: dict[str, pd.DataFrame], instrument: str) -> None:
    """Albers sweep IMB proxy: cost in bp to eat top-K depth buckets vs L1.

    Without L2 prices, approximate walk using cumulative depth near buckets and spread.
    IMBa ≈ (extra cost)/mid ; use depth imbalance of near vs mid buckets as proxy pair.
    Better proxy: notional N = median(L1 size); frac of N filled by L1; remainder assumed 1-tick worse.
    """
    for d, df in panels.items():
        book = read_cols(
            "book",
            instrument,
            d,
            columns=[
                "ts",
                "bid_px",
                "ask_px",
                "bid_qty",
                "ask_qty",
                *[f"bid_v{k:02d}" for k in range(1, 6)],
                *[f"ask_v{k:02d}" for k in range(1, 6)],
            ],
        ).sort_values("ts")
        mid = mid_from_px(book["bid_px"].to_numpy(), book["ask_px"].to_numpy())
        spread = book["ask_px"].to_numpy(float) - book["bid_px"].to_numpy(float)
        bq = book["bid_qty"].to_numpy(float)
        aq = book["ask_qty"].to_numpy(float)
        # N = median L1 size (per side avg) — Albers uses median liquidity in top 5bp
        n_ref = float(np.nanmedian(0.5 * (bq + aq)))
        n_ref = max(n_ref, 1e-12)

        def sweep_cost(qty_l1, depth_near, side_spread_sign):
            # fraction beyond L1 charged ~ 0.5..2.5 ticks using bucket index as distance proxy
            near = np.zeros(len(book))
            for k in range(1, 4):
                near = near + book[f"{'bid' if side_spread_sign < 0 else 'ask'}_v{k:02d}"].to_numpy(float)
            # cost_ask: buying N — use ask side
            return None

        # Ask sweep (buy N): consume ask_qty then ask_v01.. at +1,+2,.. ticks approx
        rem = np.full(len(book), n_ref)
        cost = np.zeros(len(book))
        # level 0 = L1 ask at 0 extra
        take0 = np.minimum(rem, aq)
        rem = rem - take0
        for k in range(1, 6):
            lvl = book[f"ask_v{k:02d}"].to_numpy(float)
            take = np.minimum(rem, lvl)
            cost = cost + take * k  # k ticks extra (proxy)
            rem = rem - take
        # leftover: assume +6 ticks
        cost = cost + rem * 6
        imb_a = np.full(len(book), np.nan)
        np.divide(cost, n_ref * np.where(mid > 0, mid, np.nan), out=imb_a, where=(n_ref > 0) & (mid > 0))
        imb_a = imb_a * 1e4  # scale

        rem = np.full(len(book), n_ref)
        cost_b = np.zeros(len(book))
        take0 = np.minimum(rem, bq)
        rem = rem - take0
        for k in range(1, 6):
            lvl = book[f"bid_v{k:02d}"].to_numpy(float)
            take = np.minimum(rem, lvl)
            cost_b = cost_b + take * k
            rem = rem - take
        cost_b = cost_b + rem * 6
        imb_b = np.full(len(book), np.nan)
        np.divide(cost_b, n_ref * np.where(mid > 0, mid, np.nan), out=imb_b, where=(n_ref > 0) & (mid > 0))
        imb_b = imb_b * 1e4

        feat = pd.DataFrame(
            {
                "ts": book["ts"].to_numpy(np.int64),
                "sweep_imba": imb_a,
                "sweep_imbb": imb_b,
                "sweep_imb": imb_a - imb_b,  # Albers uses IMBa - IMBb in regression
            }
        )
        out = pd.merge_asof(df[["ts"]].sort_values("ts"), feat, on="ts", direction="backward")
        df["sweep_imba"] = out["sweep_imba"].to_numpy()
        df["sweep_imbb"] = out["sweep_imbb"].to_numpy()
        df["sweep_imb"] = out["sweep_imb"].to_numpy()


def add_cross_btc_leader(panels_alt: dict[str, pd.DataFrame], btc_panels: dict[str, pd.DataFrame]) -> None:
    """Cross-impact: BTC hl signals as features for alt panels (same timestamps)."""
    for d, df in panels_alt.items():
        if d not in btc_panels:
            continue
        b = btc_panels[d][["ts", "imb", "ret_1_bp", "tfi_s100", "kf_innov"]].rename(
            columns={
                "imb": "btc_imb",
                "ret_1_bp": "btc_ret_1_bp",
                "tfi_s100": "btc_tfi_s100",
                "kf_innov": "btc_kf_innov",
            }
        )
        out = pd.merge_asof(df[["ts"]].sort_values("ts"), b.sort_values("ts"), on="ts", direction="backward")
        for c in ("btc_imb", "btc_ret_1_bp", "btc_tfi_s100", "btc_kf_innov"):
            df[c] = out[c].to_numpy()


def run_fold_features(panels, days, inst, paper: str):
    """Apply fold-dependent features (Stoikov/Gould) during WF; return dict name→scores."""
    results = {}
    lo, hi = load_target_clip().get((inst, "y_500ms"), (-np.inf, np.inf))

    experiments = []
    if paper in ("stoikov", "all"):
        experiments += [
            ("stoikov_g1_only", ["stoikov_g1"]),
            ("stoikov_imb_g1", ["imb", "stoikov_g1"]),
            ("hl_resid_stoikov", BASE_PACK + ["stoikov_g1"]),
        ]
    if paper in ("cont", "all"):
        experiments += [
            ("cont_ofi_depth", ["ofi_depth"]),
            ("cont_ofi_multi", ["ofi_depth", "ofi_h2_d", "ofi_h4_d", "ofi_h8_d"]),
            ("cont_bidask_of", ["bid_of", "ask_of"]),
            ("hl_resid_ofi_depth", BASE_PACK + ["ofi_depth"]),
            ("hl_resid_ofi_multi", BASE_PACK + ["ofi_depth", "ofi_h4_d", "bid_of", "ask_of"]),
        ]
    if paper in ("gould", "all"):
        experiments += [
            ("gould_ey_only", ["gould_ey"]),
            ("hl_resid_gould", BASE_PACK + ["gould_ey"]),
        ]
    if paper in ("albers_sweep", "all"):
        experiments += [
            ("sweep_imb", ["sweep_imb"]),
            ("sweep_pair", ["sweep_imba", "sweep_imbb"]),
            ("hl_resid_sweep", BASE_PACK + ["sweep_imb"]),
        ]

    # baselines without fold features
    static = {}
    if paper in ("stoikov", "cont", "gould", "albers_sweep", "all"):
        for name, cols in [
            ("imb", ["imb"]),
            ("hl_resid", list(BASE_PACK)),
        ]:
            sc, err = wf_scores(panels, cols, inst, days)
            if err:
                static[name] = {"error": err}
            else:
                static[name] = {"mean": float(np.mean(sc)), "std": float(np.std(sc)), "days": sc}

    fold_names = [e for e in experiments if "stoikov" in e[0] or "gould" in e[0]]
    static_exps = [e for e in experiments if e not in fold_names]

    for name, cols in static_exps:
        sc, err = wf_scores(panels, cols, inst, days)
        if err:
            results[name] = {"error": err}
        else:
            results[name] = {"mean": float(np.mean(sc)), "std": float(np.std(sc)), "days": sc, "cols": cols}

    # fold-wise
    need_stoikov = any("stoikov" in n for n, _ in fold_names)
    need_gould = any("gould" in n for n, _ in fold_names)
    fold_scores = {n: [] for n, _ in fold_names}

    for i, d in enumerate(days):
        if i < 3:
            continue
        tr = pd.concat([panels[x] for x in days[:i]], ignore_index=True)
        te = panels[d]
        tr2 = tr.copy()
        te2 = te.copy()
        if need_stoikov:
            # calibrate g1 on train only; apply to train+test
            tr2["stoikov_g1"] = apply_stoikov_g1_fold(tr, tr)
            te2["stoikov_g1"] = apply_stoikov_g1_fold(tr, te)
        if need_gould:
            tr2["gould_ey"] = apply_gould_fold(tr, tr)
            te2["gould_ey"] = apply_gould_fold(tr, te)

        for name, cols in fold_names:
            fitted = fit(tr2[cols].to_numpy(float), tr2["y"].to_numpy(float))
            if fitted is None:
                fold_scores[name].append(float("nan"))
                continue
            b, mx, my = fitted
            yhat = predict(te2[cols].to_numpy(float), b, mx, my)
            yc = np.clip(te2["y"].to_numpy(float), lo, hi)
            ok = np.isfinite(yhat) & np.isfinite(yc)
            fold_scores[name].append(float(pearson(yhat[ok], yc[ok])))

    for name, cols in fold_names:
        sc = [x for x in fold_scores[name] if np.isfinite(x)]
        results[name] = {
            "mean": float(np.mean(sc)) if sc else None,
            "std": float(np.std(sc)) if sc else None,
            "days": fold_scores[name],
            "cols": cols,
        }

    return static, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=["stoikov", "cont", "gould", "albers_sweep", "cross", "all"], default="stoikov")
    ap.add_argument("--instrument", default="BTCUSDT_PERP")
    ap.add_argument("--days", nargs="+", default=DAYS_DEFAULT)
    args = ap.parse_args()
    days = args.days
    inst = args.instrument
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"paper={args.paper} load {inst} {days[0]}..{days[-1]}", flush=True)
    panels = {d: load_day_panel(inst, d) for d in days}

    if args.paper in ("stoikov", "all"):
        print("add Stoikov…", flush=True)
        add_stoikov_micro(panels, days)
    if args.paper in ("cont", "all"):
        print("add Cont/Kolm OF…", flush=True)
        add_cont_ofi_depth(panels, inst)
    if args.paper in ("gould", "all"):
        print("add Gould hooks…", flush=True)
        add_gould_local_imb(panels, days)
    if args.paper in ("albers_sweep", "all"):
        print("add Albers sweep proxy…", flush=True)
        add_albers_sweep_proxy(panels, inst)

    if args.paper == "cross":
        # ETH with BTC cross features
        print("cross-impact ETH←BTC", flush=True)
        eth_days = days
        eth = {d: load_day_panel("ETHUSDT_PERP", d) for d in eth_days}
        add_cross_btc_leader(eth, panels)
        base_eth = ["imb", "imb3", "leader_ret_1_bp", "kf_innov", "tfi_100", "tfi_s100", "d_imb"]
        # ensure eth has same base cols
        packs = {
            "eth_imb": ["imb"],
            "eth_hl_resid": [c for c in base_eth if c in eth[days[5]].columns],
            "eth_hl_resid_btc": [c for c in base_eth if c in eth[days[5]].columns]
            + ["btc_imb", "btc_ret_1_bp", "btc_tfi_s100"],
        }
        out = {}
        for name, cols in packs.items():
            sc, err = wf_scores(eth, cols, "ETHUSDT_PERP", days)
            out[name] = {"error": err} if err else {"mean": float(np.mean(sc)), "std": float(np.std(sc)), "days": sc, "cols": cols}
        path = OUT_DIR / "cross_eth.json"
        path.write_text(json.dumps(out, indent=2))
        eth_base = out.get("eth_hl_resid", {}).get("mean") or 0
        lines = ["## Cross-impact (BTC→ETH) — Sirignano/Albers family", ""]
        for k, v in out.items():
            if "mean" in v:
                ver = verdict(v["mean"], out["eth_imb"]["mean"] if k != "eth_imb" else v["mean"] - 0.001)
                if k == "eth_hl_resid_btc":
                    ver = verdict(v["mean"], eth_base)
                lines.append(f"- **{k}**: mean={v['mean']:.4f} → **{ver}**")
            else:
                lines.append(f"- **{k}**: ERROR {v}")
        # relative to eth_hl_resid for btc add
        if "eth_hl_resid_btc" in out and "eth_hl_resid" in out and "mean" in out["eth_hl_resid_btc"]:
            dlt = out["eth_hl_resid_btc"]["mean"] - out["eth_hl_resid"]["mean"]
            lines.append(f"- Δ vs eth_hl_resid: {dlt:+.4f} → **{verdict(out['eth_hl_resid_btc']['mean'], out['eth_hl_resid']['mean'])}**")
        lines.append(f"- JSON: `{path}`")
        append_log("\n".join(lines))
        print("\n".join(lines))
        return

    static, results = run_fold_features(panels, days, inst, args.paper)
    payload = {"paper": args.paper, "static_baselines": static, "results": results}
    path = OUT_DIR / f"{args.paper}_{inst}.json"
    path.write_text(json.dumps(payload, indent=2))

    base_imb = static.get("imb", {}).get("mean")
    base_hl = static.get("hl_resid", {}).get("mean")
    title = {
        "stoikov": "## 1. Stoikov micro-price G1",
        "cont": "## 2. Cont / Kolm order-flow",
        "gould": "## 3. Gould local E[y|imb]",
        "albers_sweep": "## 4. Albers sweep-IMB proxy",
        "all": "## Batch",
    }[args.paper]
    lines = [title, f"Baseline imb={base_imb} hl_resid={base_hl}", ""]
    for name, v in results.items():
        if "mean" not in v or v["mean"] is None:
            lines.append(f"- **{name}**: ERROR/empty {v}")
            continue
        # compare adds-on-hl to hl_resid; single features to imb
        if "hl_resid" in name:
            ref, ref_name = base_hl or 0, "hl_resid"
        else:
            ref, ref_name = base_imb or 0, "imb"
        ver = verdict(v["mean"], ref)
        lines.append(
            f"- **{name}**: mean={v['mean']:.4f} (vs {ref_name}={ref:.4f}, Δ={v['mean']-ref:+.4f}) → **{ver}**"
        )
    lines.append(f"- JSON: `{path}`")
    append_log("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
