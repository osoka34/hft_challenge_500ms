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
    """Stoikov G1 / g(I,S): precompute I, spread, next mid-change Δ for fold calibration.

    Stores: stoikov_I in [0,1], stoikov_S (= spread_bp), next_dmid_bp (calibration label only).
    Placeholders ``stoikov_g1`` / ``micro_adj`` filled per WF fold.
    """
    del n_bins  # edges chosen inside apply_*_fold
    for d, df in panels.items():
        imb = df["imb"].to_numpy(float)
        # I = Qb/(Qb+Qa) = (1+imb)/2
        I = (1.0 + imb) * 0.5
        mid = df["mid"].to_numpy(float)
        if "spread_bp" in df.columns:
            S = df["spread_bp"].to_numpy(float)
        else:
            S = np.full(len(df), np.nan)
        n = len(mid)
        next_dmid = np.full(n, np.nan)
        # next mid-change looking forward within day (for calibration only; feature uses train g)
        for t in range(n - 2, -1, -1):
            if not np.isfinite(mid[t]) or not np.isfinite(mid[t + 1]):
                next_dmid[t] = next_dmid[t + 1] if t + 1 < n else np.nan
                continue
            if mid[t + 1] != mid[t]:
                next_dmid[t] = (mid[t + 1] / mid[t] - 1.0) * 1e4  # bp
            else:
                next_dmid[t] = next_dmid[t + 1]
        df["stoikov_I"] = I
        df["stoikov_S"] = S
        df["stoikov_next_dmid_bp"] = next_dmid
        # placeholders; filled per WF fold
        df["stoikov_g1"] = np.nan
        df["micro_adj"] = np.nan


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


def apply_micro_adj_fold(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    n_imb: int = 10,
    n_spread: int = 4,
) -> np.ndarray:
    """Stoikov g(I,S): E[next Δmid bp | I-bin, S-bin] on train; map te (I,S)."""
    I_tr = tr["stoikov_I"].to_numpy(float)
    S_tr = tr["stoikov_S"].to_numpy(float)
    y_tr = tr["stoikov_next_dmid_bp"].to_numpy(float)
    ok = np.isfinite(I_tr) & np.isfinite(S_tr) & np.isfinite(y_tr)
    if ok.sum() < 200:
        return np.full(len(te), np.nan)
    i_edges = np.linspace(0.0, 1.0, n_imb + 1)
    s_ok = S_tr[ok]
    # quantile spread edges on train (stable across ticks / regimes)
    qs = np.linspace(0.0, 1.0, n_spread + 1)
    s_edges = np.unique(np.quantile(s_ok, qs))
    if len(s_edges) < 3:
        s_edges = np.linspace(float(np.min(s_ok)), float(np.max(s_ok)) + 1e-12, min(n_spread, 2) + 1)
    n_s = len(s_edges) - 1
    g = np.full((n_imb, n_s), np.nan)
    for bi in range(n_imb):
        i_m = (I_tr >= i_edges[bi]) & (
            I_tr < i_edges[bi + 1] if bi < n_imb - 1 else I_tr <= i_edges[bi + 1]
        )
        for bs in range(n_s):
            s_m = (S_tr >= s_edges[bs]) & (
                S_tr < s_edges[bs + 1] if bs < n_s - 1 else S_tr <= s_edges[bs + 1]
            )
            m = ok & i_m & s_m
            if m.sum() >= 30:
                g[bi, bs] = float(np.mean(y_tr[m]))
    # fill empty cells: prefer same-I interp across S, then global
    for bi in range(n_imb):
        row = g[bi]
        idx = np.where(np.isfinite(row))[0]
        if len(idx) == 0:
            continue
        if len(idx) < n_s:
            g[bi] = np.interp(np.arange(n_s), idx, row[idx])
    flat = g.ravel()
    idx = np.where(np.isfinite(flat))[0]
    if len(idx) == 0:
        return np.full(len(te), np.nan)
    if len(idx) < flat.size:
        # nearest finite via simple fill from row means then global mean
        row_means = np.nanmean(g, axis=1)
        glob = float(np.nanmean(flat))
        for bi in range(n_imb):
            for bs in range(n_s):
                if not np.isfinite(g[bi, bs]):
                    g[bi, bs] = row_means[bi] if np.isfinite(row_means[bi]) else glob
    I_te = te["stoikov_I"].to_numpy(float)
    S_te = te["stoikov_S"].to_numpy(float)
    bi = np.clip(np.digitize(I_te, i_edges) - 1, 0, n_imb - 1)
    bs = np.clip(np.digitize(S_te, s_edges) - 1, 0, n_s - 1)
    out = g[bi, bs]
    out[~(np.isfinite(I_te) & np.isfinite(S_te))] = np.nan
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


def vol_side_imb_buckets(
    bid_vs: list[np.ndarray] | tuple[np.ndarray, ...],
    ask_vs: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
    """Lucchese volume-strip side imbalance: mean per-bucket (b−a)/(b+a).

    Equal weight per distance j (unlike volume-weighted ``depth_imb_*``).
    Empty both-side buckets are skipped in the mean.
    """
    if len(bid_vs) != len(ask_vs) or not bid_vs:
        raise ValueError("bid_vs and ask_vs must be same non-empty length")
    n = len(bid_vs[0])
    num = np.zeros(n, dtype=np.float64)
    den = np.zeros(n, dtype=np.float64)
    for bv, av in zip(bid_vs, ask_vs):
        b = np.asarray(bv, dtype=np.float64)
        a = np.asarray(av, dtype=np.float64)
        s = b + a
        ok = s > 0
        imb = np.zeros(n, dtype=np.float64)
        np.divide(b - a, s, out=imb, where=ok)
        num = num + np.where(ok, imb, 0.0)
        den = den + ok.astype(np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def add_lucchese_vol_side(panels: dict[str, pd.DataFrame], instrument: str) -> None:
    """Lucchese volume representation → scalar ``vol_side_imb`` (buckets 1..5)."""
    for d, df in panels.items():
        book = read_cols(
            "book",
            instrument,
            d,
            columns=[
                "ts",
                *[f"bid_v{k:02d}" for k in range(1, 6)],
                *[f"ask_v{k:02d}" for k in range(1, 6)],
            ],
        ).sort_values("ts")
        bid_vs = [book[f"bid_v{k:02d}"].to_numpy(float) for k in range(1, 6)]
        ask_vs = [book[f"ask_v{k:02d}"].to_numpy(float) for k in range(1, 6)]
        feat = pd.DataFrame(
            {
                "ts": book["ts"].to_numpy(np.int64),
                "vol_side_imb": vol_side_imb_buckets(bid_vs, ask_vs),
            }
        )
        out = pd.merge_asof(df[["ts"]].sort_values("ts"), feat, on="ts", direction="backward")
        df["vol_side_imb"] = out["vol_side_imb"].to_numpy()


def add_lucchese_vol_side_far(panels: dict[str, pd.DataFrame], instrument: str) -> None:
    """Lucchese volume representation → ``vol_side_imb_far`` (buckets 6..11)."""
    for d, df in panels.items():
        book = read_cols(
            "book",
            instrument,
            d,
            columns=[
                "ts",
                *[f"bid_v{k:02d}" for k in range(6, 12)],
                *[f"ask_v{k:02d}" for k in range(6, 12)],
            ],
        ).sort_values("ts")
        bid_vs = [book[f"bid_v{k:02d}"].to_numpy(float) for k in range(6, 12)]
        ask_vs = [book[f"ask_v{k:02d}"].to_numpy(float) for k in range(6, 12)]
        feat = pd.DataFrame(
            {
                "ts": book["ts"].to_numpy(np.int64),
                "vol_side_imb_far": vol_side_imb_buckets(bid_vs, ask_vs),
            }
        )
        out = pd.merge_asof(df[["ts"]].sort_values("ts"), feat, on="ts", direction="backward")
        df["vol_side_imb_far"] = out["vol_side_imb_far"].to_numpy()


def boltzmann_gap(
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    bid_qty: np.ndarray,
    ask_qty: np.ndarray,
    *,
    beta: float = 1.0,
) -> np.ndarray:
    """Rola equilibrium/Boltzmann price minus mid, relative (β=1 → Def. 3.3).

    P(β) = (e^{-β q_b} P_b + e^{-β q_a} P_a) / (e^{-β q_b} + e^{-β q_a}),
    q_b = Q_b/(Q_b+Q_a), q_a = 1 - q_b; gap = (P - mid) / mid.
    """
    bp = np.asarray(bid_px, dtype=np.float64)
    ap = np.asarray(ask_px, dtype=np.float64)
    bq = np.asarray(bid_qty, dtype=np.float64)
    aq = np.asarray(ask_qty, dtype=np.float64)
    mid = mid_from_px(bp, ap)
    tot = bq + aq
    out = np.full(len(bp), np.nan, dtype=np.float64)
    ok = (tot > 0) & np.isfinite(bp) & np.isfinite(ap) & np.isfinite(mid) & (mid > 0)
    if not np.any(ok):
        return out
    qb = np.zeros(len(bp), dtype=np.float64)
    np.divide(bq, tot, out=qb, where=ok)
    qa = 1.0 - qb
    wb = np.exp(-beta * qb)
    wa = np.exp(-beta * qa)
    p_b = (wb * bp + wa * ap) / (wb + wa)
    np.divide(p_b - mid, mid, out=out, where=ok)
    return out


def add_boltzmann_gap(panels: dict[str, pd.DataFrame], instrument: str) -> None:
    """Rola Boltzmann equilibrium price gap → ``boltz_gap`` (L1 book as-of)."""
    for d, df in panels.items():
        book = read_cols(
            "book",
            instrument,
            d,
            columns=["ts", "bid_px", "ask_px", "bid_qty", "ask_qty"],
        ).sort_values("ts")
        feat = pd.DataFrame(
            {
                "ts": book["ts"].to_numpy(np.int64),
                "boltz_gap": boltzmann_gap(
                    book["bid_px"].to_numpy(),
                    book["ask_px"].to_numpy(),
                    book["bid_qty"].to_numpy(),
                    book["ask_qty"].to_numpy(),
                ),
            }
        )
        out = pd.merge_asof(df[["ts"]].sort_values("ts"), feat, on="ts", direction="backward")
        df["boltz_gap"] = out["boltz_gap"].to_numpy()


def vpin_from_trades(
    ts: np.ndarray,
    buy_qty: np.ndarray,
    sell_qty: np.ndarray,
    *,
    bucket_vol: float,
    n_buckets: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Easley/Fang VPIN on aggressor volumes: mean |V^B−V^S|/(n V) over last n buckets.

    Returns (bucket_end_ts, vpin_at_bucket) for completed buckets only. Fixed ``bucket_vol``
    (not day-total/50) so a day cut at T does not resize V.
    """
    if bucket_vol <= 0:
        raise ValueError("bucket_vol must be positive")
    buy = np.nan_to_num(np.asarray(buy_qty, dtype=np.float64), nan=0.0)
    sell = np.nan_to_num(np.asarray(sell_qty, dtype=np.float64), nan=0.0)
    tsa = np.asarray(ts, dtype=np.int64)
    end_ts: list[int] = []
    oi: list[float] = []
    vb = vs = 0.0
    filled = 0.0
    for i in range(len(tsa)):
        b_left, s_left = float(buy[i]), float(sell[i])
        while b_left + s_left > 1e-18:
            need = bucket_vol - filled
            bar = b_left + s_left
            take = min(need, bar)
            if bar > 0:
                frac = take / bar
                vb += b_left * frac
                vs += s_left * frac
                b_left -= b_left * frac
                s_left -= s_left * frac
            filled += take
            if filled >= bucket_vol - 1e-12:
                oi.append(abs(vb - vs) / bucket_vol)
                end_ts.append(int(tsa[i]))
                vb = vs = 0.0
                filled = 0.0
    if not oi:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    oi_a = np.asarray(oi, dtype=np.float64)
    c = np.cumsum(oi_a)
    vpin = np.empty_like(oi_a)
    for k in range(len(oi_a)):
        lo = max(0, k + 1 - n_buckets)
        width = k + 1 - lo
        total = c[k] - (c[lo - 1] if lo > 0 else 0.0)
        vpin[k] = total / width
    return np.asarray(end_ts, dtype=np.int64), vpin


def add_vpin(panels: dict[str, pd.DataFrame], instrument: str, *, bucket_vol: float = 500.0, n_buckets: int = 50) -> None:
    """Volume-clock VPIN → ``vpin`` (trades as-of onto the grid)."""
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("trades", instrument, d):
            df["vpin"] = np.nan
            continue
        tr = read_cols(
            "trades",
            instrument,
            d,
            columns=["ts", "buy_qty", "sell_qty"],
        ).sort_values("ts")
        if tr.empty:
            df["vpin"] = np.nan
            continue
        end_ts, vpin = vpin_from_trades(
            tr["ts"].to_numpy(np.int64),
            tr["buy_qty"].to_numpy(float),
            tr["sell_qty"].to_numpy(float),
            bucket_vol=bucket_vol,
            n_buckets=n_buckets,
        )
        if len(end_ts) == 0:
            df["vpin"] = np.nan
            continue
        feat = pd.DataFrame({"ts": end_ts, "vpin": vpin})
        out = pd.merge_asof(df[["ts"]].sort_values("ts"), feat, on="ts", direction="backward")
        df["vpin"] = out["vpin"].to_numpy()


def vb_tfi_from_trades(
    ts: np.ndarray,
    buy_qty: np.ndarray,
    sell_qty: np.ndarray,
    *,
    bucket_vol: float,
    n_buckets: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed volume-bucket TFI: rolling mean of (V^B−V^S)/V over last n completed buckets.

    Albers-style aggressor flow on an Easley/Fang volume clock (fixed ``bucket_vol``).
    """
    if bucket_vol <= 0:
        raise ValueError("bucket_vol must be positive")
    buy = np.nan_to_num(np.asarray(buy_qty, dtype=np.float64), nan=0.0)
    sell = np.nan_to_num(np.asarray(sell_qty, dtype=np.float64), nan=0.0)
    tsa = np.asarray(ts, dtype=np.int64)
    end_ts: list[int] = []
    signed: list[float] = []
    vb = vs = 0.0
    filled = 0.0
    for i in range(len(tsa)):
        b_left, s_left = float(buy[i]), float(sell[i])
        while b_left + s_left > 1e-18:
            need = bucket_vol - filled
            bar = b_left + s_left
            take = min(need, bar)
            if bar > 0:
                frac = take / bar
                vb += b_left * frac
                vs += s_left * frac
                b_left -= b_left * frac
                s_left -= s_left * frac
            filled += take
            if filled >= bucket_vol - 1e-12:
                signed.append((vb - vs) / bucket_vol)
                end_ts.append(int(tsa[i]))
                vb = vs = 0.0
                filled = 0.0
    if not signed:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    s_a = np.asarray(signed, dtype=np.float64)
    c = np.cumsum(s_a)
    out = np.empty_like(s_a)
    for k in range(len(s_a)):
        lo = max(0, k + 1 - n_buckets)
        width = k + 1 - lo
        total = c[k] - (c[lo - 1] if lo > 0 else 0.0)
        out[k] = total / width
    return np.asarray(end_ts, dtype=np.int64), out


def add_vb_tfi(
    panels: dict[str, pd.DataFrame],
    instrument: str,
    *,
    bucket_vol: float = 500.0,
    n_buckets: int = 10,
) -> None:
    """Volume-clock signed TFI → ``vb_tfi`` (trades as-of onto the grid)."""
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("trades", instrument, d):
            df["vb_tfi"] = np.nan
            continue
        tr = read_cols(
            "trades",
            instrument,
            d,
            columns=["ts", "buy_qty", "sell_qty"],
        ).sort_values("ts")
        if tr.empty:
            df["vb_tfi"] = np.nan
            continue
        end_ts, vb = vb_tfi_from_trades(
            tr["ts"].to_numpy(np.int64),
            tr["buy_qty"].to_numpy(float),
            tr["sell_qty"].to_numpy(float),
            bucket_vol=bucket_vol,
            n_buckets=n_buckets,
        )
        if len(end_ts) == 0:
            df["vb_tfi"] = np.nan
            continue
        feat = pd.DataFrame({"ts": end_ts, "vb_tfi": vb})
        out = pd.merge_asof(df[["ts"]].sort_values("ts"), feat, on="ts", direction="backward")
        df["vb_tfi"] = out["vb_tfi"].to_numpy()


# Bieganowski & Ślepaczuk (arXiv:2602.00776): VWAP-to-mid on a 1-second bar.
VWAP_MID_WINDOW_MS = 1000


def vwap_mid_bp(
    trade_ts: np.ndarray,
    buy_qty: np.ndarray,
    buy_vwap: np.ndarray,
    sell_qty: np.ndarray,
    sell_vwap: np.ndarray,
    book_ts: np.ndarray,
    mid: np.ndarray,
    *,
    window_ms: int = VWAP_MID_WINDOW_MS,
) -> pd.DataFrame:
    """Qty-weighted mean of per-bar (VWAP − mid) / mid in bp over (ts − window, ts].

    Each trade bar's VWAP uses that bar's buy and sell prints only. Mid is the
    book mid as-of the bar (backward). Bars with no finite print do not enter
    the average. Returned ``ts`` is the trade-bar clock, sorted.
    """
    bars = pd.DataFrame(
        {
            "ts": np.asarray(trade_ts, dtype=np.int64),
            "bq": np.nan_to_num(np.asarray(buy_qty, dtype=np.float64), nan=0.0),
            "bv": np.asarray(buy_vwap, dtype=np.float64),
            "sq": np.nan_to_num(np.asarray(sell_qty, dtype=np.float64), nan=0.0),
            "sv": np.asarray(sell_vwap, dtype=np.float64),
        }
    ).sort_values("ts")
    if bars.empty:
        return pd.DataFrame({"ts": np.array([], dtype=np.int64), "vwap_mid_bp": np.array([], dtype=np.float64)})
    bq = bars["bq"].to_numpy()
    sq = bars["sq"].to_numpy()
    bv = bars["bv"].to_numpy()
    sv = bars["sv"].to_numpy()
    num = np.zeros(len(bars), dtype=np.float64)
    den = np.zeros(len(bars), dtype=np.float64)
    ok_b = (bq > 0) & np.isfinite(bv) & (bv > 0)
    ok_s = (sq > 0) & np.isfinite(sv) & (sv > 0)
    num[ok_b] = bq[ok_b] * bv[ok_b]
    den[ok_b] = bq[ok_b]
    num[ok_s] += sq[ok_s] * sv[ok_s]
    den[ok_s] += sq[ok_s]
    book = pd.DataFrame(
        {
            "ts": np.asarray(book_ts, dtype=np.int64),
            "mid": np.asarray(mid, dtype=np.float64),
        }
    ).sort_values("ts")
    joined = pd.merge_asof(
        pd.DataFrame({"ts": bars["ts"].to_numpy(), "num": num, "den": den}),
        book,
        on="ts",
        direction="backward",
    )
    ts = joined["ts"].to_numpy(np.int64)
    den_j = joined["den"].to_numpy(np.float64)
    mid_asof = joined["mid"].to_numpy(np.float64)
    ok = (den_j > 0) & np.isfinite(mid_asof) & (mid_asof > 0) & np.isfinite(joined["num"].to_numpy())
    dev = np.zeros(len(joined), dtype=np.float64)
    num_j = joined["num"].to_numpy(np.float64)
    dev[ok] = 1e4 * (num_j[ok] / den_j[ok] - mid_asof[ok]) / mid_asof[ok]
    w = np.where(ok, den_j, 0.0)
    c_w = np.cumsum(w)
    c_wd = np.cumsum(w * dev)
    lo = np.searchsorted(ts, ts - np.int64(window_ms), side="right")
    left = lo - 1
    take_left = left >= 0
    w_left = np.zeros(len(ts), dtype=np.float64)
    wd_left = np.zeros(len(ts), dtype=np.float64)
    w_left[take_left] = c_w[left[take_left]]
    wd_left[take_left] = c_wd[left[take_left]]
    w_win = c_w - w_left
    out = np.full(len(ts), np.nan, dtype=np.float64)
    good = w_win > 0
    out[good] = (c_wd - wd_left)[good] / w_win[good]
    return pd.DataFrame({"ts": ts, "vwap_mid_bp": out})


def add_vwap_mid(
    panels: dict[str, pd.DataFrame],
    instrument: str,
    *,
    window_ms: int = VWAP_MID_WINDOW_MS,
) -> None:
    """1s VWAP-to-mid in bp → ``vwap_mid_bp`` (trades as-of onto the grid).

    A grid time keeps the value only when the matched trade bar is inside the
    same window. Older prints are left missing rather than carried across a hole.
    """
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("trades", instrument, d) or not exists("book", instrument, d):
            df["vwap_mid_bp"] = np.nan
            continue
        tr = read_cols(
            "trades",
            instrument,
            d,
            columns=["ts", "buy_qty", "buy_vwap", "sell_qty", "sell_vwap"],
        )
        book = read_cols("book", instrument, d, columns=["ts", "bid_px", "ask_px"])
        if tr.empty or book.empty:
            df["vwap_mid_bp"] = np.nan
            continue
        feat = vwap_mid_bp(
            tr["ts"].to_numpy(np.int64),
            tr["buy_qty"].to_numpy(float),
            tr["buy_vwap"].to_numpy(float),
            tr["sell_qty"].to_numpy(float),
            tr["sell_vwap"].to_numpy(float),
            book["ts"].to_numpy(np.int64),
            mid_from_px(book["bid_px"].to_numpy(), book["ask_px"].to_numpy()),
            window_ms=window_ms,
        )
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        age = grid["ts"].to_numpy(np.int64) - out["ts"].to_numpy(np.int64)
        vals = out["vwap_mid_bp"].to_numpy(np.float64)
        vals = np.where(np.isfinite(age) & (age >= 0) & (age <= window_ms), vals, np.nan)
        df["vwap_mid_bp"] = vals


# Plerou, Gopikrishnan, Gabaix, Stanley (cond-mat/0106657) eq. (2a).
PHI_WINDOW_MS = 500


def phi_cnt(
    trade_ts: np.ndarray,
    buy_n: np.ndarray,
    sell_n: np.ndarray,
    *,
    window_ms: int = PHI_WINDOW_MS,
) -> pd.DataFrame:
    """Φ_Δt = Σ a_i over trade bars in (ts − window, ts].

    Each buy trade contributes +1 and each sell trade −1 (their a_i). Counts
    come from the bar's ``buy_n`` and ``sell_n``; the tape already labels the
    aggressor, so there is no separate indeterminate class.
    """
    bars = pd.DataFrame(
        {
            "ts": np.asarray(trade_ts, dtype=np.int64),
            "bn": np.nan_to_num(np.asarray(buy_n, dtype=np.float64), nan=0.0),
            "sn": np.nan_to_num(np.asarray(sell_n, dtype=np.float64), nan=0.0),
        }
    ).sort_values("ts")
    if bars.empty:
        return pd.DataFrame({"ts": np.array([], dtype=np.int64), "phi_cnt": np.array([], dtype=np.float64)})
    ts = bars["ts"].to_numpy(np.int64)
    signed = bars["bn"].to_numpy() - bars["sn"].to_numpy()
    c = np.cumsum(signed)
    lo = np.searchsorted(ts, ts - np.int64(window_ms), side="right")
    left = lo - 1
    prev = np.zeros(len(ts), dtype=np.float64)
    take = left >= 0
    prev[take] = c[left[take]]
    return pd.DataFrame({"ts": ts, "phi_cnt": c - prev})


def add_phi_cnt(
    panels: dict[str, pd.DataFrame],
    instrument: str,
    *,
    window_ms: int = PHI_WINDOW_MS,
) -> None:
    """Count imbalance Φ over 500 ms → ``phi_cnt`` (trades as-of onto the grid).

    A grid time keeps the value only when the matched trade bar is inside the
    same window.
    """
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("trades", instrument, d):
            df["phi_cnt"] = np.nan
            continue
        tr = read_cols("trades", instrument, d, columns=["ts", "buy_n", "sell_n"])
        if tr.empty:
            df["phi_cnt"] = np.nan
            continue
        feat = phi_cnt(
            tr["ts"].to_numpy(np.int64),
            tr["buy_n"].to_numpy(float),
            tr["sell_n"].to_numpy(float),
            window_ms=window_ms,
        )
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        age = grid["ts"].to_numpy(np.int64) - out["ts"].to_numpy(np.int64)
        vals = out["phi_cnt"].to_numpy(np.float64)
        vals = np.where(np.isfinite(age) & (age >= 0) & (age <= window_ms), vals, np.nan)
        df["phi_cnt"] = vals


def sq_micro_bp(
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    bid_qty: np.ndarray,
    ask_qty: np.ndarray,
    *,
    rebate: float = 0.0,
) -> np.ndarray:
    """Bonart–Lillo eq. (23): square-volume fundamental price minus mid, in bp.

    p̂ = (Va² (bid − r) + Vb² (ask + r)) / (Va² + Vb²), with Va the ask size and
    Vb the bid size. ``rebate`` is r in price units. This tape has no maker-rebate
    schedule, so the default is r = 0; the squares are what separate p̂ from the
    linear microprice.
    """
    vb = np.asarray(bid_qty, dtype=np.float64)
    va = np.asarray(ask_qty, dtype=np.float64)
    bt = np.asarray(bid_px, dtype=np.float64)
    at = np.asarray(ask_px, dtype=np.float64)
    vb2 = vb * vb
    va2 = va * va
    den = va2 + vb2
    mid = 0.5 * (bt + at)
    out = np.full(len(den), np.nan, dtype=np.float64)
    ok = (den > 0) & np.isfinite(den) & np.isfinite(mid) & (mid > 0) & (at > bt)
    p_hat = np.zeros(len(den), dtype=np.float64)
    p_hat[ok] = (va2[ok] * (bt[ok] - rebate) + vb2[ok] * (at[ok] + rebate)) / den[ok]
    out[ok] = 1e4 * (p_hat[ok] - mid[ok]) / mid[ok]
    return out


def add_imb_x_lead(panels: dict[str, pd.DataFrame]) -> None:
    """Book × leader interaction → ``imb_x_lead`` = imb * leader_ret_1_bp.

    Both inputs are already on the grid at t (own book and backward as-of leader).
    """
    for df in panels.values():
        imb = df["imb"].to_numpy(float)
        if "leader_ret_1_bp" in df.columns:
            lead = df["leader_ret_1_bp"].to_numpy(float)
        else:
            lead = np.full(len(df), np.nan)
        prod = imb * lead
        prod[~(np.isfinite(imb) & np.isfinite(lead))] = np.nan
        df["imb_x_lead"] = prod


def add_sq_micro(panels: dict[str, pd.DataFrame], instrument: str, *, rebate: float = 0.0) -> None:
    """Square-volume microprice gap → ``sq_micro_bp`` (book as-of onto the grid)."""
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("book", instrument, d):
            df["sq_micro_bp"] = np.nan
            continue
        book = read_cols("book", instrument, d, columns=["ts", "bid_px", "ask_px", "bid_qty", "ask_qty"])
        if book.empty:
            df["sq_micro_bp"] = np.nan
            continue
        book = book.sort_values("ts")
        gap = sq_micro_bp(
            book["bid_px"].to_numpy(),
            book["ask_px"].to_numpy(),
            book["bid_qty"].to_numpy(),
            book["ask_qty"].to_numpy(),
            rebate=rebate,
        )
        feat = pd.DataFrame({"ts": book["ts"].to_numpy(np.int64), "sq_micro_bp": gap})
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        df["sq_micro_bp"] = out["sq_micro_bp"].to_numpy()


# Same-price branch of Cont–Kukanov–Stoikov e_t, as a fraction of the queue it leaves.
Q_DEPL_WINDOW_MS = 500


def q_depl(
    bbo_ts: np.ndarray,
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    bid_qty: np.ndarray,
    ask_qty: np.ndarray,
    *,
    window_ms: int = Q_DEPL_WINDOW_MS,
) -> pd.DataFrame:
    """Share of the best queue that disappeared while that side's price stayed put.

    On an update that leaves the bid (ask) price unchanged, the bid (ask) contribution
    is the fraction of the previous queue that went away, clipped to [-1, 1]. A price change contributes
    nothing: that is the jump part of Cont e_t, not depletion of the same quote.
    The column at an event is the sum of those fractions over (ts − window, ts].
    Positive means the ask was eaten more than the bid.
    """
    ev = pd.DataFrame(
        {
            "ts": np.asarray(bbo_ts, dtype=np.int64),
            "bp": np.asarray(bid_px, dtype=np.float64),
            "ap": np.asarray(ask_px, dtype=np.float64),
            "bq": np.asarray(bid_qty, dtype=np.float64),
            "aq": np.asarray(ask_qty, dtype=np.float64),
        }
    ).sort_values("ts")
    if len(ev) < 2:
        return pd.DataFrame({"ts": np.array([], dtype=np.int64), "q_depl": np.array([], dtype=np.float64)})
    ts = ev["ts"].to_numpy(np.int64)
    bp = ev["bp"].to_numpy()
    ap = ev["ap"].to_numpy()
    bq = ev["bq"].to_numpy()
    aq = ev["aq"].to_numpy()
    contrib = np.zeros(len(ev), dtype=np.float64)
    same_bid = bp[1:] == bp[:-1]
    same_ask = ap[1:] == ap[:-1]
    bq0 = bq[:-1]
    aq0 = aq[:-1]
    bid_frac = np.zeros(len(ev) - 1, dtype=np.float64)
    ask_frac = np.zeros(len(ev) - 1, dtype=np.float64)
    bid_ok = same_bid & np.isfinite(bq0) & (bq0 > 0) & np.isfinite(bq[1:])
    ask_ok = same_ask & np.isfinite(aq0) & (aq0 > 0) & np.isfinite(aq[1:])
    bid_frac[bid_ok] = (bq0[bid_ok] - bq[1:][bid_ok]) / bq0[bid_ok]
    ask_frac[ask_ok] = (aq0[ask_ok] - aq[1:][ask_ok]) / aq0[ask_ok]
    # A dust queue that is replaced by a real one is a full replenishment, not a 10^6 event.
    bid_frac = np.clip(bid_frac, -1.0, 1.0)
    ask_frac = np.clip(ask_frac, -1.0, 1.0)
    contrib[1:] = ask_frac - bid_frac
    c = np.cumsum(contrib)
    lo = np.searchsorted(ts, ts - np.int64(window_ms), side="right")
    left = lo - 1
    prev = np.zeros(len(ts), dtype=np.float64)
    take = left >= 0
    prev[take] = c[left[take]]
    return pd.DataFrame({"ts": ts, "q_depl": c - prev})


def add_q_depl(
    panels: dict[str, pd.DataFrame],
    instrument: str,
    *,
    window_ms: int = Q_DEPL_WINDOW_MS,
) -> None:
    """500 ms same-price queue depletion → ``q_depl`` (bbo as-of onto the grid).

    A grid time with no bbo update inside the window is 0, not a carried old sum.
    """
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("bbo", instrument, d):
            df["q_depl"] = np.nan
            continue
        bbo = read_cols("bbo", instrument, d, columns=["ts", "bid_px", "ask_px", "bid_qty", "ask_qty"])
        if len(bbo) < 2:
            df["q_depl"] = np.nan
            continue
        feat = q_depl(
            bbo["ts"].to_numpy(np.int64),
            bbo["bid_px"].to_numpy(float),
            bbo["ask_px"].to_numpy(float),
            bbo["bid_qty"].to_numpy(float),
            bbo["ask_qty"].to_numpy(float),
            window_ms=window_ms,
        )
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        age = grid["ts"].to_numpy(np.int64) - out["ts"].to_numpy(np.int64)
        vals = out["q_depl"].to_numpy(np.float64)
        fresh = np.isfinite(age) & (age >= 0) & (age <= window_ms) & np.isfinite(vals)
        df["q_depl"] = np.where(fresh, vals, 0.0)


# Valenzuela, Zer, Fryźlewicz, Rheinländer (JFM 2015): depth far from the touch
# relative to depth near it. Bucket edges on this tape: v01–v04 are 0–2 bp,
# v05–v11 are 2–100 bp.
_REL_LIQ_NEAR = range(1, 5)
_REL_LIQ_FAR = range(5, 12)


def rel_liq(
    bid_vs: list[np.ndarray],
    ask_vs: list[np.ndarray],
) -> np.ndarray:
    """V(2–100 bp) / V(0–2 bp), both sides together.

    ``bid_vs`` / ``ask_vs`` are the eleven distance buckets in order. Empty near
    depth is missing, not an infinite ratio.
    """
    if len(bid_vs) != 11 or len(ask_vs) != 11:
        raise ValueError("rel_liq expects 11 bid and 11 ask buckets")
    near = np.zeros(len(bid_vs[0]), dtype=np.float64)
    far = np.zeros(len(bid_vs[0]), dtype=np.float64)
    for k in _REL_LIQ_NEAR:
        near = near + np.nan_to_num(bid_vs[k - 1], nan=0.0) + np.nan_to_num(ask_vs[k - 1], nan=0.0)
    for k in _REL_LIQ_FAR:
        far = far + np.nan_to_num(bid_vs[k - 1], nan=0.0) + np.nan_to_num(ask_vs[k - 1], nan=0.0)
    out = np.full(len(near), np.nan, dtype=np.float64)
    ok = near > 0
    out[ok] = far[ok] / near[ok]
    return out


def add_rel_liq(panels: dict[str, pd.DataFrame], instrument: str) -> None:
    """Relative liquidity → ``rel_liq`` (book as-of onto the grid)."""
    from full_eda_io import exists

    cols = ["ts", *[f"bid_v{k:02d}" for k in range(1, 12)], *[f"ask_v{k:02d}" for k in range(1, 12)]]
    for d, df in panels.items():
        if not exists("book", instrument, d):
            df["rel_liq"] = np.nan
            continue
        book = read_cols("book", instrument, d, columns=cols).sort_values("ts")
        if book.empty:
            df["rel_liq"] = np.nan
            continue
        bid_vs = [book[f"bid_v{k:02d}"].to_numpy(float) for k in range(1, 12)]
        ask_vs = [book[f"ask_v{k:02d}"].to_numpy(float) for k in range(1, 12)]
        feat = pd.DataFrame({"ts": book["ts"].to_numpy(np.int64), "rel_liq": rel_liq(bid_vs, ask_vs)})
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        df["rel_liq"] = out["rel_liq"].to_numpy()


def liq_side(
    bid_vs: list[np.ndarray],
    ask_vs: list[np.ndarray],
) -> np.ndarray:
    """Near-touch share of bid depth minus the same share of ask depth.

    Near is 0–2 bp (v01–v04), far is 2–100 bp (v05–v11). Positive when a larger
    fraction of bid depth sits near the touch than of ask depth. A side with no
    depth is missing.
    """
    if len(bid_vs) != 11 or len(ask_vs) != 11:
        raise ValueError("liq_side expects 11 bid and 11 ask buckets")
    n = len(bid_vs[0])

    def _sum(vs: list[np.ndarray], ks: range) -> np.ndarray:
        acc = np.zeros(n, dtype=np.float64)
        for k in ks:
            acc = acc + np.nan_to_num(vs[k - 1], nan=0.0)
        return acc

    def _share(near: np.ndarray, far: np.ndarray) -> np.ndarray:
        den = near + far
        out = np.full(n, np.nan, dtype=np.float64)
        ok = den > 0
        out[ok] = near[ok] / den[ok]
        return out

    bid = _share(_sum(bid_vs, _REL_LIQ_NEAR), _sum(bid_vs, _REL_LIQ_FAR))
    ask = _share(_sum(ask_vs, _REL_LIQ_NEAR), _sum(ask_vs, _REL_LIQ_FAR))
    return bid - ask


def add_liq_side(panels: dict[str, pd.DataFrame], instrument: str) -> None:
    """Signed near-touch concentration → ``liq_side`` (book as-of onto the grid)."""
    from full_eda_io import exists

    cols = ["ts", *[f"bid_v{k:02d}" for k in range(1, 12)], *[f"ask_v{k:02d}" for k in range(1, 12)]]
    for d, df in panels.items():
        if not exists("book", instrument, d):
            df["liq_side"] = np.nan
            continue
        book = read_cols("book", instrument, d, columns=cols).sort_values("ts")
        if book.empty:
            df["liq_side"] = np.nan
            continue
        bid_vs = [book[f"bid_v{k:02d}"].to_numpy(float) for k in range(1, 12)]
        ask_vs = [book[f"ask_v{k:02d}"].to_numpy(float) for k in range(1, 12)]
        feat = pd.DataFrame({"ts": book["ts"].to_numpy(np.int64), "liq_side": liq_side(bid_vs, ask_vs)})
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        df["liq_side"] = out["liq_side"].to_numpy()


# Albers et al. sweep imbalance, measured from prints rather than a price ladder.
SWEEP_PX_WINDOW_MS = 500
_TRADE_BAR_MS = 100


def sweep_px(
    trade_ts: np.ndarray,
    buy_qty: np.ndarray,
    buy_vwap: np.ndarray,
    sell_qty: np.ndarray,
    sell_vwap: np.ndarray,
    book_ts: np.ndarray,
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    *,
    window_ms: int = SWEEP_PX_WINDOW_MS,
) -> pd.DataFrame:
    """Signed distance of aggressor VWAP through the pre-bar touch, in bp.

    The touch is the book as-of 100 ms before the trade bar, so the bar's own
    prints do not move the quote they are measured against. Buys above that ask
    and sells below that bid count; prints at the touch count as zero. The value
    at a bar is the quantity-weighted sum over (ts − window, ts].
    """
    bars = pd.DataFrame(
        {
            "ts": np.asarray(trade_ts, dtype=np.int64),
            "bq": np.nan_to_num(np.asarray(buy_qty, dtype=np.float64), nan=0.0),
            "bv": np.asarray(buy_vwap, dtype=np.float64),
            "sq": np.nan_to_num(np.asarray(sell_qty, dtype=np.float64), nan=0.0),
            "sv": np.asarray(sell_vwap, dtype=np.float64),
        }
    ).sort_values("ts")
    if bars.empty:
        return pd.DataFrame({"ts": np.array([], dtype=np.int64), "sweep_px": np.array([], dtype=np.float64)})
    book = pd.DataFrame(
        {
            "ts": np.asarray(book_ts, dtype=np.int64),
            "bid": np.asarray(bid_px, dtype=np.float64),
            "ask": np.asarray(ask_px, dtype=np.float64),
        }
    ).sort_values("ts")
    probe = bars[["ts"]].copy()
    probe["ts_touch"] = probe["ts"] - _TRADE_BAR_MS
    touch = pd.merge_asof(
        probe.sort_values("ts_touch"),
        book,
        left_on="ts_touch",
        right_on="ts",
        direction="backward",
    )
    # merge_asof reorders by ts_touch; put the touch back on the trade-ts order.
    touch = touch.sort_values("ts_x")
    ts = touch["ts_x"].to_numpy(np.int64)
    bid = touch["bid"].to_numpy(np.float64)
    ask = touch["ask"].to_numpy(np.float64)
    bq = bars["bq"].to_numpy()
    sq = bars["sq"].to_numpy()
    bv = bars["bv"].to_numpy()
    sv = bars["sv"].to_numpy()
    mid = 0.5 * (bid + ask)
    buy_x = np.zeros(len(ts), dtype=np.float64)
    sell_x = np.zeros(len(ts), dtype=np.float64)
    buy_ok = (bq > 0) & np.isfinite(bv) & np.isfinite(ask) & np.isfinite(mid) & (mid > 0) & (ask > bid)
    sell_ok = (sq > 0) & np.isfinite(sv) & np.isfinite(bid) & np.isfinite(mid) & (mid > 0) & (ask > bid)
    buy_x[buy_ok] = bq[buy_ok] * np.maximum(bv[buy_ok] - ask[buy_ok], 0.0)
    sell_x[sell_ok] = sq[sell_ok] * np.maximum(bid[sell_ok] - sv[sell_ok], 0.0)
    num = buy_x - sell_x
    den = np.where(buy_ok | sell_ok, (bq + sq) * mid, 0.0)
    c_num = np.cumsum(num)
    c_den = np.cumsum(den)
    lo = np.searchsorted(ts, ts - np.int64(window_ms), side="right")
    left = lo - 1
    num_prev = np.zeros(len(ts), dtype=np.float64)
    den_prev = np.zeros(len(ts), dtype=np.float64)
    take = left >= 0
    num_prev[take] = c_num[left[take]]
    den_prev[take] = c_den[left[take]]
    num_w = c_num - num_prev
    den_w = c_den - den_prev
    out = np.zeros(len(ts), dtype=np.float64)
    good = den_w > 0
    out[good] = 1e4 * num_w[good] / den_w[good]
    return pd.DataFrame({"ts": ts, "sweep_px": out})


def add_sweep_px(
    panels: dict[str, pd.DataFrame],
    instrument: str,
    *,
    window_ms: int = SWEEP_PX_WINDOW_MS,
) -> None:
    """Realized sweep-through → ``sweep_px`` (trades and book, as-of onto the grid).

    No trade bar inside the window leaves the grid value at 0.
    """
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("trades", instrument, d) or not exists("book", instrument, d):
            df["sweep_px"] = np.nan
            continue
        tr = read_cols(
            "trades",
            instrument,
            d,
            columns=["ts", "buy_qty", "buy_vwap", "sell_qty", "sell_vwap"],
        )
        book = read_cols("book", instrument, d, columns=["ts", "bid_px", "ask_px"])
        if tr.empty or book.empty:
            df["sweep_px"] = np.nan
            continue
        feat = sweep_px(
            tr["ts"].to_numpy(np.int64),
            tr["buy_qty"].to_numpy(float),
            tr["buy_vwap"].to_numpy(float),
            tr["sell_qty"].to_numpy(float),
            tr["sell_vwap"].to_numpy(float),
            book["ts"].to_numpy(np.int64),
            book["bid_px"].to_numpy(float),
            book["ask_px"].to_numpy(float),
            window_ms=window_ms,
        )
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        age = grid["ts"].to_numpy(np.int64) - out["ts"].to_numpy(np.int64)
        vals = out["sweep_px"].to_numpy(np.float64)
        fresh = np.isfinite(age) & (age >= 0) & (age <= window_ms) & np.isfinite(vals)
        df["sweep_px"] = np.where(fresh, vals, 0.0)


def sign_age_marks(
    trade_ts: np.ndarray,
    buy_n: np.ndarray,
    sell_n: np.ndarray,
) -> pd.DataFrame:
    """Last aggressive buy and sell timestamps, updated only when that side prints.

    A bar with both sides sets both marks to that bar. The column is not the
    feature; the grid turns the two marks into a relative recency.
    """
    bars = pd.DataFrame(
        {
            "ts": np.asarray(trade_ts, dtype=np.int64),
            "bn": np.nan_to_num(np.asarray(buy_n, dtype=np.float64), nan=0.0),
            "sn": np.nan_to_num(np.asarray(sell_n, dtype=np.float64), nan=0.0),
        }
    ).sort_values("ts")
    n = len(bars)
    if n == 0:
        return pd.DataFrame(
            {
                "ts": np.array([], dtype=np.int64),
                "last_buy_ts": np.array([], dtype=np.int64),
                "last_sell_ts": np.array([], dtype=np.int64),
            }
        )
    ts = bars["ts"].to_numpy(np.int64)
    bn = bars["bn"].to_numpy()
    sn = bars["sn"].to_numpy()
    last_buy = np.full(n, -1, dtype=np.int64)
    last_sell = np.full(n, -1, dtype=np.int64)
    lb = ls = -1
    for i in range(n):
        if bn[i] > 0:
            lb = int(ts[i])
        if sn[i] > 0:
            ls = int(ts[i])
        last_buy[i] = lb
        last_sell[i] = ls
    return pd.DataFrame({"ts": ts, "last_buy_ts": last_buy, "last_sell_ts": last_sell})


def sign_age_from_marks(
    grid_ts: np.ndarray,
    last_buy_ts: np.ndarray,
    last_sell_ts: np.ndarray,
) -> np.ndarray:
    """(age_sell − age_buy) / (age_sell + age_buy) at each grid time.

    Positive when the last buy is more recent than the last sell. Missing until
    both sides have printed. A tie (both marks equal) is 0.
    """
    g = np.asarray(grid_ts, dtype=np.int64)
    lb = np.asarray(last_buy_ts, dtype=np.int64)
    ls = np.asarray(last_sell_ts, dtype=np.int64)
    out = np.full(len(g), np.nan, dtype=np.float64)
    ok = (lb >= 0) & (ls >= 0) & (g >= lb) & (g >= ls)
    age_b = (g[ok] - lb[ok]).astype(np.float64)
    age_s = (g[ok] - ls[ok]).astype(np.float64)
    den = age_b + age_s
    vals = np.zeros(age_b.shape, dtype=np.float64)
    nz = den > 0
    vals[nz] = (age_s[nz] - age_b[nz]) / den[nz]
    out[ok] = vals
    return out


def add_sign_age(panels: dict[str, pd.DataFrame], instrument: str) -> None:
    """Relative recency of the last buy versus the last sell → ``sign_age``."""
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("trades", instrument, d):
            df["sign_age"] = np.nan
            continue
        tr = read_cols("trades", instrument, d, columns=["ts", "buy_n", "sell_n"])
        if tr.empty:
            df["sign_age"] = np.nan
            continue
        marks = sign_age_marks(tr["ts"].to_numpy(np.int64), tr["buy_n"].to_numpy(float), tr["sell_n"].to_numpy(float))
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, marks, on="ts", direction="backward")
        df["sign_age"] = sign_age_from_marks(
            grid["ts"].to_numpy(np.int64),
            out["last_buy_ts"].fillna(-1).to_numpy(np.int64),
            out["last_sell_ts"].fillna(-1).to_numpy(np.int64),
        )


# Eisler, Bouchaud, Kockelkoren (arXiv:0904.0900): impact depends weakly on trade
# size, so the gap between volume imbalance and count imbalance is the size effect.
SIZE_GAP_WINDOW_MS = 500


def size_gap(
    trade_ts: np.ndarray,
    buy_qty: np.ndarray,
    sell_qty: np.ndarray,
    buy_n: np.ndarray,
    sell_n: np.ndarray,
    *,
    window_ms: int = SIZE_GAP_WINDOW_MS,
) -> pd.DataFrame:
    """Volume imbalance minus count imbalance over (ts − window, ts].

    Both imbalances are in [-1, 1]. Positive means buys are larger, relative to
    their number, than sells. A window with no quantity or no trades is missing.
    """
    bars = pd.DataFrame(
        {
            "ts": np.asarray(trade_ts, dtype=np.int64),
            "bq": np.nan_to_num(np.asarray(buy_qty, dtype=np.float64), nan=0.0),
            "sq": np.nan_to_num(np.asarray(sell_qty, dtype=np.float64), nan=0.0),
            "bn": np.nan_to_num(np.asarray(buy_n, dtype=np.float64), nan=0.0),
            "sn": np.nan_to_num(np.asarray(sell_n, dtype=np.float64), nan=0.0),
        }
    ).sort_values("ts")
    if bars.empty:
        return pd.DataFrame({"ts": np.array([], dtype=np.int64), "size_gap": np.array([], dtype=np.float64)})
    ts = bars["ts"].to_numpy(np.int64)
    parts = {k: np.cumsum(bars[k].to_numpy()) for k in ("bq", "sq", "bn", "sn")}
    lo = np.searchsorted(ts, ts - np.int64(window_ms), side="right")
    left = lo - 1
    take = left >= 0

    def _win(c: np.ndarray) -> np.ndarray:
        prev = np.zeros(len(ts), dtype=np.float64)
        prev[take] = c[left[take]]
        return c - prev

    bq, sq = _win(parts["bq"]), _win(parts["sq"])
    bn, sn = _win(parts["bn"]), _win(parts["sn"])
    out = np.full(len(ts), np.nan, dtype=np.float64)
    vden = bq + sq
    nden = bn + sn
    ok = (vden > 0) & (nden > 0)
    out[ok] = (bq[ok] - sq[ok]) / vden[ok] - (bn[ok] - sn[ok]) / nden[ok]
    return pd.DataFrame({"ts": ts, "size_gap": out})


def add_size_gap(
    panels: dict[str, pd.DataFrame],
    instrument: str,
    *,
    window_ms: int = SIZE_GAP_WINDOW_MS,
) -> None:
    """Size-versus-count imbalance → ``size_gap`` (trades as-of onto the grid)."""
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("trades", instrument, d):
            df["size_gap"] = np.nan
            continue
        tr = read_cols("trades", instrument, d, columns=["ts", "buy_qty", "sell_qty", "buy_n", "sell_n"])
        if tr.empty:
            df["size_gap"] = np.nan
            continue
        feat = size_gap(
            tr["ts"].to_numpy(np.int64),
            tr["buy_qty"].to_numpy(float),
            tr["sell_qty"].to_numpy(float),
            tr["buy_n"].to_numpy(float),
            tr["sell_n"].to_numpy(float),
            window_ms=window_ms,
        )
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        age = grid["ts"].to_numpy(np.int64) - out["ts"].to_numpy(np.int64)
        vals = out["size_gap"].to_numpy(np.float64)
        fresh = np.isfinite(age) & (age >= 0) & (age <= window_ms)
        df["size_gap"] = np.where(fresh, vals, np.nan)


# Oxenhorn (arXiv:2205.06338): a depeg is a deviation of the stablecoin from par.
# The column is that signed deviation, in bp, not the thresholded event.
USDC_PREM_MAX_AGE_MS = 3000


def add_usdc_prem(panels: dict[str, pd.DataFrame], instrument: str, *, max_age_ms: int = USDC_PREM_MAX_AGE_MS) -> None:
    """USDCUSDT mid minus 1, in bp → ``usdc_prem_bp``.

    Quotes already marked stale by more than ``max_age_ms``, and grid times
    whose last quote is older than that, are missing.
    """
    from full_eda_io import exists

    for d, df in panels.items():
        if not exists("stable", instrument, d):
            df["usdc_prem_bp"] = np.nan
            continue
        st = read_cols("stable", instrument, d, columns=["ts", "pair", "bid_px", "ask_px", "stale_ms"])
        st = st.loc[st["pair"] == "USDCUSDT"].sort_values("ts")
        if st.empty:
            df["usdc_prem_bp"] = np.nan
            continue
        mid = 0.5 * (st["bid_px"].to_numpy(float) + st["ask_px"].to_numpy(float))
        prem = np.full(len(st), np.nan, dtype=np.float64)
        fresh_src = st["stale_ms"].to_numpy(float) <= max_age_ms
        ok = fresh_src & np.isfinite(mid) & (mid > 0)
        prem[ok] = 1e4 * (mid[ok] - 1.0)
        feat = pd.DataFrame({"ts": st["ts"].to_numpy(np.int64), "usdc_prem_bp": prem})
        grid = df[["ts"]].sort_values("ts")
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        age = grid["ts"].to_numpy(np.int64) - out["ts"].to_numpy(np.int64)
        vals = out["usdc_prem_bp"].to_numpy(np.float64)
        fresh = np.isfinite(age) & (age >= 0) & (age <= max_age_ms)
        df["usdc_prem_bp"] = np.where(fresh, vals, np.nan)


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
