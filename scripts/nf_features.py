"""Causal feature builders for NeuralForecast segment experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from full_eda_io import (  # noqa: E402
    GRID_MS,
    HORIZONS,
    imbalance_from_qty,
    mid_from_px,
    read_cols,
)

# Named feature packs to ablate as hist_exog inputs.
# "of_*" packs use Cont-style order-flow / Δ inputs (stationary vs raw book levels).
FEATURE_SETS: dict[str, list[str]] = {
    "y_only": [],
    "imb": ["imb"],
    "imb_cube": ["imb", "imb3"],
    "imb_micro": ["imb", "imb3", "micro_gap"],
    "imb_ret": ["imb", "imb3", "ret_1_bp", "ret_4_bp"],
    "imb_depth": ["imb", "imb3", "depth_imb_near", "depth_imb_mid"],
    "imb_leader": ["imb", "imb3", "leader_ret_1_bp", "leader_imb"],
    "imb_plus": [
        "imb",
        "imb3",
        "micro_gap",
        "depth_imb_near",
        "ret_1_bp",
        "leader_ret_1_bp",
        "leader_imb",
    ],
    "core": [
        "imb",
        "imb3",
        "micro_gap",
        "ret_1_bp",
        "ret_4_bp",
        "depth_imb_near",
        "depth_imb_mid",
        "spread_bp",
        "leader_ret_1_bp",
        "leader_imb",
    ],
    # Order-flow / Δ packs (Kolm-style stationary inputs; still scored with NHITS)
    "of": ["ofi_best"],
    "of_imb": ["imb", "ofi_best"],
    "of_d": ["imb", "ofi_best", "d_imb"],
    "of_near": ["imb", "ofi_best", "ofi_near"],
    "of_leader": ["imb", "ofi_best", "leader_ofi", "leader_ret_1_bp"],
    "of_core": [
        "imb",
        "ofi_best",
        "d_imb",
        "ofi_near",
        "ret_1_bp",
        "leader_ofi",
        "leader_ret_1_bp",
    ],
    # Thin high-level packs (imb + leader short-lag + gap_dev)
    "hl_lead": ["imb", "leader_ret_1_bp"],
    "hl_gap": ["imb", "gap_dev"],
    "hl3": ["imb", "leader_ret_1_bp", "gap_dev"],
    "hl4": ["imb", "leader_ret_1_bp", "gap_dev", "leader_ret_300ms_bp"],
    "hl_short": ["imb", "leader_ret_200ms_bp", "leader_ret_300ms_bp", "gap_dev"],
    # Albers et al. (arXiv:2108.09750) — absolute TFI, softsign, mean-divergence
    "alb_tfi": ["imb", "tfi_500", "tfi_1000"],
    "alb_tfi_s": ["imb", "tfi_s500", "tfi_s1000"],
    "alb_div": ["imb", "gap_dev", "div_150s", "div_300s"],
    "alb_thin": ["imb", "tfi_s500", "gap_dev"],
    "alb_core": [
        "imb",
        "tfi_s500",
        "tfi_s1000",
        "pret_500_bp",
        "gap_dev",
        "div_150s",
        "leader_ret_1_bp",
        "leader_tfi_s500",
    ],
    # Causal Kalman gap-state packs (local-level / level+vel on ln(own/leader))
    "kf_gap": ["imb", "kf_innov"],
    "hl_kf": ["imb", "leader_ret_1_bp", "kf_innov"],
    "hl_kf2": ["imb", "leader_ret_1_bp", "kf_innov", "kf_vel"],
    # Best linear SCORE path so far: cubic imb + leader + KF innovation
    "hl_kf_imb3": ["imb", "imb3", "leader_ret_1_bp", "kf_innov"],
    "hl_kf_imb3_short": [
        "imb",
        "imb3",
        "leader_ret_1_bp",
        "leader_ret_200ms_bp",
        "leader_ret_300ms_bp",
        "kf_innov",
    ],
    "hl_kf_imb3_micro": ["imb", "imb3", "leader_ret_1_bp", "kf_innov", "micro_gap"],
    # Residual dig (2026-11-12): short absolute TFI + Δimb still predict OOS residual
    "hl_kf_imb3_tfi": ["imb", "imb3", "leader_ret_1_bp", "kf_innov", "tfi_100"],
    "hl_resid": [
        "imb",
        "imb3",
        "leader_ret_1_bp",
        "kf_innov",
        "tfi_100",
        "tfi_s100",
        "d_imb",
    ],
    # Marginal residual adds after hl_resid (~+0.002 WF); keep separate from core
    "hl_resid2": [
        "imb",
        "imb3",
        "leader_ret_1_bp",
        "kf_innov",
        "tfi_100",
        "tfi_s100",
        "d_imb",
        "depth_imb_near",
        "ofi_near",
    ],

    "alb_kf": [
        "imb",
        "tfi_s500",
        "tfi_s1000",
        "pret_500_bp",
        "kf_innov",
        "leader_ret_1_bp",
        "leader_tfi_s500",
    ],
    "alb_kf_imb3": [
        "imb",
        "imb3",
        "tfi_s500",
        "tfi_s1000",
        "pret_500_bp",
        "kf_innov",
        "leader_ret_1_bp",
        "leader_tfi_s500",
    ],
    "alb_core_kf": [
        "imb",
        "tfi_s500",
        "tfi_s1000",
        "pret_500_bp",
        "gap_dev",
        "div_150s",
        "kf_innov",
        "kf_vel",
        "leader_ret_1_bp",
        "leader_tfi_s500",
    ],
}

# Causal rolling window for gap_dev (120 × 500 ms = 60 s), matching PEPE EDA.
GAP_DEV_WINDOW = 120
# Albers mean-divergence lookbacks (Δ in paper ∈ {5s…600s}); on 500 ms grid.
DIV_WINDOWS = {
    "div_150s": 300,  # 150 s
    "div_300s": 600,  # 300 s
}
# Trade-flow imbalance horizons (ms). Trades stream is 100 ms; paper uses
# {100,250,500,1000,2000}. We skip 250 (not on grid) and keep 100/500/1000/2000.
TFI_HORIZONS_MS = (100, 500, 1000, 2000)
# Softsign scale: causal expanding median of |TFI| (warm-up = this many rows).
TFI_SOFT_WARMUP = 120  # ~12 s on 100 ms trades grid
# Kalman on log-gap (500 ms steps). Q/R in log-price^2; defaults ≈ slow fair-value.
KF_GAP_Q = 1e-10  # process noise (level random-walk)
KF_GAP_R = 1e-8  # observation noise
KF_VEL_Q_LEVEL = 1e-10
KF_VEL_Q_VEL = 1e-12
KF_VEL_R = 1e-8

# Preferred same-coin leaders (first existing book wins).
LEADER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "BTCUSDT_PERP": ("BTCUSD1_SPOT", "BTCUSDC_SPOT"),
    "ETHUSDT_PERP": ("ETHUSD1_SPOT", "ETHUSDC_SPOT"),
    "SOLUSDT_PERP": ("SOLUSDC_SPOT", "SOLUSDT_SPOT"),
    "1000PEPEUSDT_PERP": ("1000PEPEUSDC_PERP",),
}


def _depth_imb(book: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """Imbalance of depth buckets bid_v{lo}..bid_v{hi} vs ask (1-indexed names)."""
    bid = np.zeros(len(book), dtype=np.float64)
    ask = np.zeros(len(book), dtype=np.float64)
    for k in range(lo, hi + 1):
        bid = bid + book[f"bid_v{k:02d}"].to_numpy(dtype=np.float64)
        ask = ask + book[f"ask_v{k:02d}"].to_numpy(dtype=np.float64)
    return imbalance_from_qty(bid, ask)


def cont_ofi(
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    bid_qty: np.ndarray,
    ask_qty: np.ndarray,
) -> np.ndarray:
    """Cont–Kukanov–Stoikov L1 order-flow imbalance between consecutive snapshots.

    e_t = I{Pb_t≥Pb_{t-1}} qb_t − I{Pb_t≤Pb_{t-1}} qb_{t-1}
        − I{Pa_t≤Pa_{t-1}} qa_t + I{Pa_t≥Pa_{t-1}} qa_{t-1}
    First row is NaN (no previous snapshot).
    """
    bp = bid_px.astype(np.float64, copy=False)
    ap = ask_px.astype(np.float64, copy=False)
    bq = bid_qty.astype(np.float64, copy=False)
    aq = ask_qty.astype(np.float64, copy=False)
    n = len(bp)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    bp0, bp1 = bp[:-1], bp[1:]
    ap0, ap1 = ap[:-1], ap[1:]
    bq0, bq1 = bq[:-1], bq[1:]
    aq0, aq1 = aq[:-1], aq[1:]
    bid_part = np.where(bp1 > bp0, bq1, np.where(bp1 < bp0, -bq0, bq1 - bq0))
    ask_part = np.where(ap1 < ap0, -aq1, np.where(ap1 > ap0, aq0, aq0 - aq1))
    out[1:] = bid_part + ask_part
    return out


def _depth_volume_flow(book: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """Normalized ΣΔbid_v − ΣΔask_v over depth buckets (proxy OF; no per-level px).

    Divides by contemporaneous (bid+ask) depth so the scale stays ~O(1) for NF.
    """
    bid = np.zeros(len(book), dtype=np.float64)
    ask = np.zeros(len(book), dtype=np.float64)
    for k in range(lo, hi + 1):
        bid = bid + book[f"bid_v{k:02d}"].to_numpy(dtype=np.float64)
        ask = ask + book[f"ask_v{k:02d}"].to_numpy(dtype=np.float64)
    db = np.concatenate([[np.nan], np.diff(bid)])
    da = np.concatenate([[np.nan], np.diff(ask)])
    den = bid + ask
    out = np.full(len(book), np.nan, dtype=np.float64)
    np.divide(db - da, den, out=out, where=den > 0)
    return out


def _resolve_leader(instrument: str, day: str) -> str | None:
    from full_eda_io import exists

    for cand in LEADER_CANDIDATES.get(instrument, ()):
        if exists("book", cand, day):
            return cand
    return None


def _leader_features(leader: str, day: str) -> pd.DataFrame:
    cols = ["ts", "bid_px", "ask_px", "bid_qty", "ask_qty"]
    b = read_cols("book", leader, day, columns=cols).sort_values("ts")
    mid = mid_from_px(b["bid_px"].to_numpy(), b["ask_px"].to_numpy())
    logmid = np.log(np.clip(mid, 1e-12, None))
    ret = np.concatenate([[np.nan], np.diff(logmid)])
    bp = b["bid_px"].to_numpy()
    ap = b["ask_px"].to_numpy()
    bq = b["bid_qty"].to_numpy()
    aq = b["ask_qty"].to_numpy()
    ofi_raw = cont_ofi(bp, ap, bq, aq)
    l1 = bq.astype(np.float64) + aq.astype(np.float64)
    leader_ofi = np.full(len(b), np.nan, dtype=np.float64)
    np.divide(ofi_raw, l1, out=leader_ofi, where=l1 > 0)
    return pd.DataFrame(
        {
            "ts": b["ts"].to_numpy(dtype=np.int64),
            "leader_mid": mid,
            "leader_ret_1_bp": ret * 1e4,
            "leader_imb": imbalance_from_qty(bq, aq),
            "leader_ofi": leader_ofi,
        }
    )


def _leader_mid_series(leader: str, day: str) -> tuple[np.ndarray, np.ndarray]:
    """Prefer bbo mids for sub-500 ms leader lags; fall back to book."""
    from full_eda_io import exists

    stream = "bbo" if exists("bbo", leader, day) else "book"
    raw = read_cols(stream, leader, day, columns=["ts", "bid_px", "ask_px"]).sort_values("ts")
    ts = raw["ts"].to_numpy(dtype=np.int64)
    mid = mid_from_px(raw["bid_px"].to_numpy(), raw["ask_px"].to_numpy())
    return ts, mid


def _asof_mid(query_ts: np.ndarray, src_ts: np.ndarray, src_mid: np.ndarray) -> np.ndarray:
    """Causal last mid at each query timestamp (NaN if before first src print)."""
    idx = np.searchsorted(src_ts, query_ts, side="right") - 1
    out = np.full(len(query_ts), np.nan, dtype=np.float64)
    ok = idx >= 0
    out[ok] = src_mid[idx[ok]]
    return out


def _leader_short_rets_bp(
    query_ts: np.ndarray, src_ts: np.ndarray, src_mid: np.ndarray, lag_ms: int
) -> np.ndarray:
    """log mid(t)/mid(t−lag) on leader tape, in bp."""
    mid_now = _asof_mid(query_ts, src_ts, src_mid)
    mid_past = _asof_mid(query_ts - lag_ms, src_ts, src_mid)
    out = np.full(len(query_ts), np.nan, dtype=np.float64)
    ok = np.isfinite(mid_now) & np.isfinite(mid_past) & (mid_now > 0) & (mid_past > 0)
    out[ok] = np.log(mid_now[ok] / mid_past[ok]) * 1e4
    return out


def _softsign(x: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    """Albers-style saturating map x/(s+|x|) → (−1,1); linear near 0, flat in tails."""
    s = np.asarray(scale, dtype=np.float64)
    out = np.full(np.shape(x), np.nan, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    den = s + np.abs(xv)
    ok = np.isfinite(xv) & np.isfinite(den) & (den > 0)
    np.divide(xv, den, out=out, where=ok)
    return out


def _causal_softsign(x: np.ndarray, warmup: int = TFI_SOFT_WARMUP) -> np.ndarray:
    """Softsign with causal expanding-mean |x| scale (no peek into the future)."""
    xv = np.asarray(x, dtype=np.float64)
    n = len(xv)
    abs_x = np.abs(xv)
    csum = np.cumsum(np.where(np.isfinite(abs_x), abs_x, 0.0))
    ccnt = np.cumsum(np.isfinite(abs_x).astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        emean = csum / np.maximum(ccnt, 1.0)
    scale = np.full(n, np.nan, dtype=np.float64)
    # shift by 1 so scale at t uses only history < t
    scale[1:] = emean[:-1]
    scale[: max(warmup, 1)] = np.nan
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
    return _softsign(xv, scale)


def _tfi_from_trades(instrument: str, day: str) -> pd.DataFrame | None:
    """Absolute trade-flow imbalance B−S over Albers horizons (not relative (B−S)/(B+S))."""
    from full_eda_io import exists

    if not exists("trades", instrument, day):
        return None
    tr = read_cols(
        "trades",
        instrument,
        day,
        columns=["ts", "buy_qty", "sell_qty"],
    ).sort_values("ts")
    if tr.empty:
        return None
    buy = np.nan_to_num(tr["buy_qty"].to_numpy(dtype=np.float64), nan=0.0)
    sell = np.nan_to_num(tr["sell_qty"].to_numpy(dtype=np.float64), nan=0.0)
    flow = buy - sell  # aggressive TFI, Albers eq. (7)
    out: dict[str, np.ndarray] = {"ts": tr["ts"].to_numpy(dtype=np.int64)}
    bin_ms = 100
    for h in TFI_HORIZONS_MS:
        w = max(1, int(h // bin_ms))
        c = np.cumsum(flow)
        rolled = c.copy()
        rolled[w:] = c[w:] - c[:-w]
        rolled[: w - 1] = np.nan
        out[f"tfi_{h}"] = rolled
        out[f"tfi_s{h}"] = _causal_softsign(rolled)
    return pd.DataFrame(out)


def _raw_gap(own_mid: np.ndarray, leader_mid: np.ndarray) -> np.ndarray:
    gap = np.full(len(own_mid), np.nan, dtype=np.float64)
    ok = (
        np.isfinite(own_mid)
        & np.isfinite(leader_mid)
        & (own_mid > 0)
        & (leader_mid > 0)
    )
    gap[ok] = np.log(own_mid[ok] / leader_mid[ok])
    return gap


def _gap_dev(own_mid: np.ndarray, leader_mid: np.ndarray, window: int = GAP_DEV_WINDOW) -> np.ndarray:
    """gap = ln(own/leader); gap_dev = gap − causal rolling mean (shifted, excl. t).

    Same idea as Albers mean-divergence (eq. 12): purge pair gap of its rolling mean.
    """
    gap = _raw_gap(own_mid, leader_mid)
    min_periods = max(10, window // 5)
    mean = (
        pd.Series(gap)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .shift(1)
        .to_numpy(dtype=np.float64)
    )
    return gap - mean


def _kalman_local_level(
    z: np.ndarray,
    *,
    q: float = KF_GAP_Q,
    r: float = KF_GAP_R,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal 1D local-level KF on gap z_t.

    Returns (innovation z_t − x_{t|t−1}, filtered level x_{t|t}).
    NaN observations skip the update (state freezes).
    """
    n = len(z)
    innov = np.full(n, np.nan, dtype=np.float64)
    level = np.full(n, np.nan, dtype=np.float64)
    x = 0.0
    p = 1.0
    started = False
    for i in range(n):
        zi = z[i]
        if not np.isfinite(zi):
            continue
        if not started:
            x = float(zi)
            p = float(r)
            started = True
            level[i] = x
            continue
        x_pred = x
        p_pred = p + q
        e = float(zi) - x_pred
        s = p_pred + r
        k = p_pred / s
        x = x_pred + k * e
        p = (1.0 - k) * p_pred
        innov[i] = e
        level[i] = x
    return innov, level


def _kalman_level_vel(
    z: np.ndarray,
    *,
    q_level: float = KF_VEL_Q_LEVEL,
    q_vel: float = KF_VEL_Q_VEL,
    r: float = KF_VEL_R,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal constant-velocity KF on gap: state = [level, vel], obs = level.

    Returns (innovation, filtered level, filtered vel).
    """
    n = len(z)
    innov = np.full(n, np.nan, dtype=np.float64)
    level = np.full(n, np.nan, dtype=np.float64)
    vel = np.full(n, np.nan, dtype=np.float64)
    # x = [level, vel]
    x0, x1 = 0.0, 0.0
    # P diagonal init
    p00, p01, p11 = 1.0, 0.0, 1.0
    started = False
    for i in range(n):
        zi = z[i]
        if not np.isfinite(zi):
            continue
        if not started:
            x0, x1 = float(zi), 0.0
            p00, p01, p11 = float(r), 0.0, 1.0
            started = True
            level[i] = x0
            vel[i] = x1
            continue
        # predict: F = [[1,1],[0,1]] (dt=1 step)
        x0_p = x0 + x1
        x1_p = x1
        # P <- F P F' + Q
        p00_p = p00 + 2.0 * p01 + p11 + q_level
        p01_p = p01 + p11
        p11_p = p11 + q_vel
        e = float(zi) - x0_p
        s = p00_p + r
        k0 = p00_p / s
        k1 = p01_p / s
        x0 = x0_p + k0 * e
        x1 = x1_p + k1 * e
        # Joseph-lite: P = (I-KH) P
        # H = [1,0]
        p00 = (1.0 - k0) * p00_p
        p01 = (1.0 - k0) * p01_p
        p11 = p11_p - k1 * p01_p
        innov[i] = e
        level[i] = x0
        vel[i] = x1
    return innov, level, vel


def _kalman_gap_features(own_mid: np.ndarray, leader_mid: np.ndarray) -> dict[str, np.ndarray]:
    """KF analogues of mean-divergence on ln(own/leader)."""
    z = _raw_gap(own_mid, leader_mid)
    innov, lev = _kalman_local_level(z)
    innov2, lev2, vel = _kalman_level_vel(z)
    return {
        "kf_innov": innov,
        "kf_level": lev,
        # prefer CV-filter velocity; innovation from 1D (nearly identical to innov2)
        "kf_vel": vel,
        "kf_innov_cv": innov2,
        "kf_level_cv": lev2,
    }


def load_day_panel(instrument: str, day: str, horizon: str = "y_500ms") -> pd.DataFrame:
    """One UTC day on the 500 ms target grid with causal own-book (+ leader) features."""
    if horizon not in HORIZONS:
        raise ValueError(horizon)

    book_cols = [
        "ts",
        "bid_px",
        "ask_px",
        "bid_qty",
        "ask_qty",
        *[f"bid_v{k:02d}" for k in range(1, 6)],
        *[f"ask_v{k:02d}" for k in range(1, 6)],
    ]
    book = read_cols("book", instrument, day, columns=book_cols)
    tgt = read_cols("targets", instrument, day, columns=["ts", horizon])

    b = book.sort_values("ts").copy()
    mid = mid_from_px(b["bid_px"].to_numpy(), b["ask_px"].to_numpy())
    imb = imbalance_from_qty(b["bid_qty"].to_numpy(), b["ask_qty"].to_numpy())
    spread = b["ask_px"].to_numpy(dtype=np.float64) - b["bid_px"].to_numpy(dtype=np.float64)
    # microprice - mid (Stoikov-style), in relative units
    micro = np.where(
        (b["bid_qty"].to_numpy(dtype=np.float64) + b["ask_qty"].to_numpy(dtype=np.float64)) > 0,
        (
            b["ask_px"].to_numpy(dtype=np.float64) * b["bid_qty"].to_numpy(dtype=np.float64)
            + b["bid_px"].to_numpy(dtype=np.float64) * b["ask_qty"].to_numpy(dtype=np.float64)
        )
        / (b["bid_qty"].to_numpy(dtype=np.float64) + b["ask_qty"].to_numpy(dtype=np.float64)),
        mid,
    )
    logmid = np.log(np.clip(mid, 1e-12, None))
    ret_1 = np.concatenate([[np.nan], np.diff(logmid)])
    ret_4 = np.concatenate([np.full(4, np.nan), logmid[4:] - logmid[:-4]])
    # Albers past returns (eq. 10) on 500 ms book grid: 1 step = 500 ms, 2 = 1000 ms
    pret_500 = ret_1 * 1e4
    pret_1000 = np.concatenate([np.full(2, np.nan), (logmid[2:] - logmid[:-2]) * 1e4])
    pret_2000 = np.concatenate([np.full(4, np.nan), (logmid[4:] - logmid[:-4]) * 1e4])
    bp = b["bid_px"].to_numpy()
    ap = b["ask_px"].to_numpy()
    bq = b["bid_qty"].to_numpy()
    aq = b["ask_qty"].to_numpy()
    ofi_raw = cont_ofi(bp, ap, bq, aq)
    # Scale Cont OFI by L1 size → ~comparable to imbalance, kinder to robust scaler.
    l1 = bq.astype(np.float64) + aq.astype(np.float64)
    ofi_best = np.full(len(b), np.nan, dtype=np.float64)
    np.divide(ofi_raw, l1, out=ofi_best, where=l1 > 0)
    d_imb = np.concatenate([[np.nan], np.diff(imb)])
    feat = pd.DataFrame(
        {
            "ts": b["ts"].to_numpy(dtype=np.int64),
            "imb": imb,
            "imb3": imb**3,
            "micro_gap": (micro - mid) / np.where(mid > 0, mid, np.nan),
            "depth_imb_near": _depth_imb(b, 1, 3),
            "depth_imb_mid": _depth_imb(b, 4, 5),
            "spread_bp": spread / np.where(mid > 0, mid, np.nan) * 1e4,
            "ret_1_bp": ret_1 * 1e4,
            "ret_4_bp": ret_4 * 1e4,
            "pret_500_bp": pret_500,
            "pret_1000_bp": pret_1000,
            "pret_2000_bp": pret_2000,
            "ofi_best": ofi_best,
            "d_imb": d_imb,
            "ofi_near": _depth_volume_flow(b, 1, 3),
            "mid": mid,
        }
    )

    t = tgt.sort_values("ts").rename(columns={horizon: "y"})
    panel = pd.merge_asof(t, feat, on="ts", direction="backward")

    # Own-market absolute TFI (+ softsign), Albers §3.1.2 / §3.2
    tfi = _tfi_from_trades(instrument, day)
    if tfi is not None:
        panel = pd.merge_asof(
            panel.sort_values("ts"),
            tfi.sort_values("ts"),
            on="ts",
            direction="backward",
        )
    else:
        for h in TFI_HORIZONS_MS:
            panel[f"tfi_{h}"] = np.nan
            panel[f"tfi_s{h}"] = np.nan

    leader = _resolve_leader(instrument, day)
    if leader is not None:
        lead = _leader_features(leader, day)
        panel = pd.merge_asof(
            panel.sort_values("ts"),
            lead.sort_values("ts"),
            on="ts",
            direction="backward",
        )
        own_m = panel["mid"].to_numpy(dtype=np.float64)
        lead_m = panel["leader_mid"].to_numpy(dtype=np.float64)
        panel["gap_dev"] = _gap_dev(own_m, lead_m)
        for name, win in DIV_WINDOWS.items():
            panel[name] = _gap_dev(own_m, lead_m, window=win)
        for k, v in _kalman_gap_features(own_m, lead_m).items():
            panel[k] = v
        lead_ts, lead_mid = _leader_mid_series(leader, day)
        qts = panel["ts"].to_numpy(dtype=np.int64)
        # 100 ms is often empty vs 500 ms query grid on USD1 bbo (prints land ~200 ms
        # before :000/:500); use 200/300 ms short lags instead.
        panel["leader_ret_200ms_bp"] = _leader_short_rets_bp(qts, lead_ts, lead_mid, 200)
        panel["leader_ret_300ms_bp"] = _leader_short_rets_bp(qts, lead_ts, lead_mid, 300)
        lead_tfi = _tfi_from_trades(leader, day)
        if lead_tfi is not None:
            # rename to leader_* so own TFI columns stay distinct
            ren = {c: f"leader_{c}" for c in lead_tfi.columns if c != "ts"}
            lead_tfi = lead_tfi.rename(columns=ren)
            panel = pd.merge_asof(
                panel.sort_values("ts"),
                lead_tfi.sort_values("ts"),
                on="ts",
                direction="backward",
            )
        else:
            panel["leader_tfi_500"] = np.nan
            panel["leader_tfi_s500"] = np.nan
        panel["leader"] = leader
    else:
        panel["leader_ret_1_bp"] = np.nan
        panel["leader_imb"] = np.nan
        panel["leader_ofi"] = np.nan
        panel["leader_mid"] = np.nan
        panel["gap_dev"] = np.nan
        for name in DIV_WINDOWS:
            panel[name] = np.nan
        for k in ("kf_innov", "kf_level", "kf_vel", "kf_innov_cv", "kf_level_cv"):
            panel[k] = np.nan
        panel["leader_ret_200ms_bp"] = np.nan
        panel["leader_ret_300ms_bp"] = np.nan
        panel["leader_tfi_500"] = np.nan
        panel["leader_tfi_s500"] = np.nan
        panel["leader"] = None

    panel["day"] = day
    panel["instrument"] = instrument
    panel["horizon"] = horizon
    start = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    panel["ds"] = ((panel["ts"] - start) // GRID_MS).astype(np.int64)
    return panel


# temporal_masks needs each of train/val/test ≥ this many rows
MIN_SPLIT_PART = 32
# ceil(MIN_SPLIT_PART / 0.15) so the smallest 15% slice is still ≥ MIN_SPLIT_PART
DEFAULT_MIN_SEGMENT_LEN = int(np.ceil(MIN_SPLIT_PART / 0.15))  # 214


def cut_into_segments(
    panel: pd.DataFrame,
    n_segments: int = 100,
    *,
    min_len: int = DEFAULT_MIN_SEGMENT_LEN,
) -> pd.DataFrame:
    """Split one day into n contiguous segments; each becomes a NeuralForecast series."""
    panel = panel.sort_values("ds").reset_index(drop=True)
    n = len(panel)
    if n < n_segments * min_len:
        # fewer segments if day is short / sparse
        n_segments = max(1, n // min_len)
    edges = np.linspace(0, n, n_segments + 1, dtype=int)
    parts = []
    for i in range(n_segments):
        a, b = int(edges[i]), int(edges[i + 1])
        if b - a < min_len:
            continue
        chunk = panel.iloc[a:b].copy()
        day = chunk["day"].iloc[0]
        inst = chunk["instrument"].iloc[0]
        chunk["unique_id"] = f"{inst}|{day}|seg{i:03d}"
        chunk["segment"] = i
        # local ds restart at 0 for each segment (regular freq=1)
        chunk["ds"] = np.arange(len(chunk), dtype=np.int64)
        parts.append(chunk)
    if not parts:
        raise RuntimeError("no segments produced")
    return pd.concat(parts, ignore_index=True)


def temporal_masks(
    n: int,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    min_part: int = MIN_SPLIT_PART,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous train → val → test split inside one segment."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < min_part:
        raise ValueError(f"segment too short for split: n={n}")
    idx = np.arange(n)
    train = idx < n_train
    val = (idx >= n_train) & (idx < n_train + n_val)
    test = idx >= n_train + n_val
    return train, val, test


def columns_for_feature_sets(feature_sets: list[str]) -> list[str]:
    """Union of y + all exog in the packs (+ imb for baseline). Drop once for fair ablations."""
    cols = {"y", "imb"}
    for fs in feature_sets:
        if fs not in FEATURE_SETS:
            raise KeyError(fs)
        cols.update(FEATURE_SETS[fs])
    return sorted(cols)
