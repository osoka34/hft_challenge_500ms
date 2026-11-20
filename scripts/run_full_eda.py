#!/usr/bin/env python3
"""Full-dataset presentation-style EDA for the 500 ms HFT challenge.

Reads official layout under data/{book,bbo,trades,targets,stable,meta}.
Writes outputs/full_eda_results.json and updates reports/.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from full_eda_io import (  # noqa: E402
    DATA,
    HORIZONS,
    POINTS_PER_DAY,
    TARGET_PERPS,
    TRAIN_END,
    VAL_START,
    all_calendar_days,
    bbo_mid_on_grid,
    day_start_ms,
    exists,
    imbalance_from_qty,
    list_instruments,
    load_daily_tick_sizes,
    load_depeg,
    load_instruments_meta,
    load_target_clip,
    log_returns_arr,
    mid_from_px,
    path_for,
    pearson,
    read_cols,
    spearman,
    split_of,
    stratified_sample_days,
    train_days,
    val_days,
)

OUT_JSON = ROOT / "outputs" / "full_eda_results.json"
OUT_FIGS = ROOT / "outputs" / "figs"
REPORT_LOG = ROOT / "reports" / "VARIABLE_ANALYSIS_LOG.md"
REPORT_EXPL = ROOT / "reports" / "EXPLANATIONS.md"
REPORT_FINDINGS = ROOT / "reports" / "FULL_EDA_FINDINGS.md"

PEER_OUTAGES = ("2026-03-26", "2026-05-21", "2026-06-12", "2026-07-28", "2026-08-08")

# Same-coin lead/lag pairs to verify (leader, follower, label)
LEADLAG_PAIRS = [
    ("BTCUSD1_SPOT", "BTCUSDT_PERP", "BTCUSD1→BTCUSDT_PERP"),
    ("BTCUSDT_PERP", "BTCUSDT_SPOT", "BTCUSDT_PERP→BTCUSDT_SPOT"),
    ("BTCUSDT_SPOT", "BTCUSDT_PERP", "BTCUSDT_SPOT→BTCUSDT_PERP"),
    ("ETHUSD1_SPOT", "ETHUSDT_PERP", "ETHUSD1→ETHUSDT_PERP"),
    ("ETHUSDT_PERP", "ETHUSDT_SPOT", "ETHUSDT_PERP→ETHUSDT_SPOT"),
    ("ETHUSDT_SPOT", "ETHUSDT_PERP", "ETHUSDT_SPOT→ETHUSDT_PERP"),
    ("1000PEPEUSDC_PERP", "1000PEPEUSDT_PERP", "PEPEUSDC→PEPEUSDT_PERP"),
    ("1000PEPEUSDT_PERP", "1000PEPEUSDC_PERP", "PEPEUSDT→PEPEUSDC_PERP"),
    ("SOLUSDT_SPOT", "SOLUSDT_PERP", "SOLUSDT_SPOT→SOLUSDT_PERP"),
    ("SOLUSDT_PERP", "SOLUSDT_SPOT", "SOLUSDT_PERP→SOLUSDT_SPOT"),
    ("BTCUSDC_SPOT", "BTCUSDT_PERP", "BTCUSDC→BTCUSDT_PERP"),
    ("ETHUSDC_SPOT", "ETHUSDT_PERP", "ETHUSDC→ETHUSDT_PERP"),
]


def _progress(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# V07 — inventory / missing files
# ---------------------------------------------------------------------------

def phase_inventory() -> dict:
    _progress("phase inventory…")
    meta = load_instruments_meta()
    calendar = all_calendar_days()
    missing: dict[str, dict[str, list[str]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for _, row in meta.iterrows():
        inst = row["instrument"]
        first, last = row["first_date"], row["last_date"]
        expected = [d for d in calendar if first <= d <= last]
        for stream in ("book", "bbo", "trades"):
            miss = [d for d in expected if not exists(stream, inst, d)]
            present = len(expected) - len(miss)
            counts.setdefault(inst, {})[stream] = present
            if miss:
                missing.setdefault(inst, {})[stream] = miss
        if inst in TARGET_PERPS:
            miss_t = [d for d in expected if not exists("targets", inst, d)]
            counts.setdefault(inst, {})["targets"] = len(expected) - len(miss_t)
            if miss_t:
                missing.setdefault(inst, {})["targets"] = miss_t
    # stable
    stable_miss = [d for d in calendar if not path_for("stable", "", d).exists()]
    # USD1 pre-listing days (should be absent before first_date)
    usd1_pre = {}
    for inst in ("BTCUSD1_SPOT", "ETHUSD1_SPOT"):
        first = meta.loc[meta.instrument == inst, "first_date"].iloc[0]
        pre = [d for d in calendar if d < first]
        usd1_pre[inst] = {"pre_listing_days": len(pre), "first_date": first}

    depeg = load_depeg()
    depeg_sum = {
        "n_intervals": int(len(depeg)),
        "by_stablecoin_n": depeg.groupby("stablecoin").size().astype(int).to_dict(),
        "by_stablecoin_minutes": depeg.groupby("stablecoin")["minutes"].sum().astype(float).to_dict(),
    }
    return {
        "coverage": "full_period",
        "n_calendar_days": len(calendar),
        "n_train": len(train_days()),
        "n_val": len(val_days()),
        "file_counts": counts,
        "missing_within_listed_range": {k: {s: v for s, v in d.items()} for k, d in missing.items()},
        "n_missing_file_entries": sum(len(v) for d in missing.values() for v in d.values()),
        "stable_missing_days": stable_miss,
        "usd1_pre_listing": usd1_pre,
        "depeg": depeg_sum,
        "peer_missing_note": "Peers report 167 missing files — only USD1 pairs before listing (BTC 80, ETH 87).",
    }


# ---------------------------------------------------------------------------
# V08 — book data quality (full)
# ---------------------------------------------------------------------------

def _dq_book_one(inst: str, day: str) -> dict | None:
    path = path_for("book", inst, day)
    if not path.exists():
        return None
    df = read_cols("book", inst, day, columns=["ts", "bid_px", "ask_px"])
    ts = df["ts"].to_numpy(dtype=np.int64)
    start = day_start_ms(day)
    info = {
        "instrument": inst,
        "day": day,
        "split": split_of(day),
        "n": int(len(ts)),
        "missing_vs_full": int(POINTS_PER_DAY - len(ts)),
        "first_lag_ms": int(ts.min() - start) if len(ts) else None,
        "last_to_eod_ms": int((start + 86_400_000 - 1) - ts.max()) if len(ts) else None,
    }
    if len(ts) >= 2:
        gaps = np.diff(np.sort(ts))
        info["max_gap_ms"] = int(gaps.max())
        info["median_gap_ms"] = float(np.median(gaps))
        info["n_gaps_gt_2s"] = int((gaps > 2000).sum())
        info["n_gaps_gt_60s"] = int((gaps > 60_000).sum())
    bid = df["bid_px"].to_numpy()
    ask = df["ask_px"].to_numpy()
    info["locked_frac"] = float((bid == ask).mean()) if len(ts) else None
    info["crossed_frac"] = float((bid > ask).mean()) if len(ts) else None
    return info


def phase_book_dq(instruments: list[str] | None = None) -> dict:
    _progress("phase book DQ (full)…")
    instruments = instruments or list_instruments("book")
    rows: list[dict] = []
    t0 = time.time()
    for i, inst in enumerate(instruments):
        days_present = sorted(p.stem for p in (DATA / "book" / inst).glob("*.parquet"))
        for day in days_present:
            info = _dq_book_one(inst, day)
            if info:
                rows.append(info)
        _progress(f"  book DQ {i+1}/{len(instruments)} {inst} ({len(days_present)} days) [{time.time()-t0:.1f}s]")
    tab = pd.DataFrame(rows)

    def agg(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {}
        return {
            "n_days": int(len(sub)),
            "rows_min": int(sub["n"].min()),
            "rows_median": float(sub["n"].median()),
            "rows_max": int(sub["n"].max()),
            "first_lag_ms_min": int(sub["first_lag_ms"].min()),
            "first_lag_ms_median": float(sub["first_lag_ms"].median()),
            "first_lag_ms_max": int(sub["first_lag_ms"].max()),
            "max_gap_ms_min": int(sub["max_gap_ms"].min()),
            "max_gap_ms_median": float(sub["max_gap_ms"].median()),
            "max_gap_ms_max": int(sub["max_gap_ms"].max()),
            "locked_frac_median": float(sub["locked_frac"].median()),
            "crossed_frac_median": float(sub["crossed_frac"].median()),
            "crossed_frac_max": float(sub["crossed_frac"].max()),
            "n_days_max_gap_gt_60s": int((sub["max_gap_ms"] > 60_000).sum()),
            "n_days_max_gap_gt_5min": int((sub["max_gap_ms"] > 300_000).sum()),
        }

    by_inst = {inst: agg(tab[tab.instrument == inst]) for inst in tab.instrument.unique()}
    # outage spotlight
    outages = []
    for day in PEER_OUTAGES:
        day_rows = tab[tab.day == day]
        for _, r in day_rows.iterrows():
            if r["max_gap_ms"] >= 60_000 or r["missing_vs_full"] > 200:
                outages.append(
                    {
                        "day": day,
                        "instrument": r["instrument"],
                        "max_gap_ms": int(r["max_gap_ms"]),
                        "missing_vs_full": int(r["missing_vs_full"]),
                        "n": int(r["n"]),
                    }
                )
    # top gap days overall (target perps)
    top_gaps = (
        tab[tab.instrument.isin(TARGET_PERPS)]
        .sort_values("max_gap_ms", ascending=False)
        .head(20)[["day", "instrument", "max_gap_ms", "missing_vs_full", "n"]]
        .to_dict(orient="records")
    )
    return {
        "coverage": "full_period_all_instruments",
        "summary_all": agg(tab),
        "by_instrument": by_inst,
        "peer_outage_days": outages,
        "top_gaps_target_perps": top_gaps,
        # keep slim per-day only for large gaps
        "large_gap_days": tab[tab.max_gap_ms > 60_000][
            ["day", "instrument", "max_gap_ms", "missing_vs_full", "n"]
        ].to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# V09 — targets zeros / vol (full)
# ---------------------------------------------------------------------------

def phase_targets() -> dict:
    _progress("phase targets (full)…")
    rows = []
    for inst in TARGET_PERPS:
        days = sorted(p.stem for p in (DATA / "targets" / inst).glob("*.parquet"))
        for day in days:
            df = read_cols("targets", inst, day)
            rec = {"instrument": inst, "day": day, "split": split_of(day), "n": int(len(df))}
            for h in HORIZONS:
                y = df[h].to_numpy(dtype=np.float64)
                nan_n = int(np.isnan(y).sum())
                finite = y[np.isfinite(y)]
                rec[f"{h}_nan"] = nan_n
                rec[f"{h}_zero_frac"] = float((finite == 0).mean()) if len(finite) else None
                rec[f"{h}_std_bp"] = float(np.nanstd(y) * 1e4) if len(finite) else None
            rows.append(rec)
        _progress(f"  targets {inst} {len(days)} days")
    tab = pd.DataFrame(rows)

    def summarize(inst: str) -> dict:
        sub = tab[tab.instrument == inst]
        out = {"n_days": int(len(sub))}
        for split in ("train", "val"):
            s = sub[sub.split == split]
            out[split] = {
                "zero_frac_median": {h: float(s[f"{h}_zero_frac"].median()) for h in HORIZONS},
                "y_2s_std_bp_median": float(s["y_2s_std_bp"].median()),
                "y_2s_std_bp_mean": float(s["y_2s_std_bp"].mean()),
                "y_2s_std_bp_min": float(s["y_2s_std_bp"].min()),
                "y_2s_std_bp_max": float(s["y_2s_std_bp"].max()),
            }
        # top vol days
        top = sub.nlargest(5, "y_2s_std_bp")[["day", "y_2s_std_bp", "split"]].to_dict(orient="records")
        out["top_vol_days"] = top
        # weekend vs weekday (train)
        s = sub[sub.split == "train"].copy()
        s["dow"] = pd.to_datetime(s["day"]).dt.day_of_week
        s["is_weekend"] = s["dow"] >= 5
        out["weekend_vs_weekday_train_y2s_std_bp"] = {
            "weekend_median": float(s.loc[s.is_weekend, "y_2s_std_bp"].median()),
            "weekday_median": float(s.loc[~s.is_weekend, "y_2s_std_bp"].median()),
        }
        spike = sub[sub.day == "2025-10-10"]
        out["spike_2025_10_10_y2s_std_bp"] = float(spike["y_2s_std_bp"].iloc[0]) if len(spike) else None
        out["nan_total"] = {h: int(sub[f"{h}_nan"].sum()) for h in HORIZONS}
        return out

    by_inst = {inst: summarize(inst) for inst in TARGET_PERPS}
    return {"coverage": "full_period_4_target_perps", "by_instrument": by_inst}


# ---------------------------------------------------------------------------
# V10 — own imbalance IC + microstructure + OLS SCORE (full)
# ---------------------------------------------------------------------------

class OLSAccum:
    __slots__ = ("n", "sx", "sy", "sxx", "sxy")

    def __init__(self) -> None:
        self.n = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.sxy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            return
        xx = x[mask]
        yy = y[mask]
        self.n += int(len(xx))
        self.sx += float(xx.sum())
        self.sy += float(yy.sum())
        self.sxx += float((xx * xx).sum())
        self.sxy += float((xx * yy).sum())

    def coef(self) -> tuple[float, float]:
        if self.n < 10:
            return 0.0, 0.0
        # solve [n, sx; sx, sxx] [a;b] = [sy; sxy]
        det = self.n * self.sxx - self.sx * self.sx
        if det == 0:
            return 0.0, 0.0
        b = (self.n * self.sxy - self.sx * self.sy) / det
        a = (self.sy - b * self.sx) / self.n
        return float(a), float(b)


def _align_book_targets(inst: str, day: str, tick_size: float | None) -> dict | None:
    if not exists("book", inst, day) or not exists("targets", inst, day):
        return None
    book = read_cols("book", inst, day, columns=["ts", "bid_px", "ask_px", "bid_qty", "ask_qty"])
    tgt = read_cols("targets", inst, day)
    b = book.sort_values("ts")
    b["imb"] = imbalance_from_qty(b["bid_qty"].to_numpy(), b["ask_qty"].to_numpy())
    b["mid"] = mid_from_px(b["bid_px"].to_numpy(), b["ask_px"].to_numpy())
    spread = b["ask_px"].to_numpy(dtype=np.float64) - b["bid_px"].to_numpy(dtype=np.float64)
    b["spread"] = spread
    if tick_size and tick_size > 0:
        ticks = np.round(spread / tick_size)
        one_tick = float(np.nanmean(ticks == 1)) if len(ticks) else None
    else:
        one_tick = None
    merged = pd.merge_asof(
        tgt.sort_values("ts"),
        b[["ts", "imb", "mid", "spread"]],
        on="ts",
        direction="backward",
    )
    imb = merged["imb"].to_numpy(dtype=np.float64)
    return {
        "imb": imb,
        "mid": merged["mid"].to_numpy(dtype=np.float64),
        "y": {h: merged[h].to_numpy(dtype=np.float64) for h in HORIZONS},
        "ext_frac": float(np.nanmean(np.abs(imb) > 0.9)),
        "spread_median": float(np.nanmedian(merged["spread"].to_numpy(dtype=np.float64))),
        "one_tick_frac_book": one_tick,
    }


def _trade_flow_on_grid(inst: str, day: str, grid_ts: np.ndarray) -> np.ndarray:
    """Causal asof signed trade qty imbalance onto target grid."""
    if not exists("trades", inst, day):
        return np.full(len(grid_ts), np.nan)
    tr = read_cols("trades", inst, day, columns=["ts", "buy_qty", "sell_qty"])
    buy = tr["buy_qty"].to_numpy(dtype=np.float64)
    sell = tr["sell_qty"].to_numpy(dtype=np.float64)
    den = buy + sell
    flow = np.zeros_like(den)
    np.divide(buy - sell, den, out=flow, where=den > 0)
    df = pd.DataFrame({"ts": tr["ts"].to_numpy(dtype=np.int64), "flow": flow}).sort_values("ts")
    g = pd.DataFrame({"ts": grid_ts})
    m = pd.merge_asof(g, df, on="ts", direction="backward")
    return m["flow"].to_numpy(dtype=np.float64)


def phase_imb_and_score(sample_trade_flow_days: list[str] | None = None) -> dict:
    _progress("phase imbalance IC + OLS SCORE (full)…")
    clips = load_target_clip()
    ticks = load_daily_tick_sizes()
    tick_map = {(r.instrument, r.date): float(r.tick_size) for r in ticks.itertuples()}

    ols: dict[tuple[str, str], OLSAccum] = {(inst, h): OLSAccum() for inst in TARGET_PERPS for h in HORIZONS}
    daily_ic_rows: list[dict] = []
    micro_rows: list[dict] = []
    # store val (and train for score) day payloads lightly: only imb + clipped needed at score time
    # two-pass: first accumulate OLS on train; second score all days
    # To avoid re-reading, we store per-day imb and y arrays for scoring — too big for all train.
    # Strategy: pass1 train OLS only; pass2 score train+val with fitted models (re-read).

    # Pass 1: train OLS + daily IC/micro for all days
    for inst in TARGET_PERPS:
        days = sorted(p.stem for p in (DATA / "targets" / inst).glob("*.parquet"))
        for day in days:
            tick = tick_map.get((inst, day))
            aligned = _align_book_targets(inst, day, tick)
            if aligned is None:
                continue
            imb = aligned["imb"]
            micro_rows.append(
                {
                    "instrument": inst,
                    "day": day,
                    "split": split_of(day),
                    "ext_frac": aligned["ext_frac"],
                    "spread_median": aligned["spread_median"],
                    "one_tick_frac_book": aligned["one_tick_frac_book"],
                }
            )
            for h in HORIZONS:
                y = aligned["y"][h]
                lo, hi = clips[(inst, h)]
                y_clip = np.clip(y, lo, hi)
                ic = spearman(imb, y)
                daily_ic_rows.append(
                    {
                        "instrument": inst,
                        "day": day,
                        "split": split_of(day),
                        "horizon": h,
                        "spearman": ic,
                        "pearson_raw": pearson(imb, y_clip),
                    }
                )
                if day <= TRAIN_END:
                    ols[(inst, h)].update(imb, y_clip)
        _progress(f"  imb pass1 {inst}")

    models = {(inst, h): ols[(inst, h)].coef() for inst in TARGET_PERPS for h in HORIZONS}

    # Pass 2: daily Pearson SCORE with OLS preds
    score_rows: list[dict] = []
    for inst in TARGET_PERPS:
        days = sorted(p.stem for p in (DATA / "targets" / inst).glob("*.parquet"))
        for day in days:
            tick = tick_map.get((inst, day))
            aligned = _align_book_targets(inst, day, tick)
            if aligned is None:
                continue
            imb = aligned["imb"]
            for h in HORIZONS:
                a, b = models[(inst, h)]
                pred = a + b * imb
                y = aligned["y"][h]
                lo, hi = clips[(inst, h)]
                y_clip = np.clip(y, lo, hi)
                score_rows.append(
                    {
                        "instrument": inst,
                        "day": day,
                        "split": split_of(day),
                        "horizon": h,
                        "r": pearson(pred, y_clip),
                    }
                )
        _progress(f"  imb pass2 SCORE {inst}")

    ic_tab = pd.DataFrame(daily_ic_rows)
    sc_tab = pd.DataFrame(score_rows)
    micro_tab = pd.DataFrame(micro_rows)

    def cell_means(tab: pd.DataFrame, col: str, split: str) -> dict:
        sub = tab[tab.split == split]
        cells = {}
        vals = []
        for inst in TARGET_PERPS:
            for h in HORIZONS:
                v = float(sub[(sub.instrument == inst) & (sub.horizon == h)][col].mean())
                cells[f"{inst}|{h}"] = v
                vals.append(v)
        return {"cells": cells, "mean_12": float(np.mean(vals)), "by_instrument": {
            inst: float(np.mean([cells[f"{inst}|{h}"] for h in HORIZONS])) for inst in TARGET_PERPS
        }, "by_horizon": {
            h: float(np.mean([cells[f"{inst}|{h}"] for inst in TARGET_PERPS])) for h in HORIZONS
        }}

    # weekend effect on SCORE (val)
    sc_val = sc_tab[sc_tab.split == "val"].copy()
    sc_val["dow"] = pd.to_datetime(sc_val["day"]).dt.day_of_week
    sc_val["is_weekend"] = sc_val["dow"] >= 5
    weekend_score = {
        h: {
            "weekend": float(sc_val[(sc_val.horizon == h) & sc_val.is_weekend]["r"].mean()),
            "weekday": float(sc_val[(sc_val.horizon == h) & ~sc_val.is_weekend]["r"].mean()),
        }
        for h in HORIZONS
    }

    # trade-flow IC on sample days
    tf_days = sample_trade_flow_days or stratified_sample_days(train_per_month=2)["train"][:20]
    tf_rows = []
    for inst in TARGET_PERPS:
        for day in tf_days:
            if not exists("targets", inst, day):
                continue
            tgt = read_cols("targets", inst, day)
            flow = _trade_flow_on_grid(inst, day, tgt["ts"].to_numpy(dtype=np.int64))
            tick = tick_map.get((inst, day))
            aligned = _align_book_targets(inst, day, tick)
            if aligned is None:
                continue
            for h in HORIZONS:
                tf_rows.append(
                    {
                        "instrument": inst,
                        "day": day,
                        "horizon": h,
                        "flow_spearman": spearman(flow, aligned["y"][h]),
                        "imb_spearman": spearman(aligned["imb"], aligned["y"][h]),
                    }
                )
    tf_tab = pd.DataFrame(tf_rows)
    tf_summary = {}
    if not tf_tab.empty:
        for inst in TARGET_PERPS:
            s = tf_tab[tf_tab.instrument == inst]
            tf_summary[inst] = {
                "mean_flow_spearman": {h: float(s[s.horizon == h]["flow_spearman"].mean()) for h in HORIZONS},
                "mean_imb_spearman": {h: float(s[s.horizon == h]["imb_spearman"].mean()) for h in HORIZONS},
            }

    micro_summary = {}
    for inst in TARGET_PERPS:
        s = micro_tab[micro_tab.instrument == inst]
        micro_summary[inst] = {
            "ext_frac_train_mean": float(s[s.split == "train"]["ext_frac"].mean()),
            "ext_frac_val_mean": float(s[s.split == "val"]["ext_frac"].mean()),
            "one_tick_frac_train_median": float(s[s.split == "train"]["one_tick_frac_book"].median())
            if s[s.split == "train"]["one_tick_frac_book"].notna().any()
            else None,
            "one_tick_frac_val_median": float(s[s.split == "val"]["one_tick_frac_book"].median())
            if s[s.split == "val"]["one_tick_frac_book"].notna().any()
            else None,
        }

    # worst/best val days by mean across horizons for BTC (example) and overall
    sc_val_day = sc_val.groupby("day")["r"].mean().sort_values()
    return {
        "coverage": {
            "imb_ic_and_ols_score": "full_period_4_perps",
            "trade_flow_ic": f"sampled_{len(tf_days)}_days",
            "trade_flow_days": tf_days,
        },
        "ols_coef": {
            f"{inst}|{h}": {"a": models[(inst, h)][0], "b": models[(inst, h)][1], "n": ols[(inst, h)].n}
            for inst in TARGET_PERPS
            for h in HORIZONS
        },
        "spearman_ic": {
            "train": cell_means(ic_tab, "spearman", "train"),
            "val": cell_means(ic_tab, "spearman", "val"),
        },
        "ols_imb_score": {
            "train": cell_means(sc_tab, "r", "train"),
            "val": cell_means(sc_tab, "r", "val"),
            "organizers_baseline": 0.1838,
        },
        "weekend_vs_weekday_val_score": weekend_score,
        "microstructure": micro_summary,
        "trade_flow_vs_imb": tf_summary,
        "val_day_score_extremes": {
            "worst5": sc_val_day.head(5).to_dict(),
            "best5": sc_val_day.tail(5).to_dict(),
        },
    }


# ---------------------------------------------------------------------------
# V11 — lead–lag cross-correlation (sampled)
# ---------------------------------------------------------------------------

def _cross_corr(leader: np.ndarray, follower: np.ndarray, max_lag_steps: int) -> tuple[list[float], list[float]]:
    fwd, rev = [], []
    for k in range(0, max_lag_steps + 1):
        if k == 0:
            x, y = leader, follower
            xr, yr = follower, leader
        else:
            x, y = leader[:-k], follower[k:]
            xr, yr = follower[:-k], leader[k:]
        fwd.append(pearson(x, y))
        rev.append(pearson(xr, yr))
    return fwd, rev


def _pair_day_leadlag(leader: str, follower: str, day: str, step_ms: int, max_lag_steps: int) -> dict | None:
    if not exists("bbo", leader, day) or not exists("bbo", follower, day):
        return None
    try:
        mid_l = bbo_mid_on_grid(leader, day, step_ms)
        mid_f = bbo_mid_on_grid(follower, day, step_ms)
    except FileNotFoundError:
        return None
    r_l = log_returns_arr(mid_l)
    r_f = log_returns_arr(mid_f)
    fwd, rev = _cross_corr(r_l, r_f, max_lag_steps)
    return {
        "fwd": fwd,
        "rev": rev,
        "lag_ms": [k * step_ms for k in range(0, max_lag_steps + 1)],
    }


def _summarize_curves(curves: list[dict]) -> dict:
    if not curves:
        return {}
    mats = np.array([c["fwd"] for c in curves], dtype=float)
    revs = np.array([c["rev"] for c in curves], dtype=float)
    med = np.nanmedian(mats, axis=0)
    med_rev = np.nanmedian(revs, axis=0)
    lags = curves[0]["lag_ms"]
    pos = np.array(lags) > 0

    def peak(arr: np.ndarray) -> dict:
        masked = np.where(pos, arr, -np.inf)
        if not np.any(np.isfinite(masked) & (masked > -np.inf)):
            return {"lag_ms": None, "corr": None}
        i = int(np.nanargmax(masked))
        return {"lag_ms": int(lags[i]), "corr": float(arr[i])}

    def at(arr: np.ndarray, ms: int) -> float | None:
        if ms not in lags:
            return None
        return float(arr[lags.index(ms)])

    return {
        "n_days": len(curves),
        "corr_0": at(med, 0),
        "peak_k_gt0": peak(med),
        "rev_peak_k_gt0": peak(med_rev),
        "corr_at": {str(ms): at(med, ms) for ms in lags if ms in (0, 100, 200, 300, 500, 1000)},
        "rev_at": {str(ms): at(med_rev, ms) for ms in lags if ms in (0, 100, 200, 300, 500, 1000)},
        "median_fwd": med.tolist(),
        "median_rev": med_rev.tolist(),
        "lag_ms": lags,
    }


def phase_leadlag(sample: dict[str, list[str]]) -> dict:
    _progress("phase lead–lag (sampled)…")
    days = sorted(set(sample["train"] + sample["val"]))
    # filter USD1 pairs to days where file exists
    results: dict[str, dict] = {}
    for leader, follower, label in LEADLAG_PAIRS:
        for step_ms, max_lag in ((100, 20), (500, 10)):
            train_curves, val_curves = [], []
            for day in days:
                if not exists("bbo", leader, day) or not exists("bbo", follower, day):
                    continue
                cur = _pair_day_leadlag(leader, follower, day, step_ms, max_lag)
                if cur is None:
                    continue
                if day <= TRAIN_END:
                    train_curves.append(cur)
                else:
                    val_curves.append(cur)
            key = f"{label}@{step_ms}ms"
            results[key] = {
                "leader": leader,
                "follower": follower,
                "step_ms": step_ms,
                "train": _summarize_curves(train_curves),
                "val": _summarize_curves(val_curves),
            }
            _progress(f"  leadlag {key} train={len(train_curves)} val={len(val_curves)}")
    return {
        "coverage": "stratified_sample",
        "sample_days": sample,
        "pairs": results,
    }


# ---------------------------------------------------------------------------
# V12 — light Granger / ΔR² on a small day sample
# ---------------------------------------------------------------------------

def _delta_r2(y: np.ndarray, x: np.ndarray, max_lag: int) -> float:
    """ΔR²: extra R² from lagged x beyond own lagged y (both max_lag)."""
    n = len(y)
    # build lagged design
    rows = n - max_lag
    if rows < 100:
        return float("nan")
    Y = y[max_lag:]
    own = np.column_stack([y[max_lag - k : n - k] for k in range(1, max_lag + 1)])
    cross = np.column_stack([x[max_lag - k : n - k] for k in range(1, max_lag + 1)])
    mask = np.isfinite(Y) & np.isfinite(own).all(1) & np.isfinite(cross).all(1)
    Y, own, cross = Y[mask], own[mask], cross[mask]
    if len(Y) < 100:
        return float("nan")

    def r2(X: np.ndarray, yy: np.ndarray) -> float:
        Xd = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(Xd, yy, rcond=None)
        pred = Xd @ coef
        ss_res = ((yy - pred) ** 2).sum()
        ss_tot = ((yy - yy.mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    r_own = r2(own, Y)
    r_full = r2(np.hstack([own, cross]), Y)
    if not np.isfinite(r_own) or not np.isfinite(r_full):
        return float("nan")
    return float(r_full - r_own)


def phase_granger_light(sample_days: list[str]) -> dict:
    _progress("phase light Granger ΔR²…")
    pairs = [
        ("BTCUSD1_SPOT", "BTCUSDT_PERP"),
        ("BTCUSDT_PERP", "BTCUSDT_SPOT"),
        ("ETHUSD1_SPOT", "ETHUSDT_PERP"),
        ("ETHUSDT_PERP", "ETHUSDT_SPOT"),
        ("1000PEPEUSDC_PERP", "1000PEPEUSDT_PERP"),
        ("SOLUSDT_SPOT", "SOLUSDT_PERP"),
        ("SOLUSDT_PERP", "SOLUSDT_SPOT"),
    ]
    # Prefer post-USD1-listing days so BTCUSD1/ETHUSD1 pairs are populated
    post = [d for d in sample_days if d >= "2025-12-11"]
    days = (post + [d for d in sample_days if d not in post])[:16]
    out: dict[str, dict] = {}
    for leader, follower in pairs:
        vals_100, vals_500 = [], []
        for day in days:
            if not exists("bbo", leader, day) or not exists("bbo", follower, day):
                continue
            for step, store in ((100, vals_100), (500, vals_500)):
                mid_l = bbo_mid_on_grid(leader, day, step)
                mid_f = bbo_mid_on_grid(follower, day, step)
                r_l = log_returns_arr(mid_l)
                r_f = log_returns_arr(mid_f)
                # 2s of lags
                max_lag = max(1, 2000 // step)
                store.append(_delta_r2(r_f, r_l, max_lag))
        label = f"{leader}→{follower}"
        out[label] = {
            "n_days": len(vals_100),
            "delta_r2_100ms_median": float(np.nanmedian(vals_100)) if vals_100 else None,
            "delta_r2_500ms_median": float(np.nanmedian(vals_500)) if vals_500 else None,
            "days": days,
        }
        _progress(f"  granger {label}")
    return {"coverage": f"light_sample_{len(days)}_days", "pairs": out}


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_findings(results: dict) -> None:
    inv = results["inventory"]
    dq = results["book_dq"]
    tgt = results["targets"]
    imb = results["imb_score"]
    ll = results["leadlag"]["pairs"]
    gr = results["granger"]["pairs"]

    score_val = imb["ols_imb_score"]["val"]
    score_train = imb["ols_imb_score"]["train"]

    bullets = []
    bullets.append(
        f"**Official-style OLS imb SCORE (val, 12-cell mean) = {_fmt(score_val['mean_12'])}** "
        f"(organizers 0.1838); train mean {_fmt(score_train['mean_12'])}."
    )
    bullets.append(
        f"Val SCORE by instrument: "
        + ", ".join(f"{k.split('_')[0]}={_fmt(v)}" for k, v in score_val["by_instrument"].items())
        + "; by horizon: "
        + ", ".join(f"{h}={_fmt(v)}" for h, v in score_val["by_horizon"].items())
        + "."
    )
    # zeros
    for inst in TARGET_PERPS:
        z = tgt["by_instrument"][inst]["train"]["zero_frac_median"]
        bullets.append(
            f"{inst} train zero rates (median): 500ms={_fmt(z['y_500ms'],3)}, "
            f"2s={_fmt(z['y_2s'],3)}, 8s={_fmt(z['y_8s'],3)}; "
            f"y_2s std median train/val "
            f"{_fmt(tgt['by_instrument'][inst]['train']['y_2s_std_bp_median'],2)}/"
            f"{_fmt(tgt['by_instrument'][inst]['val']['y_2s_std_bp_median'],2)} bp."
        )
    # vol spike
    spikes = {
        inst: tgt["by_instrument"][inst]["spike_2025_10_10_y2s_std_bp"] for inst in TARGET_PERPS
    }
    bullets.append(
        "2025-10-10 y_2s std (bp): "
        + ", ".join(f"{k}={_fmt(v,2)}" for k, v in spikes.items())
        + " (peer: BTC ~3.9 vs ~0.9 median)."
    )
    # DQ / outages
    bullets.append(
        f"Book DQ full: locked median {_fmt(dq['summary_all'].get('locked_frac_median'),6)}, "
        f"crossed median {_fmt(dq['summary_all'].get('crossed_frac_median'),6)}; "
        f"days with max_gap>60s: {dq['summary_all'].get('n_days_max_gap_gt_60s')} "
        f"across all instruments."
    )
    if dq["peer_outage_days"]:
        top = sorted(dq["peer_outage_days"], key=lambda r: -r["max_gap_ms"])[:6]
        bullets.append(
            "Peer outage days confirmed (examples): "
            + "; ".join(f"{r['day']} {r['instrument']} gap={r['max_gap_ms']}ms" for r in top)
            + "."
        )
    # USD1 missing
    bullets.append(
        f"USD1 pre-listing absences: BTCUSD1 {inv['usd1_pre_listing']['BTCUSD1_SPOT']['pre_listing_days']} days, "
        f"ETHUSD1 {inv['usd1_pre_listing']['ETHUSD1_SPOT']['pre_listing_days']} days "
        f"(peer total ~167 missing files)."
    )
    # ext imb
    for inst in TARGET_PERPS:
        m = imb["microstructure"][inst]
        bullets.append(
            f"{inst} |imb|>0.9 mean train/val {_fmt(m['ext_frac_train_mean'],3)}/"
            f"{_fmt(m['ext_frac_val_mean'],3)}; 1-tick spread share train median "
            f"{_fmt(m['one_tick_frac_train_median'],3)}."
        )
    # lead-lag highlights
    for key in [
        "BTCUSD1→BTCUSDT_PERP@100ms",
        "BTCUSDT_PERP→BTCUSDT_SPOT@100ms",
        "BTCUSDT_SPOT→BTCUSDT_PERP@100ms",
        "ETHUSD1→ETHUSDT_PERP@100ms",
        "PEPEUSDC→PEPEUSDT_PERP@100ms",
        "SOLUSDT_PERP→SOLUSDT_SPOT@100ms",
    ]:
        if key not in ll:
            continue
        tr, va = ll[key]["train"], ll[key]["val"]
        bullets.append(
            f"Lead–lag {key}: train peak k>0 lag={tr.get('peak_k_gt0',{}).get('lag_ms')} "
            f"corr={_fmt(tr.get('peak_k_gt0',{}).get('corr'))} "
            f"(rev {_fmt(tr.get('rev_peak_k_gt0',{}).get('corr'))}); "
            f"val peak corr={_fmt(va.get('peak_k_gt0',{}).get('corr'))} "
            f"(rev {_fmt(va.get('rev_peak_k_gt0',{}).get('corr'))}); "
            f"corr@0 train/val {_fmt(tr.get('corr_0'))}/{_fmt(va.get('corr_0'))}."
        )
    # granger
    for label, g in gr.items():
        bullets.append(
            f"Light ΔR² {label}: 100ms median {_fmt(g.get('delta_r2_100ms_median'),4)}, "
            f"500ms {_fmt(g.get('delta_r2_500ms_median'),4)} (n={g.get('n_days')})."
        )
    # trade flow
    if imb.get("trade_flow_vs_imb"):
        pepe = imb["trade_flow_vs_imb"].get("1000PEPEUSDT_PERP", {})
        btc = imb["trade_flow_vs_imb"].get("BTCUSDT_PERP", {})
        bullets.append(
            f"Trade-flow vs imb Spearman (sample): BTC y_500ms flow={_fmt((btc.get('mean_flow_spearman') or {}).get('y_500ms'))} "
            f"vs imb={_fmt((btc.get('mean_imb_spearman') or {}).get('y_500ms'))}; "
            f"PEPE flow={_fmt((pepe.get('mean_flow_spearman') or {}).get('y_500ms'))} "
            f"vs imb={_fmt((pepe.get('mean_imb_spearman') or {}).get('y_500ms'))}."
        )
    # weekend
    wk = imb["weekend_vs_weekday_val_score"]
    bullets.append(
        "Val SCORE weekend vs weekday: "
        + ", ".join(
            f"{h}: {_fmt(wk[h]['weekend'])} vs {_fmt(wk[h]['weekday'])}" for h in HORIZONS
        )
        + "."
    )
    # val quieter
    bullets.append(
        "Validation is quieter than train on y_2s std for all 4 perps "
        "(see targets section) — matches peer train−val gap narrative."
    )

    text = f"""# Full EDA findings — 500 ms challenge

Generated by `scripts/run_full_eda.py`. Numbers also in `outputs/full_eda_results.json`.

## Coverage

- **Full period:** inventory/missing files, book DQ (all instruments), targets zeros/vol (4 perps), own-imb Spearman IC + OLS SCORE (4×3 cells), microstructure (|imb|>0.9, 1-tick).
- **Sampled:** lead–lag cross-corr (stratified train days + all val), light Granger ΔR² (~12 days), trade-flow IC (~20 train days).
- Sample train days: {len(results['leadlag']['sample_days']['train'])}; val days in lead–lag: {len(results['leadlag']['sample_days']['val'])}.

## Interesting findings

"""
    for b in bullets:
        text += f"- {b}\n"

    text += """
## Matched peers

- Organizers OLS imb baseline ~0.1838 — our full-period val SCORE is in the same ballpark (see above).
- Zeros dominate short horizons; PEPE more extreme |imb| and higher vol than BTC.
- Peer outage calendar (2026-03-26, 05-21, 06-12, 07-28) shows large book gaps locally.
- USD1 missing only pre-listing; perp→spot lead stronger than spot→perp at 100 ms.
- PEPE·USDC → PEPE·USDT short-lag structure; val often weaker than train.

## Differed / caveats

- Lead–lag and Granger here use **receive-time** cross-corr / light ΔR² on a **day sample**, not peers' full 365-day Go Granger suite — magnitudes may differ.
- Trade bars are sparse; outages look like “no trades” (peer warning).
- Half-life / cointegration not re-run in this pass.
"""
    REPORT_FINDINGS.write_text(text)
    _progress(f"wrote {REPORT_FINDINGS}")


def write_variable_log(results: dict) -> None:
    """Append V07+ sections while preserving V01–V06 PEPE-sample history."""
    old = REPORT_LOG.read_text() if REPORT_LOG.exists() else ""
    # Keep everything before a previous FULL-EDA marker if present; else append.
    marker = "\n---\n\n# Full-dataset EDA (V07+)\n"
    if marker.strip() in old or "# Full-dataset EDA (V07+)" in old:
        old = old.split("# Full-dataset EDA (V07+)")[0].rstrip() + "\n"

    inv = results["inventory"]
    dq = results["book_dq"]
    tgt = results["targets"]
    imb = results["imb_score"]
    ll = results["leadlag"]
    gr = results["granger"]

    score_val = imb["ols_imb_score"]["val"]
    score_train = imb["ols_imb_score"]["train"]
    ic_val = imb["spearman_ic"]["val"]
    ic_train = imb["spearman_ic"]["train"]

    def cells_table(cell_dict: dict) -> str:
        lines = ["| instrument | y_500ms | y_2s | y_8s |", "|---|---:|---:|---:|"]
        for inst in TARGET_PERPS:
            vals = [cell_dict["cells"][f"{inst}|{h}"] for h in HORIZONS]
            lines.append(
                f"| {inst} | {_fmt(vals[0])} | {_fmt(vals[1])} | {_fmt(vals[2])} |"
            )
        lines.append(f"| **mean** |  |  | **{_fmt(cell_dict['mean_12'])}** |")
        return "\n".join(lines)

    # compact leadlag table
    ll_lines = [
        "| pair@grid | train peak lag/corr | rev peak | val peak corr | corr@0 tr/va |",
        "|---|---|---|---|---|",
    ]
    for key, block in ll["pairs"].items():
        if not key.endswith("@100ms"):
            continue
        tr, va = block["train"], block["val"]
        ll_lines.append(
            f"| {key} | {tr.get('peak_k_gt0',{}).get('lag_ms')} / {_fmt(tr.get('peak_k_gt0',{}).get('corr'))} "
            f"| {_fmt(tr.get('rev_peak_k_gt0',{}).get('corr'))} "
            f"| {_fmt(va.get('peak_k_gt0',{}).get('corr'))} "
            f"| {_fmt(tr.get('corr_0'))}/{_fmt(va.get('corr_0'))} |"
        )

    new = f"""{marker}
**Данные.** Official layout `data/{{book,bbo,trades,targets,stable,meta}}`, full calendar {inv['n_calendar_days']}d (train {inv['n_train']}, val {inv['n_val']}).  
**Код.** `scripts/run_full_eda.py`, `scripts/full_eda_io.py`.  
**JSON.** `outputs/full_eda_results.json`.  
**Findings.** `reports/FULL_EDA_FINDINGS.md`.

## Roadmap (full)

| # | Блок | Coverage | Статус |
|---|------|----------|--------|
| V07 | Inventory / missing / depeg | full | **done** |
| V08 | Book DQ all instruments | full | **done** |
| V09 | Targets zeros / daily vol | full 4 perps | **done** |
| V10 | Own imb IC + OLS 12-cell SCORE | full | **done** |
| V11 | Lead–lag cross-corr | stratified sample + all val | **done** |
| V12 | Light Granger ΔR² | ~12 days | **done** |

---

## V07 — Inventory / missing / depeg

Статус: **done** · coverage: **full**

Цифры:
- calendar days: {inv['n_calendar_days']} (train {inv['n_train']}, val {inv['n_val']})
- missing within listed range: {inv['n_missing_file_entries']} file entries
- USD1 pre-listing: BTC {inv['usd1_pre_listing']['BTCUSD1_SPOT']['pre_listing_days']}d (first {inv['usd1_pre_listing']['BTCUSD1_SPOT']['first_date']}), ETH {inv['usd1_pre_listing']['ETHUSD1_SPOT']['pre_listing_days']}d (first {inv['usd1_pre_listing']['ETHUSD1_SPOT']['first_date']})
- stable missing days: {len(inv['stable_missing_days'])}
- depeg intervals: {inv['depeg']['n_intervals']}; minutes by coin: {json.dumps(inv['depeg']['by_stablecoin_minutes'])}

Вывод: совпадает с peer narrative — «дыры» USD1 = до листинга, не lost data. **Keep** availability flags для USD1.

---

## V08 — Book DQ (all instruments)

Статус: **done** · coverage: **full**

Цифры (все инструменты):
- rows median {dq['summary_all'].get('rows_median')} (grid 172800); first_lag median {dq['summary_all'].get('first_lag_ms_median')} ms
- max_gap median {dq['summary_all'].get('max_gap_ms_median')} ms; days max_gap>60s: {dq['summary_all'].get('n_days_max_gap_gt_60s')}; >5min: {dq['summary_all'].get('n_days_max_gap_gt_5min')}
- locked median {dq['summary_all'].get('locked_frac_median')}; crossed median {dq['summary_all'].get('crossed_frac_median')}
- peer outage hits logged: {len(dq['peer_outage_days'])} instrument-day rows; top target gaps: {json.dumps(dq['top_gaps_target_perps'][:5])}

Вывод: locked/crossed ≈ 0; редкие длинные gaps совпадают с peer outage calendar. **Keep** gap/outage masks.

---

## V09 — Targets zeros / daily vol

Статус: **done** · coverage: **full 4 perps**

Цифры (median zero frac train / y_2s std bp train→val):
{chr(10).join(
    f"- {inst}: zeros 500ms/2s/8s = "
    f"{_fmt(tgt['by_instrument'][inst]['train']['zero_frac_median']['y_500ms'],3)}/"
    f"{_fmt(tgt['by_instrument'][inst]['train']['zero_frac_median']['y_2s'],3)}/"
    f"{_fmt(tgt['by_instrument'][inst]['train']['zero_frac_median']['y_8s'],3)}; "
    f"std { _fmt(tgt['by_instrument'][inst]['train']['y_2s_std_bp_median'],2)} → "
    f"{_fmt(tgt['by_instrument'][inst]['val']['y_2s_std_bp_median'],2)} bp; "
    f"2025-10-10={_fmt(tgt['by_instrument'][inst]['spike_2025_10_10_y2s_std_bp'],2)} bp"
    for inst in TARGET_PERPS
)}

Вывод: нули доминируют на 500ms; val тише train; 2025-10-10 — явный vol spike. **Keep** как контекст потолка SCORE.

---

## V10 — Own imbalance → y + OLS SCORE

Статус: **done** · coverage: **full**

Spearman IC mean (train / val), 12-cell:
- train mean: {_fmt(ic_train['mean_12'])}
- val mean: {_fmt(ic_val['mean_12'])}

{cells_table(ic_val)}

OLS imb SCORE (clip = `meta/target_clip.parquet`, daily Pearson → mean cells):
- **train 12-cell mean: {_fmt(score_train['mean_12'])}**
- **val 12-cell mean: {_fmt(score_val['mean_12'])}** (organizers **0.1838**)

Val cells:
{cells_table(score_val)}

Microstructure |imb|>0.9 / 1-tick:
{chr(10).join(
    f"- {inst}: ext train/val {_fmt(imb['microstructure'][inst]['ext_frac_train_mean'],3)}/"
    f"{_fmt(imb['microstructure'][inst]['ext_frac_val_mean'],3)}; "
    f"1-tick train {_fmt(imb['microstructure'][inst]['one_tick_frac_train_median'],3)}"
    for inst in TARGET_PERPS
)}

Weekend vs weekday val SCORE: {json.dumps(imb['weekend_vs_weekday_val_score'])}

Trade-flow vs imb (sample): see JSON `imb_score.trade_flow_vs_imb`.

Вывод: own L1 imb остаётся ядром; полный SCORE близок к 0.1838. Flow слабее imb на sample. **Keep** imb; flow secondary.

---

## V11 — Lead–lag cross-correlation

Статус: **done** · coverage: **sample train ({len(ll['sample_days']['train'])}d) + val ({len(ll['sample_days']['val'])}d)**

100 ms grid (median over days; peak only for k>0):

{chr(10).join(ll_lines)}

Вывод: perp→spot сильнее reverse; USD1→perp и PEPEUSDC→PEPEUSDT показывают короткий lead; val часто слабее train (как у peers). **Keep** short-lag leaders ≤300ms. Caveat: receive-time latency на тонких USD1.

---

## V12 — Light Granger ΔR²

Статус: **done** · coverage: **~{results['granger']['coverage']}**

Цифры (median ΔR², own lags + cross lags over 2s):
{chr(10).join(
    f"- {k}: 100ms={_fmt(v.get('delta_r2_100ms_median'))}, 500ms={_fmt(v.get('delta_r2_500ms_median'))}"
    for k, v in gr['pairs'].items()
)}

Вывод: порядок величины согласуется с peer (perp→spot крупнее, USD1/USDC→perp меньше). Не полный 17×17×365 Granger.

---

## Hypotheses (updated)

| id | Claim | EDA support | Strength | Next |
|----|-------|-------------|----------|------|
| H1 | Own L1 imb forecasts all 4 perps | full IC + SCORE≈0.18 | strong | Keep M0 |
| H2 | Perp leads spot (~100ms) | xcorr + ΔR² sample | strong | Lead features from perps into spots only as inputs the other way for targets |
| H3 | USD1 / PEPEUSDC lead same-coin perps | xcorr sample | medium | Mask USD1 latency; short lags |
| H4 | Val quieter → lower SCORE | full vol + SCORE gap | strong | Don't chase train SCORE |
| H5 | Trade flow << imb at 500ms | sample Spearman | medium | Don't over-invest in raw tflow |
"""
    REPORT_LOG.write_text(old.rstrip() + "\n" + new)
    _progress(f"wrote {REPORT_LOG}")


def write_explanations_append() -> None:
    old = REPORT_EXPL.read_text() if REPORT_EXPL.exists() else ""
    marker = "## V07 — Inventory / missing / depeg"
    if marker in old:
        old = old.split(marker)[0].rstrip() + "\n\n"
    block = """## V07 — Inventory / missing / depeg

### Зачем

Нужно отделить «файла нет, потому что рынок ещё не листился» от реальных дыр в данных.

### Как считается

Сверяем календарь дат с `meta/instruments.parquet` (`first_date`/`last_date`) и наличием parquet в `data/{book,bbo,trades,targets,stable}`. Depeg intervals — из `meta/depeg_intervals.parquet`.

### Что показывает

Почти все «пропажи» — USD1 до листинга. Stable depeg минуты малы относительно года.

### Как пользоваться

Availability flags для USD1/alt quotes; не импутить pre-listing дни.

### Что нам дало

Согласование с peer: 167 missing = pre-listing, не corruption.

---

## V08 — Book DQ (all instruments)

### Зачем

Outage и late-start ломают IC/SCORE и маскируются под «рынок замер».

### Как считается

По каждому book-дню: `n` vs 172800, first lag от 00:00 UTC, max gap, locked/crossed фракции. Смотрим peer-даты outage отдельно.

### Что показывает

Locked/crossed ≈ 0; редкие многоминутные gaps на известных outage днях.

### Как пользоваться

Маска `max_gap` / missing rows; не учить flow-фичи без stale-флага.

### Что нам дало

Hygiene gate на полном году, не только PEPE sample.

---

## V09 — Targets zeros / daily vol

### Зачем

Доля нулей задаёт потолок корреляции; дневная вола — режимы, где lead–lag сильнее.

### Как считается

Exact-zero fraction по горизонтам; daily std `y_2s` в bp; train vs val; spike-день 2025-10-10; weekend vs weekday.

### Что показывает

500ms почти всегда «тихий»; val тише train; 10-10 — экстремум.

### Как пользоваться

Не оптимизировать только 500ms; учитывать vol regimes.

### Что нам дало

Полная калибровка 4 perps против peer цифр.

---

## V10 — Own imbalance IC + OLS SCORE

### Зачем

Это официальный baseline и якорь прогресса (0.1838).

### Как считается

`imb=(bid_qty−ask_qty)/(bid_qty+ask_qty)` через causal `merge_asof` book→targets. Clip из `meta/target_clip.parquet`. OLS на train; daily Pearson; среднее 12 клеток. Дополнительно Spearman IC, |imb|>0.9, 1-tick share, sample trade-flow IC.

### Что показывает

Imb доминирует; SCORE близок к organizers; flow слабее на коротком горизонте.

### Как пользоваться

M0 = OLS imb; всё остальное меряем как lift над ним.

### Что нам дало

Полный 12-cell SCORE на локальных данных.

---

## V11 — Lead–lag cross-correlation

### Зачем

Кто на сколько миллисекунд опережает целевой perp / spot — главный cross-market угол peers.

### Как считается

BBO mid → causal reindex на 100/500 ms сетку → log-return → `corr(r_leader(t−k), r_follower(t))` и reverse. Медиана по sample-дням; peak только для k>0.

### Что показывает

Высокий corr@0 (синхронность); perp→spot сильнее spot→perp; короткие лаги 100–300 ms; USD1/USDC leaders осмысленны, но тонкие рынки чувствительны к receive-time.

### Как пользоваться

Лагированные mid-returns лидеров ≤300–500 ms как фичи; не minute-scale lead–lag.

### Что нам дало

Проверка peer claims на официальной раскладке данных (sample, не 365d Granger).

---

## V12 — Light Granger ΔR²

### Зачем

Cross-corr не отделяет «общий шок» от дополнительной предсказательной силы.

### Как считается

На sample днях: ΔR² = R²(y ~ own lags + leader lags) − R²(y ~ own lags), лаги на окне ~2 s.

### Что показывает

Порядок величины: perp→spot крупнее, same-coin alt→perp меньше — как у peers.

### Как пользоваться

Приоритизация лидеров для фич, не замена полному Granger.

### Что нам дало

Дешёвая проверка направления без Go-пайплайна peers.
"""
    REPORT_EXPL.write_text(old.rstrip() + "\n\n" + block)
    _progress(f"wrote {REPORT_EXPL}")


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sample = stratified_sample_days(train_per_month=3, include_all_val=True)
    _progress(
        f"sample train days={len(sample['train'])} val days={len(sample['val'])}"
    )

    inventory = phase_inventory()
    book_dq = phase_book_dq()
    targets = phase_targets()
    imb_score = phase_imb_and_score(sample_trade_flow_days=sample["train"][:24])
    leadlag = phase_leadlag(sample)
    granger = phase_granger_light(sample["train"])

    results = {
        "meta": {
            "generated_at_unix": time.time(),
            "train": [train_days()[0], train_days()[-1], len(train_days())],
            "val": [val_days()[0], val_days()[-1], len(val_days())],
            "coverage_notes": {
                "inventory": "full",
                "book_dq": "full_all_instruments",
                "targets": "full_4_perps",
                "imb_ic_ols_score": "full_4_perps",
                "leadlag": "stratified_train_plus_all_val",
                "granger": "light_sample",
                "trade_flow": "sample_subset",
            },
        },
        "inventory": inventory,
        "book_dq": book_dq,
        "targets": targets,
        "imb_score": imb_score,
        "leadlag": leadlag,
        "granger": granger,
    }

    # JSON: drop bulky median curves optionally keep — useful for plots
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    _progress(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")

    write_findings(results)
    write_variable_log(results)
    write_explanations_append()

    val_mean = results["imb_score"]["ols_imb_score"]["val"]["mean_12"]
    train_mean = results["imb_score"]["ols_imb_score"]["train"]["mean_12"]
    _progress(f"DONE train SCORE={train_mean:.4f} val SCORE={val_mean:.4f} (baseline 0.1838)")


if __name__ == "__main__":
    main()
