"""Shared loaders and feature helpers for the PEPE EDA path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

MS_DAY = 86_400_000
GRID_MS = 500
POINTS_PER_DAY = MS_DAY // GRID_MS  # 172_800

TRAIN_DAYS = [f"2025-11-{d:02d}" for d in range(1, 8)]
VAL_DAYS = [f"2026-08-{d:02d}" for d in range(10, 16)]
ALL_DAYS = TRAIN_DAYS + VAL_DAYS

HORIZONS = ("y_500ms", "y_2s", "y_8s")


def day_start_ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def read_parquet(instrument: str, stream: str, day: str) -> pd.DataFrame:
    path = RAW / stream / instrument / f"{day}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def mid_from_book(df: pd.DataFrame) -> pd.Series:
    return 0.5 * (df["bid_px"].to_numpy(dtype=np.float64) + df["ask_px"].to_numpy(dtype=np.float64))


def imbalance(df: pd.DataFrame) -> np.ndarray:
    bid = df["bid_qty"].to_numpy(dtype=np.float64)
    ask = df["ask_qty"].to_numpy(dtype=np.float64)
    den = bid + ask
    out = np.zeros_like(den)
    np.divide(bid - ask, den, out=out, where=den > 0)
    return out


def align_book_to_targets(book: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """As-of merge: each target grid point gets the last book row with ts <= t."""
    b = book[["ts"]].copy()
    b["mid"] = mid_from_book(book)
    b["imb"] = imbalance(book)
    b["spread"] = book["ask_px"].to_numpy(dtype=np.float64) - book["bid_px"].to_numpy(dtype=np.float64)
    b["locked"] = book["bid_px"].to_numpy() == book["ask_px"].to_numpy()
    b["crossed"] = book["bid_px"].to_numpy() > book["ask_px"].to_numpy()
    b = b.sort_values("ts")
    t = targets.sort_values("ts").copy()
    merged = pd.merge_asof(t, b, on="ts", direction="backward")
    return merged


def load_day_usdt(day: str) -> pd.DataFrame:
    book = read_parquet("1000PEPEUSDT_PERP", "book", day)
    targets = read_parquet("1000PEPEUSDT_PERP", "targets", day)
    return align_book_to_targets(book, targets)


def load_day_pair(day: str) -> pd.DataFrame:
    """USDT targets+book joined with USDC book mids (causal asof)."""
    usdt = load_day_usdt(day)
    usdc_book = read_parquet("1000PEPEUSDC_PERP", "book", day)
    u = usdc_book[["ts"]].copy()
    u["mid_usdc"] = mid_from_book(usdc_book)
    u["imb_usdc"] = imbalance(usdc_book)
    u = u.sort_values("ts")
    out = pd.merge_asof(
        usdt.sort_values("ts"),
        u,
        on="ts",
        direction="backward",
        suffixes=("", "_usdc"),
    )
    out["gap"] = np.log(out["mid"] / out["mid_usdc"])
    return out


def bbo_mid_series(instrument: str, day: str) -> pd.Series:
    bbo = read_parquet(instrument, "bbo", day).sort_values("ts")
    mid = 0.5 * (bbo["bid_px"].to_numpy(dtype=np.float64) + bbo["ask_px"].to_numpy(dtype=np.float64))
    return pd.Series(mid, index=bbo["ts"].to_numpy(), name="mid")


def reindex_mid_grid(mid: pd.Series, day: str, step_ms: int) -> pd.Series:
    start = day_start_ms(day)
    grid = np.arange(start, start + MS_DAY, step_ms, dtype=np.int64)
    s = mid.sort_index()
    # causal: forward-fill after placing onto grid via asof
    base = pd.Series(index=grid, dtype=np.float64)
    df = pd.DataFrame({"ts": s.index, "mid": s.to_numpy()}).sort_values("ts")
    g = pd.DataFrame({"ts": grid})
    m = pd.merge_asof(g, df, on="ts", direction="backward")
    return pd.Series(m["mid"].to_numpy(), index=grid, name="mid")


def log_returns(mid: pd.Series) -> pd.Series:
    arr = mid.to_numpy(dtype=np.float64)
    r = np.full_like(arr, np.nan)
    valid = np.isfinite(arr)
    # only where both sides finite and positive
    for i in range(1, len(arr)):
        if valid[i] and valid[i - 1] and arr[i - 1] > 0 and arr[i] > 0:
            r[i] = np.log(arr[i] / arr[i - 1])
        else:
            r[i] = np.nan
    return pd.Series(r, index=mid.index, name="r")
