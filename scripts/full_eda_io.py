"""I/O helpers for the official unpacked layout under data/{book,bbo,trades,targets,stable,meta}."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MS_DAY = 86_400_000
GRID_MS = 500
POINTS_PER_DAY = MS_DAY // GRID_MS  # 172_800

TRAIN_START = "2025-09-15"
TRAIN_END = "2026-07-14"
VAL_START = "2026-07-15"
VAL_END = "2026-09-14"

HORIZONS = ("y_500ms", "y_2s", "y_8s")

TARGET_PERPS = (
    "BTCUSDT_PERP",
    "ETHUSDT_PERP",
    "SOLUSDT_PERP",
    "1000PEPEUSDT_PERP",
)

STREAMS = ("book", "bbo", "trades")


def all_calendar_days(start: str = TRAIN_START, end: str = VAL_END) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def train_days() -> list[str]:
    return all_calendar_days(TRAIN_START, TRAIN_END)


def val_days() -> list[str]:
    return all_calendar_days(VAL_START, VAL_END)


def split_of(day: str) -> str:
    return "train" if day <= TRAIN_END else "val"


def day_start_ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def path_for(stream: str, instrument: str, day: str) -> Path:
    if stream == "stable":
        return DATA / "stable" / f"{day}.parquet"
    return DATA / stream / instrument / f"{day}.parquet"


def exists(stream: str, instrument: str, day: str) -> bool:
    return path_for(stream, instrument, day).exists()


def read_cols(stream: str, instrument: str, day: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = path_for(stream, instrument, day)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pq.read_table(path, columns=columns)
    return table.to_pandas()


def list_instruments(stream: str = "book") -> list[str]:
    return sorted(p.name for p in (DATA / stream).iterdir() if p.is_dir())


def load_instruments_meta() -> pd.DataFrame:
    return pd.read_parquet(DATA / "meta" / "instruments.parquet")


def load_target_clip() -> dict[tuple[str, str], tuple[float, float]]:
    df = pd.read_parquet(DATA / "meta" / "target_clip.parquet")
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for _, r in df.iterrows():
        out[(r["instrument"], r["horizon"])] = (float(r["lo"]), float(r["hi"]))
    return out


def load_daily_tick_sizes() -> pd.DataFrame:
    return pd.read_parquet(DATA / "meta" / "daily.parquet")[["instrument", "date", "tick_size"]]


def load_depeg() -> pd.DataFrame:
    return pd.read_parquet(DATA / "meta" / "depeg_intervals.parquet")


def imbalance_from_qty(bid_qty: np.ndarray, ask_qty: np.ndarray) -> np.ndarray:
    bid = bid_qty.astype(np.float64, copy=False)
    ask = ask_qty.astype(np.float64, copy=False)
    den = bid + ask
    out = np.zeros_like(den)
    np.divide(bid - ask, den, out=out, where=den > 0)
    return out


def mid_from_px(bid_px: np.ndarray, ask_px: np.ndarray) -> np.ndarray:
    return 0.5 * (bid_px.astype(np.float64, copy=False) + ask_px.astype(np.float64, copy=False))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    x = a[mask] - a[mask].mean()
    y = b[mask] - b[mask].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    if den == 0:
        return float("nan")
    return float((x * y).sum() / den)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(a[mask]).rank().to_numpy()
    yr = pd.Series(b[mask]).rank().to_numpy()
    return pearson(xr, yr)


def log_returns_arr(mid: np.ndarray) -> np.ndarray:
    r = np.full(len(mid), np.nan, dtype=np.float64)
    valid = np.isfinite(mid) & (mid > 0)
    for i in range(1, len(mid)):
        if valid[i] and valid[i - 1]:
            r[i] = np.log(mid[i] / mid[i - 1])
    return r


def causal_reindex_mid(ts: np.ndarray, mid: np.ndarray, day: str, step_ms: int) -> np.ndarray:
    """Place irregular mids onto a regular grid with merge_asof backward (causal)."""
    start = day_start_ms(day)
    grid = np.arange(start, start + MS_DAY, step_ms, dtype=np.int64)
    order = np.argsort(ts)
    df = pd.DataFrame({"ts": ts[order], "mid": mid[order]})
    g = pd.DataFrame({"ts": grid})
    m = pd.merge_asof(g, df, on="ts", direction="backward")
    return m["mid"].to_numpy(dtype=np.float64)


def bbo_mid_on_grid(instrument: str, day: str, step_ms: int) -> np.ndarray:
    bbo = read_cols("bbo", instrument, day, columns=["ts", "bid_px", "ask_px"])
    mid = mid_from_px(bbo["bid_px"].to_numpy(), bbo["ask_px"].to_numpy())
    return causal_reindex_mid(bbo["ts"].to_numpy(dtype=np.int64), mid, day, step_ms)


def stratified_sample_days(
    *,
    train_per_month: int = 3,
    include_all_val: bool = True,
    extra_days: list[str] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """~train_per_month train days per calendar month + all validation (+ extras)."""
    rng = np.random.default_rng(seed)
    tr = train_days()
    by_month: dict[str, list[str]] = {}
    for d in tr:
        by_month.setdefault(d[:7], []).append(d)
    sampled: list[str] = []
    for days in by_month.values():
        k = min(train_per_month, len(days))
        # always include first weekday + random others for coverage
        chosen = set(days[:1])
        rest = [x for x in days if x not in chosen]
        if k > 1 and rest:
            pick = rng.choice(rest, size=min(k - 1, len(rest)), replace=False).tolist()
            chosen.update(pick)
        sampled.extend(sorted(chosen)[:k])
    # peer outage / spike anchors
    anchors = [
        "2025-10-10",
        "2025-12-04",
        "2026-03-26",
        "2026-05-21",
        "2026-06-12",
        "2026-07-28",
        "2026-08-08",
    ]
    for a in anchors:
        if TRAIN_START <= a <= TRAIN_END and a not in sampled and a in tr:
            sampled.append(a)
    sampled = sorted(set(sampled))
    if include_all_val:
        va = val_days()
    else:
        va = sorted(set(val_days()[::3]))
    if extra_days:
        for e in extra_days:
            if e <= TRAIN_END and e not in sampled:
                sampled.append(e)
            elif e >= VAL_START and e not in va:
                va.append(e)
        sampled = sorted(set(sampled))
        va = sorted(set(va))
    return {"train": sampled, "val": va}
