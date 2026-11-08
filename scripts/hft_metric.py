"""Competition-style clip + daily Pearson SCORE (PEPE-only variant)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hft_common import HORIZONS


@dataclass
class ClipBounds:
    lo: dict[str, float]
    hi: dict[str, float]


def fit_clip_bounds(train_frames: list[pd.DataFrame], q_lo: float = 0.001, q_hi: float = 0.999) -> ClipBounds:
    lo, hi = {}, {}
    for h in HORIZONS:
        vals = np.concatenate([df[h].to_numpy(dtype=np.float64) for df in train_frames])
        vals = vals[np.isfinite(vals)]
        lo[h] = float(np.quantile(vals, q_lo))
        hi[h] = float(np.quantile(vals, q_hi))
    return ClipBounds(lo=lo, hi=hi)


def clip_series(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(y, lo, hi)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    x = a[mask]
    y = b[mask]
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    if den == 0:
        return float("nan")
    return float((x * y).sum() / den)


def daily_scores(
    days: list[str],
    frames: dict[str, pd.DataFrame],
    pred_cols: dict[str, str],
    bounds: ClipBounds,
) -> pd.DataFrame:
    rows = []
    for day in days:
        df = frames[day]
        for h in HORIZONS:
            y = df[h].to_numpy(dtype=np.float64)
            y = clip_series(y, bounds.lo[h], bounds.hi[h])
            p = df[pred_cols[h]].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "day": day,
                    "horizon": h,
                    "r": pearson(p, y),
                    "n": int(np.sum(np.isfinite(y) & np.isfinite(p))),
                }
            )
    return pd.DataFrame(rows)


def mean_score(daily: pd.DataFrame) -> dict:
    by_h = daily.groupby("horizon")["r"].mean()
    return {
        "by_horizon": {k: float(v) for k, v in by_h.items()},
        "mean": float(by_h.mean()),
    }


def ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10:
        return 0.0, 0.0
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_imbalance_baseline(
    train_frames: list[pd.DataFrame],
    bounds: ClipBounds,
) -> dict[str, tuple[float, float]]:
    models = {}
    for h in HORIZONS:
        xs, ys = [], []
        for df in train_frames:
            x = df["imb"].to_numpy(dtype=np.float64)
            y = clip_series(df[h].to_numpy(dtype=np.float64), bounds.lo[h], bounds.hi[h])
            xs.append(x)
            ys.append(y)
        a, b = ols_fit(np.concatenate(xs), np.concatenate(ys))
        models[h] = (a, b)
    return models


def apply_imbalance_baseline(df: pd.DataFrame, models: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for h, (a, b) in models.items():
        out[f"pred_{h}"] = a + b * out["imb"].to_numpy(dtype=np.float64)
    return out
