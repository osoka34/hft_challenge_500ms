"""Truncation / causality templates for feature builders.

Pattern for a new column:
1. Build the feature on a full synthetic series.
2. Cut every event with ts > T (or drop rows after T).
3. Rebuild on the truncated series.
4. Assert values at t ≤ T match.

Extend this module for each new builder; prefer synthetic frames over parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nf_features import _causal_softsign, _gap_dev, cont_ofi  # noqa: E402


def test_cont_ofi_truncation():
    """L1 Cont OFI at t uses only snapshots ≤ t."""
    n = 20
    rng = np.random.default_rng(0)
    bp = 100 + np.cumsum(rng.normal(0, 0.01, n))
    ap = bp + 0.01
    bq = rng.uniform(1, 5, n)
    aq = rng.uniform(1, 5, n)
    full = cont_ofi(bp, ap, bq, aq)

    T = 12
    cut = cont_ofi(bp[: T + 1], ap[: T + 1], bq[: T + 1], aq[: T + 1])
    assert cut.shape == (T + 1,)
    np.testing.assert_allclose(full[: T + 1], cut, equal_nan=True)


def test_causal_softsign_truncation():
    """Softsign scale at t uses history before t only."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 40)
    full = _causal_softsign(x, warmup=5)

    T = 25
    cut = _causal_softsign(x[: T + 1], warmup=5)
    np.testing.assert_allclose(full[: T + 1], cut, equal_nan=True)


def test_gap_dev_truncation():
    """gap_dev rolling mean is shift(1); prefix must match after a cut."""
    rng = np.random.default_rng(2)
    own = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 80)))
    lead = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 80)))
    full = _gap_dev(own, lead, window=20)

    T = 50
    cut = _gap_dev(own[: T + 1], lead[: T + 1], window=20)
    np.testing.assert_allclose(full[: T + 1], cut, equal_nan=True)


def test_asof_join_ignores_future_prints():
    """Grid row at t keeps last event with ts ≤ t."""
    grid = pd.DataFrame({"ts": [100, 200, 300, 400]})
    events = pd.DataFrame({"ts": [100, 250, 350], "x": [1.0, 2.0, 3.0]})
    full = pd.merge_asof(grid, events, on="ts", direction="backward")

    T = 300
    truncated_events = events[events["ts"] <= T]
    cut = pd.merge_asof(grid[grid["ts"] <= T], truncated_events, on="ts", direction="backward")
    left = full[full["ts"] <= T].reset_index(drop=True)
    right = cut.reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("leaky", [False, True])
def test_truncation_detects_leak_pattern(leaky: bool):
    """Reference: a forward-looking cumsum fails truncation; causal cumsum passes."""
    x = np.arange(30, dtype=float)

    def causal(v: np.ndarray) -> np.ndarray:
        return np.cumsum(v)

    def leaky_build(v: np.ndarray) -> np.ndarray:
        # each t sees the sum of the *full* series length — depends on future rows
        return np.full(len(v), np.sum(v), dtype=float)

    build = leaky_build if leaky else causal
    full = build(x)
    T = 15
    cut = build(x[: T + 1])
    same = np.allclose(full[: T + 1], cut, equal_nan=True)
    assert same is (not leaky)
