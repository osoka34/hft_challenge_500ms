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
from run_paper_seq import (  # noqa: E402
    apply_micro_adj_fold,
    liq_side,
    phi_cnt,
    q_depl,
    sq_micro_bp,
    sign_age_from_marks,
    sign_age_marks,
    sweep_px,
    vol_side_imb_buckets,
)


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


def test_vol_side_imb_truncation():
    """Row-local helper is prefix-stable (necessary but not sufficient for the builder)."""
    n = 24
    rng = np.random.default_rng(3)
    bid_vs = [rng.uniform(0, 5, n) for _ in range(5)]
    ask_vs = [rng.uniform(0, 5, n) for _ in range(5)]
    full = vol_side_imb_buckets(bid_vs, ask_vs)

    T = 14
    cut = vol_side_imb_buckets([v[: T + 1] for v in bid_vs], [v[: T + 1] for v in ask_vs])
    assert cut.shape == (T + 1,)
    np.testing.assert_allclose(full[: T + 1], cut, equal_nan=True)


def test_vol_side_imb_asof_ignores_future_book():
    """Builder join: grid at t keeps last book strip with ts ≤ t (as-of lock)."""
    book = pd.DataFrame(
        {
            "ts": [100, 250, 350],
            **{f"bid_v{k:02d}": [float(k), float(k + 10), float(k + 20)] for k in range(1, 6)},
            **{f"ask_v{k:02d}": [1.0, 2.0, 3.0] for k in range(1, 6)},
        }
    )
    bid_vs = [book[f"bid_v{k:02d}"].to_numpy(float) for k in range(1, 6)]
    ask_vs = [book[f"ask_v{k:02d}"].to_numpy(float) for k in range(1, 6)]
    feat = pd.DataFrame(
        {"ts": book["ts"].to_numpy(np.int64), "vol_side_imb": vol_side_imb_buckets(bid_vs, ask_vs)}
    )
    grid = pd.DataFrame({"ts": [100, 200, 300, 400]})
    full = pd.merge_asof(grid, feat, on="ts", direction="backward")

    T = 300
    truncated = feat[feat["ts"] <= T]
    cut = pd.merge_asof(grid[grid["ts"] <= T], truncated, on="ts", direction="backward")
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


def test_btc_ret_asof_ignores_future_btc():
    """Cross BTC→alt join: SOL grid at t keeps last BTC ret with ts ≤ t."""
    btc = pd.DataFrame({"ts": [100, 250, 350], "btc_ret_1_bp": [1.0, 2.0, 3.0]})
    grid = pd.DataFrame({"ts": [100, 200, 300, 400]})
    full = pd.merge_asof(grid, btc, on="ts", direction="backward")
    T = 300
    cut = pd.merge_asof(grid[grid["ts"] <= T], btc[btc["ts"] <= T], on="ts", direction="backward")
    left = full[full["ts"] <= T].reset_index(drop=True)
    right = cut.reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_micro_adj_fold_ignores_test_labels():
    """Fold-fit: te micro_adj depends on train g(I,S) only, not te next_dmid / y."""
    rng = np.random.default_rng(4)
    n_tr, n_te = 800, 200
    tr = pd.DataFrame(
        {
            "stoikov_I": rng.uniform(0, 1, n_tr),
            "stoikov_S": rng.uniform(0.5, 5.0, n_tr),
            "stoikov_next_dmid_bp": rng.normal(0, 0.3, n_tr),
            "y": rng.normal(0, 1e-4, n_tr),
        }
    )
    te = pd.DataFrame(
        {
            "stoikov_I": rng.uniform(0, 1, n_te),
            "stoikov_S": rng.uniform(0.5, 5.0, n_te),
            "stoikov_next_dmid_bp": rng.normal(0, 0.3, n_te),
            "y": rng.normal(0, 1e-4, n_te),
        }
    )
    base = apply_micro_adj_fold(tr, te)
    te_poison = te.copy()
    te_poison["stoikov_next_dmid_bp"] = rng.normal(10.0, 1.0, n_te)
    te_poison["y"] = rng.normal(10.0, 1.0, n_te)
    poisoned = apply_micro_adj_fold(tr, te_poison)
    np.testing.assert_allclose(base, poisoned, equal_nan=True)


def test_phi_cnt_truncation():
    """Φ at t sums trade counts in (t−500ms, t] only; a later bar must not move it."""
    ts = np.array([100, 200, 300, 400, 500, 600], dtype=np.int64)
    buy_n = np.array([2, 0, 1, 0, 0, 10], dtype=np.float64)
    sell_n = np.array([0, 1, 0, 0, 3, 0], dtype=np.float64)
    full = phi_cnt(ts, buy_n, sell_n, window_ms=500)
    # bars with ts in (0, 500]: +2 −1 +1 +0 −3 = −1. The +10 at 600 is outside.
    at_500 = float(full.loc[full["ts"] == 500, "phi_cnt"].iloc[0])
    assert at_500 == -1.0

    T = 500
    keep = ts <= T
    cut = phi_cnt(ts[keep], buy_n[keep], sell_n[keep], window_ms=500)
    np.testing.assert_allclose(
        full.loc[full["ts"] <= T, "phi_cnt"].to_numpy(),
        cut["phi_cnt"].to_numpy(),
        equal_nan=True,
    )

    grid = pd.DataFrame({"ts": np.array([100, 400, 500, 900], dtype=np.int64)})

    def _asof(feat: pd.DataFrame) -> np.ndarray:
        out = pd.merge_asof(grid, feat, on="ts", direction="backward")
        age = grid["ts"].to_numpy(np.int64) - out["ts"].to_numpy(np.int64)
        vals = out["phi_cnt"].to_numpy(np.float64)
        return np.where(np.isfinite(age) & (age >= 0) & (age <= 500), vals, np.nan)

    on_grid = _asof(full)
    cut_grid = _asof(cut)
    m = grid["ts"].to_numpy() <= T
    np.testing.assert_allclose(on_grid[m], cut_grid[m], equal_nan=True)


def test_sq_micro_asof_ignores_future_book():
    """Square-volume gap at t uses the last book row with ts ≤ t, then eq. (23)."""
    # Heavy ask: p̂ sits closer to the bid than the linear microprice.
    book = pd.DataFrame(
        {
            "ts": np.array([100, 300, 700], dtype=np.int64),
            "bid_px": [100.0, 100.0, 110.0],
            "ask_px": [101.0, 101.0, 111.0],
            "bid_qty": [1.0, 1.0, 50.0],
            "ask_qty": [3.0, 3.0, 1.0],
        }
    )
    grid = pd.DataFrame({"ts": np.array([100, 200, 300, 500, 700], dtype=np.int64)})

    def _on(events: pd.DataFrame) -> np.ndarray:
        joined = pd.merge_asof(grid, events, on="ts", direction="backward")
        return sq_micro_bp(
            joined["bid_px"].to_numpy(),
            joined["ask_px"].to_numpy(),
            joined["bid_qty"].to_numpy(),
            joined["ask_qty"].to_numpy(),
        )

    full = _on(book)
    # At the 100/101 book with sizes 1 and 3: p̂ = (9*100 + 1*101)/10 = 100.1, mid = 100.5.
    assert full[0] == pytest.approx(1e4 * (100.1 - 100.5) / 100.5)
    T = 300
    cut = _on(book[book["ts"] <= T].reset_index(drop=True))
    m = grid["ts"].to_numpy() <= T
    np.testing.assert_allclose(full[m], cut[m], equal_nan=True)
    # The 700 print must not leak into t ≤ T, and it does move the later row.
    assert not np.allclose(full[~m], cut[~m], equal_nan=True)


def test_q_depl_truncation():
    """Depletion at t uses only bbo updates at or before t, and only same-price updates."""
    ts = np.array([100, 200, 300, 400, 800], dtype=np.int64)
    # 100→200: bid price unchanged, bid qty 10→6 (40% eaten), ask price jumps so ask contributes 0.
    # 200→300: both prices unchanged, ask 4→2 (50% eaten), bid unchanged in size.
    # 300→400: both unchanged, bid 6→12 (full replenishment, clipped).
    # 400→800 is after T and must not affect t≤400.
    bid_px = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    ask_px = np.array([101.0, 102.0, 102.0, 102.0, 102.0])
    bid_qty = np.array([10.0, 6.0, 6.0, 12.0, 1.0])
    ask_qty = np.array([4.0, 4.0, 2.0, 2.0, 2.0])
    full = q_depl(ts, bid_px, ask_px, bid_qty, ask_qty, window_ms=500)
    # At ts=200 the only in-window event is the 200 update: ask_frac=0 (price changed),
    # bid_frac=(10-6)/10=0.4, so q_depl = 0 - 0.4 = -0.4.
    at_200 = float(full.loc[full["ts"] == 200, "q_depl"].iloc[0])
    assert at_200 == pytest.approx(-0.4)
    # At ts=300, events 200 and 300 are inside (300-500, 300]. Ask eaten 0.5, bid 0 at 300,
    # plus the -0.4 from 200: 0.5 + (-0.4) = 0.1.
    at_300 = float(full.loc[full["ts"] == 300, "q_depl"].iloc[0])
    assert at_300 == pytest.approx(0.1)

    T = 400
    keep = ts <= T
    cut = q_depl(ts[keep], bid_px[keep], ask_px[keep], bid_qty[keep], ask_qty[keep], window_ms=500)
    np.testing.assert_allclose(
        full.loc[full["ts"] <= T, "q_depl"].to_numpy(),
        cut["q_depl"].to_numpy(),
        equal_nan=True,
    )


def test_sweep_px_truncation():
    """Through-touch distance uses the quote from before the bar, and ignores later prints."""
    # Bar at 200: touch is book as-of 100. Ask is 101, buy VWAP 103 walks 2.
    # Bar at 300: touch as-of 200 is ask 101, buy at 101 is not through.
    # Bar at 900 is after T.
    trade_ts = np.array([200, 300, 900], dtype=np.int64)
    buy_qty = np.array([2.0, 1.0, 5.0])
    buy_vwap = np.array([103.0, 101.0, 120.0])
    sell_qty = np.array([0.0, 0.0, 0.0])
    sell_vwap = np.array([np.nan, np.nan, np.nan])
    book_ts = np.array([100, 250, 800], dtype=np.int64)
    bid_px = np.array([100.0, 100.0, 110.0])
    ask_px = np.array([101.0, 101.0, 111.0])
    full = sweep_px(
        trade_ts, buy_qty, buy_vwap, sell_qty, sell_vwap, book_ts, bid_px, ask_px, window_ms=500
    )
    # At 200, only that bar: num=2*(103-101)=4, den=2*100.5, bp=1e4*4/(2*100.5)
    at_200 = float(full.loc[full["ts"] == 200, "sweep_px"].iloc[0])
    assert at_200 == pytest.approx(1e4 * 4.0 / (2.0 * 100.5))
    # The 300 print is at the touch, so it does not change the quantity-weighted through.
    at_300 = float(full.loc[full["ts"] == 300, "sweep_px"].iloc[0])
    # window (300-500, 300] includes 200 and 300. den = 2*100.5 + 1*100.5, num = 4 + 0
    assert at_300 == pytest.approx(1e4 * 4.0 / (3.0 * 100.5))

    T = 300
    keep = trade_ts <= T
    book_keep = book_ts <= T
    cut = sweep_px(
        trade_ts[keep],
        buy_qty[keep],
        buy_vwap[keep],
        sell_qty[keep],
        sell_vwap[keep],
        book_ts[book_keep],
        bid_px[book_keep],
        ask_px[book_keep],
        window_ms=500,
    )
    np.testing.assert_allclose(
        full.loc[full["ts"] <= T, "sweep_px"].to_numpy(),
        cut["sweep_px"].to_numpy(),
    )


def test_sign_age_truncation():
    """Relative recency at t uses only prints at or before t."""
    trade_ts = np.array([100, 300, 900], dtype=np.int64)
    buy_n = np.array([1.0, 0.0, 0.0])
    sell_n = np.array([0.0, 2.0, 4.0])
    grid = np.array([100, 300, 400, 900], dtype=np.int64)

    def _col(ts, bn, sn) -> np.ndarray:
        marks = sign_age_marks(ts, bn, sn)
        joined = pd.merge_asof(
            pd.DataFrame({"ts": grid}),
            marks,
            on="ts",
            direction="backward",
        )
        return sign_age_from_marks(
            grid,
            joined["last_buy_ts"].fillna(-1).to_numpy(np.int64),
            joined["last_sell_ts"].fillna(-1).to_numpy(np.int64),
        )

    full = _col(trade_ts, buy_n, sell_n)
    # At 100 only a buy has printed.
    assert np.isnan(full[0])
    # At 300 the sell is current and the buy is 200 ms old: (0 - 200) / 200 = -1.
    assert full[1] == pytest.approx(-1.0)
    # At 400 the sell is 100 ms old and the buy is 300 ms old: (100 - 300) / 400 = -0.5.
    assert full[2] == pytest.approx(-0.5)

    T = 400
    keep = trade_ts <= T
    cut = _col(trade_ts[keep], buy_n[keep], sell_n[keep])
    m = grid <= T
    np.testing.assert_allclose(full[m], cut[m], equal_nan=True)
    assert not np.allclose(full[~m], cut[~m], equal_nan=True)


def test_liq_side_asof_ignores_future_book():
    """Near-touch share at t uses the last book row with ts ≤ t."""

    def _buckets(near_b, far_b, near_a, far_a):
        bid = [np.array([0.0]) for _ in range(11)]
        ask = [np.array([0.0]) for _ in range(11)]
        bid[0] = np.array([near_b])  # v01, inside 0–2 bp
        bid[5] = np.array([far_b])  # v06, inside 2–100 bp
        ask[0] = np.array([near_a])
        ask[5] = np.array([far_a])
        return bid, ask

    # Bid is entirely near, ask is entirely far → shares 1 and 0.
    bid, ask = _buckets(10.0, 0.0, 0.0, 10.0)
    assert float(liq_side(bid, ask)[0]) == pytest.approx(1.0)

    book = pd.DataFrame(
        {
            "ts": np.array([100, 500], dtype=np.int64),
            "x": [1.0, -1.0],
        }
    )
    grid = pd.DataFrame({"ts": np.array([100, 200, 500], dtype=np.int64)})

    def _on(events: pd.DataFrame) -> np.ndarray:
        joined = pd.merge_asof(grid, events, on="ts", direction="backward")
        return joined["x"].to_numpy(float)

    full = _on(book)
    T = 200
    cut = _on(book[book["ts"] <= T].reset_index(drop=True))
    m = grid["ts"].to_numpy() <= T
    np.testing.assert_allclose(full[m], cut[m], equal_nan=True)
    assert full[2] == pytest.approx(-1.0)
    assert cut[2] == pytest.approx(1.0)
