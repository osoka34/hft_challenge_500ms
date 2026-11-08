# HFT homework 2 — short-horizon challenge

Domain language for CMF HFT Task 2: predict Binance perpetual mid-returns every 500 ms.

## Language

**Grid point / prediction time `t`**:
A 500 ms timestamp on the day grid (172,800 points per UTC day per target perp). Features at `t` may use only events with `ts ≤ t`.
_Avoid_: using future book/trade updates, or “closing” sparse trade gaps with future bars

**Target `y_H(t)`**:
Forward log mid-return starting after a fixed **10 ms** order latency:  
`y_H(t) = ln(mid(t + 10ms + H) / mid(t + 10ms))`, with `H ∈ {500 ms, 2 s, 8 s}`.
_Avoid_: measuring the return from mid at `t` without the 10 ms delay

**SCORE (competition metric)**:
1. Clip each `y` to train-period 0.1% / 99.9% percentiles.  
2. Pearson correlation of prediction vs clipped `y` **per calendar day**, per instrument × horizon.  
3. Average over validation days, then average over the 12 cells (4 perps × 3 horizons).  
Scale of the prediction does not matter — only co-movement.
_Avoid_: RMSE/MAE as the selection metric; pooling all days into one correlation

**Causality / leak**:
A prediction at `t` is legal only if every input timestamp is `≤ t`. Organizers re-run with truncated days; a mismatch is a disqualification.
_Avoid_: centering/windows that peek past `t`; filling trade holes with future prints

**Target perps**:
USDT-quoted perpetuals whose returns are scored: BTC, ETH, SOL, **1000PEPE**.
_Avoid_: treating PEPE·USDC as a scored target (it is a feature market only)

**Feature instruments**:
Other markets used as inputs (same-coin USDC/spot, other perps, stablecoin rates). Locally we currently have **1000PEPEUSDT_PERP** (target) + **1000PEPEUSDC_PERP** (leader / feature).
_Avoid_: assuming spot leads the perp on this tape — peer EDA finds the opposite for most pairs

**Streams**:
- `book` — 500 ms L1 + depth buckets (`bid_v01..11`, `ask_v01..11`)
- `bbo` — L1 quote changes (~100 ms resolution when quotes move)
- `trades` — sparse 100 ms bars with buy/sell qty, VWAP, counts (missing bar ≠ quiet market unless you mark stale causally)
- `targets` — precomputed `y_500ms`, `y_2s`, `y_8s` on the 500 ms grid (target perps only)

**Splits**:
- Train: 2025-09-15 → 2026-07-14  
- Validation: 2026-07-15 → 2026-09-14  
- Test: after submission  

**Baseline**:
Organizers’ OLS on own best-quote imbalance  
`imb = (bid_qty − ask_qty) / (bid_qty + ask_qty)` → validation SCORE **0.1838** (all 12 cells).

**Imbalance / micro-price**:
Canonical queue imbalance on best bid/ask; micro-price = mid adjusted by imbalance (Stoikov). Strongest simple own-book signal in peer work.
_Avoid_: using mid alone as the short-horizon fair value

**Lead–lag**:
Same-coin cross-market timing (e.g. PEPE·USDC perp → PEPE·USDT perp). Peer EDA: most of the useful lead arrives within ~100–300 ms.
_Avoid_: daily cointegration as a substitute for HF lead–lag

## Local status

Only the PEPE USDT + USDC perpetual archives are downloaded. Full SCORE needs BTC/ETH/SOL targets too; until then we report **PEPE-only** mean of 3 horizons as a pipeline check, not the competition number.
