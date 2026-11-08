# HFT analysis plan — 500 ms challenge (PEPE start)

## Claim (working)

Short-horizon mid returns are forecastable from **own book imbalance**, **same-coin USDC lead**, and **trade/OFI flow** — not from slow spot catch-up. Full competition SCORE needs 4 target perps; we bootstrap on **1000PEPE** only.

## Status

| Phase | Content | Status |
|-------|---------|--------|
| 0 | Repo scaffold (`CONTEXT`, reports, scripts, requirements) | **done** |
| 1 | Unpack PEPE sample days + DQ | **done** |
| 2 | EDA: targets, spreads, imbalance→y, lead–lag USDC→USDT | **done** |
| 3 | Local clip+Pearson metric + OLS imbalance baseline (PEPE-only) | **done** |
| 4 | Download remaining instruments | **backlog** (priority listed below) |

## Already skipped (for now)

- 6-model ensemble / GRU / Optuna
- Full 17×17 Granger on all days
- Executable option-style PnL / Avellaneda MM
- DeepLOB on raw depth

## Download priority (ROI for SCORE)

Ordered by expected SCORE impact from peer work (Meshcheryakov feature groups + EDA PDF Granger):

| Priority | What to fetch | Why |
|----------|---------------|-----|
| P0 | **BTCUSDT / ETHUSDT / SOLUSDT** perps: `book`, `bbo`, `trades`, **`targets`** | Full 12-cell competition SCORE; own imb is #1 feature everywhere |
| P1 | Same-coin leaders: **BTCUSD1**, **ETHUSD1**, SOLUSDT spot / SOLUSDC; (PEPE·USDC **already local**) | Top conditional ΔR² sources for BTC/ETH/SOL in EDA PDF |
| P2 | Spot USDT + USDC books for BTC/ETH/SOL/PEPE + **stablecoin rates** (USDC, FDUSD, USD1) | Spot / stablecoin-gap groups in LightGBM (~0.007–0.008 shuffle drop); skip obsessing over FDUSD gaps (≈0) |
| P3 | Remaining feature markets from the 17-instrument universe | Diminishing returns vs P0–P2 |

Local now: `data-1000PEPEUSDT_PERP.tar`, `data-1000PEPEUSDC_PERP.tar` (sample unpacked under `data/raw/`).

While archives download, keep iterating PEPE features (USDC lags, depth imb, OFI) on the local metric.

## Peer anchors (do not re-derive from scratch)

- Organizers baseline val SCORE: **0.1838**  
- Linear M3 + book paths (EDA PDF): ~**0.2205**  
- Ensemble 76 features (Meshcheryakov): **0.2346**  
- PEPE baseline cells (PDF): ~0.15 / 0.16 / 0.12 → mean ~0.145  
- PEPE·USDC → PEPE lead: most of ΔR² in first 100–300 ms; weaker on validation  

## Sample days (local unpack)

Train window: `2025-11-01` … `2025-11-07`  
Val window: `2026-08-10` … `2026-08-15`  

Full PEPE archives stay as `.tar`; only these days are extracted under `data/raw/`.
