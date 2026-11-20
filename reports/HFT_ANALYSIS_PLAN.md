# HFT analysis plan — 500 ms challenge

## Claim (working)

Short-horizon mid returns are forecastable from **own book imbalance**, **same-coin leaders** (USD1 / PEPE·USDC), and weak secondary flow/depth — not from slow spot catch-up. Spot is mostly a *follower* of the perp.

## Status

| Phase | Content | Status |
|-------|---------|--------|
| 0 | Repo scaffold (`CONTEXT`, reports, scripts, requirements, `.venv`) | **done** |
| 1 | Unpack **all** instruments into `data/{book,bbo,trades,targets,stable,meta}/` | **done** |
| 2 | Full-period DQ + targets + imb IC + OLS 12-cell SCORE | **done** (`reports/FULL_EDA_FINDINGS.md`) |
| 3 | Lead–lag / light Granger (sampled; USD1 post-listing) | **done** |
| 4 | Feature engineering beyond imb / first models past baseline | **next** |

## Key local anchors (our EDA)

- Val OLS imb SCORE: **0.1838** (= organizers)
- Train SCORE: **0.1971**
- Perp → spot @100 ms ≫ reverse; BTCUSD1 → BTC perp lead real but smaller (ΔR² ~1–2% vs perp→spot ~12%)
- Trade-flow alone ≪ imbalance; val quieter than train; PEPE weakest / noisiest cell (~0.145)

## Already skipped (for now)

- 6-model ensemble / GRU / Optuna
- Full 17×17 Granger on all 365 days
- Half-life / cointegration re-run
- Executable option-style PnL / Avellaneda MM
- DeepLOB on raw depth

## Peer anchors (do not re-derive from scratch)

- Organizers baseline val SCORE: **0.1838**
- Linear M3 + book paths (EDA PDF): ~**0.2205**
- Ensemble 76 features (Meshcheryakov): **0.2346**
- PEPE baseline cells (PDF): ~0.15 / 0.16 / 0.12 → mean ~0.145

## Artifacts

- `scripts/run_full_eda.py`, `scripts/full_eda_io.py`
- `outputs/full_eda_results.json`
- `reports/FULL_EDA_FINDINGS.md`, `VARIABLE_ANALYSIS_LOG.md`, `EXPLANATIONS.md`
