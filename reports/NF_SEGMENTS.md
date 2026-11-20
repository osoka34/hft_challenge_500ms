# NeuralForecast segment lab

Sandbox around [nixtla/neuralforecast](https://github.com/nixtla/neuralforecast) with an unusual split:

1. One UTC day → causal panel `y` + hist features (bp-scaled)  
2. **Drop NaNs once** on the union of feature packs (fair ablations)  
3. Cut into ~**100** contiguous segments → each segment = `unique_id`  
4. Inside each segment: **train / val / test** = 70 / 15 / 15, then **trim to common lengths** (NF needs one global `val_size`)  
5. Fit **one** small `NHITS` across all segments (pooled)  
6. Score **Pearson** of 1-step preds vs `y` (boundary + optional walk-forward)  
7. Compare to **Pearson(imb, y)** on the same points  

## Run

```bash
source .venv/bin/activate
python scripts/run_nf_segments.py --max-segments 8 --feature-sets y_only imb core --max-steps 150

python scripts/run_nf_segments.py --max-segments 0 --n-segments 100 \
  --feature-sets y_only imb core \
  --max-steps 400 --dropout 0.1 --weight-decay 1e-4 --walkforward --wf-stride 25
```

Feature packs: `FEATURE_SETS` in `scripts/nf_features.py`.

Order-flow packs (`of`, `of_imb`, `of_d`, `of_near`, `of_leader`, `of_core`) use Cont–Kukanov–Stoikov L1 OFI (size-normalized) plus `d_imb` / near-depth volume flow — stationary Δ inputs rather than raw book levels.

## Notes

- Target is competition `y_H` (default `y_500ms`), scaled to **bp** (`×1e4`).  
- **Lag alignment:** NF `h=1` forecasts the next row; we store `y[t]=y_comp[t-1]` so a forecast from cutoff `t` matches `y_comp[t]` with features at `t`.  
- With exogenous packs: `exclude_insample_y=True`.  
- Walk-forward is **strided** (`--wf-stride`) and capped per segment (`max_per_segment=12`); fail rate >10% aborts.  
- Local GPU (GTX 1080 Ti / sm_61) → training forced to **CPU**.  
- Segment `min_len` default **214** so 15% val/test slices stay ≥32 rows.

## Results (pre-review runs; treat as directional)

### v3 (lag-aligned, 50 segs)
| pack | WF Pearson | base imb |
|------|------------|----------|
| y_only | 0.190 | 0.426 |
| imb | 0.268 | 0.426 |
| core | -0.017 | 0.426 |

### v3b (100 segs, imb, 800 steps)
WF n=1200: NHITS **0.272** vs linear imb **0.381**.

## Review fixes (post code-review)

- Common `val_size` via trimmed equal-length splits  
- Pre-cut `dropna` on union of packs  
- `min_len` aligned with split floor  
- Walk-forward failures counted; high fail rate raises  
- Baseline is Pearson(imb,y) only (no in-sample OLS on test)  

## Feature ablation (BTC 2026-01-20, leader=BTCUSD1, 40 segs, WF n=440)

| pack | WF Pearson | vs linear imb 0.308 |
|------|------------|---------------------|
| **imb** | **0.196** | best neural |
| imb_leader | 0.074 | worse |
| imb_plus | 0.028 | worse |
| imb_ret | 0.017 | worse |
| imb_depth | -0.018 | worse |
| imb_micro | -0.057 | worse |

Extra features did **not** help NHITS transform them into alpha; they degraded walk-forward IC. Linear imb still wins.

## Order-flow ablation (BTC 2026-01-20, leader=BTCUSD1, 40 segs, WF n=440)

Cont L1 OFI + Δ packs (size-normalized). Same NHITS recipe (`max_steps=400`, `dropout=0.1`, `wd=1e-4`).

| pack | WF Pearson | vs linear imb 0.329 |
|------|------------|---------------------|
| **imb** | **0.147** | best neural (again) |
| of_leader | 0.083 | worse |
| of_near | 0.054 | worse |
| of_core | 0.047 | worse |
| of | 0.020 | worse |
| of_imb | -0.010 | worse |
| of_d | -0.054 | worse |

Raw day Pearson (pre-model): `imb` 0.257 > `ofi_best` ~0.19 > `d_imb` ~0.18 — OF has signal, but NHITS does not turn it into lift over `imb`. Linear imb still wins.
JSON: `outputs/nf_segments/BTCUSDT_PERP_2026-01-20_y_500ms_of_ablation.json`.

## High-level thin packs (BTC 2026-01-20, 40 segs, WF n=440)

`gap_dev` + short leader lags (200/300 ms bbo asof; **100 ms dropped** — empty vs USD1 tape on 500 ms grid).

| pack | WF Pearson | vs linear imb 0.319 |
|------|------------|---------------------|
| **hl_gap** (imb, gap_dev) | **0.080** | best neural this run |
| imb | 0.070 | |
| hl_lead | 0.048 | worse |
| hl3 | 0.032 | worse |
| hl4 | −0.008 | worse |
| hl_short | −0.014 | worse |

Slight lift of `gap_dev` over `imb`-only NHITS; still far below linear imb. Agent-facing rollup: `reports/NF_EXPERIMENT_LOG.md`.
JSON: `outputs/nf_segments/BTCUSDT_PERP_2026-01-20_y_500ms_hl_ablation.json`.

## Save / load fitted models

```bash
# after fit, write artifacts under outputs/nf_segments/models/
python scripts/run_nf_segments.py ... --loss pearson --save-models

# later: skip fit, reuse weights for boundary / walk-forward
python scripts/run_nf_segments.py ... --loss pearson --load-models --walkforward
```

Paths: `{model-dir}/{instrument}__{day_tag}__{horizon}__{feature_set}__{loss}/`  
(`nf.save`, dataset not stored). Custom `PearsonCorrLoss` is registered via `neuralforecast.register_loss`.

## Pearson loss (SCORE-aligned)

`--loss pearson` → train/valid on `1 − Pearson(y, ŷ)` (`scripts/nf_losses.py`).

| pack | MSE WF | Pearson-loss WF | linear imb |
|------|--------|-----------------|------------|
| imb | 0.070 | **0.222** | 0.319 |
| hl_gap | 0.080 | 0.167 | 0.319 |

Big lift from metric-matched loss; still below linear imb. JSON: `..._pearson_loss.json`.
