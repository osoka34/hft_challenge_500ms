# NeuralForecast / NHITS experiment log (for agents)

Last update: 2026-11-11. Raw JSON under `outputs/nf_segments/`.

## Setup (shared)

- Model: small **NHITS** (`n_blocks=[1,1,1]`, MLP `[64,64]`×3, `h=1`, `input_size=32`, CPU).
- Split: one UTC day → ~100 contiguous segments as `unique_id` → pooled fit; score Pearson on walk-forward (and boundary).
- Baseline on same points: **Pearson(imb, y)** (linear, no OLS fit).
- Typical ablation day: **BTCUSDT_PERP 2026-01-20**, leader **BTCUSD1_SPOT**, 40 segs, WF n=440, `max_steps=400`, `dropout=0.1`, `wd=1e-4`.
- Code: `scripts/nf_features.py`, `scripts/run_nf_segments.py`.

**Caveat:** `dropna` is on the **union** of requested feature packs, so absolute WF numbers are only comparable within one CLI run (same row set).

## Verdict (read this first)

1. **NHITS never beat linear imb** on this sandbox (best neural WF ~0.22 vs imb ~0.32).
2. **Loss matters:** `--loss pearson` (`1−corr`) lifts WF a lot vs MSE on the same day/packs (imb 0.07→**0.22**). Use it going forward.
3. Extra hist_exog almost always **hurt** vs `imb`-only NHITS (even with pearson, `hl_gap` 0.17 < `imb` 0.22).
4. Prefer **few high-level** features over low-level depth/OFI dumps.
5. Scale-up (many days) + pearson loss is the next NHITS test; linear/ridge on same features still the safer SCORE path.
6. **Albers et al. (arXiv:2108.09750)** ideas transferred: absolute TFI + softsign + longer mean-divergence. Linear ridge packs beat imb-only on several days (`alb_core` best). Prefer this path over more NHITS.

Loss code: `scripts/nf_losses.py` (`PearsonCorrLoss`). CLI: `--loss mse|pearson`.
Linear Albers eval: `python scripts/eval_albers_linear.py --day YYYY-MM-DD`.

## Feature inventory (causal)

| Feature | Level | Notes / day Pearson vs y (BTC 2026-01-20) |
|---------|-------|-------------------------------------------|
| `imb` | high | ~+0.26; strongest own-book |
| `imb3` | high | ~+0.28 |
| `leader_ret_1_bp` | high | ~+0.20 (500 ms book step) |
| `gap_dev` | high | ~−0.18; ln(own/leader) − causal 60s mean |
| `leader_ret_300ms_bp` | high | ~+0.13 (bbo asof) |
| `leader_ret_200ms_bp` | high | ~+0.11; **use instead of 100 ms** |
| `leader_ret_100ms_bp` | — | **degenerate on USD1**: no bbo prints in (t−100,t] vs 500 ms grid → std≈0. Dropped. |
| Cont `ofi_best` (L1/size) | mid | weak after norm; raw OF had more IC |
| depth / micro / trade flow | mid–low | weak or noisy; harmed NHITS |

Packs: see `FEATURE_SETS` in `scripts/nf_features.py` (`of_*`, `hl_*`, legacy `imb_*` / `core`).

## Ablation results (BTC 2026-01-20, 40 segs, WF n=440)

### A. Legacy handcrafted (`reports/NF_SEGMENTS.md`)

Best neural was **`imb`** (~0.20 WF); `imb_leader` / depth / micro / plus all worse. Linear imb ~0.31.

### B. Order-flow packs (`..._of_ablation.json`)

| pack | WF Pearson | linear imb |
|------|------------|------------|
| **imb** | **0.147** | 0.329 |
| of_leader | 0.083 | |
| of_near | 0.054 | |
| of_core | 0.047 | |
| of | 0.020 | |
| of_imb | −0.010 | |
| of_d | −0.054 | |

OF has raw signal but NHITS does not turn it into lift.

### C. Thin high-level packs (`..._hl_ablation.json`) — 2026-11-10

Raw day IC: imb +0.26, leader_ret_1 +0.20, gap_dev −0.18, leader_ret_300 +0.13, leader_ret_200 +0.11.

| pack | exog | WF Pearson | linear imb |
|------|------|------------|------------|
| **hl_gap** | imb, gap_dev | **0.080** | 0.319 |
| imb | imb | 0.070 | 0.319 |
| hl_lead | imb, leader_ret_1 | 0.048 | |
| hl3 | imb, leader_ret_1, gap_dev | 0.032 | |
| hl4 | + leader_ret_300 | −0.008 | |
| hl_short | imb, ret_200, ret_300, gap_dev | −0.014 | |

`hl_gap` slightly above `imb`-only in this run; still **≪ linear imb**. Adding short leader lags on top did not help NHITS.

### D. Pearson loss vs MSE (same day/setup, packs `imb` + `hl_gap`) — 2026-11-10

| pack | loss | WF Pearson | linear imb |
|------|------|------------|------------|
| **imb** | **pearson** | **0.222** | 0.319 |
| hl_gap | pearson | 0.167 | 0.319 |
| imb | mse (prior hl run) | 0.070 | 0.319 |
| hl_gap | mse (prior hl run) | 0.080 | 0.319 |

JSON: `outputs/nf_segments/BTCUSDT_PERP_2026-01-20_y_500ms_pearson_loss.json`.

Metric-aligned loss closes much of the gap but **still does not beat** linear imb.

### E. Multi-day scale-up (14 train days, 40 segs/day = **560** series, pearson loss, CPU) — 2026-11-10

Evenly spaced days `2025-09-15 … 2026-06-15`. WF n=6720. Wall ~472 s (fit cheap; walk-forward dominates).

| pack | WF Pearson | linear imb |
|------|------------|------------|
| imb | 0.173 | 0.278 |
| hl_gap | 0.149 | 0.278 |

More data did **not** beat linear imb; WF lower than 1-day pearson probe (0.22) — harder pooled regime. JSON: `outputs/nf_segments/BTCUSDT_PERP_14days_y_500ms_pearson.json`.

GPU path: `.venv-gpu` (Python 3.12 + `torch==2.7.1+cu118` for GTX 1080 Ti sm_61). See `requirements-gpu.txt`. Walk-forward batched + quiet Lightning; `--save-models` / `--load-models`.

### F. GPU 14-day rerun (batched WF + save-models) — 2026-11-10

Same 14 days / 560 segs / pearson. Wall ~302 s. Models saved under `outputs/nf_segments/models/`.

| pack | WF Pearson | linear imb | fit / WF s |
|------|------------|------------|------------|
| imb | 0.170 | 0.278 | 9.2 / 133 |
| hl_gap | 0.163 | 0.278 | 7.7 / 151 |

Still below linear imb. JSON: `outputs/nf_segments/BTCUSDT_PERP_14days_y_500ms_pearson_gpu.json`.

### G. Albers features → linear ridge (not NHITS) — 2026-11-11

Source: `other_presentations/price_info.pdf` (Albers/Cucuringu/Howison/Shestopaloff, arXiv:2108.09750).

Transferred:
- absolute TFI `tfi_{100,500,1000,2000}` = Σ(buy−sell) over window (not relative)
- softsign `tfi_s*` with causal expanding-mean |x| scale
- past returns `pret_{500,1000,2000}_bp`
- mean-divergence `div_150s` / `div_300s` (gap_dev already ≈ 60s version)
- packs `alb_*` in `FEATURE_SETS`; eval `scripts/eval_albers_linear.py`

Ridge(1) holdout Pearson (train 70% / test 30%, BTCUSDT_PERP):

| day | imb | alb_tfi | alb_tfi_s | alb_thin | hl3 | **alb_core** |
|-----|-----|---------|----------|----------|-----|--------------|
| 01-20 | 0.259 | 0.272 | 0.264 | 0.264 | 0.285 | **0.296** |
| 01-19 | 0.198 | 0.198 | 0.204 | 0.204 | 0.225 | **0.230** |
| 01-18 | 0.163 | 0.188 | 0.170 | 0.170 | 0.257 | **0.271** |
| 01-17 | 0.104 | 0.118 | 0.115 | 0.115 | 0.128 | **0.133** |

Uni IC: softsign lifts TFI (e.g. 01-20 `tfi_500` 0.088 → `tfi_s500` 0.126); still << imb. Leader ret + imb drive most of the multivariate lift; TFI adds a smaller incremental bump. Paper sweep-IMB (eq. 3–4) skipped — no L2 prices, only depth buckets.

**Expanding WF** (train past days → next day, clipped-y Pearson, BTC 2026-01-12..25, 11 test days):

| pack | mean daily SCORE | vs imb |
|------|------------------|--------|
| imb / imb IC | 0.216 | — |
| alb_tfi_s | 0.221 | +0.005 |
| hl3 (`imb+leader_ret+gap_dev`) | 0.245 | +0.029 |
| **alb_core** (OLS ridge=0) | **0.253** | **+0.037** |

Every test day alb_core ≥ imb. Most lift = leader + gap; soft TFI/pret add ~+0.005–0.008 on top of hl3. JSON: `outputs/albers_linear/BTCUSDT_PERP_wf_*.json`.

### H. Causal Kalman gap-state — 2026-11-12

Local-level + level/vel KF on `ln(own/leader)` → `kf_innov`, `kf_level`, `kf_vel` (causal).

Uni IC (BTC 01-20): `kf_innov` −0.204 > `gap_dev` −0.182; corr(gap_dev, kf_innov)≈0.79.

Expanding WF OLS (same 11 days as §G):

| pack | mean SCORE |
|------|------------|
| imb | 0.216 |
| hl3 (`gap_dev`) | 0.250 |
| **hl_kf** (`kf_innov` replaces gap) | **0.257** |
| hl_kf2 (+`kf_vel`) | 0.257 |
| alb_core | 0.253 |
| **alb_kf** / alb_core_kf | **0.258** |

`kf_innov` beats rolling `gap_dev` as a drop-in; vel adds almost nothing. JSON: `outputs/albers_linear/BTCUSDT_PERP_wf_kf.json`.

NHITS probe (01-20, pearson loss, 40 segs): `imb` WF 0.22 > `hl_kf` 0.08 > `hl3` 0.06 — still far below linear imb IC ~0.32. KF helps linear, not NHITS.

### I. `imb3` + short leader + micro on linear KF pack — 2026-11-12

OLS expanding WF (BTC 01-12..25, 11 test days). JSON: `outputs/albers_linear/BTCUSDT_PERP_wf_imb3.json`.

| pack | mean SCORE | vs imb |
|------|------------|--------|
| imb | 0.216 | — |
| hl_kf | 0.257 | +0.041 |
| **hl_kf_imb3** | **0.271** | **+0.055** |
| hl_kf_imb3_short | 0.272 | +0.056 |
| hl_kf_imb3_micro | 0.272 | +0.056 |
| alb_kf_imb3 | 0.272 | +0.056 |

`imb3` is the real lift (~+0.015 over `hl_kf`). Short leader lags / L1 microprice / extra Albers TFI add ≤0.001. Current best thin pack: `imb + imb3 + leader_ret_1_bp + kf_innov`.

### J. Residual dig after `hl_kf_imb3` — 2026-11-12

OOS residual IC (pooled) still nonzero for short aggressive flow / Δ book:
`tfi_100` +0.075, `tfi_s100` +0.061, `d_imb` +0.059. Interactions / TOD / micro ≈ 0 in WF adds.

Greedy WF adds → pack `hl_resid` = base + `tfi_100` + `tfi_s100` + `d_imb`:

| pack | mean SCORE |
|------|------------|
| imb | 0.216 |
| hl_kf_imb3 | 0.271 |
| hl_kf_imb3_tfi (+`tfi_100`) | 0.281 |
| **hl_resid** | **0.287** |

Online/adaptive: roll-3d OLS 0.286 ≈ expanding; block-RLS unstable (num. blow-up). Stick with expanding OLS.

JSON: `outputs/albers_linear/BTCUSDT_PERP_residual_dig.json`, `..._wf_resid.json`.

### K. After `hl_resid` + multi-instrument — 2026-11-12

Second residual dig: leftover IC ≤0.03. Leader short TFI ≈0. Greedy +`depth_imb_near`+`ofi_near` → **hl_resid2 ~0.289** (+0.002 only).

Multi-instrument expanding WF (01-12..25):

| instrument | imb | hl_kf_imb3 | hl_resid |
|------------|-----|------------|----------|
| BTC | 0.216 | 0.271 | **0.287** |
| ETH | 0.217 | 0.253 | **0.268** |
| SOL | 0.210 | 0.238 | **0.255** |
| PEPE | 0.203 | 0.213 | 0.214 |

Pack generalizes on ETH/SOL; PEPE almost flat. Own-book features largely exhausted on BTC; next edge is probably PEPE-specific microstructure or new data, not more BTC transforms.

JSON: `outputs/albers_linear/BTCUSDT_PERP_next_dig.json`.

## Implications for next agents

- **Keep building:** `imb`, `leader_ret_*`, `gap_dev`/`div_*`, soft TFI for **linear/ridge** SCORE path (`alb_core` / `hl3`).
- **Skip:** more NHITS exog ablations; raw L2 dumps into NHITS; `leader_ret_100ms_bp` on USD1@500ms grid.
- **If nonlinear again:** Albers softclip for linear > NHITS kitchen-sink; event/OF LSTM only if user asks.
- Re-run:

```bash
source .venv/bin/activate
python scripts/eval_albers_linear.py --instrument BTCUSDT_PERP --day 2026-01-20
python scripts/run_nf_segments.py --instrument BTCUSDT_PERP --day 2026-01-20 \
  --n-segments 100 --max-segments 40 \
  --feature-sets imb hl_gap hl3 \
  --max-steps 400 --dropout 0.1 --weight-decay 1e-4 --walkforward --wf-stride 25
```
