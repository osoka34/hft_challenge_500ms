# Paper integration log (sequential)

**Method (updated):** first screen features **one-by-one** (`scripts/screen_features_uni.py` → `reports/FEATURE_UNI_SCREEN.md`).  
Only SIGNIFICANT uni features go to later **group** tests. Earlier pack-add results below are historical.

Uni protocol: expanding WF OLS `y ~ x`, clipped-y Pearson, days `2026-01-12..25`, min_train=3.  
SIGNIFICANT: `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.

Pack baselines (historical): `imb` ≈ **0.216**, `hl_resid` ≈ **0.287**.

## Summary

| # | Idea | vs imb | vs hl_resid | Verdict |
|---|------|--------|-------------|---------|
| 1 | Stoikov G1 micro-price | +0.006 alone | +0.000 | **NOT OK** as add-on (OK alone≪pack) |
| 2 | Cont/Kolm OFI/depth / bid–ask OF | worse alone | ~0 | **NOT OK** |
| 3 | Gould local E[y\|imb] | +0.027 alone | +0.002 | **WEAK** add-on |
| 4 | Albers sweep-IMB proxy (bucket) | ~0 | ~0 | **NOT OK** (need real L2 prices) |
| 5 | Cross BTC→ETH features | — | +0.003 on ETH pack | **OK** (small) |
| 6 | Pooled BTC+ETH+SOL β | — | −0.007 macro | **NOT OK** |

Keep: cross-asset BTC→alt features; optional Gould bin-EY (**WEAK**).  
Drop: Stoikov G1 add-on, Cont OF stack, sweep proxy, multi-asset pooling.

---

## 1. Stoikov micro-price G1

Baseline imb=0.2162 hl_resid=0.2874

- **stoikov_g1_only**: mean=0.2220 (vs imb, Δ=+0.0059) → **OK** alone but weak absolute
- **stoikov_imb_g1**: mean=0.2228 (vs imb, Δ=+0.0066) → redundant with imb
- **hl_resid_stoikov**: mean=0.2876 (vs hl_resid, Δ=+0.0002) → **NOT OK**
- Note: G1 ≈ nonlinear function of imbalance; already captured by `imb`/`imb3`.
- JSON: `outputs/paper_seq/stoikov_BTCUSDT_PERP.json`

## 2. Cont / Kolm order-flow

- **cont_ofi_depth**: 0.0316 → **NOT OK**
- **cont_ofi_multi**: 0.0317 → **NOT OK**
- **cont_bidask_of**: 0.1730 → **NOT OK** (below imb)
- **hl_resid_ofi_***: Δ≈0 → **NOT OK**
- Note: snapshot Cont OFI already in `d_imb`/`ofi_*`; depth-norm doesn't beat queue imb on this grid.
- JSON: `outputs/paper_seq/cont_BTCUSDT_PERP.json`

## 3. Gould local E[y|imb]

- **gould_ey_only**: 0.2434 (vs imb +0.027) → **OK** alone
- **hl_resid_gould**: 0.2893 (vs hl_resid +0.0018) → **WEAK**
- Note: nonparametric bin-mean of y|imb; overlaps `imb3`. Marginal on full pack.
- JSON: `outputs/paper_seq/gould_BTCUSDT_PERP.json`

## 4. Albers sweep-IMB proxy

- **sweep_imb / sweep_pair**: ~0.215 → **NOT OK**
- **hl_resid_sweep**: Δ≈0 → **NOT OK**
- Note: without L2 prices, tick-distance proxy from depth buckets is too crude.
- JSON: `outputs/paper_seq/albers_sweep_BTCUSDT_PERP.json`

## 5. Cross-impact BTC→ETH

- **eth_imb**: 0.2169
- **eth_hl_resid**: 0.2684
- **eth_hl_resid + btc_{imb,ret,tfi}**: 0.2715 (Δ=+0.0030) → **OK**
- JSON: `outputs/paper_seq/cross_eth.json`

## Next in queue (not run yet)

- Briola volume representation of depth buckets
- Sirignano–Cont pooled BTC+ETH+SOL single model
- Volume-clock / VPIN (low priority)

## 6. Sirignano–Cont pooled linear (BTC+ETH+SOL)

Per-asset `hl_resid` WF means: {'BTCUSDT_PERP': 0.2874, 'ETHUSDT_PERP': 0.2684, 'SOLUSDT_PERP': 0.2553}
Pooled shared-β + instrument FE: {'BTCUSDT_PERP': 0.2781, 'ETHUSDT_PERP': 0.2636, 'SOLUSDT_PERP': 0.2475}
Macro mean per-asset=0.2704 vs pooled=0.2631 (Δ=-0.0073)
→ **NOT OK** (pooling does not beat per-name here)
