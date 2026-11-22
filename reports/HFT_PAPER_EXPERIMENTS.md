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
| 7 | Lucchese `vol_side_imb` | +0.189 alone | +0.0012 | **WEAK** add-on |
| 8 | Fang/Easley `vpin` (vol-clock) | −0.001 alone | — | **NOT OK** (uni reject) |
| 9 | Stoikov `micro_adj` g(I,S) | +0.222 alone | −0.0001 | **NOT OK** as add-on (≈G1) |
| 10 | Albers `vb_tfi` (vol-clock signed) | −0.002 alone | — | **NOT OK** (uni reject) |
| 11 | Le Coz `btc_ret_1_bp`→SOL | +0.100 alone | +0.0017 | **WEAK** add-on |
| 12 | Lucchese `vol_side_imb_far` | +0.006 alone | — | **NOT OK** (uni reject) |
| 13 | Bieganowski `vwap_mid_bp` | +0.039 alone | — | **NOT OK** (uni reject) |
| 14 | Plerou `phi_cnt` | +0.134 alone | +0.0006 | **NOT OK** as add-on (uni OK) |
| 15 | Bonart–Lillo `sq_micro_bp` | +0.115 alone | +0.0002 | **NOT OK** as add-on (uni OK) |
| 16 | Cont `q_depl` | +0.072 alone | +0.0003 | **NOT OK** as add-on (uni OK) |
| 17 | Valenzuela `rel_liq` | −0.003 alone | — | **NOT OK** (uni reject) |
| 18 | Albers `sweep_px` | +0.104 alone | +0.0009 | **NOT OK** as add-on (uni OK) |
| 19 | Dufour–Engle `sign_age` | +0.068 alone | +0.0000 | **NOT OK** as add-on (uni OK) |
| 20 | Valenzuela `liq_side` | +0.134 alone | −0.0009 | **NOT OK** as add-on (uni OK) |
| 21 | Eisler `size_gap` | +0.026 alone | — | **NOT OK** (uni reject) |
| 22 | Oxenhorn `usdc_prem_bp` | +0.001 alone | — | **NOT OK** (uni reject) |

Keep: cross-asset BTC→alt features (ETH OK; SOL WEAK alone); optional Gould bin-EY (**WEAK**); optional `vol_side_imb` (**WEAK**).  
Drop: Stoikov G1/`micro_adj` add-on, Cont OF stack, sweep proxy, multi-asset pooling, volume-clock VPIN/`vb_tfi`, far-book `vol_side_imb_far`, VWAP-to-mid `vwap_mid_bp`, count-imbalance `phi_cnt` add-on, square-volume `sq_micro_bp` add-on, same-price queue depletion `q_depl`, relative liquidity `rel_liq`, realized sweep `sweep_px`, trade-recency `sign_age`, near-touch concentration `liq_side`, size-versus-count `size_gap`, USDC premium `usdc_prem_bp`.

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

- (empty — pick from HFT_PAPERS; true MLOFI blocked without L2 prices)

### usdc_prem_bp — arXiv:2205.06338

- Status: NOT OK
- Paper: A Multivariate Hawkes Process Model for Stablecoin-Cryptocurrency Depegging Event Dynamics (Connor Oxenhorn)
- Family: other
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.0008 · sign_stability=0.55 · reject · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_usdc_prem_bp.json`
- Causality: skipped · rule — · test —
- Pack-add: Δ vs hl_resid=— · skipped · JSON —
- Note: 10⁴ (USDCUSDT mid − 1), quotes older than 3s dropped; prem stays near +5 to +12 bp and does not track 500 ms BTC returns (oos Pearson 0.0004)

### size_gap — arXiv:0904.0900

- Status: NOT OK
- Paper: The price impact of order book events: market orders, limit orders and cancellations (Zoltan Eisler, Jean-Philippe Bouchaud, Julien Kockelkoren)
- Family: flow
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.0258 · sign_stability=1.00 · reject · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_size_gap.json`
- Causality: skipped · rule — · test —
- Pack-add: Δ vs hl_resid=— · skipped · JSON —
- Note: (buy−sell)/(buy+sell) on quantity minus the same ratio on trade counts, over 500 ms; stable sign but below 0.05; corr≈0.05 vs tfi_100 on 2026-01-12

### liq_side — DOI:10.1016/j.finmar.2015.03.001

- Status: NOT OK
- Paper: Relative liquidity and future volatility (Marcela Valenzuela, Ilknur Zer, Piotr Fryźlewicz, Thorsten Rheinländer)
- Family: depth
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.1337 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_liq_side.json`
- Causality: pass · rule as-of · test `tests/test_feature_truncation.py::test_liq_side_asof_ignores_future_book`
- Pack-add: Δ vs hl_resid=-0.0009 · NOT OK · JSON `outputs/paper_seq/pack_add_BTCUSDT_PERP_liq_side.json`
- Note: near-touch share of bid depth minus the same share of ask depth (0–2 bp vs 2–100 bp); corr≈0.42 vs imb and ≈0.61 vs depth_imb_near on 2026-01-12

### sign_age — DOI:10.1111/0022-1082.00297

- Status: NOT OK
- Paper: Time and the Price Impact of a Trade (Alfonso Dufour, Robert F. Engle)
- Family: flow
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.0679 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_sign_age.json`
- Causality: pass · rule truncation · test `tests/test_feature_truncation.py::test_sign_age_truncation`
- Pack-add: Δ vs hl_resid=+0.00002 · NOT OK · JSON `outputs/paper_seq/pack_add_BTCUSDT_PERP_sign_age.json`
- Note: (age_sell−age_buy)/(age_sell+age_buy); positive when the last buy is more recent; corr≈0.09 vs tfi_100 and ≈0.16 vs imb on 2026-01-12

### sweep_px — arXiv:2108.09750

- Status: NOT OK
- Paper: Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets (Jakob Albers, Mihai Cucuringu, Sam Howison, Alexander Y. Shestopaloff)
- Family: flow
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.1042 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_sweep_px.json`
- Causality: pass · rule truncation · test `tests/test_feature_truncation.py::test_sweep_px_truncation`
- Pack-add: Δ vs hl_resid=+0.0009 · NOT OK · JSON `outputs/paper_seq/pack_add_BTCUSDT_PERP_sweep_px.json`
- Note: qty-weighted bp through the pre-bar touch (buy VWAP above ask, sell VWAP below bid) over 500 ms; corr≈0.78 vs ret_1_bp on 2026-01-12; the bucket-cost proxy stays the old NOT OK sweep_imb

### rel_liq — DOI:10.1016/j.finmar.2015.03.001

- Status: NOT OK
- Paper: Relative liquidity and future volatility (Marcela Valenzuela, Ilknur Zer, Piotr Fryźlewicz, Thorsten Rheinländer)
- Family: depth
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=-0.0025 · sign_stability=0.55 · reject · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_rel_liq.json`
- Causality: skipped · rule — · test —
- Pack-add: Δ vs hl_resid=— · skipped · JSON —
- Note: V(2–100bp)/V(0–2bp); the paper links this shape to future volatility, and it has no signed 500 ms correlation (oos Pearson −0.0017)

### q_depl — arXiv:1011.6402

- Status: NOT OK
- Paper: The Price Impact of Order Book Events (Rama Cont, Arseniy Kukanov, Sasha Stoikov)
- Family: book
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.0721 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_q_depl.json`
- Causality: pass · rule truncation · test `tests/test_feature_truncation.py::test_q_depl_truncation`
- Pack-add: Δ vs hl_resid=+0.0003 · NOT OK · JSON `outputs/paper_seq/pack_add_BTCUSDT_PERP_q_depl.json`
- Note: same-price branch of Cont e_t as a fraction of the previous queue, clipped to [-1, 1], summed over 500 ms; corr≈0.53 vs d_imb (already in hl_resid), ≈0.04 vs ofi_best on 2026-01-12

### sq_micro_bp — arXiv:1608.00756

- Status: NOT OK
- Paper: A continuous and efficient fundamental price on the discrete order book grid (Julius Bonart, Fabrizio Lillo)
- Family: book
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.1146 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_sq_micro_bp.json`
- Causality: pass · rule as-of · test `tests/test_feature_truncation.py::test_sq_micro_asof_ignores_future_book`
- Pack-add: Δ vs hl_resid=+0.0002 · NOT OK · JSON `outputs/paper_seq/pack_add_BTCUSDT_PERP_sq_micro_bp.json`
- Note: eq. (23) at r=0; corr≈0.991 vs micro_gap on 2026-01-12, so the squares do not leave the linear microprice

### phi_cnt — arXiv:cond-mat/0106657

- Status: NOT OK
- Paper: Quantifying Stock Price Response to Demand Fluctuations (Vasiliki Plerou, Parameswaran Gopikrishnan, Xavier Gabaix, H. Eugene Stanley)
- Family: flow
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.1342 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_phi_cnt.json`
- Causality: pass · rule truncation · test `tests/test_feature_truncation.py::test_phi_cnt_truncation`
- Pack-add: Δ vs hl_resid=+0.0006 · NOT OK · JSON `outputs/paper_seq/pack_add_BTCUSDT_PERP_phi_cnt.json`
- Note: Φ_500ms = Σ a_i (buy_n−sell_n); uni above tfi_500 (0.072) but corr≈0.74 vs tfi_500 and hl_resid already has tfi_100/tfi_s100

### vwap_mid_bp — arXiv:2602.00776

- Status: NOT OK
- Paper: Explainable Patterns in Cryptocurrency Microstructure (Bartosz Bieganowski, Robert Ślepaczuk)
- Family: flow
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.0387 · sign_stability=1.00 · reject · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_vwap_mid_bp.json`
- Causality: skipped · rule — · test —
- Pack-add: Δ vs hl_resid=— · skipped · JSON —
- Note: qty-weighted (VWAP−mid)/mid in bp over (t−1s, t]; stable sign but below 0.05; corr≈0.47 vs ret_1_bp, ≈0.04 vs tfi_100 on 2026-01-20

### mlofi — arXiv:1907.06230

- Status: blocked
- Paper: Multi-Level Order-Flow Imbalance in a Limit Order Book (Ke Xu / Martin D. Gould / Sam D. Howison)
- Family: depth
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: —
- Uni: wf_mean=— · sign_stability=— · — · JSON —
- Causality: skipped · rule — · test —
- Pack-add: Δ vs hl_resid=— · skipped · JSON —
- Note: missing per-level bid/ask prices; depth buckets are volumes only

### vol_side_imb_far — arXiv:2211.13777

- Status: NOT OK
- Paper: The Short-Term Predictability of Returns in Order Book Markets (Lucchese / Pakkanen / Veraart)
- Family: depth
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.0056 · sign_stability=1.00 · reject · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_vol_side_imb_far.json`
- Causality: skipped · rule — · test —
- Pack-add: Δ vs hl_resid=— · skipped · JSON —
- Note: mean per-bucket (b−a)/(b+a) over v06..v11; corr≈0 vs near vol_side_imb; no 500ms IC

### btc_ret_1_bp — arXiv:2305.16915

- Status: partial
- Paper: When is cross impact relevant? (Victor Le Coz / Iacopo Mastromatteo / Damien Challet / Michael Benzaquen)
- Family: leader
- Horizon: y_500ms · Instrument: SOLUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.1003 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_SOLUSDT_PERP_btc_ret_1_bp.json`
- Causality: pass · rule as-of · test `tests/test_feature_truncation.py::test_btc_ret_asof_ignores_future_btc`
- Pack-add: Δ vs hl_resid=+0.0017 · WEAK · JSON `outputs/paper_seq/pack_add_SOLUSDT_PERP_btc_ret_1_bp.json`
- Note: BTC ret_1_bp as-of onto SOL; corr≈0.54 vs same-coin leader_ret; ETH trio was OK earlier, SOL single-col WEAK

### vb_tfi — arXiv:2108.09750

- Status: NOT OK
- Paper: Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets (Jakob Albers / Mihai Cucuringu / Sam Howison / Alexander Y. Shestopaloff)
- Family: flow
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=-0.0021 · sign_stability=0.55 · reject · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_vb_tfi.json`
- Causality: skipped · rule — · test —
- Pack-add: Δ vs hl_resid=— · skipped · JSON —
- Note: signed volume-bucket TFI = mean_(n=10) (V^B−V^S)/V; fixed V=500; corr≈0 vs calendar tfi_100 but no 500ms IC

### micro_adj — DOI:10.1080/14697688.2018.1545250

- Status: NOT OK
- Paper: The micro-price: a high-frequency estimator of future prices (Sasha Stoikov)
- Family: book
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.2221 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_micro_adj.json`
- Causality: pass · rule fold-fit · test `tests/test_feature_truncation.py::test_micro_adj_fold_ignores_test_labels`
- Pack-add: Δ vs hl_resid=-0.0001 · NOT OK · JSON `outputs/paper_seq/pack_add_BTCUSDT_PERP_micro_adj.json`
- Note: g(I,S)=E[next Δmid bp | I-bin×S-bin] train-only; corr≈0.997 vs stoikov_g1 on BTC (spread mostly 1-tick)

### vpin — arXiv:1912.10343

- Status: NOT OK
- Paper: Design of High-Frequency Trading Algorithm Based on Machine Learning (Boyue Fang / Yutong Feng)
- Family: flow
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=-0.0010 · sign_stability=0.45 · reject · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_vpin.json`
- Causality: skipped · rule — · test —
- Pack-add: Δ vs hl_resid=— · skipped · JSON —
- Note: volume-bucket VPIN = mean_|V^B−V^S|/V over last 50 buckets; fixed V=500; aggressor buy/sell (no BVC); unsigned toxicity ≠ signed 500ms return

### vol_side_imb — arXiv:2211.13777

- Status: partial
- Paper: The Short-Term Predictability of Returns in Order Book Markets (Lucchese / Pakkanen / Veraart)
- Family: depth
- Horizon: y_500ms · Instrument: BTCUSDT_PERP · Days: 2026-01-12..25
- Uni: wf_mean=0.1890 · sign_stability=1.00 · admit · JSON `outputs/paper_seq/uni_screen_BTCUSDT_PERP_vol_side_imb.json`
- Causality: pass · rule truncation+as-of · test `tests/test_feature_truncation.py::test_vol_side_imb_truncation`
- Pack-add: Δ vs hl_resid=+0.0012 · WEAK · JSON `outputs/paper_seq/pack_add_BTCUSDT_PERP_vol_side_imb.json`
- Note: mean per-bucket (bid_vj−ask_vj)/(bid_vj+ask_vj) over v01..v05; overlaps depth_imb_near; marginal on hl_resid

## 6. Sirignano–Cont pooled linear (BTC+ETH+SOL)

Per-asset `hl_resid` WF means: {'BTCUSDT_PERP': 0.2874, 'ETHUSDT_PERP': 0.2684, 'SOLUSDT_PERP': 0.2553}
Pooled shared-β + instrument FE: {'BTCUSDT_PERP': 0.2781, 'ETHUSDT_PERP': 0.2636, 'SOLUSDT_PERP': 0.2475}
Macro mean per-asset=0.2704 vs pooled=0.2631 (Δ=-0.0073)
→ **NOT OK** (pooling does not beat per-name here)

## Forward-add recheck — old uni admits vs hl_resid

BTCUSDT_PERP · y_500ms · days 2026-01-12..25 · min_delta 0.003.
31 columns that already cleared the uni gate and are outside `hl_resid`. `vpin`, `vb_tfi`, `vwap_mid_bp`, `vol_side_imb_far` stayed out (uni reject). `btc_ret_1_bp` is a SOL probe, not this pack.

Accepted: none. Final wf_mean 0.2874, same as the base.
Shelf (WEAK, Δ ∈ [0.001, 0.003)): `gould_ey` +0.0014 (|corr| 0.98), `vol_side_imb` +0.0012, `ret_4_bp` +0.0012, `pret_1000_bp` +0.0011, `depth_imb_near` +0.0011.
Drop, duplicate (|corr|_pack ≥ 0.9): `stoikov_g1` 0.98, `micro_adj` 0.97, `sweep_imb` 0.97 (`gould_ey` same, shelved above).
Drop, Δ below gate: the other 23 (`ofi_*`, `bid_of`, `ask_of`, `phi_cnt`, `micro_gap`, `sq_micro_bp`, `gap_dev`, `div_*`, `kf_vel`, `sweep_imba`, `sweep_imbb`, leader lags, longer TFI, `ret_1_bp`, `pret_500_bp`).

Report: `reports/FORWARD_ADD_BTCUSDT_PERP.md`
JSON: `outputs/paper_seq/forward_add_BTCUSDT_PERP_gould_ey_stoikov_g1_micro_adj_plus28.json`
