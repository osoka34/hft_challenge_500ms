# HFT / microstructure papers usable for CMF 500ms

Context: best linear pack `hl_resid` ≈ **0.287** Pearson vs imb **0.216** (BTC WF).
Data: causal book L1 + depth buckets, trades 100ms, same-coin leaders. NHITS failed.

Status legend: **done** / **partial** / **todo**.

## Ranked next experiments

1. **Forward-add** among WEAK partials (`vol_side_imb`, Gould, `btc_ret_1_bp`→SOL) if rebuilding the pack.
2. **OFI / depth-normalized impact** Cont-style deepen — historical Cont stack NOT OK; only if a new scalar differs from `ofi_*`/`ofi_near`.
3. **Pooled multi-asset** Sirignano–Cont — already NOT OK as shared-β; skip unless new FE design.
4. **True MLOFI** (Xu/Gould/Howison) — **blocked** (needs per-level prices; buckets are volumes only).
5. **Albers sweep-IMB** — NOT OK without L2 prices.

Skip for now: full DeepLOB/raw LOB CNN, Hawkes full simulation, NHITS stack.

---

## Core papers

### 1. Cont, Kukanov, Stoikov — Order flow imbalance → price
- **The price impact of order book events**, arXiv:[1011.6402](https://arxiv.org/abs/1011.6402)
- Idea: short-horizon ΔP linear in L1 OFI; slope ~ 1/depth.
- Usable: multi-scale OFI, `ofi/depth`, maybe separate contrib of trades vs quotes. Tried `q_depl`, the same-price branch only: fraction of the best queue that disappears while that price stays put, summed over 500 ms.
- Difficulty: **easy** (OFI stack and `q_depl` both checked).
- Status: **partial** (OFI stack NOT OK; `q_depl` uni OK wf≈0.072, pack-add Δ≈+0.0003 NOT OK)

### 2. Gould / Cartea queue imbalance (direction)
- **Queue Imbalance as a One-Tick-Ahead Price Predictor**, arXiv:[1512.03492](https://arxiv.org/abs/1512.03492)
- Idea: logistic of next mid-move direction on queue imbalance; local logistic beats global.
- Usable: nonlinear link `I → P(up)` as feature, or piecewise β by |imb| bins (related to our `imb3`).
- Difficulty: **easy**
- Status: **partial** (`imb3` is crude version)

### 3. Stoikov — Micro-price
- **The micro-price: a high-frequency estimator of future prices** (Sasha Stoikov; QF 2018) DOI:[10.1080/14697688.2018.1545250](https://doi.org/10.1080/14697688.2018.1545250) / [code](https://github.com/sstoikov/microprice)
- Idea: \(P_\mathrm{micro}=M+g(I,S)\), \(g\) = limit of E[future mid | I,S] (Markov). Beats mid and weighted mid.
- Usable: tried `micro_adj` = train-only E[Δmid bp | I×S bins]; ≈ `stoikov_g1` on BTC (spread rarely >1 tick).
- Difficulty: **easy** (done)
- Status: **NOT OK** (uni OK wf≈0.222; pack-add Δ≈0 vs `hl_resid`)

### 4. Albers, Cucuringu, Howison, Shestopaloff — BTC cross-impact
- **Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets**, arXiv:[2108.09750](https://arxiv.org/abs/2108.09750)
- Idea: TFI, past returns, mean-divergence, softclip → linear models for **500ms** returns; leader–lagger by fee regime; sweep IMB.
- Usable: already our main playbook. Tried `sweep_px`, the realized walk: how far trade VWAP prints through the pre-bar touch. The bucket-cost proxy remains the old NOT OK sweep.
- Difficulty: **easy** (done; no L2 ladder required once VWAP is the price)
- Status: **partial** (TFI/soft/div/KF done; `vb_tfi` NOT OK; `sweep_px` uni OK wf≈0.104, pack-add Δ≈+0.0009 NOT OK)

### 5. Kolm, Turiel, Westray — Deep order flow
- **Deep Order Flow Imbalance**, SSRN:[3900141](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141) (Math Finance 2023)
- Idea: stationary **order flow** inputs beat raw LOB; architecture secondary; horizon ~ 2 price changes.
- Usable: build multi-level OF series (Δ depth buckets bid/ask separately) → linear first, LSTM only if residual path left.
- Difficulty: **medium** (snapshots ≈ OF; no true L3 events)
- Status: **partial** (`d_imb`, `ofi_near`)

### 6. Lucchese, Pakkanen, Veraart — Volume representation + DL horizons
- **The Short-Term Predictability of Returns in Order Book Markets**, arXiv:[2211.13777](https://arxiv.org/abs/2211.13777)
- Idea: predictability exists ~50–300 book updates (~ms–0.5s); **volume representation** of book matters more than net depth.
- Usable: volume-at-distance features from our `bid_v*`.`ask_v*` buckets; keep linear.
- Difficulty: **easy–medium**
- Status: **partial** (`vol_side_imb` WEAK; `vol_side_imb_far` NOT OK uni wf≈0.006)

### 7. Sirignano & Cont — Universal price formation
- **Universal features of price formation…**, arXiv:[1803.06917](https://arxiv.org/abs/1803.06917)
- Idea: path-dependent OF history; **pooled multi-stock** model beats per-name.
- Usable: pool BTC+ETH+SOL (+PEPE) linear/ridge with instrument intercept; optional shared softclip scales.
- Difficulty: **easy** for linear pool; hard for their LSTM
- Status: **todo**

### 8. Easley / Fang / Albers — volume clock
- Fang & Feng arXiv:[1912.10343](https://arxiv.org/abs/1912.10343) (VPIN); Albers arXiv:[2108.09750](https://arxiv.org/abs/2108.09750) (signed TFI).
- Idea: fixed-volume buckets; unsigned toxicity (`vpin`) or signed `(V^B−V^S)/V` (`vb_tfi`).
- Usable: both tried (V=500); neither predicts `y_500ms` on BTC WF.
- Difficulty: **easy** (done)
- Status: **NOT OK** (`vpin` wf≈−0.001; `vb_tfi` wf≈−0.002)

### 9. Cross-impact surveys
- **When is cross impact relevant?** (Victor Le Coz / Iacopo Mastromatteo / Damien Challet / Michael Benzaquen), arXiv:[2305.16915](https://arxiv.org/abs/2305.16915)
- Idea: liquid-asset flow/returns move less liquid correlated names.
- Usable: BTC→ETH pack OK (Δ=+0.003); SOL `btc_ret_1_bp` uni OK, pack-add WEAK Δ=+0.0017.
- Difficulty: **easy**
- Status: **partial** (`btc_ret_1_bp`→SOL)

### 10. Bieganowski & Ślepaczuk — VWAP-to-mid
- **Explainable Patterns in Cryptocurrency Microstructure**, arXiv:[2602.00776](https://arxiv.org/abs/2602.00776)
- Idea: on Binance futures, short-horizon trees lean on order-flow imbalance, spread, and buy/sell VWAP deviations from mid.
- Usable: tried `vwap_mid_bp` = qty-weighted (VWAP−mid)/mid over the last 1s. Spread and OFI already screened.
- Difficulty: **easy** (done)
- Status: **NOT OK** (`vwap_mid_bp` wf≈0.039, sign_stability 1.00; below the 0.05 uni gate)

### 11. Plerou, Gopikrishnan, Gabaix, Stanley — count imbalance
- **Quantifying Stock Price Response to Demand Fluctuations**, arXiv:[cond-mat/0106657](https://arxiv.org/abs/cond-mat/0106657)
- Idea: price change tracks Φ, the difference between the number of buyer-initiated and seller-initiated trades, with a concave tanh response. Share imbalance Ω is the volume analogue.
- Usable: tried `phi_cnt` = Σ a_i over the last 500 ms (`buy_n − sell_n`). Ω is already `tfi_500`.
- Difficulty: **easy** (done)
- Status: **NOT OK** (uni OK wf≈0.134, sign_stability 1.00; pack-add Δ≈+0.0006 vs `hl_resid`)

### 12. Bonart & Lillo — square-volume fundamental price
- **A continuous and efficient fundamental price on the discrete order book grid**, arXiv:[1608.00756](https://arxiv.org/abs/1608.00756)
- Idea: on large-tick books the fair price is the rebate-adjusted, square-volume weighted best quotes, and it tracks the fundamental price better than the linear volume-weighted price.
- Usable: tried `sq_micro_bp` = 10⁴ (p̂ − mid) / mid with p̂ from eq. (23) at r = 0 (no rebate on this tape). Linear microprice is already `micro_gap`.
- Difficulty: **easy** (done)
- Status: **NOT OK** (uni OK wf≈0.115, sign_stability 1.00; pack-add Δ≈+0.0002 vs `hl_resid`; corr≈0.991 vs `micro_gap`)

### 13. Valenzuela, Zer, Fryźlewicz, Rheinländer — relative liquidity
- **Relative liquidity and future volatility**, DOI:[10.1016/j.finmar.2015.03.001](https://doi.org/10.1016/j.finmar.2015.03.001)
- Idea: depth posted farther from the best quotes, relative to the rest of the book, marks disagreement about the price and is followed by higher volatility.
- Usable: tried `rel_liq` = V(2–100 bp) / V(0–2 bp), both sides together, and `liq_side` = near-touch share of bid depth minus the same share of ask depth.
- Difficulty: **easy** (done)
- Status: **NOT OK** (`rel_liq` wf≈−0.003, sign_stability 0.55; `liq_side` uni OK wf≈0.134, pack-add Δ≈−0.0009)

### 14. Dufour & Engle — trade waiting time
- **Time and the Price Impact of a Trade** (Alfonso Dufour, Robert F. Engle), DOI:[10.1111/0022-1082.00297](https://doi.org/10.1111/0022-1082.00297)
- Idea: shorter time between trades raises both the price impact of a trade and the autocorrelation of trade signs, so the more recent aggressor is the informative one.
- Usable: tried `sign_age` = (age of last sell − age of last buy) / (age of last sell + age of last buy).
- Difficulty: **easy** (done)
- Status: **NOT OK** (uni OK wf≈0.068, sign_stability 1.00; pack-add Δ≈0 vs `hl_resid`)

### 15. Eisler, Bouchaud, Kockelkoren — size versus count
- **The price impact of order book events: market orders, limit orders and cancellations**, arXiv:[0904.0900](https://arxiv.org/abs/0904.0900)
- Idea: the price impact of a market order depends only weakly on its size, so a volume imbalance and a count imbalance should carry almost the same signal.
- Usable: tried `size_gap`, the 500 ms volume imbalance minus the 500 ms count imbalance.
- Difficulty: **easy** (done)
- Status: **NOT OK** (`size_gap` wf≈0.026, sign_stability 1.00; below the 0.05 uni gate)

### 16. Oxenhorn — stablecoin depeg
- **A Multivariate Hawkes Process Model for Stablecoin-Cryptocurrency Depegging Event Dynamics**, arXiv:[2205.06338](https://arxiv.org/abs/2205.06338)
- Idea: a stablecoin's deviation from par is an excitatory event for the cryptocurrency it is used to trade.
- Usable: tried `usdc_prem_bp` = 10⁴ (USDCUSDT mid − 1), dropping quotes older than 3 s.
- Difficulty: **easy** (done)
- Status: **NOT OK** (`usdc_prem_bp` wf≈0.001, sign_stability 0.55; no 500 ms BTC correlation)

---

## Already covered / low priority

| Topic | Paper | Our status |
|-------|--------|------------|
| Classical queue imb | Cont / Gould | baseline |
| Softclip TFI + mean div | Albers | done |
| KF gap | (standard) | done |
| NHITS / deep on thin exog | — | failed vs linear |
| L1 micro_gap / `micro_adj` / G1 | Stoikov | weak / NOT OK add-on |

---

## Suggested reading order

1. Stoikov micro-price + github `sstoikov/microprice`
2. Cont et al. 1011.6402 (OFI/depth)
3. Albers 2108.09750 §3 (sweep IMB) — re-read for N-notional proxy
4. Kolm OF definition (separate bid/ask OF)
5. Sirignano–Cont (pooling) + Lucchese volume representation
