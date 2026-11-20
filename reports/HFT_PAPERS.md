# HFT / microstructure papers usable for CMF 500ms

Context: best linear pack `hl_resid` ≈ **0.287** Pearson vs imb **0.216** (BTC WF).
Data: causal book L1 + depth buckets, trades 100ms, same-coin leaders. NHITS failed.

Status legend: **done** / **partial** / **todo**.

## Ranked next experiments

1. **Stoikov micro-price \(g(I,S)\)** (not L1 VWAP) — calibrate expected mid move by imbalance×spread buckets on train → feature `micro_adj`. Easy; we only tried `micro_gap`.
2. **OFI / depth-normalized impact** Cont-style: `ofi / depth` multi-horizon + separate bid/ask OF (Kolm OF, not just net OFI). Partial (`ofi_best`); deepen.
3. **Volume-clock features** (VPIN / bucket TFI): resample own+leader flow in volume time, not calendar 100ms. Medium; may help PEPE.
4. **Pooled multi-asset linear / light path model** (Sirignano–Cont universality): train one model on BTC+ETH+SOL with instrument FE. Medium.
5. **Albers sweep-IMB** (avg execution price for notional \(N\)) via depth buckets proxy. Medium; needs price-distance assumption.

Skip for now: full DeepLOB/raw LOB CNN, Hawkes full simulation, NHITS stack.

---

## Core papers

### 1. Cont, Kukanov, Stoikov — Order flow imbalance → price
- **The price impact of order book events**, arXiv:[1011.6402](https://arxiv.org/abs/1011.6402)
- Idea: short-horizon ΔP linear in L1 OFI; slope ~ 1/depth.
- Usable: multi-scale OFI, `ofi/depth`, maybe separate contrib of trades vs quotes.
- Difficulty: **easy** (we have Cont OFI; extend horizons + depth norm).
- Status: **partial**

### 2. Gould / Cartea queue imbalance (direction)
- **Queue Imbalance as a One-Tick-Ahead Price Predictor**, arXiv:[1512.03492](https://arxiv.org/abs/1512.03492)
- Idea: logistic of next mid-move direction on queue imbalance; local logistic beats global.
- Usable: nonlinear link `I → P(up)` as feature, or piecewise β by |imb| bins (related to our `imb3`).
- Difficulty: **easy**
- Status: **partial** (`imb3` is crude version)

### 3. Stoikov — Micro-price
- **The micro-price: a high-frequency estimator of future prices** (QF 2018); SSRN / [code](https://github.com/sstoikov/microprice) / PDF mirror often circulated as OBI.pdf
- Idea: \(P_\mathrm{micro}=M+g(I,S)\), \(g\) = limit of E[future mid | I,S] (Markov). Beats mid and weighted mid.
- Usable: train \(g\) on past days → `micro_adj = g(I,S)` or `P_micro-M` as linear feature (replaces weak `micro_gap`).
- Difficulty: **easy–medium** (need mid-change events; book snapshots OK for L1 I,S)
- Status: **todo** (L1 VWAP tried; true \(g\) not)

### 4. Albers, Cucuringu, Howison, Shestopaloff — BTC cross-impact
- **Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets**, arXiv:[2108.09750](https://arxiv.org/abs/2108.09750)
- Idea: TFI, past returns, mean-divergence, softclip → linear models for **500ms** returns; leader–lagger by fee regime; sweep IMB.
- Usable: already our main playbook; remaining = sweep IMB + LASSO across more leaders.
- Difficulty: sweep **medium** without L2 prices
- Status: **partial** (TFI/soft/div/KF done)

### 5. Kolm, Turiel, Westray — Deep order flow
- **Deep Order Flow Imbalance**, SSRN:[3900141](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141) (Math Finance 2023)
- Idea: stationary **order flow** inputs beat raw LOB; architecture secondary; horizon ~ 2 price changes.
- Usable: build multi-level OF series (Δ depth buckets bid/ask separately) → linear first, LSTM only if residual path left.
- Difficulty: **medium** (snapshots ≈ OF; no true L3 events)
- Status: **partial** (`d_imb`, `ofi_near`)

### 6. Briola et al. — Volume representation + DL horizons
- **The Short-Term Predictability of Returns in Order Book Markets**, arXiv:[2211.13777](https://arxiv.org/abs/2211.13777)
- Idea: predictability exists ~50–300 book updates (~ms–0.5s); **volume representation** of book matters more than net depth.
- Usable: volume-at-distance features from our `bid_v*`.`ask_v*` buckets; keep linear.
- Difficulty: **easy–medium**
- Status: **todo**

### 7. Sirignano & Cont — Universal price formation
- **Universal features of price formation…**, arXiv:[1803.06917](https://arxiv.org/abs/1803.06917)
- Idea: path-dependent OF history; **pooled multi-stock** model beats per-name.
- Usable: pool BTC+ETH+SOL (+PEPE) linear/ridge with instrument intercept; optional shared softclip scales.
- Difficulty: **easy** for linear pool; hard for their LSTM
- Status: **todo**

### 8. Easley, López de Prado, O’Hara — VPIN
- ELO VPIN (2011/2012); critiques exist (e.g. Andersen–Bondarenko).
- Idea: volume-clock toxicity / imbalance in fixed-size volume buckets.
- Usable: volume-bucket TFI / VPIN-like on own+leader trades as slow feature; more for PEPE volatility regimes than 500ms IC.
- Difficulty: **medium**
- Status: **todo** (skeptical for SCORE; try thin)

### 9. Cross-impact surveys
- **When is cross impact relevant?**, arXiv:[2305.16915](https://arxiv.org/abs/2305.16915)
- Idea: when cross-asset flow moves another name.
- Usable: BTC→ETH/SOL leader features (not only same-coin USD1); cross-perp returns/TFI.
- Difficulty: **easy** if we allow BTC as leader for alts
- Status: **todo**

---

## Already covered / low priority

| Topic | Paper | Our status |
|-------|--------|------------|
| Classical queue imb | Cont / Gould | baseline |
| Softclip TFI + mean div | Albers | done |
| KF gap | (standard) | done |
| NHITS / deep on thin exog | — | failed vs linear |
| L1 micro_gap | Stoikov weighted mid | weak |

---

## Suggested reading order

1. Stoikov micro-price + github `sstoikov/microprice`
2. Cont et al. 1011.6402 (OFI/depth)
3. Albers 2108.09750 §3 (sweep IMB) — re-read for N-notional proxy
4. Kolm OF definition (separate bid/ask OF)
5. Sirignano–Cont (pooling) + Briola volume representation
