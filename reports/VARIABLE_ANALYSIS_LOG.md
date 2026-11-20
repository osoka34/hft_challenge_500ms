# Лог разбора переменных (HFT / PEPE start)

**Зачем этот файл.** Цифры и короткий вердикт.  
**Пояснения** — `EXPLANATIONS.md`.  
**План** — `HFT_ANALYSIS_PLAN.md`.  
**Код** — `notebooks/01_eda_pepe.ipynb`, `scripts/run_pepe_eda.py`.

**Данные.** Sample days train `2025-11-01…2025-11-07` (7d), val `2026-08-10…2026-08-15` (6d).  
JSON dump: `outputs/pepe_eda_results.json`.

---

## Roadmap

| # | Блок | Статус |
|---|------|--------|
| V01 | Data quality | **done** |
| V02 | Target zeros / daily vol | **done** |
| V03 | Own imbalance → y | **done** |
| V04 | PEPEUSDC → PEPEUSDT lead–lag | **done** |
| V05 | Same-coin mid gap | **done** |
| V06 | OLS imbalance baseline (PEPE-only) | **done** |

---

## V01 — Data quality

Статус: **done**

Цифры (USDT book/targets, 13 sample days):
- book rows / day: min 172796, median 172797, max 172798 (full grid 172800)
- first book lag from 00:00 UTC: min 1000 ms, median 1500 ms, max 2000 ms
- max intra-day book gap: min 500 ms, median 500 ms, max 500 ms
- locked quotes frac (median): 0.0000; crossed (median): 0.000000
- target NaN count across sample×3 horizons: 114 (ожидаемо у открытых точек дня)

Вывод: файлы чистые; день стартует с опозданием; дыры редкие на sample. **Keep** late-start / gap hygiene.

---

## V02 — Target zeros / daily vol

Статус: **done**

Цифры (доля exact zero, median по дням):
- y_500ms: 0.761
- y_2s: 0.445
- y_8s: 0.123
- daily std y_2s: median 1.34 bp (min 0.70, max 4.00)

Вывод: нули доминируют на коротком горизонте — как у peers. **Keep** как контекст потолка corr.

---

## V03 — Own imbalance → y

Статус: **done**

Цифры (mean Spearman IC по дням):
- train: {"y_500ms": 0.29623405567581484, "y_2s": 0.2227985311414566, "y_8s": 0.11580577362442326}
- val: {"y_500ms": 0.19551970392297247, "y_2s": 0.2396470087245376, "y_8s": 0.18536379882277165}
- доля |imb|>0.9 (mean): train 0.325, val 0.166

Вывод: imbalance связан с forward y; экстремумы часты на PEPE. **Keep** как ядро M0.

---

## V04 — PEPEUSDC → PEPEUSDT lead–lag

Статус: **done**

Цифры — 100 ms grid, median corr(r_USDC(t−k), r_USDT(t)); peak только по k>0 (lag 0 = contemporaneous):
- contemporaneous (k=0) train/val: 0.7403 / 0.7510
- train peak k>0: lag 100 ms, corr 0.1651
- val peak k>0: lag 100 ms, corr 0.0486
- train corr at 100/300 ms: 0.1651 / 0.0519 · reverse@100: 0.1426
- val corr at 100/300 ms: 0.0486 / 0.0240 · reverse@100: 0.0443

Цифры — 500 ms grid (book mids via bbo reindex), same definition:
- contemporaneous train/val: 0.9022 / 0.8375
- train peak k>0: lag 500 ms, corr 0.1330
- val peak k>0: lag 500 ms, corr 0.0888
- train@500/1000 ms: 0.1330 / 0.0442 · reverse@500: 0.0879
- val@500/1000 ms: 0.0888 / 0.0422 · reverse@500: 0.0597

Вывод: рынки почти синхронны (высокий corr@0). На 100 ms сетке USDC→USDT @100 ms сильнее reverse на train; на val lead слабее. **Keep** lagged USDC returns (≤300–500 ms).

---

## V05 — Same-coin mid gap

Статус: **done**

Цифры (mean Spearman):
- gap_dev (60s mean) → horizons train: {"gap_dev_y_500ms": -0.15087136115162614, "gap_dev_y_2s": -0.1303877410802146, "gap_dev_y_8s": -0.07958989019723474}
- gap_dev val: {"gap_dev_y_500ms": -0.11039106772484186, "gap_dev_y_2s": -0.14282306870127426, "gap_dev_y_8s": -0.12608977652745185}
- USDC last-500ms return → y_500ms: train 0.15393739195820672, val 0.10726826347849704

Вывод: `|gap_dev|` IC сопоставим с USDC return (~0.15 train на y_500ms), знак отрицательный = mean-reversion gap. **Keep** оба как M2 candidates.

---

## V06 — OLS imbalance baseline (PEPE-only SCORE)

Статус: **done**

Метрика: clip по train 0.1%/99.9% → daily Pearson → mean по 3 горизонтам PEPE (не полный 12-cell SCORE).

Цифры:
- M0 imb OLS train mean: 0.1688 · by H: {"y_2s": 0.1775573262888743, "y_500ms": 0.22947050800940733, "y_8s": 0.09944879079820204}
- M0 imb OLS val mean: **0.1457** · by H: {"y_2s": 0.16578224779861636, "y_500ms": 0.1343555690229641, "y_8s": 0.1370258305856735}
- M1 (imb, imb³, own r 0.5s/2s) val mean: **0.1503** · by H: {"y_2s": 0.1703585364446302, "y_500ms": 0.14046371887400222, "y_8s": 0.14001939193634913}
- Peer PEPE baseline cells ~0.15/0.16/0.12 (full val 62d) — наш sample короче, порядок должен быть похож.

Вывод: пайплайн метрики работает; M1 даёт небольшой lift. Дальше — depth/OFI/USDC features после догрузки остальных perps для полного SCORE.

---

## Hypotheses

| id | Claim | EDA support | Strength | Next |
|----|-------|-------------|----------|------|
| H1 | Own L1 imbalance forecasts PEPE short-horizon y | IC + baseline SCORE | strong | Keep in all models |
| H2 | Lagged PEPE·USDC returns help PEPE·USDT (≤300–500 ms) | corr@100ms > reverse on train; val weaker | medium–strong | Add lagged USDC returns |
| H3 | gap_dev mean-reverts (negative IC) | |IC|~0.11–0.15 | medium | Keep as M2 candidate |
| H4 | Zeros dominate y_500ms → corr ceiling lower than 8s | zero fractions | strong | Don't over-weight 500ms alone |

---

---

# Full-dataset EDA (V07+)

**Данные.** Official layout `data/{book,bbo,trades,targets,stable,meta}`, full calendar 365d (train 303, val 62).  
**Код.** `scripts/run_full_eda.py`, `scripts/full_eda_io.py`.  
**JSON.** `outputs/full_eda_results.json`.  
**Findings.** `reports/FULL_EDA_FINDINGS.md`.

## Roadmap (full)

| # | Блок | Coverage | Статус |
|---|------|----------|--------|
| V07 | Inventory / missing / depeg | full | **done** |
| V08 | Book DQ all instruments | full | **done** |
| V09 | Targets zeros / daily vol | full 4 perps | **done** |
| V10 | Own imb IC + OLS 12-cell SCORE | full | **done** |
| V11 | Lead–lag cross-corr | stratified sample + all val | **done** |
| V12 | Light Granger ΔR² | ~12 days | **done** |

---

## V07 — Inventory / missing / depeg

Статус: **done** · coverage: **full**

Цифры:
- calendar days: 365 (train 303, val 62)
- missing within listed range: 0 file entries
- USD1 pre-listing: BTC 80d (first 2025-12-04), ETH 87d (first 2025-12-11)
- stable missing days: 0
- depeg intervals: 149; minutes by coin: {"FDUSD": 204.99, "USD1": 4.45, "USDC": 7.68}

Вывод: совпадает с peer narrative — «дыры» USD1 = до листинга, не lost data. **Keep** availability flags для USD1.

---

## V08 — Book DQ (all instruments)

Статус: **done** · coverage: **full**

Цифры (все инструменты):
- rows median 172795.0 (grid 172800); first_lag median 2000.0 ms
- max_gap median 500.0 ms; days max_gap>60s: 8; >5min: 5
- locked median 0.0; crossed median 0.0
- peer outage hits logged: 9 instrument-day rows; top target gaps: [{"day": "2026-07-28", "instrument": "BTCUSDT_PERP", "max_gap_ms": 974000, "missing_vs_full": 2228, "n": 170572}, {"day": "2026-03-26", "instrument": "ETHUSDT_PERP", "max_gap_ms": 617500, "missing_vs_full": 1236, "n": 171564}, {"day": "2026-03-26", "instrument": "BTCUSDT_PERP", "max_gap_ms": 613500, "missing_vs_full": 1229, "n": 171571}, {"day": "2026-07-28", "instrument": "1000PEPEUSDT_PERP", "max_gap_ms": 125500, "missing_vs_full": 721, "n": 172079}, {"day": "2026-07-28", "instrument": "SOLUSDT_PERP", "max_gap_ms": 117500, "missing_vs_full": 268, "n": 172532}]

Вывод: locked/crossed ≈ 0; редкие длинные gaps совпадают с peer outage calendar. **Keep** gap/outage masks.

---

## V09 — Targets zeros / daily vol

Статус: **done** · coverage: **full 4 perps**

Цифры (median zero frac train / y_2s std bp train→val):
- BTCUSDT_PERP: zeros 500ms/2s/8s = 0.813/0.558/0.219; std 0.95 → 0.70 bp; 2025-10-10=3.92 bp
- ETHUSDT_PERP: zeros 500ms/2s/8s = 0.737/0.434/0.127; std 1.38 → 0.95 bp; 2025-10-10=8.12 bp
- SOLUSDT_PERP: zeros 500ms/2s/8s = 0.821/0.576/0.263; std 1.56 → 0.99 bp; 2025-10-10=11.50 bp
- 1000PEPEUSDT_PERP: zeros 500ms/2s/8s = 0.678/0.346/0.085; std 1.92 → 1.62 bp; 2025-10-10=19.88 bp

Вывод: нули доминируют на 500ms; val тише train; 2025-10-10 — явный vol spike. **Keep** как контекст потолка SCORE.

---

## V10 — Own imbalance → y + OLS SCORE

Статус: **done** · coverage: **full**

Spearman IC mean (train / val), 12-cell:
- train mean: 0.2736
- val mean: 0.2599

| instrument | y_500ms | y_2s | y_8s |
|---|---:|---:|---:|
| BTCUSDT_PERP | 0.2896 | 0.3333 | 0.2885 |
| ETHUSDT_PERP | 0.3012 | 0.3133 | 0.2363 |
| SOLUSDT_PERP | 0.2273 | 0.2739 | 0.2425 |
| 1000PEPEUSDT_PERP | 0.2218 | 0.2337 | 0.1568 |
| **mean** |  |  | **0.2599** |

OLS imb SCORE (clip = `meta/target_clip.parquet`, daily Pearson → mean cells):
- **train 12-cell mean: 0.1971**
- **val 12-cell mean: 0.1838** (organizers **0.1838**)

Val cells:
| instrument | y_500ms | y_2s | y_8s |
|---|---:|---:|---:|
| BTCUSDT_PERP | 0.1946 | 0.2155 | 0.1914 |
| ETHUSDT_PERP | 0.2004 | 0.2019 | 0.1583 |
| SOLUSDT_PERP | 0.1958 | 0.2212 | 0.1912 |
| 1000PEPEUSDT_PERP | 0.1550 | 0.1617 | 0.1191 |
| **mean** |  |  | **0.1838** |

Microstructure |imb|>0.9 / 1-tick:
- BTCUSDT_PERP: ext train/val 0.158/0.109; 1-tick train 0.997
- ETHUSDT_PERP: ext train/val 0.177/0.118; 1-tick train 0.998
- SOLUSDT_PERP: ext train/val 0.066/0.032; 1-tick train 0.999
- 1000PEPEUSDT_PERP: ext train/val 0.288/0.217; 1-tick train 0.994

Weekend vs weekday val SCORE: {"y_500ms": {"weekend": 0.14042377019454036, "weekday": 0.20525953891323978}, "y_2s": {"weekend": 0.1702123132749522, "weekday": 0.21227970545477587}, "y_8s": {"weekend": 0.1683949216484157, "weekday": 0.16363743326426541}}

Trade-flow vs imb (sample): see JSON `imb_score.trade_flow_vs_imb`.

Вывод: own L1 imb остаётся ядром; полный SCORE близок к 0.1838. Flow слабее imb на sample. **Keep** imb; flow secondary.

---

## V11 — Lead–lag cross-correlation

Статус: **done** · coverage: **sample train (35d) + val (62d)**

100 ms grid (median over days; peak only for k>0):

| pair@grid | train peak lag/corr | rev peak | val peak corr | corr@0 tr/va |
|---|---|---|---|---|
| BTCUSD1→BTCUSDT_PERP@100ms | 100 / 0.1451 | 0.0869 | 0.1312 | 0.5711/0.5412 |
| BTCUSDT_PERP→BTCUSDT_SPOT@100ms | 100 / 0.2917 | 0.0645 | 0.2818 | 0.4554/0.4180 |
| BTCUSDT_SPOT→BTCUSDT_PERP@100ms | 100 / 0.0645 | 0.2917 | 0.0553 | 0.4554/0.4180 |
| ETHUSD1→ETHUSDT_PERP@100ms | 100 / 0.1322 | 0.0833 | 0.1182 | 0.6104/0.5779 |
| ETHUSDT_PERP→ETHUSDT_SPOT@100ms | 100 / 0.2892 | 0.0522 | 0.3157 | 0.4794/0.4027 |
| ETHUSDT_SPOT→ETHUSDT_PERP@100ms | 100 / 0.0522 | 0.2892 | 0.0418 | 0.4794/0.4027 |
| PEPEUSDC→PEPEUSDT_PERP@100ms | 100 / 0.1176 | 0.1112 | 0.0713 | 0.7229/0.7613 |
| PEPEUSDT→PEPEUSDC_PERP@100ms | 100 / 0.1112 | 0.1176 | 0.0695 | 0.7229/0.7613 |
| SOLUSDT_SPOT→SOLUSDT_PERP@100ms | 100 / 0.1166 | 0.1599 | 0.0891 | 0.5905/0.5605 |
| SOLUSDT_PERP→SOLUSDT_SPOT@100ms | 100 / 0.1599 | 0.1166 | 0.1223 | 0.5905/0.5605 |
| BTCUSDC→BTCUSDT_PERP@100ms | 100 / 0.0872 | 0.2416 | 0.0717 | 0.5755/0.5196 |
| ETHUSDC→ETHUSDT_PERP@100ms | 100 / 0.0709 | 0.2261 | 0.0564 | 0.5842/0.5101 |

Вывод: perp→spot сильнее reverse; USD1→perp и PEPEUSDC→PEPEUSDT показывают короткий lead; val часто слабее train (как у peers). **Keep** short-lag leaders ≤300ms. Caveat: receive-time latency на тонких USD1.

---

## V12 — Light Granger ΔR²

Статус: **done** · coverage: **~light_sample_12_days**

Цифры (median ΔR², own lags + cross lags over 2s):
- BTCUSD1_SPOT→BTCUSDT_PERP: 100ms=0.0168, 500ms=0.0131
- BTCUSDT_PERP→BTCUSDT_SPOT: 100ms=0.1228, 500ms=0.0818
- ETHUSD1_SPOT→ETHUSDT_PERP: 100ms=0.0114, 500ms=0.0142
- ETHUSDT_PERP→ETHUSDT_SPOT: 100ms=0.1350, 500ms=0.0731
- 1000PEPEUSDC_PERP→1000PEPEUSDT_PERP: 100ms=0.0168, 500ms=0.0137
- SOLUSDT_SPOT→SOLUSDT_PERP: 100ms=0.0173, 500ms=0.0076

Вывод: порядок величины согласуется с peer (perp→spot крупнее, USD1/USDC→perp меньше). Не полный 17×17×365 Granger.

---

## Hypotheses (updated)

| id | Claim | EDA support | Strength | Next |
|----|-------|-------------|----------|------|
| H1 | Own L1 imb forecasts all 4 perps | full IC + SCORE≈0.18 | strong | Keep M0 |
| H2 | Perp leads spot (~100ms) | xcorr + ΔR² sample | strong | Lead features from perps into spots only as inputs the other way for targets |
| H3 | USD1 / PEPEUSDC lead same-coin perps | xcorr sample | medium | Mask USD1 latency; short lags |
| H4 | Val quieter → lower SCORE | full vol + SCORE gap | strong | Don't chase train SCORE |
| H5 | Trade flow << imb at 500ms | sample Spearman | medium | Don't over-invest in raw tflow |
