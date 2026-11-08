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
