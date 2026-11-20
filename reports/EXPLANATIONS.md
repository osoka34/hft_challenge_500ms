# Пояснения (HFT / PEPE start)

**Зачем этот файл.** Простыми словами: что считаем, зачем, как читать, как пользоваться.  
**Цифры** — в `VARIABLE_ANALYSIS_LOG.md`.

Аудитория: студент. Формат:

```
## Vxx — имя
### Зачем
### Как считается
### Что показывает
### Как пользоваться
### Что нам дало
```

---

## V01 — Data quality (coverage, gaps, NaN)

### Зачем

Перед любым сигналом нужно знать, где книга/сделки/таргеты дырявые. Иначе «alpha» = артефакт outage.

### Как считается

Считаем число строк `book` vs 172 800; задержку первого тика от 00:00; max gap между соседними `ts`; долю NaN в `targets`; crossed/locked quotes.

### Что показывает

На наших sample-днях (см. лог): день почти полный, но старт с опозданием; редкие дыры; NaN у открытых точек дня — как у peers.

### Как пользоваться

Маска outage / late-start; не учить модель на дырявых минутах без флага.

### Что нам дало

Hygiene gate — без неё нельзя доверять IC и SCORE.

---

## V02 — Target zeros / daily vol

### Зачем

На 500 ms mid часто не двигается → много точных нулей в `y`. Это задаёт потолок корреляции и режим волы.

### Как считается

Доля `y == 0` по горизонтам; daily std `y_2s` в basis points (×1e4).

### Что показывает

Короткий горизонт «тише»; PEPE волатильнее крупных монет (peer: ~2 bp vs BTC ~1 bp).

### Как пользоваться

Не удивляться низкому SCORE на 500 ms; смотреть режимы волы отдельно.

### Что нам дало

Контекст для чтения baseline и lead–lag.

---

## V03 — Own imbalance → y

### Зачем

Базовый сигнал организаторов и #1 фича у Meshcheryakov: очередь на best bid/ask.

### Как считается

`imb = (bid_qty − ask_qty) / (bid_qty + ask_qty)` на строке book в `t`. Spearman IC и mean `y` по бинам `imb`.

### Что показывает

S-shape: экстремальный imb сильнее связан с будущим mid-return.

### Как пользоваться

Keep как ядро baseline / M0; потом взаимодействия с vol и quote age.

### Что нам дало

Калибровка пайплайна против peer baseline cells на PEPE.

---

## V04 — PEPEUSDC → PEPEUSDT lead–lag

### Зачем

Peer EDA: для PEPE главный внешний источник — USDC-perp того же коина. Лекция 04.10: HF lead–lag у связанных рынков реален.

### Как считается

Mid log-returns на 100 ms и 500 ms (bbo → causal reindex на сетку). Cross-correlation `corr(r_USDC(t−k), r_USDT(t))` для лагов 0…2 s. Lag 0 = contemporaneous; сравниваем пик при **k>0** и reverse.

### Что показывает

Очень высокая синхронность (corr@0 ~0.7–0.9). На k>0 USDC→USDT чуть сильнее reverse (особенно на 500 ms train: ~0.13 vs ~0.09); на val lead слабеет.

### Как пользоваться

Keep short-lag USDC returns / missed moves как фичи; не ждать минутных лагов.

### Что нам дало

Главный cross-market угол, пока нет BTC/ETH.

---

## V05 — Same-coin mid gap

### Зачем

Если два perp на один коин временно разошлись, gap может закрываться (half-life у peers — секунды, не минуты).

### Как считается

`gap = ln(mid_USDT / mid_USDC)`; `gap_dev = gap − mean(gap over last 60 s)` (causal). IC к `y`.

### Что показывает

Слабее сырого lead return, но дешёвый same-coin признак (в духе M2 из EDA PDF).

### Как пользоваться

Keep как candidate; проверить после baseline lift.

### Что нам дало

Кандидат в M2 без остальных инструментов.

---

## V06 — OLS imbalance baseline (PEPE-only SCORE)

### Зачем

Нужна локальная копия метрики организаторов, иначе нельзя мерить прогресс до догрузки BTC/ETH/SOL.

### Как считается

Clip `y` по train 0.1%/99.9%; OLS `ŷ = a + b·imb` per horizon на train; daily Pearson на val; среднее по 3 горизонтам PEPE.

### Что показывает

Порядок ~0.14–0.16 на PEPE (peer cells), не полный 0.1838.

### Как пользоваться

Регрессионный тест пайплайна; дальше M1 (`imb³`, own returns).

### Что нам дало

Working SCORE harness на одном инструменте.

## V07 — Inventory / missing / depeg

### Зачем

Нужно отделить «файла нет, потому что рынок ещё не листился» от реальных дыр в данных.

### Как считается

Сверяем календарь дат с `meta/instruments.parquet` (`first_date`/`last_date`) и наличием parquet в `data/{book,bbo,trades,targets,stable}`. Depeg intervals — из `meta/depeg_intervals.parquet`.

### Что показывает

Почти все «пропажи» — USD1 до листинга. Stable depeg минуты малы относительно года.

### Как пользоваться

Availability flags для USD1/alt quotes; не импутить pre-listing дни.

### Что нам дало

Согласование с peer: 167 missing = pre-listing, не corruption.

---

## V08 — Book DQ (all instruments)

### Зачем

Outage и late-start ломают IC/SCORE и маскируются под «рынок замер».

### Как считается

По каждому book-дню: `n` vs 172800, first lag от 00:00 UTC, max gap, locked/crossed фракции. Смотрим peer-даты outage отдельно.

### Что показывает

Locked/crossed ≈ 0; редкие многоминутные gaps на известных outage днях.

### Как пользоваться

Маска `max_gap` / missing rows; не учить flow-фичи без stale-флага.

### Что нам дало

Hygiene gate на полном году, не только PEPE sample.

---

## V09 — Targets zeros / daily vol

### Зачем

Доля нулей задаёт потолок корреляции; дневная вола — режимы, где lead–lag сильнее.

### Как считается

Exact-zero fraction по горизонтам; daily std `y_2s` в bp; train vs val; spike-день 2025-10-10; weekend vs weekday.

### Что показывает

500ms почти всегда «тихий»; val тише train; 10-10 — экстремум.

### Как пользоваться

Не оптимизировать только 500ms; учитывать vol regimes.

### Что нам дало

Полная калибровка 4 perps против peer цифр.

---

## V10 — Own imbalance IC + OLS SCORE

### Зачем

Это официальный baseline и якорь прогресса (0.1838).

### Как считается

`imb=(bid_qty−ask_qty)/(bid_qty+ask_qty)` через causal `merge_asof` book→targets. Clip из `meta/target_clip.parquet`. OLS на train; daily Pearson; среднее 12 клеток. Дополнительно Spearman IC, |imb|>0.9, 1-tick share, sample trade-flow IC.

### Что показывает

Imb доминирует; SCORE близок к organizers; flow слабее на коротком горизонте.

### Как пользоваться

M0 = OLS imb; всё остальное меряем как lift над ним.

### Что нам дало

Полный 12-cell SCORE на локальных данных.

---

## V11 — Lead–lag cross-correlation

### Зачем

Кто на сколько миллисекунд опережает целевой perp / spot — главный cross-market угол peers.

### Как считается

BBO mid → causal reindex на 100/500 ms сетку → log-return → `corr(r_leader(t−k), r_follower(t))` и reverse. Медиана по sample-дням; peak только для k>0.

### Что показывает

Высокий corr@0 (синхронность); perp→spot сильнее spot→perp; короткие лаги 100–300 ms; USD1/USDC leaders осмысленны, но тонкие рынки чувствительны к receive-time.

### Как пользоваться

Лагированные mid-returns лидеров ≤300–500 ms как фичи; не minute-scale lead–lag.

### Что нам дало

Проверка peer claims на официальной раскладке данных (sample, не 365d Granger).

---

## V12 — Light Granger ΔR²

### Зачем

Cross-corr не отделяет «общий шок» от дополнительной предсказательной силы.

### Как считается

На sample днях: ΔR² = R²(y ~ own lags + leader lags) − R²(y ~ own lags), лаги на окне ~2 s.

### Что показывает

Порядок величины: perp→spot крупнее, same-coin alt→perp меньше — как у peers.

### Как пользоваться

Приоритизация лидеров для фич, не замена полному Granger.

### Что нам дало

Дешёвая проверка направления без Go-пайплайна peers.
