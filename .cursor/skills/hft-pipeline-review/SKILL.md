---
name: hft-pipeline-review
description: >-
  Review a pipeline run for a false SCORE, a leak, or a clip on the wrong
  horizon, then propose a new column only as a screen hypothesis. Use when the
  user asks for a pipeline review, a gross error, or the next feature to try
  on this challenge.
---

# Pipeline review

Review one run. A tape fact goes to `hft-tape-eda`. One column's signal goes to `hft-feature-screen`. Locking a builder goes to `hft-causality-tests`. Pack-add on `hl_resid` goes to `hft-group-admit`. An end-to-end hunt goes to `hft-feature-hunt`. A search for a new column in a paper goes to `hft-arxiv-features`. This skill names what the number actually is, then lists gross errors. A new column appears only after that list, only when the user asked for a next feature, and only as a hypothesis for `hft-feature-implement` (then screen).

Language (`SCORE`, the 10 ms delay, `ts ≤ t`) lives in `CONTEXT.md`. The published 12-cell baseline lives in `reports/FULL_EDA_FINDINGS.md`. Coverage of a run is whatever that script loaded, not the "local status" note in `CONTEXT.md`.

## Steps

1. Name the entry point and the number it claims.
   Done when the reply says which script or report, which instrument and horizon, which days, and whether the number is the 12-cell `SCORE` or a narrower probe.

2. Walk the checks below against that path.
   Done when each hit has a file, what the code does, and a severity: disqualification, wrong number, or weak.

3. When the user asked for a next feature, and no disqualification sits on the number you would compare against, write one hypothesis per idea and hand it to `hft-feature-implement`.
   Done when each idea names the nearest column already in `FEATURE_SETS` (`scripts/nf_features.py`) or in `CORE_CANDS` / `PAPER_CANDS`.

## What counts as SCORE

The 12-cell `SCORE` is daily Pearson of the prediction against `y` clipped with train bounds, averaged over validation days, then over 4 target perps × 3 horizons. `phase_imb_and_score` in `scripts/run_full_eda.py` reports that mean as `mean_12`.

These are probes. Call them probes:

- `scripts/hft_metric.py` `mean_score` and `scripts/run_pepe_eda.py`: PEPE, three horizons.
- `screen()` and `wf_scores`: one instrument, clip key `(instrument, "y_500ms")`, mean of daily Pearsons on the days passed in.
- `scripts/eval_albers_linear.py`: same clip key; `--mode day` is a within-day 70/30 split; `--mode wf` averages those days only.
- `scripts/run_nf_segments.py`: one Pearson pooled over rows. The training label is `y_comp` shifted by one row within the day. The default loss is `mse`; `pearson` is opt-in via `scripts/nf_losses.py`.

`scripts/hft_common.py` `TRAIN_DAYS` / `VAL_DAYS` are a short PEPE probe on `data/raw`. The challenge calendar is `TRAIN_END` / `VAL_START` in `scripts/full_eda_io.py`.

## Gross errors

**Disqualification.** A feature row at `t` uses an event with `ts > t`. A join onto the grid is not `merge_asof(..., direction="backward")`. `stoikov_next_dmid_bp` (the next mid change inside the day) is passed as a model input; it is a calibration label for `apply_stoikov_g1_fold`. `y` is rebuilt from `mid(t)` to `mid(t+H)` instead of the targets parquet. Clip bounds are fit on a day after `TRAIN_END`.

**Wrong number.** A probe from the list above is reported as the 12-cell `SCORE`. Days are pooled into one Pearson and that Pearson is called `SCORE`. The clip key stays `y_500ms` while the panel horizon is `y_2s` or `y_8s`. Imbalance is not `(bid_qty − ask_qty) / (bid_qty + ask_qty)`. On the neuralforecast path, a feature at a row is scored against the shifted `y` instead of `y_comp` on that same row.

**Weak.** A new column duplicates `imb`, `imb3`, or `micro_gap` and the run treats the duplicate as incremental `SCORE`.

## Answer shape

```
# <entry point>

Claims: <what the run calls the number>
Actually: <12-cell SCORE | probe: instrument, horizon, days, pooled|daily>

## Errors
- <severity> — <file>: <what the code does>

## Next columns
- <column>: nearest `<existing>`, family <book|leader|depth|flow|gap>. Hand to hft-feature-implement.
```

Leave the next-columns block empty when the user did not ask for a feature. A disqualification empties it too: say the number is not comparable until that hit is fixed.
