---
name: hft-tape-eda
description: >-
  Report the 500 ms tape as observed: file coverage, book gaps and locked or
  crossed quotes, stale trade bars, target zeros and volatility, own-book
  imbalance, and same-coin lead–lag. Use when the user asks for EDA, data
  quality, gaps, stale trades, target distributions, or lead–lag on this
  challenge.
---

# Tape EDA

Report a fact about the tape. A question about whether a new column predicts `y` ends here: hand it to `hft-feature-implement` if the column is missing, else `hft-feature-screen`, and stop. An end-to-end hunt goes to `hft-feature-hunt`.

Terms (`grid point`, `y_H(t)`, `SCORE`, target perps, feature instruments) live in `CONTEXT.md`. Quote that file; do not restate it.

## Steps

1. Read `CONTEXT.md` and the matching section of `reports/FULL_EDA_FINDINGS.md`.
   Done when the fact is either already published there, or you can name the gap: instrument, stream, day set, statistic.

2. If it is published, answer from that file.
   Done when the reply gives the number, full versus sampled coverage, and whether the figure is PEPE-only or the 12-cell `SCORE`.

3. If it is missing, compute the smallest slice that answers it.
   - Calendar, splits, and paths come from `scripts/full_eda_io.py` (`TRAIN_START`/`TRAIN_END`, `VAL_START`/`VAL_END`, `path_for`). `scripts/hft_common.py` `TRAIN_DAYS`/`VAL_DAYS` are a short PEPE probe on `data/raw`, not the challenge calendar.
   - Put any stream on the target grid with `merge_asof(..., direction="backward")` on `ts`. The value at grid point `t` is the last event with `ts ≤ t`.
   - A missing trade bar stays missing. Age it from the last print at or before `t`.
   - Prefer an existing phase in `scripts/run_full_eda.py` when it already computes the statistic: `phase_inventory`, `phase_book_dq`, `phase_targets`, `phase_imb_and_score`, `phase_leadlag`, `phase_granger_light`. Run the whole script only when the user asks to refresh `reports/FULL_EDA_FINDINGS.md`.
   Done when the new number names its day set and the as-of rule used.

4. Answer in the shape below. Append under `reports/` only when the user asked to record it.
   Done when the first line states scope.

## Answer shape

```
# <question>

Scope: <PEPE-only | 12-cell> · <full | sample: N days> · <train | val | both>

- <fact with the number>
- <fact>

As-of: last event with ts ≤ t. Source: <report section, JSON key, or phase>
```

The published baseline (val 12-cell OLS imbalance `SCORE`) is in `reports/FULL_EDA_FINDINGS.md`. Quote it. Refit only when that file does not contain the requested slice.
