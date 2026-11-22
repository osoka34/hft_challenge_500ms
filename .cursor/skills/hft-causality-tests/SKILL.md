---
name: hft-causality-tests
description: >-
  Write and run a truncation test that a feature at grid time t is unchanged
  once every event after t is removed. Use when the user asks for a leak test,
  a truncation check, or to lock a feature's causality on this challenge.
---

# Causality tests

Lock one builder with a truncation test. A question about the tape as observed goes to `hft-tape-eda`. A question about whether the column predicts `y` goes to `hft-feature-screen`. A claimed `SCORE` or a pass over the whole pipeline goes to `hft-pipeline-review`. This skill stops when the test has passed or failed and the verdict is logged when it failed under a hunt.

The organizer re-runs a day cut off at `T`. Rows with grid time `t ≤ T` must match the full day. That match is the truncation test.

## Steps

1. Name the builder and which rule below it has to satisfy.
   A pure row-local map (same inputs → same output per row, no history) is **not** locked by truncating its numpy helper alone — that always passes. For those builders pick **as-of** (or trade hole) on the join onto the grid.
   Done when the test file will assert one of: truncation, as-of, trade hole, fold-fit, train clip — and the chosen rule can actually fail if the builder peeks.

2. Add the smallest pytest under `tests/` that fails when the rule breaks. Prefer extending `tests/test_feature_truncation.py` (truncation / as-of templates already there) over a new file. Build the frame in the test. Open a parquet day only when the builder cannot run on a synthetic frame.
   Done when the test edits or deletes events with `ts > T` and compares rows with `t ≤ T`.

3. Run `python -m pytest tests/test_feature_truncation.py -q` (or the file you edited). Install `pytest` if it is missing.
   Done when the reply quotes pass or fail and the rule that broke.

4. On **fail**, append ledger status `NOT OK` using `reports/HFT_LEDGER_ROW.md`. On **pass**, under a hunt hand off to `hft-group-admit` (status becomes `partial` only after pack-add is recorded, or stays pending pack-add); outside a hunt, state the lock without claiming `done` or 12-cell `SCORE`.
   Done when fail has a ledger line, or pass states the next handoff.

## Rules

**Truncation.** Build the column on the full series and again after every event with `ts > T` is removed. Values at `t ≤ T` match. This is the check for `load_day_panel`, `_causal_softsign` (scale at `t` uses history before `t`), `_gap_dev` (rolling mean is `shift(1)`), and `causal_reindex_mid`.

**As-of.** A join onto the target grid is `merge_asof(..., direction="backward")` on `ts`. An update with `ts > t` leaves the row at `t` unchanged.

**Trade hole.** A grid point with no trade at `ts ≤ t` keeps the last print at or before `t`, or stays missing. A later print does not fill it. The builder is `_tfi_from_trades` followed by that as-of join.

**Fold-fit.** `apply_stoikov_g1_fold` and `apply_gould_fold` map the test frame from a model fit on the train frame only. The test-frame column is invariant to `y` and to `stoikov_next_dmid_bp` on the test frame. `stoikov_next_dmid_bp` is the next mid change inside the day (`add_stoikov_micro`); it is a calibration label, not a feature column.

**Train clip.** `fit_clip_bounds` receives days on or before `TRAIN_END` in `scripts/full_eda_io.py`. `load_target_clip` is the official table. A screen of horizon `H` clips with key `(instrument, H)`.

**Legal target.** `y` is the column in the targets parquet. A feature test does not rebuild the return from `mid(t)`. On the neuralforecast path, `run_nf_segments.py` stores the row label in `y_comp` and trains on `y_comp` shifted by one row within the day; compare features on that path to `y_comp` of the same row.

## Answer shape

```
# <builder>

Rule: <truncation | as-of | trade hole | fold-fit | train clip>
Test: tests/<file>.py::<name>
Result: pass | fail — <which rows moved after the cut>
Logged: reports/HFT_PAPER_EXPERIMENTS.md | none

Cut: events with ts > T removed. Compared: rows with t ≤ T.
```
