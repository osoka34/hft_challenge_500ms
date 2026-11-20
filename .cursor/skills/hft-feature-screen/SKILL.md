---
name: hft-feature-screen
description: >-
  Judge whether one column predicts the target under the univariate expanding
  walk-forward in scripts/screen_features_uni.py. Use when the user proposes a
  feature, asks if a column has signal, or wants a screen before any group or
  nonlinear model.
---

# Feature screen

Admit or reject one column. A question about the tape as observed goes to `hft-tape-eda`. A column that is not yet on the panel goes to `hft-feature-implement` first. A column that passes here is locked with `hft-causality-tests` before any group fit. This skill stops before the group fit, and before any nonlinear model.

The gate lives in `scripts/screen_features_uni.py`. Quote it; do not restate the candidate lists. Ledger lines follow `reports/HFT_LEDGER_ROW.md`.

## Steps

1. Read `reports/FEATURE_UNI_SCREEN.md` (and any `FEATURE_UNI_SCREEN_<feat>.md`) plus `CORE_CANDS` / `PAPER_CANDS` in `scripts/screen_features_uni.py`.
   Done when you can name the nearest published column and its `wf_mean`, or state that the column is absent from the report.
   If the name is missing from both cand lists and from a day panel, hand off to `hft-feature-implement` and stop.

2. Write the hypothesis before any run: column, family, horizon, the published column it might duplicate, and the kill rule.
   The kill rule is the script's: `|wf_mean| < 0.05` or `sign_stability < 0.7` (threshold defaults to 0.05).
   Done when those five are written.

3. Quote a published row when instrument, horizon, and days match the table.
   The published full table is `BTCUSDT_PERP`, horizon `y_500ms`, days `2026-01-12`..`2026-01-25`. Any other instrument, horizon, or day set is a new run.
   Done when the reply says the number is quoted or still to be computed.

4. A new run for **one** column:
   `python scripts/screen_features_uni.py --feat <column> --instrument BTCUSDT_PERP --horizon y_500ms`
   - Expanding OLS of `y ~ x` on past days only, three-day burn-in, Pearson of the prediction against clipped `y`.
   - Clip key is `(instrument, horizon)` from `load_target_clip()` — same horizon as `--horizon`.
   - `stoikov_g1` and `gould_ey` estimate their map on the train fold only. A new column that estimates anything from `y` uses the same fold split.
   - Single-feat runs write `reports/FEATURE_UNI_SCREEN_<feat>.md` and `outputs/paper_seq/uni_screen_{inst}_{feat}.json` (full suite still overwrites `FEATURE_UNI_SCREEN.md`).
   - Copy the row into `reports/HFT_PAPER_EXPERIMENTS.md` using `reports/HFT_LEDGER_ROW.md` before any later full-suite wipe matters.
   Done when the result names instrument, horizon, days, `wf_mean`, `sign_stability`, and the verdict.

5. On **reject**, ledger status `NOT OK`. On **admit**, hand the builder to `hft-causality-tests` (uni YES is not yet `done`).
   Done when the ledger line exists on reject, or the causality handoff is stated on admit.

## How to read a row

`wf_mean` is the Pearson of the OLS prediction, so a negative slope still produces a positive `wf_mean`. `oos_pearson` is the raw column against clipped `y` and may have the opposite sign. The gate uses `wf_mean` and `sign_stability`.

A `YES` admits the column to a causality lock, then pack-add via `hft-group-admit`. It does not measure incremental `SCORE` on top of `imb` or `hl_resid`.

## Answer shape

```
# <column>

Hypothesis: <family>, horizon <y_500ms|y_2s|y_8s>, nearest published column `<name>` (wf_mean <number or absent>)
Scope: <instrument> · <days> · quoted | new run (--feat)

- wf_mean <number>, sign_stability <number> → admit | reject
- oos_pearson <number> (raw column; sign may oppose wf_mean)
- Duplicates: <published column, or none>

Gate: |wf_mean| ≥ 0.05 and sign_stability ≥ 0.7. Source: <report row or JSON path>
Logged: reports/HFT_PAPER_EXPERIMENTS.md
Next: hft-causality-tests | stop
```
