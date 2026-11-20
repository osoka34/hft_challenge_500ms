---
name: hft-feature-implement
description: >-
  Turn one feature hypothesis into a runnable column on this 500 ms tape:
  builder code, PAPER_CANDS or panel registration, ready for screen().
  Use when the user asks to implement a paper feature, wire a column into
  the screen, or after hft-arxiv-features hands off a hypothesis.
---

# Feature implement

Make one column computable for `screen()`. A search for the idea goes to `hft-arxiv-features`. Whether it predicts `y` goes to `hft-feature-screen`. Locking the builder goes to `hft-causality-tests`. This skill stops when the column appears in a day panel under the agreed name, or when a missing field blocks the build.

## Steps

1. Take the hypothesis: column name, formula, inputs, nearest logged column.
   If any input is still a missing field (message stream, per-level prices, true sweep ladder), stop here: append status `blocked` using `reports/HFT_LEDGER_ROW.md` and hand nothing to screen.
   Done when every input is an existing stream column, or the block is written.

2. Place the builder where the screen already loads peers.
   - Always-on own-book or as-of join on the grid → `load_day_panel` in `scripts/nf_features.py` (same `merge_asof(..., direction="backward")` pattern as existing joins).
   - Paper / optional column loaded only for the uni screen → `add_*` in `scripts/run_paper_seq.py`, called from `screen()` in `scripts/screen_features_uni.py` next to `add_stoikov_micro` / `add_cont_ofi_depth` / `add_albers_sweep_proxy`.
   - Map estimated from `y` or from a calibration label on the train fold only → `apply_*_fold(tr, te)` like `apply_stoikov_g1_fold` / `apply_gould_fold`, and a branch inside `screen()` for that name.
   Register the name in `PAPER_CANDS` (or `CORE_CANDS` only if it belongs in every core list). Quote those lists; do not restate them.
   Done when the name is in a cand list and a code path assigns that column on the panel.

3. Smoke one day: load a single panel (or run the `add_*` on one day dict) and confirm the column exists, is finite on a non-trivial fraction of rows, and uses only `ts ≤ t` inputs.
   Done when you can quote instrument, day, non-null rate, and the file:function that builds it.

4. Hand the column to `hft-feature-screen`. Under a hunt, also mark the idea `todo` or `partial` in `reports/HFT_PAPERS.md` if it was absent.
   Done when the answer names the column, the registration list, and the smoke numbers.

## Answer shape

```
# <column>

Builder: <file>:<function>
Registered: PAPER_CANDS | CORE_CANDS | fold branch in screen()
Smoke: <instrument> · <day> · non-null <rate>
Nearest: `<existing>` | absent

Ready for hft-feature-screen: <column>, horizon <y_500ms|y_2s|y_8s>
```

On a missing-field block:

```
# <column>

Blocked: missing <field>
Logged: reports/HFT_PAPER_EXPERIMENTS.md
```
