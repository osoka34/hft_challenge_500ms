---
name: hft-group-admit
description: >-
  Probe whether one uni-admitted column adds walk-forward Pearson on top of
  the hl_resid pack via scripts/probe_pack_add.py. Use after causality lock,
  or when the user asks if a feature helps beyond imbalance/hl_resid.
---

# Group admit (pack-add)

Decide if one locked column earns `done` or stays `partial` / becomes `NOT OK`. Uni screen alone does not measure incremental value on `hl_resid`. This skill stops after the pack-add verdict is logged.

Gate lives in `scripts/probe_pack_add.py` / `verdict()` in `scripts/run_paper_seq.py`: Δ ≥ 0.003 → OK, ≥ 0.001 → WEAK, else NOT OK. Quote those; do not restate `FEATURE_SETS["hl_resid"]`.

## Steps

1. Confirm the column passed uni screen and causality (or the user waived causality).
   Done when you can name wf_mean, sign_stability, and the causality result.

2. Run one pack-add:
   `python scripts/probe_pack_add.py --feat <column> --instrument BTCUSDT_PERP --horizon y_500ms`
   Done when the JSON reports `wf_mean_base`, `wf_mean_add`, `delta`, and `verdict`.

3. Append a ledger row using the template in `reports/HFT_LEDGER_ROW.md`.
   - OK → status `done`
   - WEAK → status `partial`
   - NOT OK → status `NOT OK`
   Done when `reports/HFT_PAPER_EXPERIMENTS.md` has the block.

## Answer shape

```
# <column> pack-add

Base hl_resid: wf_mean <n>
With column: wf_mean <n> · Δ <n> → OK | WEAK | NOT OK
Status: done | partial | NOT OK
Logged: reports/HFT_PAPER_EXPERIMENTS.md
JSON: <path>
```
