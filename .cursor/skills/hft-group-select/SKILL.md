---
name: hft-group-select
description: >-
  Forward-stepwise choose which admitted columns enter the linear pack by
  expanding WF Pearson Δ via scripts/probe_forward_add.py. Use when several
  candidates compete for a group, or the user asks which features to keep or
  drop without elastic net.
---

# Group select (forward add)

Build or extend a pack by greedy WF adds. One column vs `hl_resid` alone goes to `hft-group-admit`. Elastic net is not the selector here: SCORE is Pearson of the prediction, so keep/drop by OOS Δ.

Gate: `scripts/probe_forward_add.py` (same Δ bands as `verdict()` — default accept only Δ ≥ 0.003). Quote the JSON; do not restate `FEATURE_SETS`.

## Steps

1. Collect candidates that already cleared uni (+ causality when under a hunt). Name the start pack (`hl_resid` unless the user said otherwise).
   Done when the candidate list and base pack name are written.

2. Run forward add:
   `python scripts/probe_forward_add.py --cand <a> --cand <b> --base-pack hl_resid --min-delta 0.003`
   Use `--min-delta 0.001` only when the user wants WEAK adds into the pack.
   Done when the report lists accepted, dropped, final pack, and per-round Δ / resid_ic / |corr|_pack.

3. Read each round:
   - Accept → stays in `final_pack`.
   - Dropped with |corr|_pack ≳ 0.9 and tiny resid_ic → duplicate; leave out.
   - Dropped with resid_ic ≈ 0 → no incremental signal; leave out.
   - WEAK under default min_delta → shelf (`partial`), not core.
   Done when every candidate is accept, shelf, or drop with a one-line reason.

4. Log a short block under `reports/HFT_PAPER_EXPERIMENTS.md` (and update `FEATURE_SETS` only when the user asked to promote the new pack).
   Done when the answer points at `reports/FORWARD_ADD_<inst>.md` and the JSON path.

## Answer shape

```
# forward-add

Base: <pack> · min_delta <n>
Accepted: <cols or none>
Dropped: <col> — <duplicate|no resid_ic|Δ below gate>
Final pack: <list> · wf_mean <n>
Report: reports/FORWARD_ADD_<inst>.md
JSON: <path>
```
