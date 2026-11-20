---
name: hft-feature-hunt
description: >-
  One-command loop: paper → implement → uni screen → causality lock → pack-add
  → log verdict. Invoke by name when hunting features end-to-end on this challenge.
disable-model-invocation: true
---

# Feature hunt

Run one closed loop that admits or rejects a column and writes the verdict. Workers stay in their own skills; this skill only sequences them and owns the stop.

Default budget: **one** column per invocation. A larger `N` only when the user set it. Stop early on a disqualification that makes numbers incomparable.

Ledger format: `reports/HFT_LEDGER_ROW.md`. Status words: `todo` → `blocked` / `NOT OK` / `partial` / `done` (see that file). Uni + causality pass is **`partial`**, not `done`.

## Loop (repeat at most N times)

1. **Pick.** Open and follow `.cursor/skills/hft-arxiv-features/SKILL.md`.
   If the user already named a column and formula, skip this step and use that hypothesis.
   Done when you have one hypothesis with mapped inputs, or a missing-field note.
   On missing field: ledger `blocked`, go to step 7.

2. **Build.** Open and follow `.cursor/skills/hft-feature-implement/SKILL.md`.
   Done when the column is registered and smoked, or blocked.
   On block: go to step 7.

3. **Screen.** Open and follow `.cursor/skills/hft-feature-screen/SKILL.md`.
   Run with `--feat <column>` (never the full suite inside a hunt).
   Done when the reply is admit or reject with `wf_mean` and `sign_stability`.
   On reject: ledger `NOT OK`, go to step 7.

4. **Lock.** Open and follow `.cursor/skills/hft-causality-tests/SKILL.md`.
   Extend `tests/test_feature_truncation.py` when possible.
   Done when the rule passes or fails.
   On fail: ledger `NOT OK`, go to step 7.
   On pass: ledger status stays heading to pack-add (`partial` until step 5 finishes).

5. **Pack-add.** Open and follow `.cursor/skills/hft-group-admit/SKILL.md`.
   Done when Δ vs `hl_resid` is OK / WEAK / NOT OK and the ledger status is `done` / `partial` / `NOT OK`.

6. **Admit summary.** Quote uni metrics, causality, and pack-add Δ. This hunt does not claim 12-cell `SCORE`.

7. **Next or stop.** If iterations left and the user allowed another pick, return to step 1 with a query that excludes logged `NOT OK` / `done` / `blocked` nearest columns. Else stop.

## When to pull the others

- Tape fact or quality blocker mid-build → `.cursor/skills/hft-tape-eda/SKILL.md`, then resume implement or mark `blocked`.
- Claimed `SCORE` or a path you would compare against → `.cursor/skills/hft-pipeline-review/SKILL.md` first. A disqualification empties the hunt until that hit is fixed.

## Answer shape (end of hunt)

```
# hunt

Budget: <k>/<N>
Column: <name> | none
Verdict: blocked | reject | partial | done | admit-leaky-reject
wf_mean / sign_stability: <numbers or —>
Causality: pass | fail | skipped
Pack-add Δ: <n or —> · OK|WEAK|NOT OK|skipped
Logged: reports/HFT_PAPER_EXPERIMENTS.md · reports/HFT_PAPERS.md
Next: <stop | queued hypothesis>
```
