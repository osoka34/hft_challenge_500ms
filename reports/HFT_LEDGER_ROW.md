# Ledger row (append-only)

Single source of truth for one hunt / screen / causality / pack-add line.
Append under `reports/HFT_PAPER_EXPERIMENTS.md`. Mirror status in `reports/HFT_PAPERS.md` when a paper section exists.

## Status words

| Status | Meaning |
|--------|---------|
| `todo` | Hypothesis logged; not screened yet |
| `blocked` | Missing field; cannot implement on this tape |
| `NOT OK` | Failed uni gate, causality, or pack-add |
| `partial` | Uni admit + causality pass; pack-add not run or WEAK |
| `done` | Pack-add vs `hl_resid` is OK (Δ ≥ 0.003) |

## Row template

Copy and fill. One column per block. Authors come from the live abstract, not a nickname in `HFT_PAPERS.md`.

```
### <column> — <arXiv:id | SSRN:id | DOI | none>

- Status: <todo | blocked | NOT OK | partial | done>
- Paper: <title> (<authors from abstract>)
- Family: <book | leader | depth | flow | gap | other>
- Horizon: <y_500ms | y_2s | y_8s> · Instrument: <inst> · Days: <first>..<last>
- Uni: wf_mean=<n> · sign_stability=<n> · admit|reject · JSON `<path>`
- Causality: <pass | fail | skipped> · rule <truncation|as-of|…> · test `<tests/…>`
- Pack-add: Δ vs hl_resid=<n> · <OK | WEAK | NOT OK | skipped> · JSON `<path>`
- Note: <one line>
```

## Close-out (end of hunt)

After the final status is known:

1. Upsert the summary table row in `HFT_PAPER_EXPERIMENTS.md`.
2. Set the matching section in `HFT_PAPERS.md` to that status (never leave «in progress»).
3. Remove the idea from «Next in queue» once it has any terminal or partial result.

## Forward-add (several candidates)

When choosing among multiple admitted columns, run `scripts/probe_forward_add.py` and point at `reports/FORWARD_ADD_<inst>.md`. Do not dump all uni-YES into `FEATURE_SETS`. Default accept gate Δ ≥ 0.003; shelf WEAK.
