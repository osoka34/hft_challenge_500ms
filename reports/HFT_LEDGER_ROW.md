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

Copy and fill. One column per block.

```
### <column> — <arXiv:id | SSRN:id | DOI | none>

- Status: <todo | blocked | NOT OK | partial | done>
- Paper: <title>
- Family: <book | leader | depth | flow | gap | other>
- Horizon: <y_500ms | y_2s | y_8s> · Instrument: <inst> · Days: <first>..<last>
- Uni: wf_mean=<n> · sign_stability=<n> · admit|reject · JSON `<path>`
- Causality: <pass | fail | skipped> · rule <truncation|as-of|…> · test `<tests/…>`
- Pack-add: Δ vs hl_resid=<n> · <OK | WEAK | NOT OK | skipped> · JSON `<path>`
- Note: <one line>
```
