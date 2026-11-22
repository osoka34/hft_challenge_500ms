# Univariate feature screen — BTCUSDT_PERP

Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson (`y_500ms`), days 2026-01-12..2026-01-25.
SIGNIFICANT if `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.
Scope: feats q_depl.

Significant: **1** / 1

| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |
|---------|---------|-------------|--------------|------|-----|--------|
| `q_depl` | 0.0721 | 0.0791 | 0.0957 | 1.00 | YES | q_depl |

## Significant only (candidates for later group tests)

- `q_depl` — WF 0.0721, spearman 0.0957

JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/uni_screen_BTCUSDT_PERP_q_depl.json`
