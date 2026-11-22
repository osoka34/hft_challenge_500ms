# Univariate feature screen — BTCUSDT_PERP

Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson (`y_500ms`), days 2026-01-12..2026-01-25.
SIGNIFICANT if `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.
Scope: feats micro_adj.

Significant: **1** / 1

| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |
|---------|---------|-------------|--------------|------|-----|--------|
| `micro_adj` | 0.2221 | 0.2310 | 0.3358 | 1.00 | YES | stoikov |

## Significant only (candidates for later group tests)

- `micro_adj` — WF 0.2221, spearman 0.3358

JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/uni_screen_BTCUSDT_PERP_micro_adj.json`
