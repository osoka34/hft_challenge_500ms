# Univariate feature screen — SOLUSDT_PERP

Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson (`y_500ms`), days 2026-01-12..2026-01-25.
SIGNIFICANT if `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.
Scope: feats btc_ret_1_bp.

Significant: **1** / 1

| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |
|---------|---------|-------------|--------------|------|-----|--------|
| `btc_ret_1_bp` | 0.1003 | 0.1072 | 0.1301 | 1.00 | YES | cross_btc |

## Significant only (candidates for later group tests)

- `btc_ret_1_bp` — WF 0.1003, spearman 0.1301

JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/uni_screen_SOLUSDT_PERP_btc_ret_1_bp.json`
