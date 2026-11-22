# Univariate feature screen — BTCUSDT_PERP

Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson (`y_500ms`), days 2026-01-12..2026-01-25.
SIGNIFICANT if `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.
Scope: feats vpin.

Significant: **0** / 1

| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |
|---------|---------|-------------|--------------|------|-----|--------|
| `vpin` | -0.0010 | -0.0008 | -0.0010 | 0.45 | no | vpin |

## Significant only (candidates for later group tests)


JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/uni_screen_BTCUSDT_PERP_vpin.json`
