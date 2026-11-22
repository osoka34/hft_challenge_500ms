# Univariate feature screen — BTCUSDT_PERP

Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson (`y_500ms`), days 2026-01-12..2026-01-25.
SIGNIFICANT if `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.
Scope: feats vol_side_imb.

Significant: **1** / 1

| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |
|---------|---------|-------------|--------------|------|-----|--------|
| `vol_side_imb` | 0.1890 | 0.1969 | 0.2582 | 1.00 | YES | lucchese_vol |

## Significant only (candidates for later group tests)

- `vol_side_imb` — WF 0.1890, spearman 0.2582

JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/uni_screen_BTCUSDT_PERP_vol_side_imb.json`
