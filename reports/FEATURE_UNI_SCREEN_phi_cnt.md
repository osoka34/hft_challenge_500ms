# Univariate feature screen — BTCUSDT_PERP

Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson (`y_500ms`), days 2026-01-12..2026-01-25.
SIGNIFICANT if `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.
Scope: feats phi_cnt.

Significant: **1** / 1

| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |
|---------|---------|-------------|--------------|------|-----|--------|
| `phi_cnt` | 0.1342 | 0.1418 | 0.2224 | 1.00 | YES | phi_cnt |

## Significant only (candidates for later group tests)

- `phi_cnt` — WF 0.1342, spearman 0.2224

JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/uni_screen_BTCUSDT_PERP_phi_cnt.json`
