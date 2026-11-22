# Univariate feature screen — BTCUSDT_PERP

Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson (`y_500ms`), days 2026-01-12..2026-01-25.
SIGNIFICANT if `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.
Scope: feats sq_micro_bp.

Significant: **1** / 1

| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |
|---------|---------|-------------|--------------|------|-----|--------|
| `sq_micro_bp` | 0.1146 | 0.0912 | 0.3377 | 1.00 | YES | sq_micro |

## Significant only (candidates for later group tests)

- `sq_micro_bp` — WF 0.1146, spearman 0.3377

JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/uni_screen_BTCUSDT_PERP_sq_micro_bp.json`
