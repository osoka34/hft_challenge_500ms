# Forward add — BTCUSDT_PERP · y_500ms

Base: `hl_resid` → ['imb', 'imb3', 'leader_ret_1_bp', 'kf_innov', 'tfi_100', 'tfi_s100', 'd_imb']
Candidates: ['gould_ey', 'stoikov_g1', 'micro_adj', 'sweep_imb', 'sweep_imba', 'sweep_imbb', 'depth_imb_near', 'vol_side_imb', 'ofi_raw', 'ofi_near', 'bid_of', 'ask_of', 'gap_dev', 'div_150s', 'div_300s', 'kf_vel', 'micro_gap', 'sq_micro_bp', 'phi_cnt', 'ret_1_bp', 'ret_4_bp', 'pret_500_bp', 'pret_1000_bp', 'leader_ret_200ms_bp', 'leader_ret_300ms_bp', 'leader_ofi', 'tfi_500', 'tfi_1000', 'tfi_s500', 'tfi_s1000', 'leader_tfi_s100']
min_delta: 0.003

Accepted: **(none)**
Final pack (0.2874): `['imb', 'imb3', 'leader_ret_1_bp', 'kf_innov', 'tfi_100', 'tfi_s100', 'd_imb']`
Dropped: ['gould_ey', 'stoikov_g1', 'micro_adj', 'sweep_imb', 'sweep_imba', 'sweep_imbb', 'depth_imb_near', 'vol_side_imb', 'ofi_raw', 'ofi_near', 'bid_of', 'ask_of', 'gap_dev', 'div_150s', 'div_300s', 'kf_vel', 'micro_gap', 'sq_micro_bp', 'phi_cnt', 'ret_1_bp', 'ret_4_bp', 'pret_500_bp', 'pret_1000_bp', 'leader_ret_200ms_bp', 'leader_ret_300ms_bp', 'leader_ofi', 'tfi_500', 'tfi_1000', 'tfi_s500', 'tfi_s1000', 'leader_tfi_s100']

## Round 1 (base wf_mean=0.2874)

| feat | Δ | resid_ic | |corr|_pack | verdict |
|------|---|----------|------------|---------|
| `gould_ey` | +0.0014 | -0.0593 | 0.98 | WEAK |
| `vol_side_imb` | +0.0012 | -0.0275 | 0.67 | WEAK |
| `ret_4_bp` | +0.0012 | +0.0027 | 0.40 | WEAK |
| `pret_1000_bp` | +0.0011 | -0.0033 | 0.58 | WEAK |
| `depth_imb_near` | +0.0011 | -0.0347 | 0.82 | WEAK |
| `ofi_near` | +0.0007 | +0.0039 | 0.60 | NOT OK |
| `phi_cnt` | +0.0006 | -0.0172 | 0.67 | NOT OK |
| `micro_gap` | +0.0002 | -0.0249 | 0.27 | NOT OK |
| `sq_micro_bp` | +0.0002 | -0.0237 | 0.29 | NOT OK |
| `stoikov_g1` | +0.0001 | -0.0624 | 0.98 | NOT OK |
| `leader_ret_300ms_bp` | +0.0001 | -0.0269 | 0.67 | NOT OK |
| `tfi_500` | +0.0001 | -0.0363 | 0.43 | NOT OK |
| `div_300s` | +0.0001 | +0.0456 | 0.58 | NOT OK |
| `leader_tfi_s100` | +0.0000 | -0.0129 | 0.21 | NOT OK |
| `div_150s` | +0.0000 | +0.0482 | 0.65 | NOT OK |
| `sweep_imbb` | -0.0000 | +0.0517 | 0.83 | NOT OK |
| `tfi_s1000` | -0.0000 | -0.0320 | 0.35 | NOT OK |
| `sweep_imba` | -0.0000 | -0.0509 | 0.83 | NOT OK |
| `gap_dev` | -0.0000 | +0.0528 | 0.76 | NOT OK |
| `leader_ret_200ms_bp` | -0.0001 | -0.0114 | 0.49 | NOT OK |
| `micro_adj` | -0.0001 | -0.0616 | 0.97 | NOT OK |
| `tfi_1000` | -0.0001 | -0.0249 | 0.30 | NOT OK |
| `sweep_imb` | -0.0001 | -0.0581 | 0.97 | NOT OK |
| `kf_vel` | -0.0001 | +0.0520 | 0.51 | NOT OK |
| `bid_of` | -0.0001 | -0.0128 | 0.46 | NOT OK |
| `ask_of` | -0.0001 | -0.0150 | 0.46 | NOT OK |
| `pret_500_bp` | -0.0002 | -0.0180 | 0.78 | NOT OK |
| `ret_1_bp` | -0.0002 | -0.0180 | 0.78 | NOT OK |
| `tfi_s500` | -0.0002 | -0.0360 | 0.44 | NOT OK |
| `ofi_raw` | -0.0003 | -0.0181 | 0.57 | NOT OK |
| `leader_ofi` | -0.0003 | -0.0080 | 0.31 | NOT OK |

Stop: best Δ=0.0013758477210183973 < min_delta=0.003

JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/forward_add_BTCUSDT_PERP_gould_ey_stoikov_g1_micro_adj_plus28.json`
