# Univariate feature screen — BTCUSDT_PERP

Protocol: expanding WF OLS `y ~ x`, clipped-y Pearson, days 2026-01-12..2026-01-25.
SIGNIFICANT if `|wf_mean| ≥ 0.05` and sign_stability ≥ 0.7.

Significant: **34** / 45

| feature | wf_mean | oos_pearson | oos_spearman | stab | sig | family |
|---------|---------|-------------|--------------|------|-----|--------|
| `gould_ey` | 0.2434 | 0.2500 | 0.3398 | 1.00 | YES | gould |
| `imb3` | 0.2407 | 0.2499 | 0.3434 | 1.00 | YES | book |
| `stoikov_g1` | 0.2220 | 0.2276 | 0.3321 | 1.00 | YES | stoikov |
| `imb` | 0.2162 | 0.2223 | 0.3434 | 1.00 | YES | book |
| `sweep_imb` | 0.2148 | 0.2216 | 0.3356 | 1.00 | YES | albers_sweep |
| `depth_imb_near` | 0.2081 | 0.2167 | 0.3050 | 1.00 | YES | book |
| `kf_innov` | 0.1858 | -0.2016 | -0.2323 | 1.00 | YES | gap_kf |
| `sweep_imbb` | 0.1854 | -0.1914 | -0.2801 | 1.00 | YES | albers_sweep |
| `sweep_imba` | 0.1842 | 0.1903 | 0.2803 | 1.00 | YES | albers_sweep |
| `leader_ret_1_bp` | 0.1837 | 0.2028 | 0.2432 | 1.00 | YES | leader |
| `ofi_raw` | 0.1731 | 0.1853 | 0.2252 | 1.00 | YES | cont_ofi |
| `gap_dev` | 0.1665 | -0.1727 | -0.2284 | 1.00 | YES | gap_kf |
| `d_imb` | 0.1585 | 0.1779 | 0.1704 | 1.00 | YES | book |
| `div_150s` | 0.1564 | -0.1590 | -0.2162 | 1.00 | YES | gap_kf |
| `div_300s` | 0.1462 | -0.1471 | -0.2028 | 1.00 | YES | gap_kf |
| `bid_of` | 0.1418 | 0.1497 | 0.2059 | 1.00 | YES | cont_ofi |
| `ask_of` | 0.1389 | 0.1484 | 0.2056 | 1.00 | YES | cont_ofi |
| `tfi_s100` | 0.1279 | 0.1322 | 0.1489 | 1.00 | YES | tfi |
| `leader_ret_300ms_bp` | 0.1196 | 0.1347 | 0.1846 | 1.00 | YES | leader |
| `pret_500_bp` | 0.1191 | 0.1311 | 0.2603 | 1.00 | YES | book |
| `ret_1_bp` | 0.1191 | 0.1311 | 0.2603 | 1.00 | YES | book |
| `micro_gap` | 0.1191 | 0.0957 | 0.3435 | 1.00 | YES | book |
| `tfi_s500` | 0.1160 | 0.1171 | 0.1701 | 1.00 | YES | tfi |
| `pret_1000_bp` | 0.1115 | 0.1193 | 0.2378 | 1.00 | YES | book |
| `tfi_100` | 0.1074 | 0.0962 | 0.1514 | 1.00 | YES | tfi |
| `ofi_near` | 0.1040 | 0.1104 | 0.1641 | 1.00 | YES | cont_ofi |
| `leader_ret_200ms_bp` | 0.1010 | 0.1137 | 0.1650 | 1.00 | YES | leader |
| `tfi_s1000` | 0.0990 | 0.0968 | 0.1522 | 1.00 | YES | tfi |
| `kf_vel` | 0.0985 | -0.0990 | -0.1510 | 1.00 | YES | gap_kf |
| `leader_ofi` | 0.0950 | 0.0876 | 0.2143 | 1.00 | YES | leader |
| `ret_4_bp` | 0.0949 | 0.1001 | 0.2012 | 1.00 | YES | book |
| `tfi_500` | 0.0720 | 0.0696 | 0.1741 | 1.00 | YES | tfi |
| `tfi_1000` | 0.0622 | 0.0579 | 0.1561 | 1.00 | YES | tfi |
| `leader_tfi_s100` | 0.0596 | 0.0587 | 0.0923 | 1.00 | YES | tfi |
| `leader_tfi_100` | 0.0412 | 0.0395 | 0.0920 | 1.00 | no | tfi |
| `ofi_best` | 0.0316 | 0.0300 | 0.2288 | 1.00 | no | cont_ofi |
| `ofi_depth` | 0.0316 | 0.0300 | 0.2288 | 1.00 | no | cont_ofi |
| `ofi_h2_d` | 0.0294 | 0.0163 | 0.2542 | 1.00 | no | cont_ofi |
| `leader_tfi_s500` | 0.0280 | 0.0264 | 0.0413 | 1.00 | no | tfi |
| `ofi_h8_d` | 0.0212 | 0.0148 | 0.2180 | 1.00 | no | cont_ofi |
| `ofi_h4_d` | 0.0209 | 0.0172 | 0.2476 | 1.00 | no | cont_ofi |
| `kf_level` | 0.0205 | -0.0024 | -0.0096 | 1.00 | no | gap_kf |
| `leader_imb` | 0.0197 | -0.0206 | -0.0320 | 1.00 | no | leader |
| `spread_bp` | -0.0024 | 0.0018 | 0.0006 | 0.55 | no | book |
| `depth_imb_mid` | 0.0002 | 0.0004 | -0.0243 | 0.45 | no | book |

## Significant only (candidates for later group tests)

- `gould_ey` — WF 0.2434, spearman 0.3398
- `imb3` — WF 0.2407, spearman 0.3434
- `stoikov_g1` — WF 0.2220, spearman 0.3321
- `imb` — WF 0.2162, spearman 0.3434
- `sweep_imb` — WF 0.2148, spearman 0.3356
- `depth_imb_near` — WF 0.2081, spearman 0.3050
- `kf_innov` — WF 0.1858, spearman -0.2323
- `sweep_imbb` — WF 0.1854, spearman -0.2801
- `sweep_imba` — WF 0.1842, spearman 0.2803
- `leader_ret_1_bp` — WF 0.1837, spearman 0.2432
- `ofi_raw` — WF 0.1731, spearman 0.2252
- `gap_dev` — WF 0.1665, spearman -0.2284
- `d_imb` — WF 0.1585, spearman 0.1704
- `div_150s` — WF 0.1564, spearman -0.2162
- `div_300s` — WF 0.1462, spearman -0.2028
- `bid_of` — WF 0.1418, spearman 0.2059
- `ask_of` — WF 0.1389, spearman 0.2056
- `tfi_s100` — WF 0.1279, spearman 0.1489
- `leader_ret_300ms_bp` — WF 0.1196, spearman 0.1846
- `pret_500_bp` — WF 0.1191, spearman 0.2603
- `ret_1_bp` — WF 0.1191, spearman 0.2603
- `micro_gap` — WF 0.1191, spearman 0.3435
- `tfi_s500` — WF 0.1160, spearman 0.1701
- `pret_1000_bp` — WF 0.1115, spearman 0.2378
- `tfi_100` — WF 0.1074, spearman 0.1514
- `ofi_near` — WF 0.1040, spearman 0.1641
- `leader_ret_200ms_bp` — WF 0.1010, spearman 0.1650
- `tfi_s1000` — WF 0.0990, spearman 0.1522
- `kf_vel` — WF 0.0985, spearman -0.1510
- `leader_ofi` — WF 0.0950, spearman 0.2143
- `ret_4_bp` — WF 0.0949, spearman 0.2012
- `tfi_500` — WF 0.0720, spearman 0.1741
- `tfi_1000` — WF 0.0622, spearman 0.1561
- `leader_tfi_s100` — WF 0.0596, spearman 0.0923

JSON: `/home/alexander/Projects/hft_challenge_500ms/outputs/paper_seq/uni_screen_BTCUSDT_PERP.json`
