# Agent index

Feature loop (user-invoked): `hft-feature-hunt` — paper → implement → uni `--feat` → causality → pack-add → log.

Workers (model-invoked): `hft-arxiv-features`, `hft-feature-implement`, `hft-feature-screen`, `hft-causality-tests`, `hft-group-admit`, `hft-group-select`.

Side doors: `hft-pipeline-review` (false SCORE / leak / wrong clip), `hft-tape-eda` (tape facts).

Pack keep/drop: one column → `hft-group-admit` / `probe_pack_add.py`; several → `hft-group-select` / `probe_forward_add.py` (greedy WF Δ, not elastic net).

Challenge language: `CONTEXT.md`.  
Ledger template: `reports/HFT_LEDGER_ROW.md` → append to `reports/HFT_PAPER_EXPERIMENTS.md`.  
Paper status: `reports/HFT_PAPERS.md`.  
Truncation templates: `tests/test_feature_truncation.py`.
