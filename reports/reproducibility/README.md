# Experiment Reproducibility Runbook

## Goal
Re-run experiments with the same config and seed and verify weighted F1 is within tolerance (default 0.01).

## Replay Steps
1. Re-run training with identical `--config`, `--seed`, and input data.
2. Evaluate the new run with:
   - `python -m benthic_model.cli evaluate --run-id <run_id> --fold-scheme spatial_blocked`
3. Inspect:
   - `reports/reproducibility/<run_id>_reproducibility.md`
   - `reports/reproducibility/<run_id>_reproducibility.json`

## Required Provenance
Each run must retain:
- code revision
- config reference and config hash
- seed
- fold scheme
- command invocation
- data split reference
- output artifact paths
