# Quickstart: Pipeline CV Improvement Investigation

**Feature**: 019-pipeline-cv-improvement  
**Date**: 2026-04-01

## Prerequisites

```powershell
cd C:\DEVlocal\BTM-Hybrid\btm
git checkout 019-pipeline-cv-improvement
$env:PYTHONPATH = "src"
```

Ensure `.venv` is activated and dependencies installed:

```powershell
.venv\Scripts\Activate.ps1
```

## 1. Run R04 Baseline (Reference)

```powershell
.venv\Scripts\python.exe -m benthic_model.cli train `
  --train-csv data/train_btm.csv `
  --bathymetry-tif data/MBES/bathymetry.tif `
  --backscatter-tif data/MBES/backscatter.tif `
  --config configs/rf-btm-fine.yaml `
  --run-type candidate --seed 42
```

Expected output: `CV weighted_f1 ≈ 0.8024`

## 2. Run Multi-Seed Variance Analysis

```powershell
foreach ($seed in 42, 123, 456, 789, 2026) {
  .venv\Scripts\python.exe -m benthic_model.cli train `
    --train-csv data/train_btm.csv `
    --bathymetry-tif data/MBES/bathymetry.tif `
    --backscatter-tif data/MBES/backscatter.tif `
    --config configs/rf-btm-fine.yaml `
    --run-type candidate --seed $seed
}
```

Then compare metrics across all runs to compute mean, std, range.

## 3. Run LightGBM Experiments

```powershell
.venv\Scripts\python.exe -m benthic_model.cli train `
  --train-csv data/train_btm.csv `
  --bathymetry-tif data/MBES/bathymetry.tif `
  --backscatter-tif data/MBES/backscatter.tif `
  --config configs/lgbm-btm-tuned.yaml `
  --run-type candidate --seed 42
```

## 4. Generate Predictions

```powershell
.venv\Scripts\python.exe -m benthic_model.cli predict `
  --run-id <RUN_ID> `
  --test-csv data/test.csv `
  --bathymetry-tif data/MBES/bathymetry.tif `
  --backscatter-tif data/MBES/backscatter.tif
```

## 5. Build Seed Ensemble

```powershell
.venv\Scripts\python.exe scripts/experiment_v14_seed_ensemble.py
```

## 6. Create Submission

```powershell
.venv\Scripts\python.exe -m benthic_model.cli make-submission `
  --predictions artifacts/predictions/<PREDICTIONS_CSV> `
  --sample-submission data/sample_submission.csv `
  --output data/submission_v14.csv
```

## 7. Run Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -m "not arcgis and not qgis" -v --tb=short
```

Expected: 236+ tests pass, 0 failures.

## Key Files

| File                                      | Purpose                            |
| ----------------------------------------- | ---------------------------------- |
| `configs/rf-btm-fine.yaml`                | R04 baseline config                |
| `configs/lgbm-candidate.yaml`             | LightGBM config (existing)         |
| `configs/lgbm-btm-tuned.yaml`             | LightGBM tuned config (new)        |
| `data/train_btm.csv`                      | Training data with BTM-10 features |
| `data/test.csv`                           | Competition test data              |
| `scripts/experiment_v14_seed_ensemble.py` | Seed ensemble script               |
| `docs/run-019-pipeline-cv-improvement.md` | Experiment run report              |
