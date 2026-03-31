# Quickstart: Combined Multi-Scale BTM + MBES-8 Features

## Prerequisites

```bash
# Activate existing venv
.venv\Scripts\activate  # Windows

# Verify dependencies (all already installed from spec-016)
python -c "import catboost, lightgbm, sklearn, rasterio; print('OK')"
```

## Required Files

| File                                        | Source           | Purpose                      |
| ------------------------------------------- | ---------------- | ---------------------------- |
| `data/MBES/bathymetry.tif`                  | Survey data      | MBES-8 raster source         |
| `data/MBES/backscatter.tif`                 | Survey data      | MBES-8 raster source         |
| `data/train.csv`                            | Competition data | Training points with labels  |
| `data/test.csv`                             | Competition data | Test points for submission   |
| `reports/metrics/cache_train_feats_v10.csv` | spec-016 output  | BTM-33 feature cache         |
| `reports/metrics/cache_test_feats_v10.csv`  | spec-016 output  | BTM-33 test cache            |
| `reports/metrics/feature_selection_v10.csv` | spec-016 output  | BTM-33 selected feature list |

## 1. Run the Experiment

```bash
# Full run (~15 min: MBES-8 extraction + 3-model CV + feature selection)
python scripts/experiment_v11.py

# Dry run (5 training points, quick validation)
python scripts/experiment_v11.py --dry-run

# MBES-8 extraction timing only
python scripts/experiment_v11.py --extract-only
```

## 2. Check Results

```bash
# CV results are in the log
type reports\experiment_v11_run.log | findstr /i "CV F1"

# Per-class recall (especially SGAM)
type reports\experiment_v11_run.log | findstr /i "recall"

# Feature importance
head reports\metrics\feature_importance_v11.csv

# Submissions
dir data\submission_v11*.csv
```

## 3. Run Tests

```bash
# Unit tests for v11 helpers
pytest tests/unit/test_experiment_v11.py -v

# Full suite (should still pass)
pytest tests/ -m "not arcgis and not qgis" -v
```

## Expected Output

```
=== Experiment v11: Combined BTM-33 + MBES-8 ===
Baseline: CV=0.8024  Kaggle=0.79518
Train: 6256 rows   Test: 98 rows
MBES-8 extraction: 6256 points in ~30s
Combined feature matrix: 6256 train × 41 features  (98 test)
  [combined] rf:  CV F1=0.XXXX ± 0.XXXX     ← expect ≥ 0.8024
  [combined] cat: CV F1=0.XXXX ± 0.XXXX
  [combined] lgb: CV F1=0.XXXX ± 0.XXXX
  [combined] Best: rf (CV F1=0.XXXX)
Feature selection: 41 → N features
Submission written: data\submission_v11.csv (98 rows)
Submission written: data\submission_v11_selected.csv (98 rows)
=== Experiment v11 complete ===
```
