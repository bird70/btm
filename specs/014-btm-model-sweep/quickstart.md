# Quickstart: Systematic BTM Model Sweep

**Branch**: `014-btm-model-sweep` | **Date**: 2026-03-30  
**Competition**: `geohab-mlwg-competition-2026`

---

## Prerequisites

All commands run from `C:\DEVlocal\BTM-Hybrid\btm` with the project venv active.

```powershell
cd C:\DEVlocal\BTM-Hybrid\btm
.\.venv\Scripts\Activate.ps1
```

### Kaggle CLI auth (one-time)

The Kaggle CLI is in a sibling venv. Auth via API token set as an environment
variable (Kaggle CLI v2):

```powershell
# Set for current session (add to your profile for persistence):
$env:KAGGLE_API_TOKEN = "<your-token-from-kaggle.com-account-settings>"

# Verify:
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions list
# Expected: geohab-mlwg-competition-2026 appears
```

To get your token: Kaggle → Account Settings → API → **Create New API Token**.
Copy the token string value from the downloaded JSON file.

---

## Step 0 — Environment check

```powershell
$env:PYTHONPATH = "src"
python -c "import btm; import benthic_model; print('OK')"
python -c "import pandas as pd; df=pd.read_csv('data/train.csv'); print(len(df), 'training points')"
python -c "import rasterio; r=rasterio.open('data/MBES/bathymetry.tif'); print('CRS:', r.crs)"
```

Expected: `OK`, ≥ 2 000 points, a valid projected CRS.

---

## Day 1 — Phase 1: Baseline verification (3 submissions)

Train all three Phase 1 configs, produce predictions, submit.

```powershell
$env:PYTHONPATH = "src"

# ── R01: Baseline RF (existing config, reproduces 0.8026 CV) ──────────────────
python -m benthic_model.cli train `
  --train-csv data/train.csv `
  --bathymetry-tif data/MBES/bathymetry.tif `
  --backscatter-tif data/MBES/backscatter.tif `
  --config configs/baseline.yaml --run-type baseline --seed 42
# Note the run_id printed: e.g. baseline-20260330HHMMSS

python -m benthic_model.cli evaluate `
  --run-id <R01_RUN_ID> --fold-scheme spatial_blocked

python -m benthic_model.cli predict `
  --run-id <R01_RUN_ID> --test-csv data/test.csv `
  --bathymetry-tif data/MBES/bathymetry.tif `
  --backscatter-tif data/MBES/backscatter.tif

# ── R02: RF core-only (no engineered features) ────────────────────────────────
python -m benthic_model.cli train `
  --train-csv data/train.csv `
  --bathymetry-tif data/MBES/bathymetry.tif `
  --backscatter-tif data/MBES/backscatter.tif `
  --config configs/rf-core-only.yaml --run-type baseline --seed 42

# (evaluate + predict same pattern as R01)

# ── R03: RF + focal stats, no interactions ────────────────────────────────────
python -m benthic_model.cli train `
  --train-csv data/train.csv `
  --bathymetry-tif data/MBES/bathymetry.tif `
  --backscatter-tif data/MBES/backscatter.tif `
  --config configs/rf-no-interactions.yaml --run-type baseline --seed 42
```

### Submit Phase 1 to Kaggle

```powershell
# R01 — always submit (anchor)
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit `
  -c geohab-mlwg-competition-2026 `
  -f data\submission_<R01_RUN_ID>.csv `
  -m "R01 baseline RF: CV=<F1>"

Start-Sleep -Seconds 90   # wait for scoring

C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submissions `
  -c geohab-mlwg-competition-2026 --csv | Select-Object -First 2

# Repeat for R02, R03 (same pattern)
# Append scores to artifacts/experiments/kaggle_scores.csv
```

---

## Day 1 (after Phase 1 trains) — Extract BTM features for Phase 2

```powershell
# Training points
btm-export-features `
  --bathy data/MBES/bathymetry.tif `
  --points data/train.csv `
  --broad-inner 10 --broad-outer 30 `
  --fine-inner 1 --fine-outer 5 `
  --outdir data/btm_rasters --output data/train_btm.csv

# Test points
btm-export-features `
  --bathy data/MBES/bathymetry.tif `
  --points data/test.csv `
  --broad-inner 10 --broad-outer 30 `
  --fine-inner 1 --fine-outer 5 `
  --output data/test_btm.csv
```

---

## Day 2 — Phase 2 training (all 5 configs) + top-3 submissions

```powershell
$env:PYTHONPATH = "src"

# Train R04–R08 (all using train_btm.csv except R07 which uses train.csv)
# R07 — texture only (no BTM columns)
python -m benthic_model.cli train `
  --train-csv data/train.csv `
  --config configs/rf-texture.yaml --run-type baseline --seed 42 [+ raster args]

# R04, R05, R06, R08 — BTM-augmented CSV
python -m benthic_model.cli train `
  --train-csv data/train_btm.csv `
  --config configs/rf-btm-full.yaml --run-type baseline --seed 42 [+ raster args]

# After all train + evaluate: submit top 3 by CV score (Day 2 budget: 4 max)
# Priority order: R07 (texture, smallest delta), R06 (full BTM), R05 (broad BTM)
```

---

## Day 3 — Remaining Phase 2 + diagnostic R13

```powershell
# Submit R04, R08, R13 (fine BTM, max features, spatial diagnostic)
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit `
  -c geohab-mlwg-competition-2026 `
  -f data\submission_<R13_RUN_ID>.csv `
  -m "R13 DIAGNOSTIC spatial-coords RF: CV=<F1> — expected Kaggle << CV"
```

**Phase 2 gate**: After all submissions scored, select the run with the
smallest CV–Kaggle gap. Copy its `feature_flags` into `configs/lgbm-candidate.yaml`,
`configs/xgb-candidate.yaml`, `configs/catboost-candidate.yaml`, and
`configs/rf-lgbm-ensemble.yaml`.

---

## Day 4 — Phase 3: Model family sweep (4 submissions)

```powershell
$env:PYTHONPATH = "src"

# Train R09–R12 using Phase 2 winning feature set and corresponding CSV
python -m benthic_model.cli train `
  --train-csv data/<winner_train_csv> `
  --config configs/catboost-candidate.yaml --run-type candidate --seed 42 [+ raster args]

# Submit in order: R11 (CatBoost), R09 (LightGBM), R10 (XGBoost), R12 (ensemble)
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit `
  -c geohab-mlwg-competition-2026 `
  -f data\submission_<R11_RUN_ID>.csv `
  -m "R11 CatBoost Phase2-winner features: CV=<F1>"
```

---

## After all submissions — Write run report

```powershell
# Review all registry entries
Get-Content artifacts\experiments\run_registry.jsonl | ConvertFrom-Json

# Review Kaggle scores
Import-Csv artifacts\experiments\kaggle_scores.csv | Format-Table

# Write docs/run-013-btm-model-sweep.md
```

---

## Referencing the run registry

```powershell
# List all runs sorted by CV F1
Get-Content artifacts\experiments\run_registry.jsonl |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Sort-Object metric_weighted_f1 -Descending |
  Select-Object run_id, run_type, metric_weighted_f1 |
  Format-Table

# Join with Kaggle scores for gap analysis
$registry = Get-Content artifacts\experiments\run_registry.jsonl |
  ForEach-Object { $_ | ConvertFrom-Json }
$scores = Import-Csv artifacts\experiments\kaggle_scores.csv
# Cross-reference on run_id
```
