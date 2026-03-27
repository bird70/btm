# Quickstart: OBIA + Pixel-Based Hybrid Classification

**Branch**: `009-obia-pixel-hybrid`

## Prerequisites

```bash
# Python 3.11+ required; activate virtual environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies (already installed on branch)
pip install scikit-image scipy lightgbm xgboost catboost scikit-learn rasterio pandas numpy
```

## Run the experiment

From the repository root:

```bash
python scripts/experiment_v6.py
```

Expected runtime: approximately 3–8 minutes (raster derivation + segmentation dominate).

## Expected output

```
Loading rasters...  bathymetry & backscatter (4040 x 4743, cell=0.25 m)
Computing pixel-based features...  slope, vrm, complexity, curvature, northness, eastness
Segmenting rasters (SLIC, n_segments=4000)...  mean segment area: ~XXX px
Computing segment statistics...  bathy/back/vrm mean/std/skew
Extracting features for train (566) and test (98) points...

=== Spatial Block CV (10 blocks) ===
  lgbm...    lgbm spatial CV weighted-F1 = X.XXXX
  xgb...     xgb  spatial CV weighted-F1 = X.XXXX
  catboost...catboost spatial CV weighted-F1 = X.XXXX
  rf...      rf   spatial CV weighted-F1 = X.XXXX

  ENSEMBLE spatial CV weighted-F1 = X.XXXX

=== Training final models on all data ===
  ...

Submission written to data/submission_v6_best.csv   (98 rows)
Run report written to docs/run-009-obia-pixel-hybrid.md
```

## Output files

| File | Description |
|------|-------------|
| `data/submission_v6_best.csv` | Kaggle submission (98 rows, `ID` + `class` columns) |
| `docs/run-009-obia-pixel-hybrid.md` | Full run report with CV metrics and feature importance |

## Run unit tests only

```bash
pytest tests/unit/test_experiment_v6_features.py -v
```

## Run all tests

```bash
pytest tests/ -v
```
