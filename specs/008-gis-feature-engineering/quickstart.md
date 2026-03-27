# Quickstart: GIS Feature Engineering Experiment (Branch 008)

**Script**: `scripts/experiment_v5.py`  
**Branch**: `008-gis-feature-engineering`

---

## Prerequisites

### 1. Activate the virtual environment

```powershell
# Windows (from repo root)
.venv\Scripts\Activate.ps1
```

### 2. Verify dependencies

```bash
python -c "import btm, lightgbm, xgboost, catboost, pykrige, sklearn, skimage, rasterio; print('All deps OK')"
```

If any import fails, install the missing package:

```bash
pip install pykrige scikit-image lightgbm xgboost catboost
```

### 3. Verify input data

```bash
python - <<'EOF'
from pathlib import Path
import rasterio, pandas as pd

for p in ["data/train.csv", "data/test.csv",
          "data/MBES/bathymetry.tif", "data/MBES/backscatter.tif"]:
    assert Path(p).exists(), f"Missing: {p}"
    print(f"OK  {p}")

for p in ["outputs/gis_layers/rasters/slope.tif",
          "outputs/gis_layers/rasters/backscatter_zones.tif",
          "outputs/gis_layers/rasters/acoustic_facies.tif"]:
    if Path(p).exists():
        print(f"OK  {p}")
    else:
        print(f"MISSING {p}  -> run step1_prepare_gis_layers.py first")
EOF
```

If any derived raster is missing:

```bash
python scripts/step1_prepare_gis_layers.py
```

---

## Running the Experiment

```bash
python scripts/experiment_v5.py
```

The script runs three sequential phases and prints progress to stdout:

| Phase        | Description                                                               | Est. duration |
| ------------ | ------------------------------------------------------------------------- | ------------- |
| **baseline** | v2 features only → spatial-block CV (10 folds) → `baseline_f1`            | ~5–10 min     |
| **run_a**    | v2 + raw GIS features → spatial-block CV → `run_a_f1` + submission CSV    | ~5–10 min     |
| **run_b**    | v2 + kriged GIS features → spatial-block CV → `run_b_f1` + submission CSV | ~15–25 min    |
| **report**   | Compare A vs B, copy best, write run report                               | < 1 min       |

Total estimated wall-clock: **25–45 minutes** on a modern CPU.

---

## Outputs

After the script completes, the following files are created or updated:

```
data/
├── submission_v5_gis_features.csv         # Run A predictions (ID, class)
├── submission_v5_gis_features_kriging.csv # Run B predictions (ID, class)
└── submission_v5_best.csv                 # Copy of the better-scoring run

docs/
└── run-008-gis-feature-engineering.md    # Run report with metrics comparison table
```

---

## Reading the Run Report

Open `docs/run-008-gis-feature-engineering.md`. The report contains:

1. **Metrics table** — baseline, Run A, and Run B weighted-F1 and per-class F1
2. **Top-3 features by mean GBDT gain** for Run A and Run B
3. **Best run identification** — which submission CSV to use for Kaggle
4. **Interpretation notes** — what the results mean for the feature engineering hypothesis

---

## Re-running from Scratch

To force re-derivation of the GIS rasters before re-running the experiment:

```bash
python scripts/step1_prepare_gis_layers.py   # re-derives all GIS rasters
python scripts/experiment_v5.py              # re-runs all three phases
```

---

## Troubleshooting

| Symptom                            | Likely cause                     | Fix                                                            |
| ---------------------------------- | -------------------------------- | -------------------------------------------------------------- |
| `ModuleNotFoundError: pykrige`     | pykrige not installed            | `pip install pykrige`                                          |
| `FileNotFoundError: slope.tif`     | Derived rasters missing          | `python scripts/step1_prepare_gis_layers.py`                   |
| `ConvergenceWarning` from pykrige  | Zone with sparse training points | Expected — fallback to KNN kicks in; check log for zone counts |
| Kriging phase runs > 1 hour        | Very slow variogram fitting      | Reduce `nlags` in `_krige_gis_features()` from 6 to 4          |
| Submission CSV has wrong row count | test.csv mismatch                | Verify `data/test.csv` has not been modified                   |
