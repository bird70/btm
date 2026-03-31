# Quickstart: Multi-Scale Terrain Features

## Prerequisites

```bash
# Activate existing venv
.venv\Scripts\activate  # Windows

# Install with all extras
pip install -e ".[dev,ml,benthic]"
```

## 1. Extract Multi-Scale Features

```python
from btm.features.extract import extract_btm_features
import pandas as pd

# Load sample points
train = pd.read_csv("data/train.csv")
sample_points = train[["ID", "x", "y"]].copy()

# Extract BTM features with multi-scale enabled
btm_feats = extract_btm_features(
    sample_points,
    bathymetry_tif="data/MBES/bathymetry.tif",
    include_interactions=True,
    include_eco_features=True,
    scales=[3, 7, 11, 15, 21],          # Multi-scale window sizes
    include_glcm=True,                   # GLCM texture from backscatter
    backscatter_tif="data/MBES/backscatter.tif",
)

# Inspect multi-scale columns
ms_cols = [c for c in btm_feats.columns if any(f"_{s}" in c for s in ["_3", "_7", "_11", "_15", "_21"])]
print(f"Multi-scale features: {len(ms_cols)}")
print(btm_feats[ms_cols].describe())
```

## 2. Run Experiment

```bash
python scripts/experiment_v10.py
```

This runs the full pipeline: multi-scale extraction → model training → CV evaluation → feature importance → feature selection → retrain → submission CSV.

## 3. Run Tests

```bash
# Unit tests for new modules
pytest tests/unit/test_multiscale.py tests/unit/test_glcm.py tests/unit/test_feature_selection.py -v

# Integration test
pytest tests/integration/test_multiscale_pipeline.py -v

# Full suite (should still pass)
pytest tests/ -m "not arcgis and not qgis" -v
```
