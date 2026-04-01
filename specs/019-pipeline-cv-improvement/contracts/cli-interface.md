# CLI Interface Contracts: Pipeline CV Improvement

**Feature**: 019-pipeline-cv-improvement  
**Date**: 2026-04-01

## Existing CLI Commands (consumed, not modified)

### `benthic_model.cli train`

```
python -m benthic_model.cli train \
  --train-csv <path>           # Training CSV (e.g., data/train_btm.csv)
  --bathymetry-tif <path>      # Bathymetry raster (data/MBES/bathymetry.tif)
  --backscatter-tif <path>     # Backscatter raster (data/MBES/backscatter.tif)
  --config <path>              # YAML configuration file
  --run-type <baseline|candidate>
  --seed <int>                 # Random seed (default: 42)
```

**Output**: `artifacts/runs/{run_id}/` containing model.joblib, metrics.json, provenance.json, feature_columns.json, class_vocab.json

**Exit code**: 0 on success, 1 on failure

### `benthic_model.cli predict`

```
python -m benthic_model.cli predict \
  --run-id <string>            # Run ID from train step
  --test-csv <path>            # Test CSV (data/test.csv)
  --bathymetry-tif <path>
  --backscatter-tif <path>
```

**Output**: `artifacts/predictions/{run_id}_test_predictions.csv` with columns: ID, class

### `benthic_model.cli make-submission`

```
python -m benthic_model.cli make-submission \
  --predictions <path>         # Predictions CSV from predict step
  --sample-submission <path>   # data/sample_submission.csv
  --output <path>              # Output submission CSV
```

**Output**: Validated submission CSV with 98 rows, columns: ID, class

## YAML Configuration Contract

```yaml
# Required fields
model_type: rf | lgbm # Model type selector
seed: 42 # Random seed

# Cross-validation settings
cv:
  n_splits: 5 # Number of folds
  fold_scheme: spatial_blocked # MUST be spatial_blocked
  random_state: 42 # Fold assignment seed
  spatial_bins: 4 # Spatial grid resolution

# Feature engineering flags
feature_flags:
  include_focal_stats: false
  include_interactions: false
  include_spatial_z_scores: false
  include_btm_features: true
  include_eco_features: false
  exclude_coords: false

# Optional: model hyperparameter overrides (RF only)
model_params:
  n_estimators: 300
  max_depth: null
  min_samples_leaf: 2
  max_features: sqrt
  class_weight: balanced
```

## Metrics JSON Contract

```json
{
  "run_id": "candidate-20260401HHMMSS",
  "run_type": "candidate",
  "weighted_f1": 0.8024,
  "per_class_f1": {
    "ALG": 0.85,
    "FMAT": 0.78,
    "NVB": 0.92,
    "SGAM": 0.27,
    "SGZ": 0.65
  },
  "fold_scheme": "spatial_blocked"
}
```
