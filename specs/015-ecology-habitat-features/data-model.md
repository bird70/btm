# Data Model: Ecology-Informed Habitat Feature Engineering

**Branch**: `015-ecology-habitat-features` | **Date**: 2026-03-31

---

## Entities

### 1. `EcoFeatureTransformer`

**Location**: `src/benthic_model/features/eco_features.py`  
**Responsibility**: In-pipeline transformer that adds training-time eco-features (depth zone + SGAM niche indicator) to a feature DataFrame. Does **not** handle raster derivatives (those arrive pre-extracted in the CSV).

| Attribute         | Type                     | Description                                               |
| ----------------- | ------------------------ | --------------------------------------------------------- |
| `depth_bin_edges` | `np.ndarray`, shape (5,) | Quantile boundaries for 4 depth bins; set at `fit()` time |
| `p25_bpi`         | `float`                  | 25th percentile of `btm_fine_bpi` in training set         |
| `p25_slope`       | `float`                  | 25th percentile of `btm_slope` in training set            |
| `sgam_depth_min`  | `float`                  | Minimum bathymetry value of SGAM-labelled training points |
| `sgam_depth_max`  | `float`                  | Maximum bathymetry value of SGAM-labelled training points |
| `is_fitted`       | `bool`                   | True after `fit()` has been called                        |

**Methods**:

| Method                 | Signature                                       | Description                                                                                          |
| ---------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `fit(df, y=None)`      | `(pd.DataFrame, Iterable[str]                   | None) -> self`                                                                                       | Compute all thresholds from training data. `y` is the class labels (needed for SGAM depth range). |
| `transform(df)`        | `(pd.DataFrame) -> pd.DataFrame`                | Add `btm_depth_zone` and `btm_sgam_niche` columns. Raises `NotFittedError` if called before `fit()`. |
| `fit_transform(df, y)` | `(pd.DataFrame, Iterable[str]) -> pd.DataFrame` | Convenience: `fit(df, y)` then `transform(df)`.                                                      |
| `to_dict()`            | `() -> dict`                                    | Serialise thresholds to a plain dict for JSON artifact storage.                                      |
| `from_dict(d)`         | `classmethod (dict) -> EcoFeatureTransformer`   | Reconstruct from artifact dict (used at predict time).                                               |

**State transitions**:

```
Unfitted  ──fit()──▶  Fitted  ──transform()──▶  (returns enriched DataFrame)
                              ──to_dict()──▶    (returns threshold dict for artifact)
Unfitted  ──from_dict()──▶  Fitted (from artifact)
```

---

### 2. `EcoThresholds` (data container, no class required — plain dict stored in artifact)

**Location**: `artifacts/runs/<run_id>/eco_thresholds.json`  
**Responsibility**: Persist the fit-time thresholds so `transform()` at predict time uses identical values as training.

| Key               | Type                    | Example                              |
| ----------------- | ----------------------- | ------------------------------------ |
| `depth_bin_edges` | `list[float]`, length 5 | `[-0.2, -12.1, -21.4, -35.7, -52.0]` |
| `p25_bpi`         | `float`                 | `-8.3`                               |
| `p25_slope`       | `float`                 | `1.4`                                |
| `sgam_depth_min`  | `float`                 | `-18.5`                              |
| `sgam_depth_max`  | `float`                 | `-3.2`                               |
| `sgam_n_points`   | `int`                   | `42`                                 |
| `fit_timestamp`   | `str`                   | `"2026-03-31T14:22:00"`              |

---

### 3. `FeatureFlags` (extended)

**Location**: `src/benthic_model/config.py`  
**Responsibility**: Boolean toggles for each feature group; extended with one new field.

| Field                      | Type   | Default     | Description                                               |
| -------------------------- | ------ | ----------- | --------------------------------------------------------- |
| `include_focal_stats`      | `bool` | `True`      | Focal statistics (std, mean at window radii)              |
| `include_interactions`     | `bool` | `True`      | Pairwise interaction terms (bathymetry × backscatter)     |
| `include_spatial_z_scores` | `bool` | `True`      | Z-score normalisation of bathymetry and backscatter       |
| `include_btm_features`     | `bool` | `True`      | BTM library derivatives (BPI, VRM, slope, etc.)           |
| `include_eco_features`     | `bool` | **`False`** | **NEW**: Eco-features (depth zone + SGAM niche indicator) |

The default of `False` ensures backward compatibility: existing configs without `include_eco_features` behave identically to current behaviour.

---

### 4. `PipelineConfig` (extended)

**Location**: `src/benthic_model/config.py`  
**Responsibility**: Top-level config for a training run; extended with optional model_params.

| Field             | Type                    | Default         | Description                        |
| ----------------- | ----------------------- | --------------- | ---------------------------------- | ---------------------------------------------------- |
| `data_dir`        | `str`                   | `"data"`        | Root directory for input CSVs      |
| `artifacts_dir`   | `str`                   | `"artifacts"`   | Root directory for run artifacts   |
| `reports_dir`     | `str`                   | `"reports"`     | Root directory for reports         |
| `submissions_dir` | `str`                   | `"submissions"` | Root directory for submission CSVs |
| `seed`            | `int`                   | `42`            | Global random seed                 |
| `cv`              | `CrossValidationConfig` | default         | CV fold settings                   |
| `model_type`      | `str                    | None`           | `None`                             | One of: rf, xgb, lgbm, catboost, rf_lgbm_ensemble    |
| `feature_flags`   | `FeatureFlags           | None`           | `None`                             | Feature group toggles                                |
| `spatial_coords`  | `bool`                  | `False`         | Include x,y columns as features    |
| `model_params`    | `dict                   | None`           | **`None`**                         | **NEW**: Extra kwargs forwarded to model constructor |

**Validation**: `model_params` is accepted as a plain dict and not type-checked against the model constructor; a passing test run validates it implicitly.

---

### 5. Raster Derivative Functions (in `btm/features/extract.py`)

**Responsibility**: Pure-Python raster-level computation of the four spatial derivatives from a loaded NumPy elevation array.

| Function                                         | Signature                                              | Outputs                                                                                              |
| ------------------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `compute_northness_eastness(dem, cell_size)`     | `(np.ndarray, float) -> tuple[np.ndarray, np.ndarray]` | `(northness, eastness)` arrays, same shape as DEM                                                    |
| `compute_max_curvature(dem, cell_size)`          | `(np.ndarray, float) -> np.ndarray`                    | `max_curvature` array                                                                                |
| `compute_complexity(dem, cell_size)`             | `(np.ndarray, float) -> np.ndarray`                    | `complexity` array                                                                                   |
| `extract_eco_raster_features(points, bathy_tif)` | `(pd.DataFrame, Path) -> pd.DataFrame`                 | DataFrame with 4 new columns: `btm_northness`, `btm_eastness`, `btm_max_curvature`, `btm_complexity` |

`extract_eco_raster_features` is called from within `extract_btm_features()` when `include_eco_features=True`.

---

## Data Flow Diagram

```
INPUTS
──────
data/MBES/bathymetry.tif       data/train.csv (or train_btm.csv)
         │                              │
         ▼                              │
btm-export-features                     │
  --include-eco-features                │
  [extract_btm_features()               │
   + extract_eco_raster_features()]     │
         │                              │
         ▼                              │
data/train_btm_eco.csv ◄────────────────┘
  Added columns:
    btm_northness, btm_eastness,
    btm_max_curvature, btm_complexity
  (plus all existing btm_* columns)
         │
         ▼
benthic-model train --config configs/rf-btm-eco-*.yaml
  [engineer_features(df, flags)]
     └── if flags.include_eco_features:
           EcoFeatureTransformer.fit_transform(df, y)
             → adds btm_depth_zone (int 1-4)
             → adds btm_sgam_niche (int 0/1)
         │
         ▼
artifacts/runs/<run_id>/
  model.joblib
  metrics.json          ← per-class F1 for all 5 classes
  eco_thresholds.json   ← threshold values for reproducibility
  feature_columns.json
  provenance.json
         │
         ▼
benthic-model predict --run-id <run_id>
  [EcoFeatureTransformer.from_dict(eco_thresholds)]
  → produces data/submission_<run_id>.csv
         │
         ▼
kaggle competitions submit -f data/submission_<run_id>.csv -m "<run_id>"
```

---

## Validation Rules

| Rule                             | Constraint                                                     | Fail Action                                  |
| -------------------------------- | -------------------------------------------------------------- | -------------------------------------------- |
| `btm_northness`, `btm_eastness`  | Values in [−1.001, 1.001] (float precision)                    | Log warning + clamp to [−1, 1]               |
| `btm_depth_zone`                 | Integer in {1, 2, 3, 4} after transform                        | Assert; raise if violated                    |
| `btm_sgam_niche`                 | Integer in {0, 1}                                              | Assert; raise if violated                    |
| NaN rate per eco-column          | ≤ 10% before fillna                                            | Log warning if exceeded                      |
| SGAM niche — minimum SGAM points | ≥ 5 SGAM training points for depth range to be valid           | If < 5, skip depth condition; log warning    |
| `eco_thresholds.json` written    | Present in every run artifact when `include_eco_features=True` | Train should raise if file cannot be written |
