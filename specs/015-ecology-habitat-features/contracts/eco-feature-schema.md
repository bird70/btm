# Contract: Eco-Feature Column Schema

**Branch**: `015-ecology-habitat-features` | **Date**: 2026-03-31  
**Scope**: Internal ML pipeline — defines the contract between the raster extraction step and the training/prediction steps.

---

## Raster-Extracted Columns (produced by `btm-export-features --include-eco-features`)

These columns are added to the CSV by sampling the MBES bathymetry raster. They travel through to the model trainer alongside all existing `btm_*` and `bathymetry`/`backscatter` columns.

| Column              | dtype   | Range    | Units           | Source Algorithm                          | Reference                  |
| ------------------- | ------- | -------- | --------------- | ----------------------------------------- | -------------------------- | --- | ------- | --------------------------- | ------------------- |
| `btm_northness`     | float64 | [−1, 1]  | dimensionless   | Horn (1981) gradient → sin(atan2(dy, dx)) | Wilson et al. 2007 Table 1 |
| `btm_eastness`      | float64 | [−1, 1]  | dimensionless   | Horn (1981) gradient → cos(atan2(dy, dx)) | Wilson et al. 2007 Table 1 |
| `btm_max_curvature` | float64 | (−∞, +∞) | m⁻¹ (curvature) | max(                                      | plan                       | ,   | profile | ) from 2nd-order polynomial | Schmidt et al. 2003 |
| `btm_complexity`    | float64 | [0, +∞)  | °/cell          | slope of slope (second derivative of DEM) | Wilson et al. 2007 Table 1 |

**NaN handling**: All four columns are filled with `0.0` where raster sampling produces NaN (edge pixels, nodata, or out-of-bounds points). NaN rate is logged as a warning if > 10%.

---

## Training-Time Columns (produced by `EcoFeatureTransformer.fit_transform()`)

These columns are computed inside the ML pipeline from the already-extracted feature values. They require prior `fit()` to establish thresholds.

| Column           | dtype | Values       | Semantics                                                                                                                          |
| ---------------- | ----- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `btm_depth_zone` | int64 | {1, 2, 3, 4} | Ordinal depth bin. 1 = very shallow, 4 = deepest. Boundaries at p25/p50/p75 of training `bathymetry`.                              |
| `btm_sgam_niche` | int64 | {0, 1}       | 1 if: `btm_fine_bpi < p25(btm_fine_bpi)` AND `btm_slope < p25(btm_slope)` AND `bathymetry` in SGAM depth range (training min/max). |

---

## Threshold Artifact Schema (`eco_thresholds.json`)

Written to `artifacts/runs/<run_id>/eco_thresholds.json` whenever `include_eco_features: true`. Must be present for predict-time `EcoFeatureTransformer.from_dict()` to reconstruct the fitted state.

```json
{
  "depth_bin_edges": [-0.2, -12.1, -21.4, -35.7, -52.0],
  "p25_bpi": -8.3,
  "p25_slope": 1.4,
  "sgam_depth_min": -18.5,
  "sgam_depth_max": -3.2,
  "sgam_n_points": 42,
  "fit_timestamp": "2026-03-31T14:22:00"
}
```

| Key               | Type               | Constraint                                                                       |
| ----------------- | ------------------ | -------------------------------------------------------------------------------- |
| `depth_bin_edges` | list[float], len=5 | Strictly increasing (ascending depth values, typically negative for ocean depth) |
| `p25_bpi`         | float              | Any finite float                                                                 |
| `p25_slope`       | float              | ≥ 0.0 (slope is non-negative)                                                    |
| `sgam_depth_min`  | float              | ≤ `sgam_depth_max`                                                               |
| `sgam_depth_max`  | float              | ≥ `sgam_depth_min`                                                               |
| `sgam_n_points`   | int                | ≥ 0 (0 or < 5 triggers depth-condition skip + warning)                           |
| `fit_timestamp`   | str                | ISO 8601 format                                                                  |

---

## Config YAML Schema Extension

The four new experiment configs extend the existing YAML contract with the addition of:

1. `feature_flags.include_eco_features: true` — enables EcoFeatureTransformer
2. `model_params` (optional) — forwarded as `**kwargs` to the model constructor

```yaml
# Minimal eco-feature config example
model_type: rf
seed: 42
cv:
  n_splits: 5
  fold_scheme: spatial_blocked
  random_state: 42
  spatial_bins: 4
feature_flags:
  include_btm_features: true
  include_eco_features: true
  include_focal_stats: false
  include_interactions: false
  include_spatial_z_scores: false

# Optional: override RF hyperparameters
model_params:
  n_estimators: 500
  max_features: sqrt
  min_samples_leaf: 1
```

**Backward compatibility**: Configs without `include_eco_features` or `model_params` continue to work unchanged (defaults to `False` / `None`).

---

## Training CSV Column Requirements

Configs with `include_eco_features: true` require the training CSV to contain the following columns (in addition to existing requirements):

| Required column     | Used for                                          |
| ------------------- | ------------------------------------------------- |
| `bathymetry`        | Depth zone bins + SGAM depth range                |
| `btm_fine_bpi`      | SGAM niche indicator p25 threshold                |
| `btm_slope`         | SGAM niche indicator p25 threshold                |
| `btm_northness`     | Raster-extracted (must be in CSV before training) |
| `btm_eastness`      | Raster-extracted (must be in CSV before training) |
| `btm_max_curvature` | Raster-extracted (must be in CSV before training) |
| `btm_complexity`    | Raster-extracted (must be in CSV before training) |

If any of `btm_northness`, `btm_eastness`, `btm_max_curvature`, `btm_complexity` are absent from the training CSV and `include_eco_features: true`, the pipeline logs a warning and skips those columns (graceful degradation). The training-time features (`btm_depth_zone`, `btm_sgam_niche`) can always be computed as long as `bathymetry`, `btm_fine_bpi`, and `btm_slope` are present.
