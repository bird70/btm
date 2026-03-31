# Data Model: Combined Multi-Scale BTM + MBES-8 Features

## Entities

### CombinedFeatureSet

The merged feature matrix used for model training and prediction.

| Field         | Type             | Description                                                   |
| ------------- | ---------------- | ------------------------------------------------------------- |
| point_id      | int              | Unique sample point identifier (from `data/train.csv` ID col) |
| mbes_features | dict[str, float] | 8 MBES point-sample features sampled from rasters             |
| btm_features  | dict[str, float] | 33 selected multi-scale BTM features from spec-016 cache      |
| combined_df   | pd.DataFrame     | Merged DataFrame with ~41 columns per sample point            |

### MBES-8 Features

The 8 raster-sampled features from bathymetry and backscatter GeoTIFFs.

| Column        | Algorithm                       | Source Raster   | Reference                |
| ------------- | ------------------------------- | --------------- | ------------------------ |
| depth         | Direct pixel value              | bathymetry.tif  | —                        |
| backscatter   | Direct pixel value              | backscatter.tif | —                        |
| slope         | Horn 3×3 gradient kernel        | bathymetry.tif  | Horn (1981)              |
| vrm           | Vector Ruggedness Measure       | bathymetry.tif  | Sappington et al. (2007) |
| complexity    | 1/cos(focal_mean(slope_rad))    | bathymetry.tif  | Wilson et al. (2007)     |
| max_curvature | Laplacian / cell_size²          | bathymetry.tif  | Evans (1980)             |
| northness     | cos(aspect) from Horn gradients | bathymetry.tif  | Horn (1981)              |
| eastness      | sin(aspect) from Horn gradients | bathymetry.tif  | Horn (1981)              |

### BTM-33 Selected Features

The 33 features retained by spec-016 permutation importance selection.

| Group                     | Count | Features                                                                                              |
| ------------------------- | ----- | ----------------------------------------------------------------------------------------------------- |
| Base BTM                  | 7     | broad_bpi, slope, vrm, northness, eastness, max_curvature, complexity                                 |
| Multi-scale northness     | 4     | northness_3, northness_7, northness_11, northness_21                                                  |
| Multi-scale eastness      | 5     | eastness_3, eastness_7, eastness_11, eastness_15, eastness_21                                         |
| Multi-scale max_curvature | 2     | max_curvature_3, max_curvature_11                                                                     |
| Multi-scale complexity    | 1     | complexity_21                                                                                         |
| Multi-scale slope         | 1     | slope_3                                                                                               |
| RDMV                      | 3     | rdmv_3, rdmv_7, rdmv_15                                                                               |
| GLCM contrast             | 5     | glcm_contrast_3, glcm_contrast_7, glcm_contrast_11, glcm_contrast_15, glcm_contrast_21                |
| GLCM homogeneity          | 5     | glcm_homogeneity_3, glcm_homogeneity_7, glcm_homogeneity_11, glcm_homogeneity_15, glcm_homogeneity_21 |

All prefixed with `btm_` in the DataFrame.

### ModelResult

Per-model cross-validation output.

| Field            | Type             | Description                                            |
| ---------------- | ---------------- | ------------------------------------------------------ |
| model_name       | str              | "rf", "cat", or "lgb"                                  |
| cv_f1_mean       | float            | Mean weighted-F1 across 5 folds                        |
| cv_f1_std        | float            | Std of weighted-F1 across 5 folds                      |
| per_class_recall | dict[str, float] | Mean recall per class (ALG, FMAT, NVB, SGAM, SGZ)      |
| oof_proba        | np.ndarray       | Out-of-fold probability matrix (n_samples × n_classes) |

### FeatureSelectionResult

Reuses `FeatureSelectionResult` from `src/benthic_model/features/selection.py` (spec-016).

| Field           | Type  | Description                          |
| --------------- | ----- | ------------------------------------ |
| feature_name    | str   | Column name in the DataFrame         |
| importance_mean | float | Mean permutation importance score    |
| importance_std  | float | Standard deviation across repeats    |
| selected        | bool  | Whether the feature was retained     |
| rank            | int   | Importance rank (1 = most important) |

## Relationships

```
Bathymetry Raster ──→ MBES-8 Features (depth, slope, VRM, complexity, max_curvature, northness, eastness)
Backscatter Raster ──→ MBES-8 Features (backscatter)
Spec-016 CSV Cache ──→ BTM-33 Selected Features
                          │
                          ▼
                   CombinedFeatureSet (~41 columns)
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           RF Model   CatBoost    LightGBM
              │           │           │
              └───────────┼───────────┘
                          ▼
               5-fold spatial-blocked CV
                          │
                          ▼
              Best Model → Feature Selection
                          │
                          ▼
              Selected Model → Submission CSV
```

## State Transitions

```
CSV cache + rasters → Extract MBES-8 → Merge with BTM-33 → Combined ~41 features
  → 5-fold spatial-blocked CV (RF, CatBoost, LGB) → Report per-model CV F1 + per-class recall
  → Feature importance → Permutation selection → Reduced set ≤ 25 features
  → Retrain on selected set → CV comparison → Submission CSV
```

## Expected Feature Overlap

The following MBES-8 and BTM features may be highly correlated:

| MBES-8        | BTM equivalent    | Expected Spearman | r                   |     | Resolution |
| ------------- | ----------------- | ----------------- | ------------------- | --- | ---------- |
| northness     | btm_northness     | > 0.95            | Corr filter drops 1 |
| eastness      | btm_eastness      | > 0.95            | Corr filter drops 1 |
| max_curvature | btm_max_curvature | > 0.95            | Corr filter drops 1 |
| complexity    | btm_complexity    | > 0.95            | Corr filter drops 1 |
| slope         | btm_slope         | > 0.95            | Corr filter drops 1 |

After correlation pre-filter, the effective combined set may be ~36 features (41 − 5 overlaps). Feature selection further reduces to ≤ 25.
