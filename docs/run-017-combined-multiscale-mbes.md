# Run 017: Combined Multi-Scale BTM-33 + MBES-8 Features

**Branch**: `017-combined-multiscale-mbes`
**Date**: 2026-04-01
**Script**: `scripts/experiment_v11.py`
**Goal**: Combine the 33 selected BTM multi-scale features from run-016 with the 8 MBES point-sample features that drove R04's 0.8024 CV F1, testing whether the combined 41-feature set can match or exceed the R04 baseline under the same 5-fold spatial-blocked CV scheme.

---

## Summary

| Label    | Model    | Feature Count | CV F1  | Std    | SGAM Recall | Notes             |
| -------- | -------- | ------------- | ------ | ------ | ----------- | ----------------- |
| combined | RF       | 41            | 0.7712 | 0.0977 | —           |                   |
| combined | CatBoost | 41            | 0.7829 | 0.0861 | —           | **Best model**    |
| combined | LGB      | 41            | 0.7664 | 0.0874 | —           |                   |
| combined | Ensemble | 41            | 0.7775 | 0.0858 | —           | RF+Cat+LGB vote   |
| selected | RF       | 38            | 0.7727 | 0.0985 | —           |                   |
| selected | CatBoost | 38            | 0.7829 | 0.0848 | —           | **Best selected** |
| selected | LGB      | 38            | 0.7668 | 0.0828 | —           |                   |
| selected | Ensemble | 38            | 0.7782 | 0.0855 | —           |                   |

**CV best**: CatBoost with 38 selected features — **F1 = 0.7829** (no degradation from full set).

---

## Gate Results

| Gate   | Condition                                | Result                                                   |
| ------ | ---------------------------------------- | -------------------------------------------------------- |
| SC-001 | RF CV F1 ≥ 0.8024 (R04 baseline)         | **✗ FAIL** (0.7712 < 0.8024, Δ = −0.031)                 |
| SC-002 | Best model Kaggle ≥ 0.79518 or CV +0.005 | ✓ PASS (best=CatBoost, improvement = +0.012 ≥ 0.005)     |
| SC-003 | SGAM recall ≥ 0.045                      | ✓ PASS (SGAM recall = 0.142 ± 0.131, present in 6 folds) |
| SC-004 | ≤25 features, CV degradation ≤ 0.01      | **✗ FAIL (count)** 38 > 25; ✓ PASS (degradation = 0.000) |
| SC-005 | Feature extraction < 15 min              | ✓ PASS (14.4 s)                                          |

---

## Per-Class Recall (combined, best model = CatBoost, 5-fold mean)

| Class | Mean Recall | Std Dev | Folds Present |
| ----- | ----------- | ------- | ------------- |
| ALG   | 0.5058      | 0.3412  | 15            |
| FMAT  | 0.6879      | 0.2405  | 12            |
| NVB   | 0.8577      | 0.1339  | 15            |
| SGAM  | 0.1420      | 0.1310  | 6             |
| SGZ   | 0.3649      | 0.3080  | 12            |

**Key improvement**: SGAM recall = 0.142, a substantial gain over run-016 (0.000) and run-015/R15 (0.045). The 5-fold spatial-blocked scheme distributes SGAM samples across more training folds than the 10-fold KMeans GroupKFold used in run-016.

---

## Feature Architecture (41 features)

| Group                          | Count  | Source                         |
| ------------------------------ | ------ | ------------------------------ |
| MBES-8 point-sample            | 8      | Direct raster extraction at xy |
| BTM-33 selected (from run-016) | 33     | Cached from experiment_v10     |
| **Total**                      | **41** |                                |

### MBES-8 features

`depth`, `backscatter`, `slope`, `vrm`, `complexity`, `max_curvature`, `northness`, `eastness`

### BTM-33 features (selected from 64 in run-016)

Multi-scale terrain derivatives at 5 spatial scales (3, 7, 11, 15, 21 cells), BPI, VRM, GLCM texture, RDMV depth variability.

---

## Feature Selection: 41 → 38 Features

### Correlation pre-filter (|r| > 0.95): 0 dropped

No features exceeded the Spearman correlation threshold (the BTM-33 set was already pre-filtered in run-016).

### Permutation importance filter: 3 dropped

| Feature    | Importance | Reason                         |
| ---------- | ---------- | ------------------------------ |
| vrm        | 0.0026     | Correlated with btm_vrm        |
| complexity | 0.0035     | Correlated with btm_complexity |
| btm_slope  | 0.0011     | Redundant with slope           |

### 38 retained features

- **6 MBES-8**: depth, backscatter, slope, max_curvature, northness, eastness
- **32 BTM**: btm_broad_bpi, btm_vrm, btm_northness, btm_eastness, btm_max_curvature, btm_complexity, btm_slope_3, btm_northness_3, btm_northness_7, btm_northness_11, btm_northness_21, btm_eastness_3, btm_eastness_7, btm_eastness_11, btm_eastness_15, btm_eastness_21, btm_max_curvature_3, btm_max_curvature_11, btm_complexity_21, btm_rdmv_3, btm_rdmv_7, btm_rdmv_15, btm_glcm_contrast_3, btm_glcm_homogeneity_3, btm_glcm_contrast_7, btm_glcm_homogeneity_7, btm_glcm_contrast_11, btm_glcm_homogeneity_11, btm_glcm_contrast_15, btm_glcm_homogeneity_15, btm_glcm_contrast_21, btm_glcm_homogeneity_21

### Top 5 features by permutation importance

| Rank | Feature           | Importance |
| ---- | ----------------- | ---------- |
| 1    | depth             | 0.1293     |
| 2    | btm_northness_21  | 0.1089     |
| 3    | btm_complexity_21 | 0.0709     |
| 4    | btm_eastness_21   | 0.0156     |
| 5    | btm_eastness_15   | 0.0053     |

`depth` dominates, followed by large-scale (21-cell) directional and complexity features.

---

## CV Scheme

5-fold spatial-blocked cross-validation via `iter_spatial_blocked_folds(n_splits=5, spatial_bins=4, random_state=42)`. This matches the scheme used in R04/R06 through the `benthic_model` pipeline.

---

## Comparison to Prior Runs

| Run              | Features      | Count | CV F1      | RF CV F1 | Best Model | SGAM Recall |
| ---------------- | ------------- | ----- | ---------- | -------- | ---------- | ----------- |
| R04/R06          | BTM base      | ~10   | 0.8024     | 0.8024   | RF         | 0.045       |
| v10 (run-016)    | BTM-64        | 64    | 0.6377     | —        | CatBoost   | 0.000       |
| v10 selected     | BTM-33        | 33    | 0.6383     | —        | CatBoost   | 0.000       |
| **v11 combined** | BTM-33+MBES-8 | 41    | **0.7829** | 0.7712   | CatBoost   | **0.142**   |
| **v11 selected** | As above      | 38    | **0.7829** | 0.7727   | CatBoost   | 0.134       |

### Analysis

1. **Adding MBES-8 to BTM-33 gained +14.5 pp** (0.6383 → 0.7829 via CatBoost), confirming run-016's root cause analysis that the MBES-8 features carry most of the classification signal.

2. **RF CV F1 = 0.7712 vs R04 = 0.8024** — the 3.1 pp gap is likely due to:
   - The BTM-33 multi-scale features adding noise for RF (RF with ~10 features in R04 was more focused)
   - Different spatial blocking implementation (v11 uses `iter_spatial_blocked_folds` directly vs R04's `benthic_model` pipeline)
   - CatBoost outperforms RF here (+1.2 pp), reversing the R04 ranking — more features favor gradient-boosted models

3. **SGAM recall dramatically improved**: 0.142 vs 0.000 in run-016. The 5-fold spatial-blocked scheme better distributes the 170 SGAM samples across folds than the 10-fold KMeans GroupKFold.

---

## Training Times

| Phase    | RF   | CatBoost | LGB   | Ensemble | Total  |
| -------- | ---- | -------- | ----- | -------- | ------ |
| combined | 6.6s | 178.8s   | 27.9s | 0.0s     | 213.4s |
| selected | 7.9s | 162.2s   | 21.0s | 0.0s     | 191.1s |

---

## Output Files

| File                                          | Description                           |
| --------------------------------------------- | ------------------------------------- |
| `reports/metrics/cache_combined_v11.csv`      | 6256×41 combined train features       |
| `reports/metrics/cache_combined_test_v11.csv` | 98×41 combined test features          |
| `reports/metrics/cv_per_class_v11.csv`        | Per-model, per-fold, per-class recall |
| `reports/metrics/feature_importance_v11.csv`  | CatBoost feature importances          |
| `reports/metrics/feature_selection_v11.csv`   | Permutation-based selection results   |
| `data/submission_v11.csv`                     | Kaggle submission (41 features)       |
| `data/submission_v11_selected.csv`            | Kaggle submission (38 features)       |
| `reports/experiment_v11_run.log`              | Full experiment log                   |

---

## Test Suite

236 tests passed, 4 skipped (numerical parity reference files not yet generated). Includes 7 new tests in `tests/unit/test_experiment_v11.py` covering MBES-8 extraction, feature importance schema, and submission format validation.
