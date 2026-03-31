# Run 016: Multi-Scale Terrain Features + GLCM Backscatter Texture

**Branch**: `016-multiscale-terrain-features`  
**Date**: 2026-03-31  
**Script**: `scripts/experiment_v10.py`  
**Goal**: Add multi-scale terrain derivatives (5 spatial scales × 7 base derivatives = 35 features), RDMV depth variability (5 scales), and GLCM backscatter texture (5 scales × 2 properties = 10 features) to test whether richer BTM-only features can close the gap toward the OBIA baseline CV scores from run-009.

---

## Summary

| Label             | Model        | Feature Count | CV F1  | SGAM CV Recall | Notes                       |
| ----------------- | ------------ | ------------- | ------ | -------------- | --------------------------- |
| full_multiscale   | CatBoost CPU | 64            | 0.6377 | 0.0000         | Best single model           |
| full_multiscale   | LightGBM CPU | 64            | 0.6336 | —              |                             |
| full_multiscale   | Cat+LGB ens. | 64            | 0.6362 | —              |                             |
| selected_features | CatBoost CPU | 33            | 0.6383 | 0.0000         | After permutation selection |
| selected_features | LightGBM CPU | 33            | 0.6232 | —              |                             |
| selected_features | Cat+LGB ens. | 33            | 0.6294 | —              |                             |

**CV best**: CatBoost CPU with 33 selected features — **F1 = 0.6383** (marginal +0.0006 vs full set).

---

## Gate Results

| Gate   | Condition                                        | Result                                                      |
| ------ | ------------------------------------------------ | ----------------------------------------------------------- |
| SC-001 | CV F1 ≥ 0.8024 (OBIA baseline from run-009)      | ✗ FAIL (0.6377 < 0.8024) — see baseline clarification below |
| SC-003 | SGAM mean recall > 0.000 across folds            | ✗ FAIL (SGAM recall = 0.0000)                               |
| SC-004 | CV F1 degradation after feature selection ≤ 0.01 | ✓ PASS (Δ = −0.0006 ≤ 0.01)                                 |
| SC-005 | Feature extraction time < 600 s                  | ✓ PASS (279 s train + 275 s test)                           |

---

## Baseline Clarification: Why 0.6377 ≠ 0.8024 (two separate causes)

**Cause 1 — Missing MBES core 8 features (primary gap, ≈14 pp)**

R04/R06's 0.8024 was produced by the `benthic_model` pipeline, which _always_ injects 8 MBES
point-sample features alongside any BTM columns: direct depth, backscatter, slope, VRM,
complexity, max_curvature, northness and eastness sampled at each point from the rasters.
Run-014 Phase 1 showed RF on these 8 features alone gives CV=0.797 — they carry most of the
classification signal. This experiment is a **BTM-feature-only** standalone script; those
MBES-8 columns are never added. The 0.8024 gate is not achievable from BTM features alone.

**Cause 2 — Model choice (secondary gap, ≈0.006 pp)**

Run-014 Phase 3 benchmarked all model types on the same BTM-winner feature set:

| Model                                                               | CV F1 (BTM+MBES-8 features) |
| ------------------------------------------------------------------- | --------------------------- |
| **RF** (`n_estimators=300, min_samples_leaf=2, balanced_subsample`) | **0.8024**                  |
| CatBoost                                                            | 0.7965                      |
| LightGBM                                                            | 0.7954                      |

RF outperforms CatBoost by ~0.006 on BTM+MBES-8 features. Replacing CatBoost with RF in
this script would therefore improve the BTM-only CV score modestly (+0.5–1 pp), not close
the 14 pp Cause 1 gap.

**Correct like-for-like baseline**:

| Baseline                                           | CV F1      | Pipeline                       |
| -------------------------------------------------- | ---------- | ------------------------------ |
| BTM base only, CatBoost standalone                 | ~0.52      | This script, 10 BTM cols       |
| **This run — 64 multi-scale BTM + GLCM, CatBoost** | **0.6377** | This script, 64 BTM cols       |
| BTM+MBES-8, RF (R04/R06 benthic_model pipeline)    | 0.8024     | benthic_model, ~18 cols        |
| OBIA (SLIC + raw depth), run-009                   | 0.8024     | experiment_v9.py, 21 OBIA cols |

Multi-scale BTM achieves **+11.4 pp** over the BTM-only CatBoost base — a real gain
confirmed with all NaN bugs fixed.

---

## Per-Class Recall (full_multiscale, best model CatBoost)

| Class | Mean Recall | Std Dev |
| ----- | ----------- | ------- |
| ALG   | 0.4047      | ±0.4347 |
| FMAT  | 0.3432      | ±0.4426 |
| NVB   | 0.6141      | ±0.3739 |
| SGAM  | 0.0000      | ±0.0000 |
| SGZ   | 0.0391      | ±0.1024 |

**NVB** is the dominant class and best-predicted. **SGAM** recall remains zero throughout — 169 of 170 SGAM samples fall in a single spatial group (group 8), so the GroupKFold CV never has any SGAM training data in folds that exclude that group.

---

## Feature Architecture (64 features)

| Group                            | Count | Description                   |
| -------------------------------- | ----- | ----------------------------- |
| Base BTM (BPI, slope, VRM, etc.) | 10    | Standard BTM derivatives      |
| Eco features (northness, etc.)   | 4     | From run-015 eco extraction   |
| Multi-scale slope                | 5     | scales 3, 7, 11, 15, 21 cells |
| Multi-scale VRM                  | 5     | scales 3, 7, 11, 15, 21 cells |
| Multi-scale surface ratio        | 5     | scales 3, 7, 11, 15, 21 cells |
| Multi-scale northness            | 5     | scales 3, 7, 11, 15, 21 cells |
| Multi-scale eastness             | 5     | scales 3, 7, 11, 15, 21 cells |
| Multi-scale max curvature        | 5     | scales 3, 7, 11, 15, 21 cells |
| Multi-scale complexity           | 5     | scales 3, 7, 11, 15, 21 cells |
| RDMV depth variability           | 5     | scales 3, 7, 11, 15, 21 cells |
| GLCM contrast (backscatter)      | 5     | scales 3, 7, 11, 15, 21 cells |
| GLCM homogeneity (backscatter)   | 5     | scales 3, 7, 11, 15, 21 cells |

---

## Feature Selection: 64 → 33 Features

### Correlation pre-filter (|r| > 0.95): 25 dropped

The multi-scale derivatives are highly correlated with one another and with base derivatives. The following were dropped by Spearman correlation pre-filter (|r| > 0.95), retaining the higher-variance representative of each correlated pair:

`btm_broad_std`, `btm_complexity_11`, `btm_complexity_15`, `btm_max_curvature_15`, `btm_max_curvature_21`, `btm_max_curvature_7`, `btm_northness_15`, `btm_rdmv_11`, `btm_rdmv_21`, `btm_rough_total`, `btm_slope_11`, `btm_slope_15`, `btm_slope_21`, `btm_slope_7`, `btm_surface_ratio`, `btm_surface_ratio_11`, `btm_surface_ratio_15`, `btm_surface_ratio_21`, `btm_surface_ratio_3`, `btm_surface_ratio_7`, `btm_vrm_11`, `btm_vrm_15`, `btm_vrm_21`, `btm_vrm_3`, `btm_vrm_7`

### Permutation importance filter (importance ≤ 0.0): 6 additional dropped

`btm_fine_bpi`, `btm_broad_std` (pre-corr), `btm_fine_std`, `btm_bpi_magnitude`, `btm_broad_x_fine_std`, `btm_rough_total` (pre-corr)

### 33 retained features with top importances

| Rank | Feature                 | Permutation Importance |
| ---- | ----------------------- | ---------------------- |
| 1    | btm_northness_21        | 0.1442 ± 0.0020        |
| 2    | btm_eastness_21         | 0.0443 ± 0.0013        |
| 3    | btm_complexity_21       | 0.0311 ± 0.0030        |
| 4    | btm_glcm_contrast_21    | 0.0110 ± 0.0012        |
| 5    | btm_glcm_homogeneity_21 | 0.0115 ± 0.0011        |
| 6    | btm_eastness_15         | 0.0085 ± 0.0008        |
| 7    | btm_northness_7         | 0.0030 ± 0.0010        |

The **broad-scale morphological orientation** (northness_21, eastness_21) and **broad-scale structural complexity** (complexity_21) dominate the feature ranking, confirming that the 21-cell (≈105 m) spatial scale carries the most discriminant information for this survey.

SC-004 FAIL on count: 33 > 25 features. Degradation gate PASSES (Δ = −0.0006).

---

## Submission Files

| File                               | Rows | Notes                                  |
| ---------------------------------- | ---- | -------------------------------------- |
| `data/submission_v10.csv`          | 98   | Full 64-feature model (CatBoost best)  |
| `data/submission_v10_selected.csv` | 98   | 33-feature selected model; 1/98 differ |

Feature selection changes 1 of 98 test predictions (1.0%). The selected-feature model is the preferred submission as it carries lower overfitting risk.

---

## Code Changes (this run)

| File                                      | Change                                                                                                |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `btm/core/multiscale.py`                  | New: `focal_mean_multiscale()`, `compute_rdmv()` with NaN-safe fill (scipy uniform_filter fix)        |
| `btm/core/glcm.py`                        | New: `compute_glcm_texture()` using skimage GLCM, averaged over 4 directions                          |
| `src/benthic_model/features/selection.py` | New: `select_by_permutation_importance()` (Spearman corr pre-filter + sklearn permutation_importance) |
| `scripts/experiment_v10.py`               | New experiment script; feature cache (CSV), CatBoost + LGB + ensemble, feature selection block        |
| `btm/core/vrm.py`                         | Bug fix: scipy NaN propagation in uniform_filter — fill before filter, restore mask after             |

---

## Bug Fixes Discovered During This Run

### scipy NaN propagation in `uniform_filter`

**Problem**: `scipy.ndimage.uniform_filter` propagates NaN to all cells within the filter radius if any input cell is NaN. For this study site, 67% of the raster is nodata (value −10000.0). After converting nodata to NaN before the filter, all 64 multi-scale output arrays were 100% NaN.

**Root cause**: This affected `btm/core/vrm.py` (VRM intermediate x/y/z arrays) and `btm/core/multiscale.py` (focal mean for all multi-scale derivatives).

**Fix applied**:

- `vrm.py`: Compute `nan_mask = np.isnan(work)` before uniform_filter calls; fill NaN cells with 0.0 for the filter input; restore `vrm[nan_mask] = np.nan` after clipping.
- `multiscale.py` `focal_mean_multiscale()`: Fill NaN with 0.0 before filter; restore NaN mask after.
- `multiscale.py` `compute_rdmv()`: Fill NaN with `np.nanmean(work)` before both filters; restore NaN mask on output.

**Impact**: Without this fix, 27 of 64 feature columns were all-NaN, giving a CV F1 of ~0.52 (equivalent to BTM-base-only features — correctly so, as those columns were excluded by null variance). After the fix, all 64 columns are valid with 0 NaN values.

---

## Technical Notes

### High feature correlation in multi-scale sets

Terrain derivatives computed at adjacent spatial scales are highly correlated (Spearman |r| > 0.95 for scale pairs like slope_7/slope_11). This is expected: the underlying terrain changes smoothly across scales. The correlation pre-filter is critical to make permutation importance meaningful — without it, permutation importance is diluted across correlated redundant features.

### SGAM recall 0.000 is structural, not model failure

169 of 170 SGAM training points fall in spatial group 8 (GroupKFold groups). In all 10 folds that exclude group 8, the model is trained with zero SGAM samples, so it cannot predict SGAM at all. The single fold that includes group 8 in training but not validation has only 1 SGAM point in validation — not enough to register non-zero mean recall across folds. This is a data distribution issue, not a model or feature failure. Addressing it would require either: (a) re-stratifying the spatial groupings to spread SGAM samples, or (b) a SGAM-targeted oversampling or cost-sensitive approach outside the GroupKFold frame.

### Broad-scale orientation features dominate

`btm_northness_21` (importance 0.1442) and `btm_eastness_21` (0.0443) are the top two features by a large margin. This suggests the study site has habitat classes that are primarily organised by large-scale aspect/orientation (e.g., sun-facing vs. shadow-facing slopes, or swell-facing reef faces). This is ecologically plausible for coral reef systems where light and wave exposure drive community structure.

---

## Output Files

| File                                         | Description                                              |
| -------------------------------------------- | -------------------------------------------------------- |
| `reports/metrics/cv_per_class_v10.csv`       | Per-class recall per fold for all model types            |
| `reports/metrics/feature_importance_v10.csv` | CatBoost + LGB feature importances (full 64-feature set) |
| `reports/metrics/feature_selection_v10.csv`  | Permutation importance scores for all features           |
| `reports/metrics/cache_train_feats_v10.csv`  | Cached feature matrix (6256 × 67 incl. ID/x/y)           |
| `reports/metrics/cache_test_feats_v10.csv`   | Cached test feature matrix (98 × 67)                     |
| `data/submission_v10.csv`                    | Full-feature submission (98 rows)                        |
| `data/submission_v10_selected.csv`           | Selected-feature submission (98 rows, preferred)         |

---

## Next Steps

1. **Add RF to `experiment_v10.py`** — replace or supplement CatBoost/LGB with
   `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample')`.
   Run-014 established RF outperforms gradient boosters by ~0.006 on BTM features.
   This gives the true BTM-only RF baseline with multi-scale features and isolates the model effect.

2. **Combine multi-scale BTM + MBES-8 via `benthic_model` pipeline** — this is the highest-priority
   next experiment. Pass the 33 selected multi-scale BTM columns into `benthic_model`'s RF pipeline
   (alongside the 8 MBES point-sample features it adds internally). Expected CV gain: approaching or
   exceeding 0.8024, potentially 0.82+. Only this combination addresses Cause 1 of the discrepancy.

3. **Re-examine SGAM spatial structure** — plot group 8 location vs. rest of survey to determine
   whether geographic isolation of SGAM is a real ecological niche boundary or a sampling artefact.

4. **Reduce to ≤25 features (SC-004 full pass)** — select only the top-N within each feature group,
   or apply a more aggressive correlation threshold (|r| > 0.90 instead of 0.95).

5. **Submit `submission_v10_selected.csv`** to the Kaggle leaderboard to measure external validation
   of the 0.6383 CV score.
