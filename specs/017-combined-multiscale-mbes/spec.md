# Feature Specification: Combined Multi-Scale BTM + MBES-8 Features Experiment

**Feature Branch**: `017-combined-multiscale-mbes`  
**Created**: 2026-03-31  
**Status**: Draft  
**Input**: User description: "Run-016 multi-scale BTM features (CV F1=0.6377) fell short of R04/R06 baseline (0.8024) due to two identified causes: (1) missing MBES-8 point-sample features (~14 pp gap) and (2) CatBoost instead of RF (~0.6 pp gap). This experiment creates experiment_v11.py combining 33 selected multi-scale BTM features with the 8 MBES point-sample features and adding RF as a third model to close both gaps."

## User Scenarios & Testing _(mandatory)_

### User Story 1 — Combine BTM + MBES Features and Train Models (Priority: P1)

A data scientist runs `experiment_v11.py` on the training data. The script extracts the 33 selected multi-scale BTM features (from spec-016 feature selection) alongside the 8 MBES point-sample features (depth, backscatter, slope, VRM, complexity, max_curvature, northness, eastness), producing a combined ~41-feature set. Three models (Random Forest, CatBoost, LightGBM) are trained and evaluated via 5-fold spatial-blocked CV (matching R04/R06's scheme via `iter_spatial_blocked_folds(n_splits=5, spatial_bins=4)`). The experiment reports per-model CV weighted-F1 and per-class recall, enabling direct comparison against the R04/R06 baseline (CV F1=0.8024).

**Why this priority**: This is the core deliverable — the "very high priority" experiment identified in the runsheet root-cause analysis. Without combining both feature sources, the 14 pp Cause 1 gap cannot be closed.

**Independent Test**: Run the script end-to-end on the existing training data and verify CV weighted-F1 is reported for all three models. RF on the combined set should match or exceed 0.8024.

**Acceptance Scenarios**:

1. **Given** bathymetry/backscatter rasters and `data/train.csv`, **When** `experiment_v11.py` is run, **Then** the output DataFrame contains both `btm_*` multi-scale columns and the 8 MBES columns (depth, backscatter, slope, VRM, complexity, max_curvature, northness, eastness)
2. **Given** combined features are extracted, **When** 5-fold spatial-blocked CV is run, **Then** per-model CV weighted-F1 and per-class recall are reported for RF, CatBoost, and LightGBM
3. **Given** RF is trained with R04's proven hyperparameters, **When** CV results are compared against R04/R06, **Then** RF CV F1 on the combined set ≥ 0.8024

---

### User Story 2 — Feature Selection on Combined Set (Priority: P2)

A data scientist applies permutation-importance feature selection to the combined ~41-feature set to identify the optimal reduced subset. The selected features should retain classification performance while reducing dimensionality and overfitting risk.

**Why this priority**: The combined set may contain redundant features (e.g., BTM eco-northness overlapping with MBES northness). Feature selection identifies which features from each source carry independent signal, improving interpretability and generalisability.

**Independent Test**: Apply feature selection and verify the reduced set produces CV performance within 0.01 of the full combined set.

**Acceptance Scenarios**:

1. **Given** the full combined feature set (~41 features), **When** Spearman correlation pre-filter (|r| > 0.95) and permutation-importance selection are applied, **Then** a reduced set of ≤ 25 features is identified
2. **Given** the reduced feature set, **When** models are retrained, **Then** CV F1 degradation is ≤ 0.01 compared to the full combined set
3. **Given** feature importance results, **When** the ranking is inspected, **Then** the relative contribution of MBES-8 vs BTM multi-scale features is clearly reported

---

### User Story 3 — Produce Kaggle Submission (Priority: P2)

A data scientist generates a Kaggle-format submission CSV from the best-performing model on the combined feature set, enabling direct leaderboard comparison against R04/R06's Kaggle F1 of 0.79518.

**Why this priority**: Kaggle submission validates that CV improvements translate to held-out test performance, which is the ultimate measure of model quality for this competition.

**Independent Test**: Run the script to completion and verify a valid submission CSV is written with exactly 98 rows matching `data/sample_submission.csv` format.

**Acceptance Scenarios**:

1. **Given** a trained model on the combined feature set, **When** predictions are generated for the test points, **Then** a submission CSV is written with 98 rows and the correct column format
2. **Given** both full-feature and selected-feature models, **When** submissions are generated, **Then** the number of differing predictions between the two is reported

---

### User Story 4 — SGAM Minority Class Monitoring (Priority: P3)

A data scientist reviews per-class recall, specifically SGAM recall, to ensure the combined feature set does not degrade rare-class performance relative to baseline. SGAM has proven difficult (recall = 0.000 in run-016, 0.045 in R04/R06) due to extreme spatial clustering (169/170 samples in one group).

**Why this priority**: SGAM recall is an explicit success criterion. Adding MBES features may help (baseline R04 had non-zero SGAM recall) or hurt (more features increasing overfitting on the majority class).

**Independent Test**: Check SGAM recall in the CV output and compare against the R04/R06 baseline.

**Acceptance Scenarios**:

1. **Given** 5-fold spatial-blocked CV results, **When** per-class recall is reported, **Then** SGAM mean recall is explicitly listed
2. **Given** SGAM recall from the combined model, **When** compared against the R04/R06 baseline (SGAM recall ≈ 0.045), **Then** SGAM recall is maintained or improved

---

### Edge Cases

- What happens when MBES point-sample extraction returns NaN for points outside raster bounds? NaN values should be handled consistently (filled with column median or flagged).
- What happens when BTM multi-scale features and MBES features have highly correlated columns (e.g., `btm_northness_21` vs MBES `northness`)? The correlation pre-filter should detect and resolve these cross-source redundancies.
- What happens when the RF model with `class_weight='balanced_subsample'` encounters folds with zero SGAM samples? The per-fold SGAM recall should be reported as NaN or 0.0 for that fold without crashing.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: The system MUST extract the 8 MBES point-sample features (depth, backscatter, slope, VRM, complexity, max_curvature, northness, eastness) directly from bathymetry and backscatter rasters for each training and test point
- **FR-002**: The system MUST extract the 33 selected multi-scale BTM features (per spec-016 permutation-importance selection) for each training and test point
- **FR-003**: The system MUST merge the MBES-8 and BTM-33 feature sets into a single combined feature matrix (~41 features) keyed by point ID
- **FR-004**: The system MUST train and evaluate a Random Forest model with hyperparameters matching R04's proven configuration: `n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample'`
- **FR-005**: The system MUST train and evaluate CatBoost and LightGBM models on the same combined feature set, using the same hyperparameters as experiment_v10
- **FR-006**: The system MUST use 5-fold spatial-blocked cross-validation via `iter_spatial_blocked_folds(n_splits=5, spatial_bins=4, random_state=42)` from `src/benthic_model/evaluation/cv.py` (same scheme as R04/R06)
- **FR-007**: The system MUST report per-model CV weighted-F1 and per-class recall (including SGAM explicitly) for all three models
- **FR-008**: The system MUST apply Spearman correlation pre-filter (|r| > 0.95) followed by permutation-importance feature selection on the combined feature set
- **FR-009**: The system MUST write a feature importance CSV with columns: feature_name, importance_mean, importance_std, rank, selected
- **FR-010**: The system MUST produce a Kaggle submission CSV for the best-performing model configuration, matching `data/sample_submission.csv` format
- **FR-011**: The system MUST cache extracted features to avoid re-extraction on subsequent runs
- **FR-012**: The system MUST report total feature extraction time and total training time separately
- **FR-013**: The system MUST support `--dry-run` mode for rapid testing with reduced data and scales

### Key Entities

- **Combined Feature Set**: Union of 33 selected multi-scale BTM features and 8 MBES point-sample features, producing ~41 total columns per sample point
- **MBES-8 Features**: The 8 raster-sampled features that the `benthic_model` pipeline always includes — depth, backscatter, slope, VRM, complexity, max_curvature, northness, eastness — sampled directly at each point's (x, y) coordinates from the bathymetry and backscatter GeoTIFFs
- **BTM-33 Selected Features**: The 33 features retained after spec-016's correlation pre-filter and permutation-importance selection, including multi-scale terrain derivatives at scales 3–21, RDMV depth variability, and GLCM backscatter texture
- **Spatial Groups**: Quantile-based spatial blocks (4×4 grid) assigned round-robin to 5 folds via `iter_spatial_blocked_folds()`, preventing spatial autocorrelation leakage between train and validation sets

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: RF on the combined feature set achieves CV weighted-F1 ≥ 0.8024 (matching the R04/R06 baseline under the same 5-fold spatial-blocked CV scheme)
- **SC-002**: Best model (the single model or ensemble with the highest CV weighted-F1) achieves Kaggle F1 ≥ 0.79518 OR CV weighted-F1 improvement ≥ 0.005 over the SC-001 RF baseline
- **SC-003**: SGAM per-class recall is maintained or improved compared to the R04/R06 baseline (≈ 0.045 mean recall)
- **SC-004**: Feature selection reduces the combined set to ≤ 25 features with CV weighted-F1 degradation ≤ 0.01 compared to the full combined set
- **SC-005**: Total feature extraction time (train + test) completes within 15 minutes on a standard workstation

## Assumptions

- The 33 selected BTM features from spec-016 remain valid for the combined context; re-running selection on the combined set may yield a different optimal subset, which is expected and desirable
- The MBES-8 extraction pattern from experiment_v9.py (rasterio point sampling + derived terrain metrics) can be reused directly in the new script
- The existing raster files (`data/MBES/bathymetry.tif`, `data/MBES/backscatter.tif`) are available and in the expected CRS
- RF hyperparameters from R04 are used without further tuning; hyperparameter optimisation is out of scope
- CatBoost and LightGBM hyperparameters match those used in experiment_v10 for direct comparability
- The 5-fold spatial-blocked CV scheme is identical to R04/R06 (via `benthic_model.evaluation.cv.iter_spatial_blocked_folds`) for direct score comparability; this differs from v10's 10-fold KMeans GroupKFold, so v10↔v11 scores are not directly comparable
- Feature caching from experiment_v10 can be extended or reused for the BTM-33 columns; MBES-8 extraction adds minimal overhead
- The `benthic_model` package is installed in the same Python environment for any shared utility functions; if not, MBES-8 extraction logic is self-contained in the new script
