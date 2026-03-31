# Feature Specification: Multi-Scale Terrain Features for Improved Benthic Habitat Classification

**Feature Branch**: `016-multiscale-terrain-features`  
**Created**: 2026-05-04  
**Status**: Draft  
**Input**: User description: "Multi-scale terrain features and improved classification for benthic habitat mapping"

## Context & Motivation

The current best Kaggle weighted-F1 score is **0.79518**, achieved by three different RF+BTM model configurations. Ten scored submissions produce only five distinct Kaggle scores (0.731, 0.762, 0.764, 0.795), because the test set contains just **98 points** — models that are architecturally different still agree on 93–95 of 98 predictions.

To move the score, the system must correctly reclassify the 4–5 "swing points" at habitat boundaries where current models disagree. Research literature identifies three evidence-based strategies for improving benthic habitat classification accuracy that have not yet been applied to this dataset.

### Research Evidence

**Ierodiaconou et al. (2018)** — the reference paper for this exact competition dataset — found that a combined pixel-based + object-based model was **statistically significantly more accurate** than either approach alone. The top 3 most important variables were: (1) object-based mean backscatter, (2) pixel-based bathymetry, (3) object-based standard deviation of rugosity. Object-based features (segment-level mean, SD, skewness) are not yet computed in our pipeline.

**Nemani et al. (2022)** — "A multi-scale feature selection approach for predicting benthic assemblages" — found that **none of the terrain features calculated from the default 3×3 window were selected** in the best model. Multi-scale features (3×3 through 21×21) with Boruta wrapper feature selection improved accuracy by over 25%. The best XGB model used only 11 of 120 multi-scale features, with bathymetry, slope at 210m scale, RDMV at 50m scale, and backscatter GLCM contrast at 190m scale ranking highest.

**Current gap**: All BTM derivatives (slope, VRM, surface ratio) are computed at a single fixed 3×3 neighbourhood. The literature demonstrates this is insufficient — broader-scale terrain context is critical for distinguishing habitat types.

## Improvement Options

Three research-backed improvement strategies were evaluated. **Options A and B are in scope** for this feature branch. Option C is deferred to a follow-up branch.

### Option A: Multi-Scale Terrain Derivatives ✅ IN SCOPE

Compute existing BTM terrain derivatives at multiple window sizes (e.g., 3×3, 7×7, 11×11, 15×15, 21×21) instead of the current single 3×3 scale. Add Relative Difference to Mean Value (RDMV) as a new terrain variable. Apply importance-based feature selection (Boruta or permutation importance) to reduce dimensionality before modelling.

**Evidence**: Nemani et al. (2022) showed multi-scale features with feature selection increased accuracy by 25%+. No 3×3 features were retained in their best model — broader scales captured more ecologically meaningful terrain variation. Slope at the broadest scale (210m) and RDMV at 50m were among the top 4 most important features.

**Expected impact**: HIGH — directly addresses the known limitation that single-scale features miss broader geomorphic context.  
**Implementation effort**: MEDIUM — existing compute functions need parameterisation for variable kernel sizes; feature selection adds a new pipeline step.

### Option B: Backscatter GLCM Texture Features ✅ IN SCOPE

Extract Grey-Level Co-occurrence Matrix (GLCM) texture features from the backscatter raster at multiple window sizes. Compute contrast and homogeneity metrics that quantify local variation and similarity in seabed acoustic texture.

**Evidence**: Nemani et al. (2022) found backscatter GLCM contrast at 190m scale was the 4th most important feature. Ierodiaconou et al. (2018) found object-based mean backscatter was the single most important variable. Current pipeline uses raw backscatter and basic summary statistics but does not compute spatial texture features.

**Expected impact**: MEDIUM-HIGH — fills a gap in the existing feature set for characterising seabed substrate composition.  
**Implementation effort**: LOW-MEDIUM — GLCM computation is available in standard image processing libraries; integration into the pipeline follows the existing feature extraction pattern.

### Option C: Object-Based Segment Statistics on Terrain Derivatives ⏳ DEFERRED

Compute segment-level summary statistics (mean, standard deviation, skewness) for bathymetry, backscatter, and rugosity within image segments. These object-based features capture within-segment variability that pixel-level features cannot represent.

**Evidence**: Ierodiaconou et al. (2018) Table 2 shows the combined model used segment-level mean backscatter, bathymetry skewness, rugosity SD, and rugosity skewness — all of which were absent from the pixel-only model. The combined model was statistically significantly more accurate.

**Expected impact**: MEDIUM — the v9 pipeline already performs segmentation, but does not compute per-derivative segment statistics.  
**Implementation effort**: MEDIUM-HIGH — requires computing segment-level statistics for each terrain derivative and joining them back to point samples.

## User Scenarios & Testing _(mandatory)_

### User Story 1 — Compute Multi-Scale Features for Training Data (Priority: P1)

A data scientist runs the feature extraction pipeline and obtains terrain derivatives computed at multiple neighbourhood scales for all training points, enabling the model to learn from both fine-grained and broad-scale terrain patterns.

**Why this priority**: Multi-scale features are the highest-impact improvement identified by the literature and can be combined with any downstream model.

**Independent Test**: Run feature extraction on the training CSV with multi-scale enabled and verify the output DataFrame contains the expected columns at each scale.

**Acceptance Scenarios**:

1. **Given** a bathymetry raster and training point CSV, **When** multi-scale feature extraction is run with scales [3, 7, 11, 15, 21], **Then** the output DataFrame contains terrain derivative columns at each specified scale
2. **Given** multi-scale features are extracted, **When** the user inspects the output, **Then** values at larger scales show smoother spatial patterns compared to the 3×3 baseline
3. **Given** a point near the raster edge, **When** the neighbourhood extends beyond raster bounds, **Then** the system handles the boundary gracefully (NaN or reduced-window fallback)

---

### User Story 2 — Train and Evaluate an Improved Model (Priority: P1)

A data scientist trains a classification model using the multi-scale feature set and evaluates it with cross-validation, comparing performance against the current single-scale baseline.

**Why this priority**: The ultimate value of multi-scale features is realised only when they improve classification accuracy. This story validates the hypothesis.

**Independent Test**: Train RF/XGB/CatBoost with multi-scale features and compare CV weighted-F1 against the current best (0.8024) from baseline runs.

**Acceptance Scenarios**:

1. **Given** multi-scale features for training data, **When** a classification model is trained and evaluated with 5-fold spatial CV, **Then** the mean weighted-F1 is reported alongside the current baseline score
2. **Given** a trained model, **When** feature importance is computed, **Then** the relative importance of each feature and scale is reported to guide feature selection
3. **Given** a full pipeline run, **When** predictions are generated for the test set, **Then** a valid submission CSV is produced for Kaggle upload

---

### User Story 3 — Feature Selection to Reduce Dimensionality (Priority: P2)

A data scientist applies importance-based feature selection to the high-dimensional multi-scale feature set to identify the optimal subset of features, reducing overfitting risk and improving model interpretability.

**Why this priority**: Multi-scale extraction can produce 50–100+ features. Feature selection is needed to avoid the curse of dimensionality and identify which scales matter most for each derivative.

**Independent Test**: Apply feature selection and verify the reduced set produces CV performance equal to or better than the full set.

**Acceptance Scenarios**:

1. **Given** the full multi-scale feature set, **When** feature selection is applied, **Then** a reduced set of 10–25 features is identified
2. **Given** the reduced feature set, **When** a model is trained, **Then** CV performance matches or exceeds the full-set model

---

### User Story 4 — Backscatter Texture Features (Priority: P2)

A data scientist computes GLCM texture features (contrast, homogeneity) from the backscatter raster at multiple scales and adds them to the feature set.

**Why this priority**: Backscatter texture was among the top 4 most important features in Nemani et al. (2022) and is currently absent from the pipeline.

**Independent Test**: Extract GLCM features from the backscatter raster and verify they produce valid numeric arrays with expected statistical properties.

**Acceptance Scenarios**:

1. **Given** a backscatter raster, **When** GLCM features are computed at specified scales, **Then** contrast and homogeneity columns are added to the feature DataFrame
2. **Given** GLCM features are available, **When** they are added to the model training, **Then** they appear among the top feature importances

---

### User Story 5 — Segment-Level Statistics (Priority: P3)

A data scientist computes object-based summary statistics (mean, SD, skewness) for bathymetry, backscatter, and rugosity within image segments and adds them to the feature set.

**Why this priority**: Object-based features were critical in the reference paper's combined model but represent a more complex pipeline change.

**Independent Test**: Generate segment statistics and verify each point receives valid segment-level summary values.

**Acceptance Scenarios**:

1. **Given** a segmented raster and terrain derivatives, **When** segment statistics are computed, **Then** each training point receives mean, SD, and skewness values for its parent segment
2. **Given** segment-level features are added, **When** a model is trained, **Then** CV performance is compared against the pixel-only baseline

### Edge Cases

- What happens when a point falls at the raster boundary where the full neighbourhood window cannot be applied? The system should return NaN or use a reduced window.
- How does the system handle segments with very few pixels (< 5)? Skewness and SD become unreliable — flag or exclude these.
- What happens when correlation between multi-scale features of the same derivative is very high (> 0.95)? The feature selection step must handle collinear features.
- What if the backscatter raster has nodata gaps? GLCM computation must skip or handle nodata within the window.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST compute terrain derivatives (slope, VRM, surface ratio) at multiple scales using the calculate-then-average method: compute derivative at native 3×3 resolution, then apply focal mean over increasing window sizes. BPI retains its existing dual-scale annular configuration (broad/fine) and is not recomputed at additional window sizes
- **FR-002**: System MUST support at least 5 different scales in a single extraction run (e.g., 3, 7, 11, 15, 21)
- **FR-003**: System MUST produce consistently named output columns that encode both the derivative type and the scale (e.g., `btm_slope_7`, `btm_vrm_21`)
- **FR-004**: System MUST compute Relative Difference to Mean Value (RDMV) as a new terrain derivative at each specified scale
- **FR-005**: System MUST handle boundary cells where the full window extends beyond the raster extent using `mode='reflect'` in `scipy.ndimage.uniform_filter`, mirroring values at the raster edge (research decision R7); NaN is not produced under normal boundary conditions
- **FR-006**: System MUST provide a feature importance ranking after model training, showing the contribution of each feature and scale
- **FR-007**: System MUST compute GLCM texture features (at minimum contrast and homogeneity) from the backscatter raster at specified scales
- **FR-008**: System MUST apply permutation importance-based feature selection to reduce the multi-scale feature set before final model training
- **FR-009**: System MUST produce a valid Kaggle submission CSV from the trained model's test set predictions
- **FR-010**: System MUST log which features were selected and which were dropped, including their importance scores

### Key Entities

- **Terrain Derivative**: A continuous spatial variable computed from bathymetry (slope, VRM, BPI, surface ratio, RDMV, northness, eastness, max curvature, complexity). Each derivative can be computed at multiple neighbourhood scales.
- **Scale**: The spatial neighbourhood window size (in cells) used to compute a derivative. Determines the spatial extent of terrain context captured.
- **GLCM Texture Feature**: A second-order statistical measure (contrast, homogeneity) computed from a grey-level co-occurrence matrix of backscatter pixel values within a neighbourhood window.
- **Feature Selection Result**: A record of which features were retained and which were dropped, including importance scores, used to guide final model configuration.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: At least one model configuration using multi-scale features achieves a cross-validated weighted-F1 equal to or above the current best of 0.8024
- **SC-002**: Either the Kaggle public leaderboard score meets or exceeds the current best of 0.79518, OR cross-validated weighted-F1 improves by at least 0.005 over the current best of 0.8024 (since the 98-point test set makes Kaggle scores an unreliable sole gate)
- **SC-003**: The minority class (SGAM, 170 training samples) recall is maintained or improved compared to baseline runs
- **SC-004**: Feature selection reduces the multi-scale feature set to fewer than 25 features without degrading CV performance by more than 0.01
- **SC-005**: Total feature extraction time for the full training set (6256 points) completes within 10 minutes on a standard workstation

## Clarifications

### Session 2026-03-31

- Q: Which improvement options are in scope for this feature branch? → A: Options A + B together (multi-scale terrain derivatives + GLCM backscatter texture). Option C (segment statistics) deferred to follow-up.
- Q: What multi-scale aggregation method should be used? → A: Calculate-then-average (compute derivative at native 3×3, then focal mean over window sizes), per Nemani et al. (2022) and Misiuk et al. (2021).
- Q: Should BPI be recomputed at additional scale windows beyond broad/fine? → A: No — BPI already captures two ecological scales via annular radii. Additional BPI windows would duplicate the RDMV concept.
- Q: Which feature selection method? → A: Permutation importance — simpler, built into scikit-learn, sufficient for the 50–80 feature range expected.
- Q: Should SC-002 (Kaggle score gate) be strict given the 98-point test set quantization? → A: Softened — accept either Kaggle improvement OR CV improvement ≥ 0.005, since 98-point test set makes Kaggle scores unreliable as a sole gate.

## Assumptions

- The existing bathymetry and backscatter rasters (`data/MBES/`) have sufficient resolution to support multi-scale computation up to 21×21 cell windows
- BPI retains its existing annular (broad/fine) configuration and is not subject to multi-scale windowing — RDMV and other derivatives cover that role
- The existing spatial-blocked cross-validation strategy remains appropriate for evaluating multi-scale models
- Feature selection will be performed after full multi-scale extraction, not during the extraction step itself
- The Kaggle submission budget will allow at least 2 submissions to test multi-scale model predictions
- Python libraries for GLCM computation (scikit-image or equivalent) are compatible with the current environment
