# Feature Specification: OBIA + Pixel-Based Hybrid Classification

**Feature Branch**: `009-obia-pixel-hybrid`  
**Created**: 2026-03-27  
**Status**: Draft  
**Reference**: Ierodiaconou et al. (2018) Marine Geodesy DOI: 10.1007/s11001-017-9338-z

## Background & Motivation

This feature is directly modelled on Ierodiaconou et al. (2018), a peer-reviewed study that used the **exact same dataset** (Refuge Cove, MBES at 0.25 m, same five benthic classes: ALG, FMAT, NVB, SGAM, SGZ). The paper compared three classification strategies:

| Approach | Overall Accuracy | Kappa |
| --- | --- | --- |
| Pixel-Based (PB) only | 72.5% | 0.62 |
| Object-Based (OB) only | 78.5% | 0.70 |
| Combined PB + OB | **83.6%** | **0.78** |

The combined model was statistically significantly better than either approach alone. Our current best Kaggle submission scores 0.72750, leaving substantial room to improve by following the paper's combined approach using modern Python tooling.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Combined Feature Extraction and Baseline Comparison (Priority: P1)

A data scientist runs the new experiment script on the existing MBES rasters and obtains a spatial cross-validation F1 score that exceeds the current best (~0.69 CV, ~0.72 Kaggle). The script produces a submission file and a run report documenting all steps.

**Why this priority**: Validates the core hypothesis that the combined PB+OB approach lifts F1. Everything else builds on this.

**Independent Test**: Run `scripts/experiment_v6.py` end-to-end. The script completes, produces `data/submission_v6_best.csv` (98 rows), and `docs/run-009-obia-pixel-hybrid.md` with CV metrics.

**Acceptance Scenarios**:

1. **Given** MBES bathymetry and backscatter rasters exist, **When** the script runs, **Then** it completes without error, produces pixel-based features, segment labels, and object-level statistics for all 98 test points.
2. **Given** the experiment completes, **When** spatial cross-validation is evaluated, **Then** the combined CV F1 equals or exceeds 0.72 (vs. current ~0.69 CV for v2 baseline).
3. **Given** the script runs, **When** CV scores are compared across PB-only, OB-only, and combined configurations, **Then** the combined configuration produces the highest F1 score.

---

### User Story 2 - Pixel-Based Feature Completeness (Priority: P2)

All pixel-based derivates from Table 1 of the paper are computed and available for model input: bathymetry, backscatter, slope, rugosity, complexity, maximum curvature, northness (cos of aspect), and eastness (sin of aspect).

**Why this priority**: Pixel features form the baseline that the OB features augment. Missing any key feature leaves signal on the table.

**Independent Test**: Run the feature extraction alone with a unit test confirming all 8 pixel-based feature columns are non-NaN for all training points.

**Acceptance Scenarios**:

1. **Given** the rasters are loaded, **When** pixel features are extracted for training points, **Then** all 8 pixel-level features (depth, backscatter, slope, rugosity, complexity, max_curvature, northness, eastness) are present and finite.
2. **Given** aspect is computed from bathymetry, **When** northness and eastness are derived, **Then** values are in range [-1, 1].

---

### User Story 3 - Object-Based Segmentation and Segment Statistics (Priority: P2)

MBES rasters are segmented into spatially coherent objects using an image segmentation algorithm. For each segmentation object, summary statistics (mean, standard deviation, skewness) are computed for bathymetry, backscatter, and rugosity. Training and test points are assigned to their containing segment and inherit its statistics.

**Why this priority**: The object-based features (especially Rugosity SD and Backscatter Mean) were the most important predictors in the paper's combined model.

**Independent Test**: Unit test confirms segment labels array has correct shape and no label is unassigned for any training/test point. Segment statistics have no NaNs.

**Acceptance Scenarios**:

1. **Given** the segmentation target rasters, **When** segmentation is applied, **Then** the resulting label array has the same shape as the input rasters with integer labels ≥ 0.
2. **Given** segment labels, **When** statistics (mean, std, skewness) of bathymetry/backscatter/rugosity are aggregated per segment, **Then** all training and test points receive finite values for all OB features.
3. **Given** segments, **When** the mean segment size is computed, **Then** the target object size approximates 300 m² (equivalent to the paper's scale parameter of 41 on a 0.25 m grid), within ±50%.

---

### User Story 4 - Run Report and Documentation Trail (Priority: P3)

The experiment produces a Markdown run report that documents the approach, CV scores for each configuration, feature importances, and a brief description of segment parameters used. The report serves as a reproducible record of the decision-making process.

**Why this priority**: Enables tracing the thinking, required by the user.

**Independent Test**: File `docs/run-009-obia-pixel-hybrid.md` exists after running the script and contains at minimum: segment parameters, per-configuration CV F1 table, top 5 feature importances for the combined model.

**Acceptance Scenarios**:

1. **Given** the experiment completes, **When** the report is opened, **Then** it contains segmentation parameters, CV metrics for PB, OB, and combined configurations, and feature importance ranking.

---

### Edge Cases

- What happens when a test point falls in a segment with only one pixel (edge of raster)? The per-segment statistics degrade gracefully — if std or skewness cannot be computed, the global mean of that statistic is substituted.
- What if segmentation produces a segment with no training labels at all? This is expected; test points in such segments still obtain valid raster statistics through the lookup.
- What happens at raster NoData boundaries? NaN pixel values are excluded from segment statistic aggregation; if an entire segment is NoData, the global median is substituted.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The experiment script MUST compute all 8 pixel-based spatial derivatives from the MBES data (bathymetry, backscatter, slope, rugosity, complexity, maximum curvature, northness, eastness) at native 0.25 m resolution with a 3×3 kernel, matching the paper.
- **FR-002**: The script MUST segment the MBES bathymetry and backscatter rasters (plus rugosity) into spatially contiguous objects targeting a mean object size of approximately 300 m² (≈ 4800 pixels on a 0.25 m grid), the equivalent of the paper's eCognition scale=41 setting.
- **FR-003**: For each segmentation object, the script MUST compute mean, standard deviation, and skewness of bathymetry, backscatter, and rugosity.
- **FR-004**: The script MUST assign object-level statistics to every training and test point by segment membership.
- **FR-005**: The script MUST train and evaluate three configurations using spatial cross-validation: PB-only, OB-only, and combined PB+OB.
- **FR-006**: The script MUST select the best-performing configuration and produce a submission CSV in the required format (ID, class columns).
- **FR-007**: The script MUST produce a Markdown run report at `docs/run-009-obia-pixel-hybrid.md` documenting approach, segment parameters, CV metrics, and top feature importances.
- **FR-008**: All pixel and segment features MUST be free of NaN values at prediction time; fallback to global statistics for any missing segment values.
- **FR-009**: Unit tests MUST verify: feature completeness for all training/test points, segment label assignment, submission format compliance, and report file existence.

### Key Entities

- **Pixel Feature Table**: One row per spatial point; columns for the 8 pixel-level MBES derivatives.
- **Segment Label Map**: A 2D array co-registered with the rasters; integer label for every valid pixel identifying which segment it belongs to.
- **Segment Statistics Table**: One row per unique segment label; columns for mean/std/skewness of bathymetry, backscatter, and rugosity — 9 OB features in total.
- **Combined Feature Matrix**: Pixel features joined with segment statistics for each training/test point; forms the input to the classifier.
- **Submission CSV**: 98 rows; columns: ID and the 5 class probability/prediction columns per competition format.
- **Run Report**: Markdown document linking segmentation parameters to CV outcomes and feature importances.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Spatial cross-validation weighted F1 for the combined PB+OB configuration exceeds 0.72 (vs. 0.6927 baseline from v2).
- **SC-002**: The combined configuration's CV F1 exceeds both the PB-only and OB-only configurations on the same fold structure.
- **SC-003**: All 98 test points produce valid (non-null) predictions in the submission file.
- **SC-004**: Segment mean object size falls within 200–500 m², maintaining spatial coherence equivalent to the paper's segmentation (306 m²).
- **SC-005**: Run report is produced in under 5 minutes of wall-clock time after feature extraction completes.
- **SC-006**: All unit tests pass.

## Assumptions

- The MBES rasters `data/MBES/bathymetry.tif` and `data/MBES/backscatter.tif` are the same data as used in Ierodiaconou et al. (2018) — confirmed by study area, resolution, and author affiliation in the METADATA.MD file.
- `eCognition` multi-resolution segmentation is proprietary software not available in Python. We use `scikit-image` SLIC/SLIC0 or Felzenszwalb segmentation as a modern, open-source equivalent. The target object size of ~300 m² is matched by tuning the `n_segments` or `scale` parameter.
- The existing raster stack from `outputs/gis_layers/rasters/` (slope.tif) is reused; rugosity is re-computed identically to the v2 method (rasterio + numpy 3×3 window).
- Northness and eastness are derived from bathymetry aspect (gradient direction using numpy arctan2 of dz/dy, dz/dx).
- Complexity is defined as surface area ratio (actual surface area / planar area) in the 3×3 window — equivalent to the paper's "complexity" variable.
- Maximum curvature is computed using the standard second-derivative approach on the bathymetry DEM.
- Ground truth labels and train/test split follow the existing `data/train.csv` and `data/test.csv` convention; the Kaggle competition format is preserved.
- The ensemble classifier (LightGBM + XGBoost + CatBoost + RF) from `scripts/experiment_v2.py` is retained and extended with the new features; the paper used Random Forest alone but our ensemble is expected to perform at least as well.
- Runtime may be significant (minutes to tens of minutes); the user has explicitly accepted long runtimes.

