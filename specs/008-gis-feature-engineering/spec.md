# Feature Specification: GIS-Derived Feature Engineering for Benthic Habitat Prediction

**Feature Branch**: `008-gis-feature-engineering`
**Created**: 2026-03-27
**Status**: Draft
**Input**: User description: "Engineer GIS-derived features (slope, backscatter zones, acoustic facies) with 80/20/20 weighting, generate predictions, validate with kriging-improved inputs, create best submission file, document and PR"

## Clarifications

### Session 2026-03-27

- Q: Does the new feature matrix ADD the three GIS-derived features ON TOP of the full `experiment_v2.py` feature set, or replace it? -> A: Add on top (Option A -- retain all v2 features and append slope, backscatter zones, acoustic facies).
- Q: How is the 80/20/20 weighting applied to categorical features (backscatter zones, acoustic facies)? -> A: One-hot encode the categorical zone columns, then multiply the resulting binary columns by the respective weight factor before training.
- Q: In the kriging run, is kriging applied to ALL features or ONLY the three new GIS-derived features? -> A: Only the three new GIS-derived features are kriged; all other v2 features are passed through unchanged.
- Q: How many clusters should be used for backscatter zones and acoustic facies? -> A: Fixed to match prior work -- 5 clusters for backscatter zones, 8 clusters for acoustic facies.
- Q: What is the known v4 weighted-F1 baseline to compare against? -> A: Compute from the same spatial-block CV protocol at the start of the experiment script (v4 features only) and document the result as the baseline number in the run report.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - GIS-Derived Feature Engineering and Baseline Prediction (Priority: P1)

A marine data scientist wants to improve habitat classification accuracy by adding well-established GIS-derived spatial layers as model input features alongside the existing bathymetry and backscatter signals. The three new feature groups are appended to the full `experiment_v2.py` feature set (multi-scale terrain derivatives, curvature, GLCM texture, VRM, x/y coordinates):

- **Slope** (primary, weight ~80%): Horn-1981 slope in degrees derived from bathymetry
- **Backscatter zones** (secondary, weight ~20%): Categorical intensity zones derived from k-means clustering of smoothed backscatter (5 clusters, fixed)
- **Acoustic facies** (secondary, weight ~20%): Combined categorical zones from joint depth x backscatter k-means segmentation (8 clusters, fixed)

The derived layers already exist under `outputs/gis_layers/rasters/` from prior runs; this story covers re-deriving them on demand so the workflow is reproducible, and then incorporating them into a prediction pipeline. Slope is scaled by its weight factor directly; backscatter zones and acoustic facies are one-hot encoded before scaling.

**Why this priority**: Slope, backscatter zones, and acoustic facies are established BTM variables known to correlate with benthic habitat type. Adding them to the proven v2 feature set tests whether these explicit spatial signals improve upon the already-strong ensemble.

**Independent Test**: Run the derivation script; confirm `outputs/gis_layers/rasters/slope.tif`, `backscatter_zones.tif`, and `acoustic_facies.tif` exist or are regenerated. Then run the prediction script and confirm a valid Kaggle-format `data/submission_v5_gis_features.csv` is produced.

**Acceptance Scenarios**:

1. **Given** the MBES bathymetry and backscatter rasters on disk, **When** the derivation step runs, **Then** the three GIS-derived rasters (slope, backscatter zones, acoustic facies) are present in `outputs/gis_layers/rasters/` with correct CRS and dimensions matching the source rasters.
2. **Given** the three derived rasters, **When** values are sampled at each train and test point, **Then** no more than 1% of points have NaN feature values (points outside raster extent receive the global median as fallback; all test points must receive a valid prediction).
3. **Given** the augmented feature matrix (v2 features + new GIS features, weighted), **When** the ensemble model is trained with spatial-block cross-validation, **Then** the weighted-F1 score, per-class F1 scores, and the v4 baseline weighted-F1 (computed from v2 features only in the same run) are printed and written to the run report.
4. **Given** the trained model, **When** predictions are generated for all test points, **Then** a `data/submission_v5_gis_features.csv` with columns `ID,class` matching the format of prior submissions is created.

---

### User Story 2 - Kriging-Smoothed Feature Improvement and Comparative Validation (Priority: P2)

The scientist wants to know whether applying ordinary kriging to spatially interpolate the three GIS-derived features (slope, backscatter zone value, acoustic facies value) from train to test points improves prediction quality. Kriging is applied ONLY to these three features; all other v2 features are passed through unchanged.

**Why this priority**: Kriging leverages spatial autocorrelation and has shown improvements in prior experiments (`exp_backscatter_zones.py`, `exp_kriging.py`). Scoping kriging to only the new features keeps the comparison clean: same v2 base, same ensemble, but kriged vs raw for the three new additions.

**Independent Test**: Run the kriging-enhanced prediction script. Confirm a second submission CSV (`data/submission_v5_gis_features_kriging.csv`) is produced, and that the run report shows both the baseline (Story 1) weighted-F1 and the kriging-enhanced weighted-F1 side by side.

**Acceptance Scenarios**:

1. **Given** the sampled slope, backscatter zone, and acoustic facies values at train points, **When** ordinary kriging is fitted per feature per spatial zone and predictions are interpolated to test points (with fallback to nearest-neighbour when a zone has fewer than 10 training points), **Then** the kriged features form a valid numeric input matrix with no missing values.
2. **Given** the kriging-smoothed GIS features at test points and the raw-sampled GIS features at train points, **When** the same ensemble model configuration is applied with spatial-block CV (CV uses raw-sampled values, as in Run A; kriging applies only at final test-point inference), **Then** the Run B weighted-F1 from CV is computed and stored alongside the Run A result for comparison. The difference between Run A and Run B manifests in the test-point predictions (and thus the Kaggle submission), not in the CV F1 values.
3. **Given** both runs complete, **When** the run report is generated, **Then** it contains a comparison table showing: run name, weighted-F1, top-3 features by importance, and wall-clock runtime.

---

### User Story 3 - Best Submission Selection and Pull Request (Priority: P3)

After both runs complete, the scientist selects the better-performing submission file and creates a clean PR containing the new scripts, the best submission CSV, and a run report.

**Why this priority**: The Kaggle submission and PR are the deliverables of this branch.

**Independent Test**: Confirm that exactly one submission CSV is designated as the best run in the PR, the run report exists at `docs/run-008-gis-feature-engineering.md`, and the PR description references the weighted-F1 improvement over the v4 baseline.

**Acceptance Scenarios**:

1. **Given** both submission CSVs are created, **When** the run report is finalised, **Then** it clearly identifies which run produced the higher weighted-F1 and that CSV is designated `data/submission_v5_best.csv` (a copy of the best run file).
2. **Given** the PR is created, **When** reviewed, **Then** all new scripts are present, the run report is linked in the PR description, and the best submission CSV is included in the changed files.

---

### Edge Cases

- What happens when a derived raster already exists on disk? The derivation step checks for existence and skips re-computation by default; a `--force` flag re-derives from scratch.
- What if kriging fails to converge for a zone (e.g., too few training points)? Fall back to the nearest-neighbour interpolated value for that zone; log a warning per zone.
- What if a test point falls outside the raster extent? Assign the global median feature value for that feature; log the count of such points.
- What if slope or backscatter zones produce no variance in a spatial block? That feature is excluded from the block's kriging and a warning is emitted.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: The system MUST derive slope, backscatter zones, and acoustic facies rasters from the MBES inputs in a reproducible, scripted manner (re-running the derivation produces consistent outputs given the same random seed).
- **FR-002**: The system MUST sample the three derived rasters at all train and test point coordinates and APPEND the resulting values to the full `experiment_v2.py` feature set (multi-scale bathymetry/backscatter derivatives, curvature, GLCM texture, VRM, x/y coordinates), yielding an augmented per-point feature vector.
- **FR-003**: The system MUST apply approximate relative importance weighting of 80 (slope) / 20 (backscatter zones) / 20 (acoustic facies). Slope (continuous) is scaled by its weight factor directly. Backscatter zones and acoustic facies are first one-hot encoded (5 and 8 clusters respectively), then every resulting binary column is multiplied by the respective weight factor before model training.
- **FR-004**: The system MUST train an ensemble model using spatial-block cross-validation and report per-run weighted-F1 and per-class F1 scores.
- **FR-005**: The system MUST produce a Kaggle-format submission CSV (`ID,class`) for both the raw-features run and the kriging-enhanced run.
- **FR-006**: The system MUST output a run report (Markdown) at `docs/run-008-gis-feature-engineering.md` comparing both runs with: weighted-F1 (including v4 baseline), per-class F1, top features by importance, and runtime.
- **FR-007**: The system MUST designate the best submission as `data/submission_v5_best.csv`.
- **FR-008**: The kriging step MUST apply ordinary kriging ONLY to the three new GIS-derived features (slope, backscatter zone value, acoustic facies value). All other features (the full v2 derivative set) are passed through to the model unchanged. The kriging step MUST fall back gracefully (nearest-neighbour or global median) when a zone has fewer than 10 training points for variogram fitting.
- **FR-009**: All scripts MUST accept the existing `data/MBES/bathymetry.tif` and `data/MBES/backscatter.tif` as source rasters and write outputs under `outputs/` and `data/`.

### Key Entities

- **Derived Raster**: A single-band GeoTIFF produced from MBES inputs. Entities: `slope.tif` (continuous, degrees), `backscatter_zones.tif` (categorical integer, 5 clusters), `acoustic_facies.tif` (categorical integer, 8 clusters).
- **Feature Matrix**: Per-point tabular data comprising the full v2 feature set augmented with the three new GIS-derived feature groups (weighted and one-hot encoded as applicable). One row per sample point.
- **Submission CSV**: Kaggle-format file with columns `ID` and `class`. Two variants: raw-features run and kriging-enhanced run.
- **Run Report**: Markdown document at `docs/run-008-gis-feature-engineering.md` summarising both experimental runs with metrics (including the v4 baseline weighted-F1), parameters, and interpretation.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: The weighted-F1 score of the GIS-feature-engineered model exceeds the v4 baseline weighted-F1 as measured by the same spatial-block CV protocol. The v4 baseline is computed at the start of the experiment script (v2 features only, no GIS features) and documented as a numeric reference point in the run report.
- **SC-002**: Both experiment runs (raw features and kriging-enhanced) complete without unhandled errors on the development machine.
- **SC-003**: The run report clearly identifies which run performs better, and the best submission CSV is correctly designated as `data/submission_v5_best.csv`.
- **SC-004**: All produced submission CSVs contain the correct number of rows (matching `data/test.csv` row count) with no missing or duplicate IDs.
- **SC-005**: The PR is self-contained: a reviewer can checkout the branch and reproduce both runs by following only the instructions in the run report.

## Assumptions

- The three derived rasters (`slope.tif`, `backscatter_zones.tif`, `acoustic_facies.tif`) already exist under `outputs/gis_layers/rasters/` from previous runs; re-derivation logic is added for reproducibility but is not required to re-run for this first experiment attempt.
- The "80/20/20" weighting is applied as pre-training feature scaling: slope is scaled by its weight factor; backscatter zones and acoustic facies are one-hot encoded first, then their binary columns are scaled by the respective weight factor. The ensemble model learns further optimal weights on top of this initial scaling.
- Backscatter zones use 5 k-means clusters; acoustic facies use 8 k-means clusters -- matching the values established in `exp_backscatter_zones.py` and `step1_prepare_gis_layers.py`.
- Kriging is applied ONLY to the three new GIS-derived features (not to the full v2 feature set), per spatial zone, using ordinary kriging with a fallback to nearest-neighbour when a zone has fewer than 10 training points.
- The ensemble model configuration (LightGBM + XGBoost + CatBoost + RandomForest) mirrors `experiment_v2.py`. No architectural changes to the ensemble are in scope.
- Expert annotation layers from branch `007-gis-expert-annotation` are not incorporated in this branch; the annotation workflow remains independent.
- No expert has edited the annotation layers at this stage; the GIS layers are used as-is from the automated derivation in `step1_prepare_gis_layers.py`.
- The target CRS is EPSG:28355 (GDA94 MGA Zone 55), consistent with all prior outputs.
- A PR is created against the main/default branch after both runs complete and the best submission is identified.
