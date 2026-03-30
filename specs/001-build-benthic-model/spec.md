# Feature Specification: Benthic Habitat Classification Pipeline

**Feature Branch**: `001-build-benthic-model`  
**Created**: 2026-03-24  
**Status**: Draft  
**Input**: User description: "Build an AI/ML model for benthic terrain modelling using train/test CSVs and MBES bathymetry/backscatter GeoTIFF inputs, optimized for weighted F1 competition scoring and CSV submission output."

## Clarifications

### Session 2026-03-24

- Q: Which validation strategy is authoritative for model selection in this geospatial task? -> A: Spatial blocked k-fold is primary; stratified random k-fold is secondary diagnostic.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train and Evaluate Habitat Classifier (Priority: P1)

As a competition participant, I want to train and evaluate a habitat classification pipeline using only the provided train/test tables and MBES rasters so that I can produce valid, high-quality class predictions.

**Why this priority**: This is the core value of the project. Without a valid training and evaluation pipeline, no competitive submission can be generated.

**Independent Test**: Can be fully tested by running the pipeline on the provided training data, producing weighted F1 metrics and a reproducible evaluation report.

**Acceptance Scenarios**:

1. **Given** valid `train.csv`, `test.csv`, `bathymetry.tif`, and `backscatter.tif`, **When** the training workflow is executed, **Then** the system produces a trained candidate model and weighted F1 evaluation results.
2. **Given** a baseline run and a candidate run, **When** evaluation completes, **Then** the system produces a side-by-side metric summary including weighted F1 and per-class F1.
3. **Given** class imbalance in training data, **When** the model is trained, **Then** optimization and evaluation use weighted F1-aligned handling of class imbalance.

---

### User Story 2 - Generate Competition Submission (Priority: P2)

As a competition participant, I want to generate a submission file in the exact required format so that it can be uploaded without manual correction.

**Why this priority**: A model has no competition value unless predictions can be packaged in the required submission format.

**Independent Test**: Can be fully tested by producing predictions for all test IDs and validating that the file format matches the competition template.

**Acceptance Scenarios**:

1. **Given** a trained model and valid test predictors, **When** inference runs, **Then** the system creates a CSV with headers `ID,class` and one prediction for each test ID.
2. **Given** a generated submission file, **When** format checks are run, **Then** it confirms unique IDs, complete row coverage, and only valid class labels.

---

### User Story 3 - Reproduce and Compare Experiments (Priority: P3)

As a project collaborator, I want to reproduce model runs and compare experiments consistently so that improvements are trustworthy and regressions are easy to detect.

**Why this priority**: Reproducibility and traceability are required by the constitution and critical for stable model iteration.

**Independent Test**: Can be fully tested by rerunning an experiment from recorded metadata and verifying that metrics are within defined tolerance.

**Acceptance Scenarios**:

1. **Given** experiment metadata (seed, config, data split reference, code revision), **When** another collaborator reruns the experiment, **Then** reported metrics remain within the documented tolerance band.
2. **Given** multiple experiment runs, **When** comparison artifacts are reviewed, **Then** each run can be traced to the data, configuration, and model output used.

### Edge Cases

- Training or test points falling outside raster extent MUST be handled without crashing, with those rows flagged in output diagnostics.
- Raster cells with missing/no-data values at sampled coordinates MUST trigger deterministic fallback handling and clear reporting.
- Coordinate reference mismatch between tabular coordinates and rasters MUST fail fast with an actionable error message.
- Single-class prediction collapse (all predictions same class) MUST be detected and flagged as invalid candidate behavior.
- Severe class imbalance MUST not suppress minority classes from evaluation reporting.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST ingest the provided files: `train.csv`, `test.csv`, `sample_submission.csv`, `MBES/bathymetry.tif`, `MBES/backscatter.tif`, and `METADATA.MD`, and validate required metadata fields used for CRS and data-processing assumptions.
- **FR-002**: System MUST enforce the competition data-use rule by restricting model inputs to only provided competition data artifacts.
- **FR-003**: System MUST extract raster predictor values at each sample location from both bathymetry and backscatter rasters for train and test datasets.
- **FR-004**: System MUST support derived geospatial predictor generation from MBES rasters, including neighborhood/context features around sample coordinates.
- **FR-005**: System MUST train at least one supervised multi-class classifier to predict benthic habitat class.
- **FR-006**: System MUST evaluate candidate models using weighted F1 as the primary optimization and selection metric.
- **FR-007**: System MUST report per-class performance alongside weighted F1 to expose minority-class behavior.
- **FR-008**: System MUST support baseline-versus-candidate evaluation output for each model-affecting change in both machine-readable (JSON/CSV) and human-readable report formats.
- **FR-009**: System MUST generate test-set predictions and export a submission CSV with required schema `ID,class`.
- **FR-010**: System MUST validate submission outputs for complete ID coverage, uniqueness, and valid class labels.
- **FR-011**: System MUST persist experiment provenance for each promoted run, including config identifier, random seed, data split reference, and code revision reference.
- **FR-012**: System MUST provide clear run summaries for users, including key metrics, output artifact locations, and any data-quality warnings.
- **FR-013**: System MUST use spatial blocked k-fold cross-validation as the authoritative model-selection protocol and MAY run stratified random k-fold only as a secondary diagnostic.

### Non-Functional Requirements *(mandatory)*

- **NFR-001 Code Quality**: All changes MUST pass linting, formatting checks, and automated tests before merge; changed behavior MUST include corresponding tests.
- **NFR-002 Model Performance**: Every candidate model run MUST be compared to a declared baseline using weighted F1 and per-class F1; weighted F1 degradation greater than 0.005 versus baseline MUST block merge unless an explicit approval note and rollback/next-step plan are recorded in the PR evidence.
- **NFR-003 UX Consistency**: User-facing outputs (console messages, reports, generated files, and docs snippets) MUST use consistent terms, metric names, units, and class-label formatting.
- **NFR-004 Reproducibility**: A collaborator MUST be able to rerun a promoted experiment from documented artifacts and reproduce weighted F1 within an absolute tolerance of 0.01.

### Key Entities *(include if feature involves data)*

- **Sample Point**: A record identified by `ID` with `x` and `y` coordinates, sourced from train/test tables.
- **Habitat Class Label**: The categorical response value (`class`) present in training data and predicted for test rows.
- **Raster Predictor Layer**: A geospatial grid layer (bathymetry or backscatter) used to derive predictor variables.
- **Feature Vector**: The combined predictor representation for each sample point (raw extracted raster values plus derived contextual features).
- **Experiment Run Record**: Metadata and metrics for a model run, including baseline/candidate designation and provenance attributes.
- **Submission Artifact**: The final CSV file containing `ID,class` predictions for all test observations.

## Assumptions

- Training and test coordinate fields are in UTM Zone 55S and align to MBES raster coordinate reference as described in metadata.
- Competition class vocabulary is fully represented by training labels and must be reused exactly in submissions.
- Competition scoring uses weighted F1 on hidden test labels; local validation is an estimate only.
- Spatial blocked k-fold is the canonical local-validation method for model selection; stratified random k-fold is used only for secondary comparison.
- Initial feature set includes raster value extraction and local neighborhood-derived descriptors; additional complex feature families are optional.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of train and test rows are processed into model-ready feature vectors or explicitly flagged with reason codes in diagnostics.
- **SC-002**: Baseline and candidate weighted F1 plus per-class F1 are reported for every model-affecting run in a single comparison artifact.
- **SC-003**: Submission generation completes with 100% test ID coverage, 0 duplicate IDs, and 0 invalid class labels.
- **SC-004**: Re-running the same promoted experiment configuration reproduces weighted F1 within ±0.01.
- **SC-005**: 100% of relevant CI quality gates (lint, format, tests) pass for merged changes.
- **SC-006**: Before final competition submission, either (a) at least one candidate iteration improves baseline weighted F1 by >= 0.005, or (b) baseline is explicitly selected with documented rationale and next-step mitigation plan.
- **SC-007**: User-facing run outputs pass a documented UX consistency checklist with no unresolved critical issues.
