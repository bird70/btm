# Feature Specification: Ecology-Informed Habitat Feature Engineering

**Feature Branch**: `015-ecology-habitat-features`  
**Created**: 2026-03-31  
**Status**: Draft  
**Input**: User description: "Ecology-informed habitat feature engineering to improve SGAM class detection: add bathymetric derivatives (northness, eastness, complexity, max curvature), ecology-guided depth and BTM zone features for A. antarctica and seagrass habitat. Submit GPU CatBoost run plus 4 new eco-feature experiments. RF + interactions + BTM, RF hyperparameter tuning."

## Background & Ecological Context

The Amphibolis antarctica seagrass class (SGAM) is the hardest class to detect in the current models (per-class F1 ≤ 0.043 in all runs so far). Unlike rigid algorithmic classification (dominant BTM zone, ML rules applied uniformly), the ecology of this class imposes knowable habitat constraints directly derivable from MBES data:

| Class | Substrate                        | Depth character               | Terrain character                                 | Distinguishing terrain signal                              |
| ----- | -------------------------------- | ----------------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| SGAM  | Fine sand / muddy sand           | Shallow, sheltered            | Flat to gently sloping, depositional benthic zone | Low BPI + low slope + low rugosity + specific depth window |
| SGZ   | Fine sand / muddy sand           | Shallow, sheltered            | Flat, bioturbation-active                         | Similar to SGAM but less pronounced mounding               |
| FMAT  | Fine sand / muddy sand           | Sheltered, shallow            | Near-flat                                         | Overlaps SGAM/SGZ spectrally                               |
| NVB   | Coarse sand / gravel             | Variable                      | Flat to moderate                                  | Low backscatter variation, no biota signal                 |
| ALG   | Hard substrate (granite/boulder) | Moderate depth, higher energy | High relief, rugose                               | High rugosity + high slope + high BPI magnitude            |

The reference paper's Table 1 lists six spatial derivatives from MBES bathymetry used in the original study. Four of these — **northness**, **eastness**, **maximum curvature**, and **complexity** — are **not yet computed** in the current feature engineering pipeline but can be derived from the existing bathymetry raster.

The paper further describes a pixel-based (PB) model, object-based (OB) model, and a combined PB+OB model. The current pipeline is pixel-based only. Incorporating the additional spatial derivatives brings the feature set closer to the paper's PB+OB combined model without requiring full image segmentation.

---

## Clarifications

### Session 2026-03-31

- Q: How should SGAM niche indicator thresholds ("low BPI", "low slope", "shallow depth") be defined? → A: Data-driven percentile cutoffs computed from the training set distribution (e.g., BPI < 25th percentile AND slope < 25th percentile AND depth within the observed SGAM depth range), making thresholds fully reproducible and survey-specific.
- Q: How should `btm_depth_zone` be encoded in the feature table? → A: Ordinal integer (1 = shallowest bin, increasing with depth) — a single column preserving rank ordering, handled natively by tree-based models without column-count inflation.
- Q: How many depth zone bins should be used and how should bin boundaries be set? → A: 4 bins at quantile (equal-frequency) boundaries derived from the training depth distribution, ensuring each bin contains training samples. Labels: 1 = very shallow, 2 = shallow, 3 = mid, 4 = deeper.
- Q: Where should the eco-feature extraction code live in the codebase? → A: New dedicated module `src/benthic_model/features/eco_features.py`, following the existing `features/` subpackage pattern alongside `engineering.py` and `spatial_context.py`.
- Q: How should Kaggle submissions be executed (US1, US5)? → A: Kaggle CLI (`kaggle competitions submit -f <csv> -m "<message>"`) invoked from the project root, consistent with how prior submissions were handled.

---

## User Scenarios & Testing _(mandatory)_

### User Story 1 — Submit GPU CatBoost to Kaggle (Priority: P1)

As a researcher, I can generate a Kaggle submission CSV from the existing GPU CatBoost artifact and submit it as part of today's 5-submission batch.

**Why this priority**: The GPU CatBoost model (run `candidate-20260330203952`) achieved the highest CV F1 of any single model in the sweep (0.8139 vs previous best RF+BTM 0.8024). Whether this translates to Kaggle is the most urgent question. The artifact already exists — only a predict + CSV step is needed.

**Independent Test**: Running `btm predict` with run ID `candidate-20260330203952` produces a submission CSV with the same shape as `data/sample_submission.csv`, and the file is accepted by the Kaggle submission API without format errors.

**Acceptance Scenarios**:

1. **Given** the artifact directory `artifacts/runs/candidate-20260330203952/` exists and contains a trained model, **When** prediction is run against `data/test.csv`, **Then** a submission CSV matching the required format is written to `data/`.
2. **Given** the submission CSV is produced, **When** it is uploaded to the Kaggle competition, **Then** a public leaderboard F1 score is returned with no submission errors.
3. **Given** the Kaggle score is returned, **When** it is recorded in `artifacts/experiments/kaggle_scores.csv`, **Then** the row includes `run_id`, `cv_f1`, `kaggle_f1`, `cv_kaggle_gap`, and `notes` columns.

---

### User Story 2 — SGAM-Targeted Ecology Feature Engineering (Priority: P1)

As a researcher, I can add four new bathymetric spatial derivatives (northness, eastness, maximum curvature, complexity) and ecology-informed depth zone indicators to the feature extraction pipeline to give the model signal that distinguishes depositional seagrass habitat from other flat-sediment classes.

**Why this priority**: The current model has nearly zero SGAM recall. SGAM occupies a specific ecological niche — shallow, sheltered, flat, fine-sediment — that is distinguishable in MBES by low BPI + low slope + specific depth range + directional shelter (northness/eastness encode aspect, capturing sheltered south-facing bays in the southern hemisphere). These features are computable from the existing bathymetry raster.

**Independent Test**: Running the feature extraction pipeline with eco-features enabled produces a feature table containing columns `btm_northness`, `btm_eastness`, `btm_max_curvature`, `btm_complexity`, `btm_depth_zone`, and at least one composite column encoding the SGAM ecological niche (e.g., `btm_flat_shallow_zone`). Values are non-NaN for ≥ 90% of all training points.

**Acceptance Scenarios**:

1. **Given** the MBES bathymetry raster, **When** eco-features are extracted, **Then** `btm_northness` = sin(aspect), `btm_eastness` = cos(aspect), computed from a 3×3 analysis window, matching the reference paper specification.
2. **Given** the MBES bathymetry raster, **When** eco-features are extracted, **Then** `btm_complexity` (rate of change of slope) and `btm_max_curvature` are computed at 3×3 neighbourhood and are non-NaN for ≥ 90% of training points.
3. **Given** a depth raster or bathymetry column in the training CSV, **When** eco-features are extracted, **Then** `btm_depth_zone` assigns each point to one of at least 3 ecologically meaningful bins (shallow / mid / deep) based on thresholds drawn from the depth distribution of the training data.
4. **Given** the full eco-feature set, **When** a Random Forest is trained, **Then** feature importances are recorded in metrics and SGAM per-class F1 is logged separately, enabling comparison to the baseline (F1 = 0.043).

---

### User Story 3 — RF + Interactions + BTM Features (Priority: P2)

As a researcher, I can train an RF model that combines the BTM terrain feature set with pairwise interaction terms (previously used only on the core MBES features) to test whether adding feature interactions to the BTM winner feature set further improves performance.

**Why this priority**: Run R04 used BTM features with no interaction terms. Earlier runs showed interactions can increase overfitting. With BTM as the established winner feature set, one controlled experiment combining both is now warranted.

**Independent Test**: A config YAML `rf-btm-interactions.yaml` with both BTM and interaction flags enabled produces a training run whose CV F1 can be directly compared to R04 (BTM only, CV=0.8024) and R02 (interactions only, CV=0.8018).

**Acceptance Scenarios**:

1. **Given** `rf-btm-interactions.yaml` config, **When** training runs, **Then** the feature table includes both BTM columns and pairwise interaction columns.
2. **Given** the run completes, **When** CV F1 and per-class F1 are logged, **Then** results are stored in `run_registry.jsonl` with a clear `notes` field indicating combined feature set.

---

### User Story 4 — RF Hyperparameter Tuning on BTM Feature Set (Priority: P2)

As a researcher, I can run a targeted hyperparameter sweep over RF `n_estimators`, `max_features`, and `min_samples_leaf` on the BTM winner feature set to determine whether default settings are near-optimal or if a tuned configuration yields higher CV weighted F1.

**Why this priority**: All RF runs to date have used default or near-default hyperparameters. With the BTM feature set established, a low-cost tuning step could yield further improvement at no additional data cost.

**Independent Test**: At least one config YAML with non-default RF hyperparameters is trained and logs a CV F1 directly comparable to R04 (CV=0.8024).

**Acceptance Scenarios**:

1. **Given** a tuned RF config (e.g., `n_estimators=500`, `max_features="sqrt"`, `min_samples_leaf=2`), **When** training runs, **Then** per-class and overall weighted F1 are recorded in `run_registry.jsonl`.
2. **Given** at least two hyperparameter configs are evaluated, **When** results are compared, **Then** any config with CV F1 > 0.8024 is flagged as a Kaggle submission candidate with priority.

---

### User Story 5 — 5-Submission Kaggle Batch (Priority: P1)

As a researcher, I can execute today's full daily Kaggle submission budget (5 submissions) covering: (1) GPU CatBoost, (2) RF+BTM+eco-depth zones, (3) RF+BTM+all derivatives, (4) RF+BTM+interactions, (5) RF+BTM hyperparameter tuned — and record all scores before the session ends.

**Why this priority**: Kaggle submissions are rate-limited to 5/day; using the full budget maximises empirical signal collected today.

**Independent Test**: `artifacts/experiments/kaggle_scores.csv` contains 5 new rows dated 2026-03-31, each with a non-null `kaggle_f1` value.

**Acceptance Scenarios**:

1. **Given** 5 trained models with submission CSVs, **When** each is submitted to Kaggle, **Then** each receives a numeric leaderboard score, which is recorded in `kaggle_scores.csv`.
2. **Given** the 5 Kaggle scores, **When** compared to current best (0.79518), **Then** any new best is noted in the run summary document and the `run_registry.jsonl` `notes` field is updated.

---

### Edge Cases

- What if the bathymetry raster lacks pre-computed aspect or curvature bands? Northness, eastness, curvature, and complexity must be derived entirely from the raw depth raster using finite-difference kernels; the extraction code must not assume pre-existing derived bands.
- What if `btm_northness` / `btm_eastness` are near-constant across training points (low aspect variation in a flat survey area)? These features may have near-zero importance; they must not break the pipeline but can be noted as low-information in the run report.
- What if eco-feature extraction produces NaN for ≥ 10% of training points (edge pixels or missing depth values)? NaN handling defaults to `fillna(0)`, consistent with existing BTM feature handling.
- What if the Kaggle daily submission budget is already partially consumed when this session begins? Priority for submission order is: GPU CatBoost > eco-feature RF runs > RF+interactions > RF hyperparameter tuning.
- What if GPU CatBoost Kaggle F1 is lower than RF+BTM (0.79518)? This would suggest that the GPU CV uplift partially reflects in-distribution overfitting from symmetric-tree bias. Document in run notes as a finding rather than a failure.

---

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST generate a prediction CSV from existing artifact `candidate-20260330203952` against `data/test.csv` in the Kaggle submission format.
- **FR-002**: System MUST compute `btm_northness` = sin(aspect) at a 3×3 analysis window from the bathymetry raster. This computation MUST reside in `src/benthic_model/features/eco_features.py`.
- **FR-003**: System MUST compute `btm_eastness` = cos(aspect) at a 3×3 analysis window from the bathymetry raster. This computation MUST reside in `src/benthic_model/features/eco_features.py`.
- **FR-004**: System MUST compute `btm_max_curvature` (maximum of plan curvature and profile curvature) at a 3×3 analysis window. This computation MUST reside in `src/benthic_model/features/eco_features.py`.
- **FR-005**: System MUST compute `btm_complexity` (second derivative of slope / rate of change of slope) at a 3×3 analysis window. This computation MUST reside in `src/benthic_model/features/eco_features.py`.
- **FR-006**: System MUST assign a `btm_depth_zone` ordinal integer feature (1 = very shallow, 2 = shallow, 3 = mid, 4 = deeper) using 4 quantile-based bin boundaries computed from the training depth distribution, ensuring equal sample representation across bins. Boundaries are logged in the run artifact for reproducibility. The integer column (not one-hot) is used directly as a model input.
- **FR-007**: System MUST support a feature flag (`eco_features: true/false`) in config YAMLs to enable/disable eco-feature columns without affecting existing feature sets.
- **FR-008**: System MUST create a composite indicator feature (`btm_sgam_niche`) encoding the joint condition: BPI below the 25th percentile of the training set AND slope below the 25th percentile of the training set AND depth within the observed depth range of SGAM-labelled training points. Percentile thresholds are computed from the training CSV and must be logged with each run for reproducibility.
- **FR-009**: System MUST record per-class F1 (including SGAM class separately) in `metrics.json` for every experiment run in this spec.
- **FR-010**: System MUST create config YAMLs for at least 4 new experiment variants using eco-features on the RF model.
- **FR-011**: System MUST support RF hyperparameter overrides (`n_estimators`, `max_features`, `min_samples_leaf`) via config YAML.
- **FR-012**: System MUST record all Kaggle submissions in `artifacts/experiments/kaggle_scores.csv` with date, run ID, CV F1, Kaggle F1, CV–Kaggle gap, and notes.
- **FR-013**: All Kaggle submissions (US1, US5) MUST be executed via the Kaggle CLI (`kaggle competitions submit -f <csv_path> -m "<run_id>"`) from the project root. The CLI command used MUST be logged alongside the run ID in `kaggle_scores.csv`.

### Key Entities

- **Eco-feature set**: The four new bathymetric spatial derivatives (northness, eastness, max curvature, complexity) plus depth zone and SGAM niche indicator — all computable from the existing bathymetry raster without new survey data.
- **Depth zones**: 4 depth bins encoded as ordinal integers (1 = very shallow, 2 = shallow, 3 = mid, 4 = deeper), with bin boundaries set at the 25th, 50th, and 75th percentiles of bathymetric depth in the training data. Boundaries are computed at fit time and stored in the run artifact for full reproducibility.
- **SGAM niche indicator** (`btm_sgam_niche`): A binary feature set to 1 when BPI < 25th percentile (training set) AND slope < 25th percentile (training set) AND depth falls within the observed depth range of SGAM-labelled training points. Threshold values (p25_bpi, p25_slope, sgam_depth_min, sgam_depth_max) are computed at fit time and stored in the run artifact for full reproducibility.
- **Submission batch**: The ordered set of up to 5 Kaggle submissions for today's budget: (1) GPU CatBoost holdover, (2) RF+BTM+eco-depth zones, (3) RF+BTM+all derivatives, (4) RF+BTM+interactions, (5) RF+BTM hyperparameter tuning.

---

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: GPU CatBoost submission is executed and a numeric leaderboard score is returned without errors.
- **SC-002**: SGAM per-class F1 exceeds 0.043 in at least one eco-feature run (current best across all sweep runs).
- **SC-003**: At least one eco-feature or tuned run achieves overall CV weighted F1 ≥ 0.8024 (current RF+BTM best).
- **SC-004**: All submissions within today's daily budget are executed and recorded with numeric Kaggle F1 scores.
- **SC-005**: Northness, eastness, max curvature, and complexity features are non-NaN for ≥ 90% of training points.
- **SC-006**: The eco-feature extraction pipeline is fully reproducible: given the same bathymetry raster and training CSV, identical feature values are produced on re-run.
- **SC-007**: Per-class F1 for all 5 habitat classes is logged in every run's `metrics.json`, enabling comparison across the full sweep history.

---

## Assumptions

- The existing bathymetry raster (`data/MBES/bathymetry.tif`) contains sufficient spatial resolution to compute second-order finite-difference derivatives (complexity, curvature) at a 3×3 window without prohibitive NaN edge effects.
- Depth values are available per training point, derivable from the bathymetry raster extraction (depth = −elevation for ocean depth data, or direct column if already present in the CSV).
- Ecologically appropriate depth bins for the Fagatele Bay study area will be defined based on inspecting the depth distribution in `data/train.csv` after raster extraction. Exact thresholds are a finding of this work, not a prior constraint.
- The SGAM niche hypothesis (shallow + flat + fine sediment ↔ low BPI + low slope) is a testable starting point; feature importance analysis will confirm or refute it. The hypothesis may be wrong or redundant with existing BTM features without invalidating the feature engineering effort.
- Object-based image analysis (full segmentation-based feature extraction) is explicitly out of scope for this specification. The new derivatives bring the pixel-based pipeline closer to the paper's Table 1 feature set.
- The GPU CatBoost model may not generalise as well as its CV score suggests due to symmetric-tree behaviour differing from the test distribution. A CV–Kaggle gap larger than the RF+BTM gap of 0.007 is considered a valid finding and will be documented, not treated as a system failure.
- Daily Kaggle submission budget is capped at 5 per day. Submission priority order follows User Story 5.
