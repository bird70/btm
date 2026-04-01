# Feature Specification: Pipeline CV Improvement Investigation

**Feature Branch**: `019-pipeline-cv-improvement`  
**Created**: 2026-04-01  
**Status**: Draft  
**Input**: User description: "Investigate and improve pipeline cross-validation score beyond R04 baseline through systematic experimentation"

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Systematic CV Improvement Investigation (Priority: P1)

A data scientist wants to determine whether the current best pipeline configuration (R04: RF with BTM-10 features, spatial blocked 5-fold CV = 0.8024, Kaggle = 0.7952) can be meaningfully improved through hyperparameter tuning, feature additions, or model type changes (RF and LightGBM) — while avoiding the trap of submitting changes that only appear better in local CV but degrade on the held-out Kaggle leaderboard.

**Why this priority**: The core objective is improving competition score. Without rigorous local evaluation, any submission risks the catastrophic failures seen previously (e.g., btm_complexity_21 scored 0.6896 on Kaggle despite CV=0.8033).

**Independent Test**: Can be validated by running the pipeline with varied configurations and comparing CV scores with statistical awareness of seed variance.

**Acceptance Scenarios**:

1. **Given** the R04 baseline config and train_btm.csv, **When** alternative hyperparameter configurations are evaluated, **Then** each experiment's CV score and per-class F1 are recorded with the seed used
2. **Given** multiple experiment results, **When** comparing to R04, **Then** improvements smaller than 2× the seed standard deviation (noise floor ≈ 0.035) are flagged as statistically indistinguishable from noise
3. **Given** a candidate improvement, **When** it involves features with large spatial windows (≥15 cells), **Then** it is rejected due to known spatial non-stationarity risk

---

### User Story 2 - Seed Variance Quantification (Priority: P2)

A data scientist wants to understand the random seed sensitivity of the R04 model so that apparent CV improvements can be distinguished from noise.

**Why this priority**: Without quantifying seed variance, small CV changes (e.g., +0.001) cannot be interpreted. This is essential for sound decision-making.

**Independent Test**: Run the same config across multiple seeds and measure the range and standard deviation of CV scores.

**Acceptance Scenarios**:

1. **Given** the R04 config and data, **When** trained with 5 different random seeds, **Then** the CV variance (min, max, mean, std) is reported
2. **Given** seed variance results, **When** a candidate experiment shows CV improvement, **Then** only improvements exceeding 2× the standard deviation are considered potentially meaningful

---

### User Story 3 - Seed Ensemble for Prediction Stability (Priority: P3)

A data scientist wants to know whether ensembling predictions from multiple seed runs produces a more robust submission than any single seed.

**Why this priority**: If individual seeds disagree on a fraction of test samples, majority voting could smooth out noise and produce more reliable predictions.

**Independent Test**: Generate predictions from multiple seed models, compute majority vote, and compare to single-seed predictions.

**Acceptance Scenarios**:

1. **Given** predictions from 5 seed models, **When** majority voting is applied, **Then** the ensemble prediction for each sample is the most common class across seeds
2. **Given** the ensemble predictions, **When** compared to R04 single-seed predictions, **Then** the number of changed predictions and vote confidence distribution are reported

---

### Edge Cases

- What happens when the majority vote is tied (e.g., 2-2-1 with 5 seeds)? Use the prediction from the highest-CV seed as tiebreaker.
- How does the system handle features that are spatially stationary in training but non-stationary in test? Flag any feature where train-test distribution shift exceeds 0.5 standard deviations.
- What happens when a model type (e.g., CatBoost with symmetric trees) shows very high CV but poor Kaggle score? Document the discrepancy and exclude from submission consideration.
- What happens when ALL experiments fail to beat R04 beyond the noise floor? Document findings, confirm R04 remains the best validated configuration, and close the investigation without additional Kaggle submissions.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST run the R04 baseline pipeline configuration and record CV score as the reference benchmark
- **FR-002**: System MUST evaluate at least 5 hyperparameter variations (tree count, depth, leaf size, feature sampling, class weighting) against the R04 baseline using both RF and LightGBM model types
- **FR-003**: System MUST evaluate small-window BTM feature additions (scale ≤ 11 cells) for potential CV improvement
- **FR-004**: System MUST evaluate pipeline feature flags (eco features, spatial z-scores) individually
- **FR-005**: System MUST run the R04 config across at least 5 random seeds to quantify CV variance
- **FR-006**: System MUST generate predictions from each seed model and compute a majority-vote ensemble
- **FR-007**: System MUST NOT submit any configuration to Kaggle unless its CV improvement exceeds 2× the measured seed standard deviation
- **FR-008**: System MUST NOT use large-window BTM features (≥15 cells) in any submission candidate due to known spatial non-stationarity
- **FR-009**: System MUST use at most 2 Kaggle submission slots for this investigation, reserving remaining slots for future work
- **FR-010**: If no experiment exceeds the noise floor, system MUST document findings and confirm R04 as the best validated configuration without consuming Kaggle submissions
- **FR-011**: System MUST document all experiment results with configuration, CV score, per-class F1, and seed used
- **FR-012**: System MUST produce a final submission CSV in the required competition format

### Key Entities

- **Experiment Run**: A single pipeline execution with specific config, data, seed. Produces CV score, per-class metrics, and predictions.
- **Configuration**: A YAML file defining model type, hyperparameters, and feature flags.
- **Submission**: A CSV file mapping test sample IDs to predicted benthic habitat classes.
- **Seed Ensemble**: A set of predictions from multiple seed runs combined via majority voting.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: All planned experiments (hyperparameters, features, seeds) complete and results are documented in a structured report
- **SC-002**: Seed variance is quantified with at least 5 seeds, providing a statistical threshold for meaningful improvement
- **SC-003**: Any submitted configuration demonstrates CV improvement exceeding the noise floor (2× seed standard deviation) or provides a clear rationale (e.g., variance reduction through ensembling)
- **SC-004**: No submission regresses on Kaggle by more than 0.01 compared to the R04 baseline (0.7952)
- **SC-005**: All experiment artifacts (configs, predictions, metrics) are version-controlled for reproducibility

## Assumptions

- The R04 pipeline configuration (RF, BTM-10, spatial blocked 5-fold CV) is the correct baseline, validated at Kaggle = 0.7952
- Spatial blocked cross-validation is the appropriate evaluation strategy for this geospatial dataset
- The training data (train_btm.csv with 14 columns) and test data are fixed for the competition
- Previous findings (CatBoost symmetric tree overfitting, btm_complexity_21 spatial shift, x/y coordinate importance) remain valid
- The competition metric is weighted F1 score
- Kaggle submission slots are limited and should only be used for well-validated candidates
- Maximum 2 Kaggle submissions are budgeted for this investigation
- Deep learning, external data sources, and alternative CV schemes (e.g., stratified random) are out of scope
- Only RF and LightGBM model types are in scope; CatBoost and XGBoost are excluded based on prior findings

## Clarifications

### Session 2026-04-01

- Q: Should the investigation include model types beyond RF? → A: RF + LightGBM (two model types, enables cross-type ensemble)
- Q: What is the exit strategy if no experiment beats R04 beyond the noise floor? → A: Document findings and close; R04 remains best validated configuration
- Q: How many Kaggle submission slots are budgeted? → A: Maximum 2 submissions for this investigation
- Q: What approaches are explicitly out of scope? → A: Deep learning, external data, alternative CV schemes, CatBoost, XGBoost
