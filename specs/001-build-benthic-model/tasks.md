# Tasks: Benthic Habitat Classification Pipeline

**Input**: Design documents from `/specs/001-build-benthic-model/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included because the specification explicitly requires automated tests for changed behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize project skeleton, Python environment workflow, and quality tooling.

- [X] T001 Create package directory scaffold in src/benthic_model/__init__.py
- [X] T002 Create data and artifact root placeholders in data/.gitkeep
- [X] T003 [P] Create experiment artifact directory placeholder in artifacts/.gitkeep
- [X] T004 [P] Create report directory placeholder in reports/.gitkeep
- [X] T005 [P] Create submission directory placeholder in submissions/.gitkeep
- [X] T006 [P] Create notebook workspace placeholder in notebooks/.gitkeep
- [X] T007 Define Python dependencies with pinned critical versions in ./requirements.txt
- [X] T008 [P] Configure lint and format settings in ./pyproject.toml
- [X] T009 [P] Configure pytest discovery in ./pytest.ini
- [X] T010 [P] Add environment bootstrap script in scripts/setup_env.sh
- [X] T061 [P] Add pinned-dependency validation script in scripts/check_pinned_deps.sh

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build shared infrastructure required by all user stories.

**⚠️ CRITICAL**: No user story implementation begins until this phase is complete.

- [X] T011 Implement global pipeline configuration schema in src/benthic_model/config.py
- [X] T012 [P] Implement shared data path resolver in src/benthic_model/data/ingest.py
- [X] T062 [P] Implement competition-data whitelist guard for allowed input artifacts in src/benthic_model/data/ingest.py
- [X] T063 [P] Implement METADATA.MD parser and required-field validation in src/benthic_model/data/ingest.py
- [X] T013 [P] Implement coordinate and CRS validation utilities in src/benthic_model/data/validation.py
- [X] T014 [P] Implement experiment metadata schema and serializer in src/benthic_model/experiment/metadata.py
- [X] T015 [P] Implement run registry persistence utilities in src/benthic_model/experiment/registry.py
- [X] T016 Implement weighted F1 and per-class metric helpers in src/benthic_model/evaluation/metrics.py
- [X] T017 Implement spatial blocked fold generator in src/benthic_model/evaluation/cv.py
- [X] T018 [P] Implement stratified random fold generator for diagnostics in src/benthic_model/evaluation/cv.py
- [X] T019 [P] Create UX consistency checklist template in reports/templates/ux_consistency_checklist.md
- [X] T020 Implement CLI command routing skeleton for train/evaluate/predict/make-submission in src/benthic_model/cli.py
- [X] T021 [P] Add foundational unit tests for config and metadata schema in tests/unit/test_config_and_metadata.py
- [X] T022 [P] Add foundational unit tests for fold generation in tests/unit/test_cv_protocols.py
- [X] T064 [P] Add unit tests for competition-data whitelist enforcement in tests/unit/test_data_use_restrictions.py
- [X] T065 [P] Add unit tests for METADATA.MD required-field validation in tests/unit/test_metadata_ingestion.py

**Checkpoint**: Foundation complete; user story work can proceed.

---

## Phase 3: User Story 1 - Train and Evaluate Habitat Classifier (Priority: P1) 🎯 MVP

**Goal**: Train baseline/candidate classifiers from MBES-derived features and evaluate with authoritative spatial blocked CV.

**Independent Test**: Run training and evaluation on sample competition-format data and produce weighted F1 + per-class F1 comparison report.

### Tests for User Story 1

- [X] T023 [P] [US1] Add contract tests for train/evaluate CLI commands in tests/contract/test_cli_train_evaluate_contract.py
- [X] T024 [P] [US1] Add integration test for train-to-evaluate flow in tests/integration/test_training_evaluation_pipeline.py
- [X] T025 [P] [US1] Add unit tests for raster extraction edge cases in tests/unit/test_raster_extract.py
- [X] T026 [P] [US1] Add unit tests for feature engineering outputs in tests/unit/test_feature_engineering.py

### Implementation for User Story 1

- [X] T027 [P] [US1] Implement raster point sampling for bathymetry and backscatter in src/benthic_model/data/raster_extract.py
- [X] T028 [P] [US1] Implement neighborhood and interaction feature engineering in src/benthic_model/features/engineering.py
- [X] T029 [P] [US1] Implement spatial context feature utilities in src/benthic_model/features/spatial_context.py
- [X] T030 [P] [US1] Implement baseline random forest trainer in src/benthic_model/models/baseline.py
- [X] T031 [P] [US1] Implement candidate XGBoost trainer in src/benthic_model/models/candidate.py
- [X] T032 [US1] Implement unified training orchestrator in src/benthic_model/models/train.py
- [X] T033 [US1] Implement evaluation report comparison writer in src/benthic_model/evaluation/compare.py
- [X] T066 [US1] Implement machine-readable metric summary writer (JSON/CSV) in src/benthic_model/evaluation/compare.py
- [X] T034 [US1] Implement train command workflow in src/benthic_model/cli.py
- [X] T035 [US1] Implement evaluate command workflow enforcing spatial_blocked selection in src/benthic_model/cli.py
- [X] T036 [US1] Persist baseline-vs-candidate metric evidence in reports/metrics/README.md
- [X] T037 [US1] Add notebook workflow for feature and model experimentation in notebooks/03_model_experiments.ipynb
- [X] T067 [US1] Enforce weighted F1 degradation threshold policy (>0.005 requires explicit override evidence) in src/benthic_model/evaluation/compare.py
- [X] T068 [US1] Add contract test for machine-readable metrics output schema in tests/contract/test_metrics_output_contract.py
- [X] T069 [US1] Add unit tests for weighted F1 degradation-threshold enforcement in tests/unit/test_performance_threshold_policy.py

**Checkpoint**: User Story 1 delivers an MVP training and evaluation pipeline.

---

## Phase 4: User Story 2 - Generate Competition Submission (Priority: P2)

**Goal**: Generate valid `ID,class` submission files from trained model predictions.

**Independent Test**: Produce submission from test data and validate schema, ID coverage, uniqueness, and class vocabulary constraints.

### Tests for User Story 2

- [X] T038 [P] [US2] Add contract tests for predict and make-submission CLI commands in tests/contract/test_cli_predict_submission_contract.py
- [X] T039 [P] [US2] Add integration test for prediction-to-submission workflow in tests/integration/test_submission_generation_pipeline.py
- [X] T040 [P] [US2] Add unit tests for submission validation rules in tests/unit/test_submission_writer.py

### Implementation for User Story 2

- [X] T041 [P] [US2] Implement prediction pipeline for test samples in src/benthic_model/inference/predict.py
- [X] T042 [P] [US2] Implement submission writing and validation logic in src/benthic_model/submission/writer.py
- [X] T043 [US2] Implement predict command workflow in src/benthic_model/cli.py
- [X] T044 [US2] Implement make-submission command workflow in src/benthic_model/cli.py
- [X] T045 [US2] Add submission schema and contract documentation in submissions/README.md
- [X] T046 [US2] Add notebook walkthrough for submission generation checks in notebooks/02_feature_exploration.ipynb

**Checkpoint**: User Story 2 produces competition-ready submission artifacts independently.

---

## Phase 5: User Story 3 - Reproduce and Compare Experiments (Priority: P3)

**Goal**: Ensure experiment traceability and reproducibility for trusted model iteration.

**Independent Test**: Re-run a recorded experiment and confirm weighted F1 falls within tolerance while preserving provenance links.

### Tests for User Story 3

- [X] T047 [P] [US3] Add integration test for experiment rerun reproducibility in tests/integration/test_experiment_reproducibility.py
- [X] T048 [P] [US3] Add unit tests for run registry query and lineage in tests/unit/test_experiment_registry.py
- [X] T049 [P] [US3] Add contract test for reproducibility metadata fields in tests/contract/test_experiment_metadata_contract.py

### Implementation for User Story 3

- [X] T050 [P] [US3] Implement experiment comparison and drift summary utilities in src/benthic_model/experiment/compare_runs.py
- [X] T051 [P] [US3] Implement reproducibility report writer in src/benthic_model/evaluation/reproducibility.py
- [X] T052 [US3] Extend train/evaluate workflows to persist full provenance metadata in src/benthic_model/cli.py
- [X] T053 [US3] Create experiment runbook and replay instructions in reports/reproducibility/README.md
- [X] T054 [US3] Add notebook for reproducibility and comparison diagnostics in notebooks/01_data_validation.ipynb

**Checkpoint**: User Story 3 provides reproducible, traceable experiment lifecycle management.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Strengthen quality, documentation, and release-readiness across all stories.

- [X] T055 [P] Update project usage and architecture documentation in ./README.md
- [X] T056 Run full quality gate and capture results in reports/quality/ci_gate_report.md
- [X] T070 Run pinned-dependency validation and capture output in reports/quality/dependency_pinning_report.md
- [X] T057 [P] Add end-to-end smoke test covering train/evaluate/predict/submission in tests/integration/test_full_pipeline_smoke.py
- [X] T058 Aggregate final baseline-versus-best-candidate summary in reports/metrics/final_model_summary.md
- [X] T059 Validate final UX consistency checklist completion in reports/ux/final_checklist.md
- [X] T060 Validate quickstart commands and expected outputs in specs/001-build-benthic-model/quickstart.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: Can start immediately.
- **Phase 2 (Foundational)**: Depends on Phase 1 and blocks all user stories.
- **Phase 3 (US1)**: Depends on Phase 2.
- **Phase 4 (US2)**: Depends on Phase 2; can run in parallel with US1 after shared interfaces exist.
- **Phase 5 (US3)**: Depends on Phase 2; benefits from US1 outputs but test scaffolding can start earlier.
- **Phase 6 (Polish)**: Depends on completion of desired user stories.

### User Story Dependencies

- **US1 (P1)**: No dependency on other user stories.
- **US2 (P2)**: No hard dependency on US1 for contract/unit validation; full integration uses trained model artifacts.
- **US3 (P3)**: Uses provenance captured by US1/US2 flows for full value.

### Within Each User Story

- Tests first (must fail before implementation).
- Data and model primitives before orchestration.
- CLI integration after core modules.
- Evidence/report tasks after implementation passes tests.

### Parallel Opportunities

- Setup tasks marked `[P]` can run concurrently.
- Foundational utilities in separate modules (`config`, `validation`, `registry`, `cv`) can run concurrently.
- In US1, raster extraction, feature engineering, and model implementations can run concurrently.
- In US2, prediction and submission writer modules can run concurrently.
- In US3, comparison utilities and reproducibility report writer can run concurrently.

---

## Parallel Example: User Story 1

```bash
Task: "T027 [US1] Implement raster point sampling in src/benthic_model/data/raster_extract.py"
Task: "T028 [US1] Implement neighborhood and interaction feature engineering in src/benthic_model/features/engineering.py"
Task: "T030 [US1] Implement baseline random forest trainer in src/benthic_model/models/baseline.py"
Task: "T031 [US1] Implement candidate XGBoost trainer in src/benthic_model/models/candidate.py"
```

## Parallel Example: User Story 2

```bash
Task: "T041 [US2] Implement prediction pipeline in src/benthic_model/inference/predict.py"
Task: "T042 [US2] Implement submission writing and validation in src/benthic_model/submission/writer.py"
Task: "T040 [US2] Add unit tests for submission validation in tests/unit/test_submission_writer.py"
```

## Parallel Example: User Story 3

```bash
Task: "T050 [US3] Implement run comparison utilities in src/benthic_model/experiment/compare_runs.py"
Task: "T051 [US3] Implement reproducibility report writer in src/benthic_model/evaluation/reproducibility.py"
Task: "T048 [US3] Add unit tests for experiment registry in tests/unit/test_experiment_registry.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2.
2. Deliver Phase 3 (US1) as the MVP.
3. Validate weighted F1 and per-class F1 reporting and reproducibility metadata.

### Incremental Delivery

1. Deliver US1 training/evaluation pipeline.
2. Add US2 submission generation and validation.
3. Add US3 reproducibility and experiment comparison.
4. Finish with polish and quality evidence.

### Parallel Team Strategy

1. Developer A: data/features path (US1).
2. Developer B: CLI + submission path (US2).
3. Developer C: experiment registry and reproducibility path (US3).
4. Coordinate through shared foundational contracts from Phase 2.
