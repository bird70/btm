# Tasks: Hybrid Segmentation Ensemble

**Input**: Design documents from `/specs/021-hybrid-segmentation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare dependencies, configs, and runnable scaffolding for hybrid segmentation work

- [ ] T001 Add segmentation and tooling dependencies (`torch`, `transformers`, `datasets`, `evaluate`, `huggingface_hub`, `pydensecrf`, `kaggle`) to `pyproject.toml`
- [ ] T002 [P] Create hybrid segmentation config skeleton in `configs/segmentation-hybrid.yaml`
- [ ] T003 [P] Add segmentation artifacts directories and README placeholders in `artifacts/segmentation/.gitkeep` and `artifacts/segmentation/README.md`
- [ ] T004 Document local `.venv` activation and package install commands for this feature in `specs/021-hybrid-segmentation/quickstart.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build shared contracts, schemas, and CLI wiring required by all user stories

**⚠️ CRITICAL**: No user story work begins until this phase is complete

- [ ] T005 Create segmentation package scaffold in `src/benthic_model/segmentation/__init__.py`
- [ ] T006 [P] Implement segmentation artifact schema validators in `src/benthic_model/segmentation/io.py`
- [ ] T007 [P] Implement class-weight strategy helper (`inverse_freq_capped`) in `src/benthic_model/segmentation/weights.py`
- [ ] T008 [P] Add shared segmentation dataclasses for run metadata in `src/benthic_model/segmentation/types.py`
- [ ] T009 Add CLI command stubs for `segmentation-benchmark` and `hybrid-stack` in `src/benthic_model/cli.py`
- [ ] T010 Add contract tests for required CLI arguments and output artifacts in `tests/contract/test_hybrid_segmentation_cli_contract.py`

**Checkpoint**: Foundation complete, story work can proceed

---

## Phase 3: User Story 1 - Build Segmentation Masks (Priority: P1) 🎯 MVP

**Goal**: Convert point labels into deterministic 5x5 training masks

**Independent Test**: Run mask generation on fixture points and verify every label maps to a centered 5x5 region with deterministic overlap behavior

### Tests for User Story 1 (write first, must fail first)

- [ ] T011 [P] [US1] Add unit tests for centered 5x5 expansion rules in `tests/unit/test_segmentation_masking.py`
- [ ] T012 [P] [US1] Add integration test for mask generation from labeled points in `tests/integration/test_segmentation_mask_pipeline.py`

### Implementation for User Story 1

- [ ] T013 [US1] Implement point-to-pixel index mapping utilities in `src/benthic_model/segmentation/grid_mapping.py`
- [ ] T014 [US1] Implement 5x5 mask builder with deterministic overlap policy in `src/benthic_model/segmentation/masking.py`
- [ ] T015 [US1] Implement mask artifact writer/loader (`mask_metadata.json`, arrays) in `src/benthic_model/segmentation/io.py`
- [ ] T016 [US1] Add `segmentation-build-masks` CLI command flow in `src/benthic_model/cli.py`
- [ ] T017 [US1] Document mask command usage and expected outputs in `specs/021-hybrid-segmentation/quickstart.md`

**Checkpoint**: US1 independently testable via mask generation command and tests

---

## Phase 4: User Story 2 - Train and Compare Low-Effort Segmentation Paths (Priority: P2)

**Goal**: Benchmark exactly three candidates: SegFormer, SegFormer+CRF, and DeepLabV3+

**Independent Test**: Execute benchmark run and verify candidate metrics report contains exactly 3 required candidates and required metrics

### Tests for User Story 2 (write first, must fail first)

- [ ] T018 [P] [US2] Add unit tests for candidate registry enforcing exactly three approaches in `tests/unit/test_segmentation_candidates.py`
- [ ] T019 [P] [US2] Add integration test for benchmark artifact generation in `tests/integration/test_segmentation_benchmark_pipeline.py`
- [ ] T020 [P] [US2] Add contract test for `val_location_predictions.csv` and `test_location_predictions.csv` schema in `tests/contract/test_segmentation_artifact_schema.py`

### Implementation for User Story 2

- [ ] T021 [US2] Implement candidate runner orchestration (`segformer_ft`, `segformer_ft_crf`, `deeplabv3_ft`) in `src/benthic_model/segmentation/benchmark.py`
- [ ] T022 [P] [US2] Implement Hugging Face SegFormer fine-tuning adapter in `src/benthic_model/segmentation/segformer_adapter.py`
- [ ] T023 [P] [US2] Implement DeepLabV3+ local training adapter in `src/benthic_model/segmentation/deeplab_adapter.py`
- [ ] T024 [P] [US2] Implement CRF post-processing wrapper for SegFormer outputs in `src/benthic_model/segmentation/crf.py`
- [ ] T025 [US2] Wire `segmentation-benchmark` CLI command to produce candidate metrics and location outputs in `src/benthic_model/cli.py`

**Checkpoint**: US2 independently testable with benchmark command and artifact schema checks

---

## Phase 5: User Story 3 - Stack Segmentation with Existing RF/MLP Ensemble (Priority: P3)

**Goal**: Combine RF/MLP and segmentation outputs using multinomial logistic regression and enforce promotion gate

**Independent Test**: Run hybrid stacking on validation artifacts and confirm promotion gate logic (`weighted_f1_delta >= 0.02` and `sgam_recall_delta >= 0.0`)

### Tests for User Story 3 (write first, must fail first)

- [ ] T026 [P] [US3] Add unit tests for meta-feature assembly and alignment in `tests/unit/test_hybrid_meta_features.py`
- [ ] T027 [P] [US3] Add unit tests for promotion gate logic in `tests/unit/test_hybrid_promotion_gate.py`
- [ ] T028 [P] [US3] Add integration test for end-to-end hybrid stacking output artifacts in `tests/integration/test_hybrid_stacking_pipeline.py`

### Implementation for User Story 3

- [ ] T029 [US3] Implement meta-feature dataset builder from RF/MLP/segmentation probabilities in `src/benthic_model/segmentation/stacking_features.py`
- [ ] T030 [US3] Implement multinomial logistic regression stacker training/inference in `src/benthic_model/segmentation/stacking.py`
- [ ] T031 [US3] Implement promotion decision artifact writer (`promotion_decision.json`) in `src/benthic_model/segmentation/io.py`
- [ ] T032 [US3] Wire `hybrid-stack` CLI command with baseline fallback behavior in `src/benthic_model/cli.py`

**Checkpoint**: US3 independently testable with hybrid-stack command and promotion-decision artifact

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final hardening, reproducibility, and documentation across stories

- [ ] T033 [P] Add experiment runbook for Kaggle/HF/gh CLI workflow in `docs/runsheet-hybrid-segmentation.md`
- [ ] T034 [P] Add regression test matrix entry for segmentation feature set in `TESTING.md`
- [ ] T035 Run quickstart validation and record exact command transcript in `specs/021-hybrid-segmentation/quickstart.md`
- [ ] T036 Summarize benchmark outcomes and recommendation in `docs/run-021-hybrid-segmentation.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: Start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1, blocks all user stories
- **Phase 3 (US1)**: Depends on Phase 2
- **Phase 4 (US2)**: Depends on Phase 2 (can use fixture masks for independent testing), typically executed after US1 for production workflow
- **Phase 5 (US3)**: Depends on Phase 2 and segmentation artifacts from US2
- **Phase 6 (Polish)**: Depends on completion of targeted user stories

### User Story Dependencies

- **US1 (P1)**: No dependency on other user stories
- **US2 (P2)**: Independent test path via fixture masks; production path consumes US1 mask outputs
- **US3 (P3)**: Consumes outputs from US2 and baseline RF/MLP predictions

### Within Each User Story

- Tests must be written and failing before implementation
- Data/schema handling before CLI wiring
- Core implementation before end-to-end integration checks

---

## Parallel Opportunities

- Setup: T002 and T003 can run in parallel
- Foundational: T006, T007, and T008 can run in parallel
- US1: T011 and T012 can run in parallel
- US2: T018, T019, and T020 can run in parallel; T022, T023, and T024 can run in parallel after T021 starts interfaces
- US3: T026, T027, and T028 can run in parallel; T029 and T031 can run in parallel after schemas stabilize
- Polish: T033 and T034 can run in parallel

---

## Parallel Example: User Story 2

```bash
# Parallel test creation
T018 tests/unit/test_segmentation_candidates.py
T019 tests/integration/test_segmentation_benchmark_pipeline.py
T020 tests/contract/test_segmentation_artifact_schema.py

# Parallel adapter implementation
T022 src/benthic_model/segmentation/segformer_adapter.py
T023 src/benthic_model/segmentation/deeplab_adapter.py
T024 src/benthic_model/segmentation/crf.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2
2. Deliver US1 mask generation end-to-end
3. Validate US1 independently via integration test and CLI run

### Incremental Delivery

1. Add US2 benchmarking and validate required three-candidate report
2. Add US3 stacking and promotion gating
3. Finish with Phase 6 docs and runbook evidence

### Team Parallel Strategy

1. One developer handles Foundational + CLI contracts
2. One developer handles US1 mask pipeline
3. One developer handles US2 model adapters
4. One developer handles US3 stacker and gating

