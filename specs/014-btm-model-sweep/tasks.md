---
description: "Task list for 014-btm-model-sweep: Systematic BTM Model Sweep"
---

# Tasks: Systematic BTM Model Sweep

**Input**: Design documents from `specs/014-btm-model-sweep/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Organization**: Tasks are grouped by user story. US1 (baseline reproduce + submit)
is the MVP and can be tested independently on Day 1. US2 (feature sweep) depends on
US1's Kaggle score to calibrate comparisons. US3 (model family sweep) depends on the
Phase 2 gate from US2. US4 (run report) depends on all submissions being scored.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Pre-run pipeline extensions — code changes that enable all 14 configs
to run through the existing `benthic_model.cli` without bespoke scripts.

- [x] T001 Scaffold `artifacts/experiments/` dir and commit empty `kaggle_scores.csv` with headers `run_id,submission_file,kaggle_public_f1,submitted_at,message` in `artifacts/experiments/kaggle_scores.csv`
- [X] T002 [P] Add `FeatureFlags` dataclass (4 bool fields, all default `True`) to `src/benthic_model/config.py`
- [X] T003 [P] Add `model_type: str | None = None` and `feature_flags: FeatureFlags | None = None` fields to `PipelineConfig` in `src/benthic_model/config.py`
- [X] T004 Extend `PipelineConfig.from_dict()` to parse `feature_flags` mapping into `FeatureFlags` instance, and validate `model_type` against allowed set `{rf, xgb, lgbm, catboost, rf_lgbm_ensemble}` — raise `ValueError` on unknown value in `src/benthic_model/config.py`
- [X] T005 [P] Add `model_type_used: str` and `feature_flags_used: dict` fields (with `field(default=...)`) to `ExperimentMetadata` in `src/benthic_model/experiment/metadata.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Code changes that ALL user story runs depend on.  
**⚠️ CRITICAL**: All Phase 2 tasks must be complete and all existing tests green before any experiment run.  
**🔴 TDD ORDER**: Write failing tests FIRST, then implement the minimum code to make them pass (Red → Green → Refactor).

### 2a — Write failing tests first (Red phase)

- [X] T006 [P] Write failing tests for `FeatureFlags` and `PipelineConfig` extensions in `tests/unit/test_config_and_metadata.py`: parse config with `model_type`/`feature_flags`, invalid `model_type` raises `ValueError`, absent keys use defaults
- [X] T007 [P] Write failing tests for `engineer_features()` with flags in `tests/unit/test_feature_engineering.py`: each flag `False` individually omits expected columns; all flags `False` leaves only raw MBES core columns
- [X] T008 [P] Write failing tests for new model builders in `tests/unit/test_model_builders.py`: each builder trains on 20-row synthetic 5-class data and `predict()` returns labels in `{ALG, FMAT, NVB, SGAM, SGZ}`
- [X] T009 Run test suite and confirm T006–T008 tests **fail** (Red): `.\.venv\Scripts\pytest.exe tests/unit/test_config_and_metadata.py tests/unit/test_feature_engineering.py tests/unit/test_model_builders.py -x -q`

### 2b — Implement to make tests pass (Green phase)

- [X] T010 [P] Add `FeatureFlags` dataclass (4 bool fields, all default `True`) to `src/benthic_model/config.py`
- [X] T011 [P] Add `model_type: str | None = None` and `feature_flags: FeatureFlags | None = None` fields to `PipelineConfig` in `src/benthic_model/config.py`
- [X] T012 Extend `PipelineConfig.from_dict()` to parse `feature_flags` mapping into `FeatureFlags` instance, and validate `model_type` against allowed set `{rf, xgb, lgbm, catboost, rf_lgbm_ensemble}` — raise `ValueError` on unknown value in `src/benthic_model/config.py`
- [X] T013 [P] Add `model_type_used: str` and `feature_flags_used: dict` fields (with `field(default=...)`) to `ExperimentMetadata` in `src/benthic_model/experiment/metadata.py`
- [X] T014 Extend `engineer_features()` in `src/benthic_model/features/engineering.py` to accept optional `flags: FeatureFlags | None` parameter; when flag is `False` skip: interactions (`bathymetry_x_backscatter`, `acoustic_hardness_proxy`, `relief_index`), spatial z-scores (`add_spatial_context_features()` call), and dynamically-computed focal stats (`*_std_*`, `tpi_*`)
- [X] T015 Extend `_build_model()` in `src/benthic_model/models/train.py` to read `model_type` from config and dispatch to the appropriate builder; fallback to existing `run_type`-based logic when `model_type` is absent
- [X] T016 Extend `train_and_register_run()` in `src/benthic_model/models/train.py` to: (a) pass `FeatureFlags` from parsed config into `engineer_features()`, (b) resolve and record `model_type_used` and `feature_flags_used` in `ExperimentMetadata`
- [X] T017 [P] Add `CandidateLGBMModel` class and `build_lgbm_model()` factory to `src/benthic_model/models/candidate.py` — `LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, class_weight='balanced', random_state=seed)`
- [X] T018 [P] Add `CandidateCatBoostModel` class and `build_catboost_model()` factory to `src/benthic_model/models/candidate.py` — `CatBoostClassifier(iterations=800, depth=7, learning_rate=0.05, auto_class_weights='Balanced', verbose=0, random_seed=seed)`
- [X] T019 [P] Add `CandidateEnsembleModel` class and `build_rf_lgbm_ensemble_model()` factory to `src/benthic_model/models/candidate.py` — soft-vote (equal weights) of one RF + one LightGBM; `predict()` returns label with highest mean probability

### 2c — Contract test and config files

- [X] T020 Extend `tests/contract/test_cli_train_evaluate_contract.py` to train with `configs/rf-core-only.yaml` (once created in T021), assert `feature_flags_used` present in registry entry
- [X] T021 [P] Create `configs/rf-core-only.yaml` (R02) — `model_type: rf`, all feature flags `false` in `configs/rf-core-only.yaml`
- [X] T022 [P] Create `configs/rf-no-interactions.yaml` (R03) — `model_type: rf`, `include_focal_stats: true`, `include_interactions: false`, `include_spatial_z_scores: true` in `configs/rf-no-interactions.yaml`
- [X] T023 [P] Create `configs/rf-btm-fine.yaml` (R04) — `model_type: rf`, all feature flags `false`, `include_btm_features: true` in `configs/rf-btm-fine.yaml`
- [X] T024 [P] Create `configs/rf-btm-broad.yaml` (R05) — `model_type: rf`, all feature flags `false`, `include_btm_features: true` in `configs/rf-btm-broad.yaml`
- [X] T025 [P] Create `configs/rf-btm-full.yaml` (R06) — `model_type: rf`, all feature flags `false`, `include_btm_features: true` in `configs/rf-btm-full.yaml`
- [X] T026 [P] Create `configs/rf-texture.yaml` (R07) — `model_type: rf`, `include_focal_stats: true`, `include_interactions: false`, `include_spatial_z_scores: false` in `configs/rf-texture.yaml`
- [X] T027 [P] Create `configs/rf-btm-texture.yaml` (R08) — `model_type: rf`, `include_focal_stats: true`, `include_interactions: false`, `include_spatial_z_scores: false`, `include_btm_features: true` in `configs/rf-btm-texture.yaml`
- [X] T028 [P] Create `configs/lgbm-candidate.yaml` (R09) — `model_type: lgbm`, placeholder feature flags all `true` (to be updated after Phase 2 gate) in `configs/lgbm-candidate.yaml`
- [X] T029 [P] Create `configs/xgb-candidate.yaml` (R10) — `model_type: xgb`, placeholder feature flags all `true` in `configs/xgb-candidate.yaml`
- [X] T030 [P] Create `configs/catboost-candidate.yaml` (R11) — `model_type: catboost`, placeholder feature flags all `true` in `configs/catboost-candidate.yaml`
- [X] T031 [P] Create `configs/rf-lgbm-ensemble.yaml` (R12) — `model_type: rf_lgbm_ensemble`, placeholder feature flags all `true` in `configs/rf-lgbm-ensemble.yaml`
- [X] T032 [P] Create `configs/rf-spatial-diag.yaml` (R13) — `model_type: rf`, all feature flags `false`, `include_spatial_z_scores: true`, `spatial_coords: true` in `configs/rf-spatial-diag.yaml`
- [X] T033 [P] Create `configs/rf-6fold-baseline.yaml` (R14) — `model_type: rf`, all flags same as `baseline.yaml`, `cv.n_splits: 6` in `configs/rf-6fold-baseline.yaml`
- [X] T034 Run full test suite and confirm all tests pass (Green phase complete): `.\venv\Scripts\pytest.exe tests/ -x -q`

**Checkpoint**: All tests green. Pipeline accepts `model_type` and `feature_flags` from YAML. 13 new configs exist in `configs/`.

---

## Phase 3: User Story 1 — Reproduce and Submit the Baseline (Priority: P1) 🎯 MVP

**Goal**: Establish the true Kaggle score for the existing baseline RF config, creating
the ground-truth anchor for all subsequent comparisons.

**Independent Test**: `run_registry.jsonl` has an entry for R01 with `weighted_f1 ≥ 0.75`;
`data/submission_<R01_RUN_ID>.csv` exists with correct row count; `kaggle_scores.csv`
has a row for the R01 run_id with a real `kaggle_public_f1` value.

### Implementation for User Story 1 (Day 1 — Phase 1 runs)

- [X] T035 [US1] Verify Kaggle CLI auth: run `C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions list` and confirm `geohab-mlwg-competition-2026` appears; set `$env:KAGGLE_API_TOKEN` if not already set
- [X] T036 [US1] Run R01 train: `$env:PYTHONPATH="src"; python -m benthic_model.cli train --train-csv data/train.csv --bathymetry-tif data/MBES/bathymetry.tif --backscatter-tif data/MBES/backscatter.tif --config configs/baseline.yaml --run-type baseline --seed 42`; record `run_id`
- [X] T037 [US1] Run R01 evaluate: `python -m benthic_model.cli evaluate --run-id <R01_RUN_ID> --fold-scheme spatial_blocked`; assert `weighted_f1 ≥ 0.75`
- [X] T038 [US1] Run R01 predict: `python -m benthic_model.cli predict --run-id <R01_RUN_ID> --test-csv data/test.csv --bathymetry-tif data/MBES/bathymetry.tif --backscatter-tif data/MBES/backscatter.tif`; verify output CSV row count matches `data/test.csv`
- [X] T039 [P] [US1] Run R02 train+evaluate+predict using `configs/rf-core-only.yaml` (same data paths as T036–T038); record `run_id`
- [X] T040 [P] [US1] Run R03 train+evaluate+predict using `configs/rf-no-interactions.yaml`; record `run_id`
- [X] T041 [US1] Submit R01 to Kaggle: `C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit -c geohab-mlwg-competition-2026 -f data\submission_<R01_RUN_ID>.csv -m "R01 baseline RF: CV=<F1>"`; wait 90s; fetch score; append to `artifacts/experiments/kaggle_scores.csv`
- [X] T042 [US1] Submit R02 to Kaggle (same pattern as T041, message: `"R02 rf-core-only: CV=<F1>"`); append score to `artifacts/experiments/kaggle_scores.csv`
- [X] T043 [US1] Submit R03 to Kaggle (same pattern as T041, message: `"R03 rf-no-interactions: CV=<F1>"`); append score to `artifacts/experiments/kaggle_scores.csv`
- [X] T044 [US1] Calculate Phase 1 CV–Kaggle gaps for R01, R02, R03; record in `artifacts/experiments/kaggle_scores.csv` comments or a markdown note; confirm Phase 1 gate passes (at least one Kaggle score recorded)

**Checkpoint**: Phase 1 complete. Baseline Kaggle score established. Gap analysis available.

---

## Phase 4: User Story 2 — Sweep Feature Sets Against the RF Baseline (Priority: P2)

**Goal**: Identify which feature set (from R04–R08) produces the smallest CV–Kaggle gap,
and designate it as the Phase 2 winner for model family sweep in US3.

**Independent Test**: All 5 Phase 2 configs have `run_registry.jsonl` entries;
top-3 Kaggle scores are recorded in `kaggle_scores.csv`; a Phase 2 winner is
declared by smallest CV–Kaggle gap.

### Prerequisite: BTM feature extraction (Day 1 pm)

- [X] T045 [US2] Run `btm-export-features` for train points: `btm-export-features --bathy data/MBES/bathymetry.tif --points data/train.csv --broad-inner 10 --broad-outer 30 --fine-inner 1 --fine-outer 5 --outdir data/btm_rasters --output data/train_btm.csv`
- [X] T046 [US2] Run `btm-export-features` for test points: `btm-export-features --bathy data/MBES/bathymetry.tif --points data/test.csv --broad-inner 10 --broad-outer 30 --fine-inner 1 --fine-outer 5 --output data/test_btm.csv`
- [X] T047 [US2] Verify `data/train_btm.csv` has expected `btm_*` columns with no all-NaN columns: `python -c "import pandas as pd; df=pd.read_csv('data/train_btm.csv'); print([c for c in df.columns if c.startswith('btm_')]); print(df[[c for c in df.columns if c.startswith('btm_')]].isna().all()[lambda s: s].index.tolist())"`

### Phase 2 training (Day 2)

- [X] T048 [US2] Run R07 train+evaluate+predict using `configs/rf-texture.yaml` with `--train-csv data/train.csv`; record `run_id`
- [X] T049 [P] [US2] Run R04 train+evaluate+predict using `configs/rf-btm-fine.yaml` with `--train-csv data/train_btm.csv`; record `run_id`
- [X] T050 [P] [US2] Run R05 train+evaluate+predict using `configs/rf-btm-broad.yaml` with `--train-csv data/train_btm.csv`; record `run_id`
- [X] T051 [P] [US2] Run R06 train+evaluate+predict using `configs/rf-btm-full.yaml` with `--train-csv data/train_btm.csv`; record `run_id`
- [X] T052 [P] [US2] Run R08 train+evaluate+predict using `configs/rf-btm-texture.yaml` with `--train-csv data/train_btm.csv`; record `run_id`

### Phase 2 submissions (Day 2 — top 3 by CV, Day 3 — remaining)

- [X] T053 [US2] Rank R04–R08 by CV weighted-F1; submit top 3 to Kaggle (Day 2 budget: max 4 total; use messages `"R0N <config>: CV=<F1>"`); append scores to `artifacts/experiments/kaggle_scores.csv`
- [X] T054 [US2] Submit remaining Phase 2 runs (R04–R08) with CV ≥ 0.76 on Day 3 (alongside T057); append scores
- [X] T055 [US2] After all Phase 2 Kaggle scores received: calculate CV–Kaggle gap per run; identify Phase 2 winner (smallest gap); record winner config name in `artifacts/experiments/kaggle_scores.csv` or a comment in `specs/014-btm-model-sweep/plan.md`
- [X] T056 [US2] Copy Phase 2 winner `feature_flags` block verbatim into `configs/lgbm-candidate.yaml`, `configs/xgb-candidate.yaml`, `configs/catboost-candidate.yaml`, `configs/rf-lgbm-ensemble.yaml`; also copy the correct `--train-csv` path (plain vs BTM-augmented) into the quickstart Phase 3 section

**Checkpoint**: Phase 2 complete. Phase 3 configs updated with winner feature flags.

---

## Phase 5: User Story 3 — Sweep Model Families on the Winning Feature Set (Priority: P3)

**Goal**: Determine whether LightGBM, CatBoost, XGBoost, or a soft-vote ensemble
outperforms the Phase 2 RF winner on the Kaggle leaderboard.

**Independent Test**: All four Phase 3 configs have registry entries; all four
Kaggle scores are recorded; the model-family sweep winner is identified.

### Diagnostic runs (Day 3 — alongside remaining Phase 2)

- [X] T057 [DIAG] Run R13 train+evaluate+predict using `configs/rf-spatial-diag.yaml` with `--train-csv data/train.csv`; record `run_id` (Day 3 alongside remaining Phase 2)
- [X] T058 [DIAG] Submit R13 to Kaggle (Day 3 budget): message `"R13 DIAGNOSTIC spatial-coords RF: CV=<F1> expected Kaggle << CV"`; wait 90s; append score — expected result is a large CV–Kaggle gap confirming spatial memorisation
- [X] T059 Run R14 train+evaluate using `configs/rf-6fold-baseline.yaml` with `--train-csv data/train.csv` (Day 3); compare CV score against R01 CV to test fold-count effect — NO Kaggle submission for R14

### Phase 3 training and submissions (Day 4 — budget: 4 submissions)

- [X] T060 [US3] Run R11 train+evaluate+predict using `configs/catboost-candidate.yaml` with Phase 2 winner `--train-csv`; record `run_id`
- [X] T061 [P] [US3] Run R09 train+evaluate+predict using `configs/lgbm-candidate.yaml` with Phase 2 winner `--train-csv`; record `run_id`
- [X] T062 [P] [US3] Run R10 train+evaluate+predict using `configs/xgb-candidate.yaml` with Phase 2 winner `--train-csv`; record `run_id`
- [X] T063 [P] [US3] Run R12 train+evaluate+predict using `configs/rf-lgbm-ensemble.yaml` with Phase 2 winner `--train-csv`; record `run_id`
- [X] T064 [US3] Submit R11 (CatBoost) to Kaggle first: `C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit -c geohab-mlwg-competition-2026 -f data\submission_<R11_RUN_ID>.csv -m "R11 CatBoost Phase2-winner features: CV=<F1>"`; append score
- [X] T065 [US3] Submit R09 (LightGBM) to Kaggle; append score to `artifacts/experiments/kaggle_scores.csv`
- [X] T066 [US3] Submit R10 (XGBoost) to Kaggle; append score to `artifacts/experiments/kaggle_scores.csv`
- [X] T067 [US3] Submit R12 (RF+LightGBM ensemble) to Kaggle; append score to `artifacts/experiments/kaggle_scores.csv`
- [X] T068 [US3] Identify overall sweep winner (Phase 1–3) by highest Kaggle score; verify CV–Kaggle gap ≤ 0.04 for winner; assert all recorded `kaggle_public_f1` values ≥ 0.65 (SC-006 regression guard — flag any violation immediately); verify at least one run has per-class SGZ F1 ≥ 0.30, noting result in the run summary (SC-004)

**Checkpoint**: All 13 Kaggle submissions made (+R14 CV-only). Sweep winner identified. SC-004 and SC-006 checked.

---

## Phase 6: User Story 4 — Document Findings as a Canonical Run Report (Priority: P4)

**Goal**: Publish a structured run report summarising all run results, the CV–Kaggle
gap analysis, and lessons learned for future branches.

**Independent Test**: `docs/run-013-btm-model-sweep.md` exists and follows the format
of `docs/run-012-v9-catboost-lgb-texture.md`; README best result updated if beaten.

- [X] T069 [US4] Compile final comparative table of all 14 runs: `run_id`, config, model, feature set, CV weighted-F1, Kaggle public F1, CV–Kaggle gap, spatial-coords flag; include per-class SGZ F1 column to surface SC-004 status
- [X] T070 [US4] Write `docs/run-013-btm-model-sweep.md` following `docs/run-012-v9-catboost-lgb-texture.md` format: date, branch, approach summary, CV results table, Kaggle results table, per-class F1 for winner, top feature importances, interpretation, lessons learned
- [X] T071 [US4] If any Kaggle score > 0.76394: update `README.md` "Best result to date" line with new run_id and score in `README.md`
- [X] T072 [US4] Commit all artifacts: `configs/*.yaml`, `artifacts/experiments/kaggle_scores.csv`, `docs/run-013-btm-model-sweep.md`; tag commit `sweep-014-complete`

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T073 [P] Run full test suite one final time: `.\.venv\Scripts\pytest.exe tests/ -x -q`; confirm no regressions introduced by Phase 2 pipeline changes
- [X] T074 [P] Verify `artifacts/experiments/run_registry.jsonl` has entries for all 14 runs; count total lines matches expected
- [X] T075 Update `specs/014-btm-model-sweep/plan.md` Execution Schedule table with actual dates and outcomes

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion — BLOCKS all runs; tests (T006–T009) must go Red before impl (T010–T019 + T034)
- **US1 Phase 3**: Depends on Phase 2 (T034 green); R02 and R03 (T039/T040) can run in parallel with R01 after T036
- **US2 Phase 4**: Depends on Phase 3 (R01 Kaggle score needed as anchor); BTM extraction (T045–T047) can run in parallel with Phase 3 runs
- **US3 Phase 5**: Depends on Phase 2 gate (T055–T056); R09/R10/R12 can run in parallel; R13/R14 (T057–T059) run on Day 3 alongside remaining Phase 2 submissions
- **US4 Phase 6**: Depends on Phase 5 completion (all Kaggle scores recorded)
- **Polish Phase 7**: Depends on Phase 6 completion

### User Story Dependencies

- **US1 (P1)**: Starts immediately after Phase 2 code tasks complete (T034 green)
- **US2 (P2)**: Can execute BTM extraction in parallel with US1 runs; requires R01 Kaggle score before declaring gate pass
- **US3 (P3)**: Strictly depends on US2 Phase 2 gate (winner feature flags must be copied to Phase 3 configs in T056)
- **US4 (P4)**: Strictly depends on all US3 Kaggle submissions being scored (T064–T067 complete)

### Parallel Execution Opportunities

**Within Phase 2 (code changes)**:  
T006 (config tests), T007 (engineering tests), T008 (model builder tests) — write in parallel.  
T010 (FeatureFlags), T017 (LGBM builder), T018 (CatBoost builder), T019 (Ensemble builder), T021–T033 (config files) — all can be implemented in parallel after Red phase confirmed.

**Within Phase 3 (Day 1 runs)**:  
T039 (R02) and T040 (R03) can run in parallel after T036 (R01 train) completes.

**Within Phase 4 (Day 2 runs)**:  
T049 (R04), T050 (R05), T051 (R06), T052 (R08) can run in parallel after T045–T047 complete. T048 (R07) uses plain CSV so can start immediately.

**Within Phase 5 (Day 4 runs)**:  
T061 (R09), T062 (R10), T063 (R12) can run in parallel with T060 (R11).

### Submission Budget Binding

| Day   | Tasks                                 | Max Submissions   |
| ----- | ------------------------------------- | ----------------- |
| Day 1 | T041, T042, T043                      | 3 (R01, R02, R03) |
| Day 2 | T053 (top-3 Phase 2)                  | 3–4               |
| Day 3 | T054 (remaining Phase 2) + T058 (R13) | up to 4           |
| Day 4 | T064, T065, T066, T067 (Phase 3)      | 4                 |

---

## Implementation Strategy

**MVP (Day 0–1)**: Complete all Phase 1–2 code tasks + Phase 3 US1 runs.
After T044, you have a verified Kaggle score for the baseline and can confirm or
disprove the central hypothesis (whether the 0.76394 ceiling is from baseline
features or from OBIA complexity). This is independently verifiable and valuable
even if Phase 2–3 are never executed.

**Incremental delivery**: Each phase gate produces a usable result:

- After Phase 3: baseline Kaggle score known
- After Phase 4: optimal feature set known
- After Phase 5: optimal model family known; SC-004 and SC-006 verified
- After Phase 6: full sweep documented

Total estimated execution: ~12–16 hours over 4 calendar days.
