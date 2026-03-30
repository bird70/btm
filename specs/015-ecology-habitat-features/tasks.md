# Tasks: Ecology-Informed Habitat Feature Engineering

**Branch**: `015-ecology-habitat-features` | **Date**: 2026-03-31  
**Input**: Design documents from `specs/015-ecology-habitat-features/`  
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no unmet dependencies)
- **[Story]**: Which user story this task belongs to
- Exact file paths included in all descriptions

---

## Phase 1: Setup

**Purpose**: Verify the working branch and existing artifact before any code changes

- [X] T001 Confirm branch `015-ecology-habitat-features` is checked out and `artifacts/runs/candidate-20260330203952/model.joblib` exists locally

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Config + data-model changes that all user stories depend on. No experiment configs or eco-feature code can work until these are done.

**⚠️ CRITICAL**: Must be complete before US2–US5 implementation begins

- [X] T002 Add `include_eco_features: bool = False` field to `FeatureFlags` dataclass in `src/benthic_model/config.py`
- [X] T003 Add `model_params: dict[str, Any] | None = None` field to `PipelineConfig` dataclass in `src/benthic_model/config.py`; update `from_dict()` to pass it through
- [X] T004 Write failing tests for `include_eco_features` flag and `model_params` field in `tests/unit/test_config_and_metadata.py` (TDD Red phase)
- [X] T005 Run `pytest tests/unit/test_config_and_metadata.py` to confirm T004 tests fail (Red confirmed)
- [X] T006 Implement T002 + T003 changes and verify T004 tests pass (Green)
- [X] T007 Run full test suite `pytest tests/ -q` and confirm 0 regressions
- [X] T060 [US2/US3/US4] Verify `src/benthic_model/models/train.py` writes per-class F1 for all 5 habitat classes to `metrics.json`; if absent, extend metrics output before any training runs execute (SC-007 prerequisite — blocks all training tasks T008, T033, T038, T043, T044)

**Checkpoint**: Config changes done; all existing 154 tests still pass; per-class F1 in `metrics.json` confirmed for all run types

---

## Phase 3: User Story 1 — GPU CatBoost Kaggle Submission (Priority: P1) 🎯

**Goal**: Submit the already-trained GPU CatBoost model to Kaggle and record the score

**Independent Test**: `data/submission_catboost_gpu.csv` exists, has same row count as `data/sample_submission.csv`, and a Kaggle leaderboard score appears in `artifacts/experiments/kaggle_scores.csv`

### Implementation for User Story 1

- [ ] T008 [US1] Run `benthic-model predict --run-id candidate-20260330203952 --test-csv data/test.csv --output data/submission_catboost_gpu.csv` and verify shape matches `data/sample_submission.csv`
- [ ] T009 [US1] Submit to Kaggle: `kaggle competitions submit -c <competition> -f data/submission_catboost_gpu.csv -m "candidate-20260330203952 CatBoost GPU symmetric-tree CV=0.8139"` and record returned score
- [ ] T010 [US1] Add row to `artifacts/experiments/kaggle_scores.csv`: date=2026-03-31, run_id=candidate-20260330203952, cv_f1=0.8139, kaggle_f1=<returned>, cv_kaggle_gap=<gap>, notes="CatBoost GPU task_type=GPU symmetric trees"

**Checkpoint**: Submission 1 of 5 complete; leaderboard score recorded

---

## Phase 4: User Story 2 — Eco-Feature Engineering (Priority: P1)

**Goal**: New `eco_features.py` module + raster extraction producing all 6 eco-feature columns, gated by `include_eco_features` flag

**Independent Test**: `EcoFeatureTransformer().fit_transform(df, y)` with a synthetic DataFrame adds `btm_depth_zone` (int 1–4) and `btm_sgam_niche` (0/1); `extract_eco_raster_features()` produces 4 new columns with NaN rate ≤ 10%

### Tests for User Story 2 (TDD Red phase first) ⚠️

> **Write these tests FIRST — all must FAIL before implementation**

- [ ] T011 [P] [US2] Write failing tests for `EcoFeatureTransformer.fit()`: fit from a DataFrame with `bathymetry`, `btm_fine_bpi`, `btm_slope` + SGAM labels stores `depth_bin_edges` (len=5), `p25_bpi`, `p25_slope`, `sgam_depth_min`, `sgam_depth_max` in `tests/unit/test_eco_features.py`
- [ ] T012 [P] [US2] Write failing tests for `EcoFeatureTransformer.transform()`: adds `btm_depth_zone` ∈ {1,2,3,4} and `btm_sgam_niche` ∈ {0,1}; raises `sklearn.exceptions.NotFittedError` when called before `fit()` in `tests/unit/test_eco_features.py`
- [ ] T013 [P] [US2] Write failing tests for `EcoFeatureTransformer.to_dict()` / `from_dict()` round-trip: serialise then reconstruct; transformed output is identical in `tests/unit/test_eco_features.py`
- [X] T014 [P] [US2] Write failing tests for `FeatureFlags.include_eco_features` integration in `engineer_features()`: when `True`, output DataFrame contains `btm_depth_zone`; when `False` (default), output DataFrame does not contain it in `tests/unit/test_feature_engineering.py`
- [ ] T015 [US2] Run `pytest tests/unit/test_eco_features.py tests/unit/test_feature_engineering.py -q` — confirm all new tests fail (Red phase)

### Implementation for User Story 2

- [X] T016 [US2] Create `src/benthic_model/features/eco_features.py` with `EcoFeatureTransformer` class: `N_DEPTH_BINS = 4` constant; `fit(df, y)` computing depth quantiles + p25_bpi + p25_slope + SGAM depth range; `transform(df)` adding `btm_depth_zone` and `btm_sgam_niche`; `to_dict()` / `from_dict()`; docstrings citing Wilson et al. 2007
- [X] T017 [US2] Modify `src/benthic_model/features/engineering.py`: import `EcoFeatureTransformer`; when `flags.include_eco_features` is True, call `EcoFeatureTransformer.fit_transform(features, y=None)` and append results (pass y through from caller if available)
- [X] T018 [US2] Run `pytest tests/unit/test_eco_features.py tests/unit/test_feature_engineering.py -q` — confirm all new tests pass (Green phase)
- [X] T019 [P] [US2] Add `compute_northness_eastness(dem, cell_size)` function to `btm/features/extract.py`: Horn (1981) gradient kernels via `scipy.ndimage.convolve`; returns `(northness, eastness)` tuple; NaN where flat; docstring citing Horn 1981 + Wilson 2007
- [X] T020 [P] [US2] Add `compute_max_curvature(dem, cell_size)` function to `btm/features/extract.py`: Hessian eigenvalue approach (Evans 1980 / Schmidt 2003); return max(|k1|, |k2|); always defined (no NaN at zero-slope); docstring citing Schmidt 2003
- [X] T021 [P] [US2] Add `compute_complexity(dem, cell_size)` function to `btm/features/extract.py`: slope-of-slope (apply Horn slope kernel twice); return absolute value; docstring citing Wilson 2007
- [X] T022 [US2] Add `extract_eco_raster_features(points, bathy_tif)` function to `btm/features/extract.py`: open raster → read full band as NumPy array → call T019–T021 → sample at point (row,col) indices → return DataFrame with added `btm_northness`, `btm_eastness`, `btm_max_curvature`, `btm_complexity` columns; NaN → 0.0 fillna; log warning if NaN rate > 10%
- [X] T023 [US2] Add `include_eco_features: bool = False` param to `extract_btm_features()` in `btm/features/extract.py`; when True, call `extract_eco_raster_features()` and join result before returning
- [X] T024 [US2] Add `--include-eco-features` flag to `btm/cli/export_features.py` argument parser; pass through to `extract_btm_features()` call
- [X] T025 [US2] Write unit tests for `compute_northness_eastness`, `compute_max_curvature`, `compute_complexity` using synthetic elevation arrays with known analytical solutions in `tests/unit/test_eco_raster_features.py`
- [X] T026 [US2] Run `pytest tests/ -q` — full suite must pass; check T025 tests pass
- [X] T027 [US2] Modify `src/benthic_model/models/train.py`: after model fit, if `flags.include_eco_features` is True, write `eco_thresholds.json` to the run artifact directory using `EcoFeatureTransformer.to_dict()`
- [X] T028 [US2] Modify `src/benthic_model/inference/predict.py`: if `eco_thresholds.json` exists in the run artifact directory, reconstruct `EcoFeatureTransformer.from_dict()` and apply `transform()` before prediction
- [ ] T029 [US2] Extract eco raster features for training data: run `btm-export-features --bathy data/MBES/bathymetry.tif --points data/train_btm.csv --output data/train_btm_eco.csv --include-eco-features` and verify `btm_northness`, `btm_eastness`, `btm_max_curvature`, `btm_complexity` columns present with NaN rate < 10%
- [ ] T030 [US2] Extract eco raster features for test data: run same command on test CSV → `data/test_btm_eco.csv`
- [ ] T031 [US2] Run `pytest tests/ -q` — full suite still passes; 0 regressions

**Checkpoint**: All eco-feature extraction and in-pipeline transformation is working; raster derivatives in train/test CSVs; test suite green

---

## Phase 5: User Story 3 — RF + Interactions + BTM (Priority: P2)

**Goal**: Config YAML for RF with both BTM features and pairwise interactions; one trained run with CV F1 comparable to R04

**Independent Test**: Training with `rf-btm-interactions.yaml` produces a run in `run_registry.jsonl` whose feature table contains both `btm_*` and `bathymetry_x_backscatter` interaction columns

### Implementation for User Story 3

- [X] T032 [P] [US3] Create `configs/rf-btm-interactions.yaml`: `model_type: rf`, `feature_flags.include_btm_features: true`, `feature_flags.include_interactions: true`, all other flags false; `# R17 — RF + BTM + interactions` header comment
- [ ] T033 [US3] Run `benthic-model train --config configs/rf-btm-interactions.yaml --train-csv data/train_btm.csv`; record CV weighted F1 from output
- [ ] T034 [US3] Note CV F1 vs R04 (0.8024) and SGAM per-class F1; update `run_registry.jsonl` `notes` field with comparison

**Checkpoint**: US3 run complete; CV F1 recorded

---

## Phase 6: User Story 4 — RF Hyperparameter Tuning (Priority: P2)

**Goal**: Config YAML with non-default RF hyperparameters on the BTM winner feature set; CV F1 comparable or better than R04

**Independent Test**: Training with `rf-btm-tuned.yaml` runs without error and logs CV F1 + per-class F1 in `run_registry.jsonl`

### Implementation for User Story 4

- [X] T035 [P] [US4] Create `configs/rf-btm-tuned.yaml`: `model_type: rf`, `model_params: {n_estimators: 500, max_features: sqrt, min_samples_leaf: 1}`, `feature_flags.include_btm_features: true`, all other flags false; `# R18 — RF + BTM + tuned hyperparameters` header comment
- [X] T036 [US4] Write failing test for `model_params` forwarding in `tests/unit/test_model_builders.py`: given `model_params={n_estimators: 500}`, built RF has `n_estimators == 500` (TDD Red — must FAIL before T037)
- [X] T037 [US4] Update `src/benthic_model/models/train.py` (or baseline.py builder): pass `pipeline_cfg.model_params` as `**kwargs` to `RandomForestClassifier` constructor when `model_params` is not None; run T036 test to confirm Green
- [ ] T038 [US4] Run `benthic-model train --config configs/rf-btm-tuned.yaml --train-csv data/train_btm.csv`; record CV weighted F1
- [ ] T039 [US4] Note CV F1 vs R04 (0.8024); flag in `run_registry.jsonl` `notes` if CV > 0.8024
- [ ] T040 [US4] Run `pytest tests/ -q` — full suite still passes

**Checkpoint**: US4 run complete; hyperparameter tuning result recorded

---

## Phase 7: User Story 2 Continued — Eco-Feature RF Experiments (Priority: P1)

**Goal**: Two eco-feature RF training runs producing `metrics.json` with SGAM per-class F1; both ready for Kaggle submission

**Independent Test**: Both eco-feature runs appear in `run_registry.jsonl`; each `metrics.json` contains per-class F1 for all 5 classes

### Implementation

- [X] T041 [P] [US2] Create `configs/rf-btm-eco-depth.yaml`: `model_type: rf`, `feature_flags.include_btm_features: true`, `feature_flags.include_eco_features: true`, all other flags false; `# R15 — RF + BTM + eco depth zones + SGAM niche` header comment
- [X] T042 [P] [US2] Create `configs/rf-btm-eco-full.yaml`: `model_type: rf`, `feature_flags.include_btm_features: true`, `feature_flags.include_eco_features: true`, all other flags false — same as eco-depth but use `train_btm_eco.csv` (includes all 4 raster derivatives); `# R16 — RF + BTM + all eco derivatives` header comment
- [ ] T043 [US2] Run `benthic-model train --config configs/rf-btm-eco-depth.yaml --train-csv data/train_btm_eco.csv`; record CV weighted F1 and SGAM F1 from `metrics.json`
- [ ] T044 [US2] Run `benthic-model train --config configs/rf-btm-eco-full.yaml --train-csv data/train_btm_eco.csv`; record CV weighted F1 and SGAM F1
- [ ] T045 [US2] Compare SGAM per-class F1 across R15, R16 vs baseline (0.043); note if SC-002 is met (SGAM F1 > 0.043 in at least one run)

**Checkpoint**: Eco-feature runs trained; SGAM F1 improvement assessed

---

## Phase 8: User Story 5 — Remaining Kaggle Submissions (Priority: P1)

**Goal**: Submit remaining 4 experiments from today's daily budget; all scores recorded in `kaggle_scores.csv`

**Independent Test**: `artifacts/experiments/kaggle_scores.csv` has 5 new rows dated 2026-03-31 with non-null `kaggle_f1` values

### Implementation

- [ ] T046 [US5] Generate submission CSV for R15 (eco-depth): `benthic-model predict --run-id <r15_run_id> --test-csv data/test_btm_eco.csv --output data/submission_r15.csv`
- [ ] T047 [US5] Submit R15: `kaggle competitions submit -c <competition> -f data/submission_r15.csv -m "<r15_run_id> RF+BTM+eco-depth CV=<r15_cv>"`; record score in `kaggle_scores.csv`
- [ ] T048 [US5] Generate submission CSV for R16 (eco-full): `benthic-model predict --run-id <r16_run_id> --test-csv data/test_btm_eco.csv --output data/submission_r16.csv`
- [ ] T049 [US5] Submit R16: `kaggle competitions submit -c <competition> -f data/submission_r16.csv -m "<r16_run_id> RF+BTM+all-eco-derivatives CV=<r16_cv>"`; record score in `kaggle_scores.csv`
- [ ] T050 [US5] Generate submission CSV for R17 (interactions): `benthic-model predict --run-id <r17_run_id> --test-csv data/test.csv --output data/submission_r17.csv` (uses non-eco test CSV)
- [ ] T051 [US5] Submit R17: `kaggle competitions submit -c <competition> -f data/submission_r17.csv -m "<r17_run_id> RF+BTM+interactions CV=<r17_cv>"`; record score in `kaggle_scores.csv`
- [ ] T052 [US5] Generate submission CSV for R18 (tuned RF): `benthic-model predict --run-id <r18_run_id> --test-csv data/test.csv --output data/submission_r18.csv`
- [ ] T053 [US5] Submit R18: `kaggle competitions submit -c <competition> -f data/submission_r18.csv -m "<r18_run_id> RF+BTM+tuned n_estimators=500 max_features=sqrt CV=<r18_cv>"`; record score in `kaggle_scores.csv`
- [ ] T054 [US5] Verify `kaggle_scores.csv` has 5 new rows for 2026-03-31 with non-null `kaggle_f1`; if any new best > 0.79518 found, update `notes` field with "NEW BEST" flag

**Checkpoint**: Full 5-submission daily batch complete; all scores recorded (SC-001, SC-004 met)

---

## Final Phase: Polish & Cross-Cutting Concerns

**Purpose**: Gate validation, documentation update, and commit

- [X] T055 [P] Run `pytest tests/ -q` — full suite passes (192 passed, 4 skipped + 0 failures)
- [X] T056 [P] Run `ruff check src/ btm/ tests/` — zero lint errors in new/modified files
- [ ] T057 Update `docs/run-014-btm-model-sweep.md` → `docs/run-015-ecology-habitat-features.md` (or create new run doc): add R15–R18 rows to summary table; record SGAM per-class F1 for each; note if SC-002 (SGAM improvement) and SC-003 (CV ≥ 0.8024) are met
- [ ] T058 Update `artifacts/experiments/run_registry.jsonl` to ensure all 5 new runs have `notes` fields referencing this spec (015)
- [ ] T059 Commit all changes: `git add -A && git commit -m "feat(015): eco-feature engineering + 5-submission Kaggle batch"`

---

## Dependencies

```
T001
 └── T002–T007, T060 (foundational config + per-class F1 gate)
       ├── T060 (metrics.json per-class F1 gate — MUST precede all training tasks)
       ├── T008–T010 (US1: GPU CatBoost — no code changes needed; after T060)
       ├── T011–T031 (US2: eco-feature module + raster extraction)
       │    ├── T041–T045 (US2 continued: eco-feature RF experiments — needs T029-T030)
       │    └── T046–T054 (US5: Kaggle batch — needs all training runs)
       ├── T032–T034 (US3: RF+interactions)
       └── T035–T040 (US4: RF tuned hyperparameters)
T055–T059 (polish — depends on all preceding)
```

## Parallel Execution Examples

**US1 runs immediately** (no code changes — artifact already exists): T008 → T009 → T010

**While submitting US1 (T009 Kaggle upload wait)**: Start T011–T015 (write failing tests)

**After T007 (config changes green)**:

- T019, T020, T021 in parallel (3 separate raster functions, independent files)
- T032, T035, T041, T042 in parallel (4 YAML files, all independent)

**After T022–T028 (eco pipeline complete)**:

- T029 and T030 can run sequentially (test CSV then train CSV extraction)
- T033 and T038 can run in parallel (independent training runs on different configs)

## Implementation Strategy

**MVP = US1 + US3 + US4** (zero new code; test Kaggle signal from existing artifacts and simple config changes): T001 → T002–T007 → T008–T010 (submit GPU CatBoost) + T032–T034 (RF+interactions) + T035–T040 (RF tuned)

**Full delivery adds US2**: extends `btm/features/extract.py` and adds `eco_features.py` — the most substantial work but the most scientifically novel

**Suggested order today** (submission budget filling gradually while code work proceeds):

1. T001–T010: Submit GPU CatBoost immediately (no code changes)
2. T002–T007 + T060 + T032 + T035: Config infrastructure + per-class F1 gate while waiting for result
3. T011–T031: Eco-feature module (TDD, raster functions, pipeline integration)
4. T041–T054: Train eco runs + final submissions
5. T055–T059: Polish and commit
