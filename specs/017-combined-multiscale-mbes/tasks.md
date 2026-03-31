---
description: "Task list for Combined Multi-Scale BTM + MBES-8 Features (spec-017)"
---

# Tasks: Combined Multi-Scale BTM + MBES-8 Features

**Input**: Design documents from `/specs/017-combined-multiscale-mbes/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, quickstart.md ✅
**Branch**: `017-combined-multiscale-mbes`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependencies on incomplete tasks)
- **[Story]**: User story this task belongs to (US1–US4)
- Exact file paths included in every task description

## Dependency Note

US1 (Combine + Train, P1) is the core deliverable. US2 (Feature Selection, P2) and US3 (Submission, P2) depend on US1 completion. US4 (SGAM Monitoring, P3) is observational — addressed by logging within US1.

---

## Phase 1: Setup

**Purpose**: Create the experiment script stub and test file so TDD tests can be written.

- [x] T001 Create empty script file `scripts/experiment_v11.py` with module docstring (hypothesis, design principles, feature set, baselines, success criteria, usage) following the `experiment_v10.py` pattern; include `argparse` with `--dry-run` and `--extract-only` flags; import logging; create empty `main()` function
- [x] T002 [P] Create test file `tests/unit/test_experiment_v11.py` with pass-through stubs for `test_mbes8_extraction_returns_8_columns`, `test_feature_importance_output_schema`, `test_submission_csv_format`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No new infrastructure needed — all dependencies (scikit-learn RF, CatBoost, LightGBM, rasterio, spec-016 CSV cache, `benthic_model.features.selection`) already exist. Verify prerequisites.

- [x] T003 Verify spec-016 cached features exist: assert `reports/metrics/cache_train_feats_v10.csv`, `reports/metrics/cache_test_feats_v10.csv`, and `reports/metrics/feature_selection_v10.csv` are present and have non-zero size; log column counts and row counts
- [x] T004 [P] Verify rasters are accessible: open `data/MBES/bathymetry.tif` and `data/MBES/backscatter.tif` with rasterio; log shape, CRS, nodata value, and cell size; assert both open without error

**Checkpoint**: Prerequisites confirmed — user story implementation can begin.

---

## Phase 3: User Story 1 — Combine BTM + MBES Features and Train Models (P1) 🎯 MVP

**Goal**: Extract MBES-8 features, merge with BTM-33 from cache, train RF + CatBoost + LightGBM via 5-fold spatial-blocked CV (matching R04/R06), report per-model F1 and per-class recall.

**Independent Test**: Run `python scripts/experiment_v11.py --dry-run` and verify it completes without error, logs CV F1 for all three models, and writes a submission CSV.

### TDD: Tests First

- [x] T005 [US1] Write test `tests/unit/test_experiment_v11.py::test_mbes8_extraction_returns_8_columns` — given a mock 20×20 bathymetry array, mock 20×20 backscatter array, cell_size=0.25, and 5 point coordinates within bounds, assert `_extract_mbes8_features(bathy, back, cell_size, xy_coords)` returns a DataFrame with exactly 8 columns (`depth`, `backscatter`, `slope`, `vrm`, `complexity`, `max_curvature`, `northness`, `eastness`) and 5 rows, all values finite
- [x] T006 [P] [US1] Write test `tests/unit/test_experiment_v11.py::test_feature_importance_output_schema` — given mock importance arrays and feature names, assert `_write_feature_importance(importances_mean, importances_std, feature_names, path)` writes a CSV with columns `["feature_name", "importance_mean", "importance_std", "rank", "selected"]` (FR-009)
- [x] T007 [P] [US1] Write test `tests/unit/test_experiment_v11.py::test_submission_csv_format` — given a mock predictions DataFrame, assert `_write_submission(predictions, path)` writes a CSV matching `data/sample_submission.csv` column format (FR-010)

### Implementation

- [x] T008 [US1] Implement `_load_rasters(bathy_path, back_path)` in `scripts/experiment_v11.py`: open both GeoTIFFs with rasterio; replace nodata with NaN; return bathy, back, transform, cell_size; reuse pattern from `scripts/experiment_v9.py` lines 141–157
- [x] T009 [US1] Implement `_extract_mbes8_features(bathy, back, cell_size, xy_coords, transform)` in `scripts/experiment_v11.py`: compute 8 MBES features (depth, backscatter, slope, VRM, complexity, max_curvature, northness, eastness) using Horn gradient kernels, `btm.core.slope.compute_slope`, `btm.core.vrm.compute_vrm`, Laplacian curvature, and trig aspect; sample at point locations via `rasterio.transform.rowcol`; fill NaN values (from out-of-bounds or nodata pixels) with column median before returning; return DataFrame with 8 columns; reuse logic from `scripts/experiment_v9.py::_compute_pb_features` (FR-001, EC-1)
- [x] T010 [US1] Implement `_load_btm33_features(train_df, test_df)` in `scripts/experiment_v11.py`: read `reports/metrics/cache_train_feats_v10.csv` and `cache_test_feats_v10.csv`; read `reports/metrics/feature_selection_v10.csv` to get the 33 selected feature names (where `selected == True`); filter both DataFrames to those 33 columns plus `ID`; return train and test DataFrames (FR-002)
- [x] T011 [US1] Implement `_merge_features(mbes_df, btm_df, point_ids)` in `scripts/experiment_v11.py`: merge MBES-8 DataFrame with BTM-33 DataFrame on explicit point ID alignment (not positional index); log combined column count; assert no duplicate column names; return combined DataFrame (~41 columns) (FR-003)
- [x] T012 [US1] Implement `_extract_and_cache(train_raw, test_raw, dry_run)` in `scripts/experiment_v11.py`: coordinate MBES-8 extraction and BTM-33 loading; implement CSV caching to `reports/metrics/cache_combined_v11.csv` and `cache_combined_test_v11.csv`; if cache exists, load from cache; otherwise extract features (reduced data if `--dry-run`) and write cache; log extraction time (FR-011, FR-012)
- [x] T013 [US1] Implement `_run_cv(X, y, xy_coords, label)` in `scripts/experiment_v11.py`: 5-fold spatial-blocked CV via `iter_spatial_blocked_folds(x, y, n_splits=5, spatial_bins=4, random_state=42)` from `src/benthic_model/evaluation/cv.py` with 3 models — RF (`n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample'`, FR-004), CatBoost (iterations=800, lr=0.03, depth=7, FR-005), LightGBM (n_estimators=600, lr=0.03, num_leaves=31, FR-005); compute per-fold weighted-F1; compute per-class classification_report per fold (handle folds with zero SGAM samples gracefully — report recall as NaN for that fold, exclude from mean; EC-3); compute soft-vote RF+Cat+LGB ensemble (FR-014); log per-model and ensemble CV F1 ± std; log total CV training time per model and overall (FR-006, FR-007, FR-012); return best model/ensemble name (highest CV weighted-F1), metrics dict, per-class rows
- [x] T014 [US1] Implement `_write_feature_importance(importances_mean, importances_std, feature_names, path)` in `scripts/experiment_v11.py`: write CSV with columns `feature_name, importance_mean, importance_std, rank`; reuse pattern from `experiment_v10.py` (partial FR-009 — `selected` column added in T018 after feature selection)
- [x] T015 [US1] Implement `_write_submission(predictions, path)` in `scripts/experiment_v11.py`: write Kaggle submission CSV matching `data/sample_submission.csv` format; reuse pattern from `experiment_v10.py` (FR-010)
- [x] T016 [US1] Implement `main(dry_run, extract_only)` entry point in `scripts/experiment_v11.py`: load CSVs; call `_extract_and_cache()`; log feature matrix shape and SC-005 extraction time; if `--extract-only`, exit; call `_run_cv()` with label `"combined"`; log SC-001 comparison (CV F1 vs 0.8024); log SC-003 SGAM recall comparison; write `cv_per_class_v11.csv`; write `feature_importance_v11.csv`; write `submission_v11.csv`; add argparse `--dry-run` (5 train points, reduces verbosity) and `--extract-only` (FR-013)
- [x] T017 [US1] Run `scripts/experiment_v11.py --dry-run` and verify: no errors; CV F1 logged for rf, cat, lgb, and ensemble; submission CSV written with correct format; feature importance CSV written

**Checkpoint**: US1 complete — combined feature CV results available for all 3 models + ensemble.

---

## Phase 4: User Story 2 — Feature Selection on Combined Set (P2)

**Goal**: Apply Spearman correlation pre-filter + permutation importance to the ~41-feature combined set; retrain on the selected subset; verify CV F1 degradation ≤ 0.01 (SC-004).

**Independent Test**: After selection, retrain and compare CV F1 before vs after; assert degradation ≤ 0.01.

- [x] T018 [US2] Integrate feature selection into `scripts/experiment_v11.py`: after the initial CV training (T016), call `select_by_permutation_importance()` from `src/benthic_model/features/selection.py` using the best full-set model; write `reports/metrics/feature_selection_v11.csv` (FR-008, FR-009)
- [x] T019 [US2] Add selected-feature CV run to `scripts/experiment_v11.py`: retrain all 3 models + ensemble on only the selected features; log CV F1 before and after selection; log SC-004 degradation and feature count; log which MBES-8 vs BTM features were retained and dropped (FR-010)
- [x] T020 [US2] Run `scripts/experiment_v11.py` full pipeline with feature selection; verify `feature_selection_v11.csv` written; verify selected feature count; verify CV F1 degradation ≤ 0.01

**Checkpoint**: US2 complete — feature selection applied and validated.

---

## Phase 5: User Story 3 — Produce Kaggle Submission (P2)

**Goal**: Generate Kaggle submission CSVs for both the full-feature and selected-feature best models.

**Independent Test**: Verify submission CSVs have 98 rows and match `data/sample_submission.csv` column format.

- [x] T021 [US3] Generate `data/submission_v11.csv` from the full-feature best model in `scripts/experiment_v11.py`: train best model on full training set; predict test set; write submission; log row count (FR-010)
- [x] T022 [US3] Generate `data/submission_v11_selected.csv` from the selected-feature best model in `scripts/experiment_v11.py`: train selected-feature model on full training set; predict test set; write submission; log prediction differences vs full-feature submission (FR-010)

**Checkpoint**: US3 complete — both submission CSVs ready for Kaggle upload.

---

## Phase 6: User Story 4 — SGAM Minority Class Monitoring (P3)

**Goal**: SGAM recall is already logged in US1 (T013/T016). This phase adds explicit SC-003 comparison against the R04/R06 baseline.

**Independent Test**: Check log output for SGAM recall value and SC-003 pass/fail.

- [x] T023 [US4] Add explicit SC-003 logging to `scripts/experiment_v11.py::main()`: after CV, compare SGAM mean recall against R04/R06 baseline (0.045); log `SC-003 PASS` or `SC-003 WARNING` with the delta; include SGAM recall in `reports/metrics/cv_per_class_v11.csv` (FR-007)

**Checkpoint**: US4 complete — SGAM monitoring active.

---

## Phase 7: Full Run & Polish

**Purpose**: Execute the full experiment, validate all success criteria, document results.

- [x] T024 Run full experiment: `python scripts/experiment_v11.py 2>&1 | Tee-Object -FilePath "reports\experiment_v11_run.log"`; verify all output files created: `cv_per_class_v11.csv`, `feature_importance_v11.csv`, `feature_selection_v11.csv`, `submission_v11.csv`, `submission_v11_selected.csv`
- [x] T025 [P] Run full test suite: `pytest tests/ -m "not arcgis and not qgis" -v`; assert all pre-existing tests still pass plus new `test_experiment_v11.py` tests pass; log final test count
- [x] T026 [P] Write `docs/run-017-combined-multiscale-mbes.md`: document approach (BTM-33 + MBES-8 combined), per-model CV F1 results, per-class recall, feature selection results (MBES-8 vs BTM contribution), SC-001 through SC-005 gate outcomes, comparison to R04/R06 and v10 baselines
- [x] T027 Update `docs/runsheet-hybrid-kaggle.md`: add row(s) for experiment_v11 with CV score, Kaggle score (if submitted), feature count, model used, and link to `docs/run-017-combined-multiscale-mbes.md`

---

## Dependencies (Story Completion Order)

```
T001–T002 (Setup)
    │
    T003–T004 (Foundational: verify prerequisites)
    │
    T005–T007 (US1: TDD tests) ──→ T008–T017 (US1: implementation + dry-run)
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                   T018–T020 (US2)  T021–T022 (US3)  T023 (US4)
                          │               │               │
                          └───────────────┼───────────────┘
                                          ▼
                                   T024–T027 (Polish)
```

**US2, US3, and US4 are independent of each other** — can be implemented in parallel after US1 completes.

---

## Parallel Execution Examples

### Stream A: Script scaffold + tests (Phase 1–2 + US1 tests)

```
T001 → T005
T002 → T006, T007 (parallel)
T003, T004 (parallel prerequisites)
```

### Stream B: US1 implementation (after tests written)

```
T008 → T009 → T010, T011 (parallel) → T012 → T013 → T014, T015 (parallel) → T016 → T017
```

### Stream C: US2 + US3 + US4 (after US1 complete, all parallel)

```
T018 → T019 → T020
T021, T022 (parallel with US2)
T023 (parallel with US2 and US3)
```

---

## Implementation Strategy

**MVP Scope**: US1 only (Tasks T001–T017) — delivers combined-feature 3-model CV results and answers the primary question: does adding MBES-8 to BTM-33 reach the 0.8024 baseline?

**Increment 2**: US2 (T018–T020) + US3 (T021–T022) + US4 (T023) — feature selection, submissions, SGAM monitoring.

**Increment 3**: T024–T027 — full run, test validation, documentation.

---

## Format Validation

All tasks follow the required format: `- [ ] [TaskID] [P?] [Story?] Description with file path`

- Total tasks: 27
- Phase 1 (Setup): 2 tasks
- Phase 2 (Foundational): 2 tasks
- Phase 3 (US1 — combine + train, P1): 13 tasks (3 tests, 10 implementation)
- Phase 4 (US2 — feature selection, P2): 3 tasks
- Phase 5 (US3 — submission, P2): 2 tasks
- Phase 6 (US4 — SGAM monitoring, P3): 1 task
- Phase 7 (Polish): 4 tasks

**Parallel opportunities**: T002 with T001; T004 with T003; T006, T007 with T005; T010, T011 parallel; T014, T015 parallel; T021, T022 parallel with T018–T020; T023 parallel with US2/US3; T025, T026 with T024.

**Suggested MVP**: T001–T017 (US1 only) — delivers testable combined-feature CV results in 17 tasks.
