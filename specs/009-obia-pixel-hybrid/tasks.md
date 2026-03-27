---
description: "Task list for 009-obia-pixel-hybrid implementation"
---

# Tasks: 009 — OBIA + Pixel-Based Hybrid Classification

**Input**: Design documents from `/specs/009-obia-pixel-hybrid/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, quickstart.md ✅

**Note on tests**: Tests are requested in FR-009 and explicitly listed in plan.md. All test tasks use TDD: write test first, confirm it fails, then implement.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm dependencies, directory structure, and scaffold the experiment script shell.

- [ ] T001 Confirm `scikit-image` is installed; add to `pyproject.toml` dependencies if absent (pyproject.toml)
- [ ] T002 [P] Create `scripts/experiment_v6.py` with module-level docstring, imports (rasterio, numpy, scipy.ndimage, skimage.segmentation, pandas, sklearn, lightgbm, xgboost, catboost), top-level constants (BATHY_PATH, BACK_PATH, TRAIN_CSV, TEST_CSV, OUTPUT_SUBMISSION, OUTPUT_REPORT), and an empty `main()` guarded by `if __name__ == "__main__":`
- [ ] T003 [P] Create `tests/unit/test_experiment_v6_features.py` with module-level imports and 7 empty stub test functions: `test_pb_features_all_finite`, `test_northness_eastness_range`, `test_segment_labels_shape`, `test_segment_labels_non_negative`, `test_segment_mean_size_range`, `test_ob_features_all_finite`, `test_combined_feature_count` — each raises `NotImplementedError` so they fail immediately

**Checkpoint**: Imports resolve, stubs raise NotImplementedError, project structure confirmed

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Raster loading and baseline helper used by all three CV configurations.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Implement `_load_rasters(bathy_path, back_path) -> tuple[np.ndarray, np.ndarray, affine.Affine, float]` in `scripts/experiment_v6.py`: open both GeoTIFFs with rasterio, read band 1 as float32, mask NoData to NaN, return (bathy_arr, back_arr, transform, cell_size_metres) — cell_size derived from `abs(transform.a)`
- [ ] T005 [P] Reuse `_get_spatial_cv_groups(coords_xy)` from `scripts/experiment_v2.py` (copy-paste + adapt): KMeans with `n_clusters=10`, `random_state=42`, returns integer group array of same length as coords — place in `scripts/experiment_v6.py`
- [ ] T006 [P] Reuse `get_models()` from `scripts/experiment_v2.py` (copy-paste + adapt) into `scripts/experiment_v6.py`; confirm LightGBM, XGBoost, CatBoost, RandomForest constructor calls still work with current installed versions

**Checkpoint**: `_load_rasters` returns arrays + transform; `_get_spatial_cv_groups` returns 10-class groups; `get_models()` returns dict of 4 classifiers

---

## Phase 3: User Story 1 — Combined Feature Baseline (Priority: P1) 🎯 MVP

**Goal**: Produce a valid `data/submission_v6_best.csv` using all 18 features (8 PB + 9 OB + 1 size) with spatial CV ≥ 0.72 weighted F1.

**Independent Test**: Run `python scripts/experiment_v6.py`; verify `data/submission_v6_best.csv` exists with 98 rows and correct column headers; CV F1 printed to stdout exceeds 0.72.

### Tests for User Story 1 (TDD — write failing tests FIRST)

- [ ] T007 [P] [US1] Fill in `test_pb_features_all_finite` in `tests/unit/test_experiment_v6_features.py`: load a 10×10 synthetic bathy array; call `_compute_pb_features(bathy, back, cell_size=0.25)`; assert output dict contains exactly the 8 keys (`depth, backscatter, slope, vrm, complexity, max_curvature, northness, eastness`) and all values are finite numpy arrays with shape (10,10) — confirm test FAILS (function not implemented)
- [ ] T008 [P] [US1] Fill in `test_northness_eastness_range` in `tests/unit/test_experiment_v6_features.py`: same synthetic array; assert `northness` values are in [-1, 1] and `eastness` values are in [-1, 1] — confirm test FAILS
- [ ] T009 [P] [US1] Fill in `test_combined_feature_count` in `tests/unit/test_experiment_v6_features.py`: build a minimal mock dataframe with 18 columns (8 PB names + 9 OB names + `seg_pixel_count`); assert `len(feature_cols) == 18` — confirm test FAILS (feature_cols constant not yet defined)

### Implementation for User Story 1

- [ ] T010 [US1] Implement `_compute_pb_features(bathy, back, cell_size) -> dict[str, np.ndarray]` in `scripts/experiment_v6.py`:
  - `depth`: bathy array itself
  - `backscatter`: back array itself
  - `slope`: `btm.core.slope.compute_slope(bathy, cell_size)` (reuse existing btm core)
  - `vrm`: `btm.core.vrm.compute_vrm(bathy)` (reuse existing btm core)
  - `complexity`: `1 / np.cos(np.deg2rad(slope_arr))` using already-computed slope
  - `max_curvature`: `np.abs(scipy.ndimage.laplace(bathy)) / (cell_size ** 2)`
  - `northness`: `np.cos(np.arctan2(dz_dy, dz_dx))` where dz/dx and dz/dy from `np.gradient(bathy, cell_size)`
  - `eastness`: `np.sin(np.arctan2(dz_dy, dz_dx))`
  - NaN pixels (from NoData masking) propagate through; return dict of 8 arrays
- [ ] T011 [US1] Define `PB_FEATURE_COLS` and `COMBINED_FEATURE_COLS` constants in `scripts/experiment_v6.py` matching the 18 feature names from plan.md; run T009 test — it should now PASS
- [ ] T012 [US1] Implement `_extract_point_features(feature_dict, xy_coords, transform) -> pd.DataFrame` in `scripts/experiment_v6.py`: for each (x, y) point, convert to raster row/col using rasterio `rowcol(transform, xs, ys)`, index each feature array, return DataFrame with PB_FEATURE_COLS columns; clip indices to valid raster bounds
- [ ] T013 [US1] Run T007 and T008 tests — both should now PASS; fix any issues with `_compute_pb_features` until they do
- [ ] T014 [US1] Implement `_run_cv_config(X_train, y_train, groups, feature_cols, label) -> dict` in `scripts/experiment_v6.py`: accepts a feature matrix slice, runs `GroupKFold(n_splits=10)` CV loop with all 4 models, computes weighted F1 per fold per model, returns `{"label": label, "best_model_name": ..., "cv_f1_mean": ..., "oof_preds": ...}`
- [ ] T015 [US1] Implement full `main()` flow for PB-only and Combined configurations in `scripts/experiment_v6.py`:
  1. `_load_rasters()`
  2. `_compute_pb_features()`
  3. Load `train.csv`, `test.csv` → extract coordinates
  4. `_extract_point_features()` for train and test
  5. `_run_cv_config()` for PB-only (8 features)
  6. `_run_cv_config()` for Combined (18 features — OB columns will be zeros/NaN at this stage, patched in US2)
  7. Select best config by `cv_f1_mean`; train final model on full train set; predict test set
  8. Write `data/submission_v6_best.csv` with correct columns

**Checkpoint**: `python scripts/experiment_v6.py` runs end-to-end with PB features; submission CSV is written; T007, T008, T009, T011 tests pass

---

## Phase 4: User Story 2 — Pixel-Based Feature Completeness (Priority: P2)

**Goal**: All 8 pixel-based features demonstrably computed and validated; PB-only CV baseline documented.

**Independent Test**: Run `pytest tests/unit/test_experiment_v6_features.py::test_pb_features_all_finite tests/unit/test_experiment_v6_features.py::test_northness_eastness_range` — both pass. Print PB-only CV F1 during script run.

### Tests for User Story 2 (TDD)

- [ ] T016 [P] [US2] Expand `test_pb_features_all_finite` to also assert no NaN in the 8 arrays at non-masked pixels: pass a fully-valid synthetic array (no NaN) and assert `np.all(np.isfinite(v[~np.isnan(v)]))` for each feature — confirm passes after T013

### Implementation for User Story 2

- [ ] T017 [US2] Add explicit NaN-guard to `_compute_pb_features`: after computing each feature, replace any `np.inf` or `-np.inf` with `np.nan` so downstream extraction gets clean NaN rather than inf — this prevents inf from propagating into the feature table
- [ ] T018 [US2] Add structured logging (Python `logging` module, INFO level) to `_compute_pb_features` — log array shape and nan-pixel count per feature so the run report can reference these numbers
- [ ] T019 [US2] Re-run T016 — confirm it passes; fix any issues

**Checkpoint**: PB features are confirmed NaN-clean at non-NoData pixels; all US2 tests pass

---

## Phase 5: User Story 3 — Object-Based Segmentation & Statistics (Priority: P2)

**Goal**: SLIC segmentation + per-segment stats implemented; OB-only CV baseline documented; combined feature matrix complete.

**Independent Test**: Run `pytest tests/unit/test_experiment_v6_features.py -k "segment"` — all 3 segment tests pass. Script prints OB-only CV F1.

### Tests for User Story 3 (TDD — write failing tests FIRST)

- [ ] T020 [P] [US3] Fill in `test_segment_labels_shape` in `tests/unit/test_experiment_v6_features.py`: build a 20×20 synthetic 3-channel normalised stack; call `_segment_rasters(stack)`; assert output shape == (20, 20) and dtype is integer — confirm FAILS
- [ ] T021 [P] [US3] Fill in `test_segment_labels_non_negative` in `tests/unit/test_experiment_v6_features.py`: call `_segment_rasters(stack)` on valid input; assert `np.all(labels >= 0)` — confirm FAILS
- [ ] T022 [P] [US3] Fill in `test_segment_mean_size_range` in `tests/unit/test_experiment_v6_features.py`: use a 200×200 synthetic stack; call `_segment_rasters(stack, n_segments=100)`; compute mean segment size as `(200*200) / n_unique_labels`; assert mean size is between 0.5× and 2× the expected size (200×200/100=400) — confirm FAILS
- [ ] T023 [P] [US3] Fill in `test_ob_features_all_finite` in `tests/unit/test_experiment_v6_features.py`: build a small sythetic (bathy, back, vrm) + labels array; call `_compute_segment_stats(bathy, back, vrm, labels)`; assert output DataFrame has exactly 10 columns (`seg_bathy_mean, seg_bathy_std, seg_bathy_skew, seg_back_mean, seg_back_std, seg_back_skew, seg_vrm_mean, seg_vrm_std, seg_vrm_skew, seg_pixel_count`) and no NaN values — confirm FAILS

### Implementation for User Story 3

- [ ] T024 [US3] Implement `_segment_rasters(bathy, back, vrm, n_segments=4000, compactness=0.01) -> np.ndarray` in `scripts/experiment_v6.py`:
  - Normalise each channel to [0,1] using `(arr - nanmin) / (nanmax - nanmin)`, replace NaN with 0 before passing to SLIC
  - Stack to shape (H, W, 3)
  - Call `skimage.segmentation.slic(stack, n_segments=n_segments, compactness=compactness, enforce_connectivity=True, start_label=0)`
  - Return label array (H, W) int32
- [ ] T025 [US3] Run T020, T021, T022 — all should now PASS; fix any issues
- [ ] T026 [US3] Implement `_compute_segment_stats(bathy, back, vrm, labels) -> pd.DataFrame` in `scripts/experiment_v6.py`:
  - For each unique label: collect bathy/back/vrm pixel values within that segment (ignoring NaN)
  - Compute mean, std, skew (`scipy.stats.skew`) for each — 9 values per segment
  - Plus `seg_pixel_count` = int count of pixels in segment
  - Return DataFrame indexed by segment label, columns = OB_FEATURE_COLS (10 columns)
  - **Fallback**: if a segment has < 3 valid pixels, use the global mean for std and skew (global median for NaN-entire-segment case per spec edge case)
- [ ] T027 [US3] Run T023 — should now PASS; fix any issues
- [ ] T028 [US3] Implement `_assign_segment_features(point_df, labels, seg_stats_df, transform) -> pd.DataFrame` in `scripts/experiment_v6.py`:
  - For each point: look up row/col → `segment_label = labels[row, col]`
  - Join with `seg_stats_df` to get the 10 OB features
  - If a point's segment label is not in `seg_stats_df` (raster boundary edge case), fill with global column medians
  - Return the input `point_df` with 10 new OB columns appended
- [ ] T029 [US3] Integrate segmentation into `main()`: after PB features, compute VRM (already in T010 via `_compute_pb_features`), call `_segment_rasters(bathy, back, vrm_arr)`, call `_compute_segment_stats()`, call `_assign_segment_features()` for both train and test points, completing the 18-column feature matrices; re-run `_run_cv_config()` for OB-only (10 OB features) and Combined (18 features)

**Checkpoint**: All 7 unit tests pass; `python scripts/experiment_v6.py` completes with PB, OB, and Combined CV F1 scores printed; submission CSV updated with best configuration

---

## Phase 6: User Story 4 — Run Report (Priority: P3)

**Goal**: Markdown run report generated at `docs/run-009-obia-pixel-hybrid.md` after each run.

**Independent Test**: After running script, open `docs/run-009-obia-pixel-hybrid.md`; verify it contains: segment parameters, table of CV F1 for each config, top-5 feature importances for combined model, and (if combined is not best) an explicit flag note.

### Implementation for User Story 4

- [ ] T030 [US4] Implement `_write_run_report(config_results, seg_params, feature_importances, best_label, output_path)` in `scripts/experiment_v6.py`:
  - Write YAML-style frontmatter (date, branch, script version)
  - Section: Segmentation parameters (n_segments, compactness, actual unique segment count, actual mean segment size in pixels and m²)
  - Section: CV results table — one row per config (PB-only, OB-only, Combined) with weighted F1 mean ± std
  - Section: Top-10 feature importances for the best model in the combined configuration (use LightGBM or RF feature_importances_ attribute; average across folds)
  - Section: Best configuration selected (with note if combined was not best, quoting the delta, per SC-002)
  - Write to `output_path`
- [ ] T031 [US4] Call `_write_run_report()` at the end of `main()` in `scripts/experiment_v6.py`; pass actual results from `_run_cv_config()` calls
- [ ] T032 [US4] Add an integration-style assertion in the test file: `test_report_file_exists` — after running the script (skip if environment variable `BTM_SKIP_INTEGRATION_TESTS=1`), check `docs/run-009-obia-pixel-hybrid.md` exists and contains the string "## CV Results" — add as T033 placeholder only; mark as optional/manual for CI

**Checkpoint**: Report file written; contains all required sections; flag present if combined is not best

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, linting, documentation, and commit

- [ ] T033 [P] Run `ruff check scripts/experiment_v6.py tests/unit/test_experiment_v6_features.py` and fix all lint errors; run `ruff format` if configured
- [ ] T034 [P] Run full test suite `pytest tests/unit/test_experiment_v6_features.py -v` — confirm all 7 unit tests pass (not raising NotImplementedError or assertion errors)
- [ ] T035 Run `python scripts/experiment_v6.py` end-to-end; record actual CV F1 scores; verify `data/submission_v6_best.csv` has 98 rows and correct columns; verify `docs/run-009-obia-pixel-hybrid.md` is written
- [ ] T036 [P] Update `CHANGELOG` with a one-line entry for branch 009 describing the OBIA hybrid approach and resulting CV score
- [ ] T037 Commit all new files (`scripts/experiment_v6.py`, `tests/unit/test_experiment_v6_features.py`, `docs/run-009-obia-pixel-hybrid.md`, `data/submission_v6_best.csv`, `CHANGELOG`) with message: `feat(009): implement OBIA+pixel hybrid experiment (experiment_v6)`
- [ ] T038 Push branch and create PR: `gh pr create --title "009: OBIA + pixel-based hybrid classification (experiment_v6)" --body "Implements Ierodiaconou et al. 2018 OBIA+PB hybrid. See docs/run-009-obia-pixel-hybrid.md for results."`

**Checkpoint**: All tasks complete, branch pushed, PR open

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately; T002 and T003 can run in parallel
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user story phases; T005 and T006 can run in parallel after T004
- **Phase 3 (US1)**: Depends on Phase 2 — T007/T008/T009 can run in parallel; T010 → T011 → T012 → T013 → T014 → T015 sequential
- **Phase 4 (US2)**: Depends on T015 (US1 complete) — primarily validating and hardening existing PB functions
- **Phase 5 (US3)**: Depends on Phase 2; T020/T021/T022/T023 can run in parallel; T024 → T025 → T026 → T027 → T028 → T029 sequential; can begin in parallel with Phase 4 once Phase 2 is done
- **Phase 6 (US4)**: Depends on Phase 5 (T029) being complete — report needs all three CV results
- **Phase 7 (Polish)**: Depends on all preceding phases complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational phase; no dependency on US2, US3, US4
- **US2 (P2)**: Depends on US1 T010 (PB features implemented); validates and hardens US1 output
- **US3 (P2)**: Can start after Foundational phase, independent of US2; integrates with US1 in T029
- **US4 (P3)**: Depends on US3 T029 (all three configs have CV results)

### Within Each User Story

- Tests (stub → fail → implement → pass) precede each implementation block
- `_compute_pb_features` (T010) before `_extract_point_features` (T012) before CV (T014)
- `_segment_rasters` (T024) before `_compute_segment_stats` (T026) before `_assign_segment_features` (T028) before integration (T029)

---

## Parallel Opportunities

### Parallel Example: User Story 1 + User Story 3 (Phase 3 + Phase 5)

```text
After Phase 2 completes:

Thread A (US1):          Thread B (US3):
  T007 write test          T020 write test
  T008 write test          T021 write test
  T009 write test          T022 write test
  T010 implement PB →      T023 write test
  T011 constants      →    T024 implement SLIC
  T012 point extract  →    T025 run tests
  T013 run tests      →    T026 implement stats
  T014 CV config           T027 run tests
  T015 main() PB+Comb      T028 assign features
                        ↘
                    T029 integrate OB into main()
                          ↓
                    Phase 6 (US4)
```

### Setup parallelism (Phase 1)

```text
T002 (create experiment_v6.py) ──┐
                                  ├──> T004 (load rasters)
T003 (create test stubs)      ──┘
```

### Within Phase 5 (tests parallel)

```text
T020, T021, T022, T023  ← all can be written simultaneously (different test functions)
```

---

## Implementation Strategy

### MVP Scope (Suggested first delivery)

Complete **Phase 1 → Phase 2 → Phase 3 only**:
- Install scikit-image
- Scaffold experiment_v6.py
- Implement PB features only (8 features)
- Run CV with PB configuration
- Write submission CSV
- Verify SC-001 viability with PB-only baseline

This gives a runnable experiment in minimal time. OB segmentation (US3) and the run report (US4) layer on top.

### Incremental Delivery Order

1. **T001–T006**: Infrastructure (30 min)
2. **T007–T015**: PB features + US1 complete (60–90 min)
3. **T016–T019**: US2 hardening (15 min)
4. **T020–T029**: SLIC + OB stats + combined (60–90 min)
5. **T030–T032**: Run report (20 min)
6. **T033–T038**: Polish + commit + PR (15 min)

### Key Risk Mitigation

- **RD-001 (SLIC parameters)**: If mean segment size is outside 200–500 m² (SC-004), adjust `n_segments` by factor `actual_mean / target_mean` and re-run segmentation only (T024 is isolated)
- **RD-002 (VRM reuse)**: `btm.core.vrm.compute_vrm()` already tested in unit suite; if API changes, fall back to the inline 3×3 numpy implementation in plan.md pseudocode
- **RD-003 (complexity)**: If `compute_slope` returns values in degrees, ensure the `np.deg2rad` conversion in complexity formula is present (T010)
- **Kaggle CV mismatch**: Primary success metric is CV F1 (SC-001); Kaggle submission is secondary — select best-CV model always ensures a valid submission even if combined CV < 0.72
