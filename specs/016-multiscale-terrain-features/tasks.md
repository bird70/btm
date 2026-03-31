---
description: "Task list for Multi-Scale Terrain Features (spec-016)"
---

# Tasks: Multi-Scale Terrain Features

**Input**: Design documents from `/specs/016-multiscale-terrain-features/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, quickstart.md ✅
**Branch**: `016-multiscale-terrain-features`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependencies on incomplete tasks)
- **[Story]**: User story this task belongs to (US1–US4)
- Exact file paths included in every task description

## Dependency Note

US4 (GLCM, P2) is scheduled before US2 (Train, P1) because US2 requires a complete feature set to evaluate the full hypothesis. Feature extraction (US1 + US4) must be complete before model training is meaningful. Feature selection (US3) requires a trained model from US2.

---

## Phase 1: Setup

**Purpose**: Create empty module stubs so test files can import from them (TDD red-phase setup).

- [x] T001 Create empty module file `btm/core/multiscale.py` with module-level docstring and placeholder function signatures: `focal_mean_multiscale()`, `compute_rdmv()`
- [x] T002 [P] Create empty module file `btm/core/glcm.py` with module-level docstring and placeholder function signature: `compute_glcm_texture()`
- [x] T003 [P] Create empty module file `src/benthic_model/features/selection.py` with module-level docstring and placeholder function signature: `select_by_permutation_importance()`
- [x] T004 [P] Create empty test files: `tests/unit/test_multiscale.py`, `tests/unit/test_glcm.py`, `tests/unit/test_feature_selection.py` with pass-through stubs
- [x] T005 [P] Create `tests/integration/` directory with `__init__.py` and empty `tests/integration/test_multiscale_pipeline.py`; create `tests/data/016/` directory containing small (≤ 50×50 cell) GeoTIFF raster fixtures clipped from `data/MBES/bathymetry.tif` and `data/MBES/backscatter.tif` using rasterio windowed read — these fixtures MUST be committed to the repository so the integration test suite runs without accessing `data/MBES/` at runtime (constitution Principle I: integration tests MUST use real raster data under `tests/data/`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No shared infrastructure tasks are needed — all dependencies (scipy, scikit-image, scikit-learn) are already declared in `pyproject.toml`. Proceed directly to story phases.

_This phase intentionally empty for spec-016._

---

## Phase 3: User Story 1 — Compute Multi-Scale Features (P1)

**Story Goal**: The feature extraction pipeline computes terrain derivatives at 5 neighbourhood scales [3, 7, 11, 15, 21] using calculate-then-average, plus RDMV at each scale, returning consistently named columns (e.g., `btm_slope_7`, `btm_rdmv_21`).

**Independent Test**: Run `extract_btm_features(sample_points, bathy_tif, scales=[3,7,11,15,21])` and assert the output DataFrame contains 40 multi-scale columns (7 derivatives × 5 scales + 5 RDMV columns) with no all-NaN columns.

### TDD: Tests First

- [x] T006 [US1] Write test `tests/unit/test_multiscale.py`: `test_focal_mean_multiscale_output_shape` — given a 20×20 float32 array and scales=[3,7], assert `focal_mean_multiscale("slope", arr, [3,7])` returns a dict with keys 3 and 7, each an ndarray of the same shape as input
- [x] T007 [P] [US1] Write test `tests/unit/test_multiscale.py`: `test_focal_mean_multiscale_column_names` — assert the helper `multiscale_column_names("slope", [3,7,11])` returns `["btm_slope_3","btm_slope_7","btm_slope_11"]`
- [x] T008 [P] [US1] Write test `tests/unit/test_multiscale.py`: `test_focal_mean_multiscale_smoothing` — assert values at scale=21 have lower standard deviation than values at scale=3 for a synthetic noisy array (larger scale → smoother)
- [x] T009 [P] [US1] Write test `tests/unit/test_multiscale.py`: `test_focal_mean_boundary_reflect` — given a 10×10 array, assert a point at corner [0,0] at scale=7 returns a non-NaN float (mode="reflect" boundary)
- [x] T010 [P] [US1] Write test `tests/unit/test_multiscale.py`: `test_compute_rdmv_formula` — given a flat 5×5 array (all same value), assert `compute_rdmv(arr, scale=3)` returns all-zeros (std=0 edge case handled gracefully)
- [x] T011 [P] [US1] Write test `tests/unit/test_multiscale.py`: `test_compute_rdmv_nonflat` — given a synthetic array with a central ridge, assert RDMV values at the ridge are positive and values in the depression are negative

### Implementation

- [x] T012 [US1] Implement `focal_mean_multiscale(derivative_name: str, base_array: np.ndarray, scales: list[int]) -> dict[int, np.ndarray]` in `btm/core/multiscale.py`: apply `scipy.ndimage.uniform_filter(base_array, size=s, mode="reflect")` for each scale `s`; cite Misiuk et al. (2021) in docstring
- [x] T013 [US1] Implement `compute_rdmv(depth_array: np.ndarray, scale: int) -> np.ndarray` in `btm/core/multiscale.py`: compute `focal_mean` and `focal_std` via `uniform_filter`; RDMV = `(depth - focal_mean) / focal_std`; set to 0.0 where `focal_std == 0`; cite Lecours et al. (2017) in docstring
- [x] T014 [US1] Implement helper `multiscale_column_names(derivative: str, scales: list[int]) -> list[str]` in `btm/core/multiscale.py`: returns `[f"btm_{derivative}_{s}" for s in scales]`
- [x] T015 [US1] Modify `btm/features/extract.py`: add `scales: list[int] | None = None` parameter to `extract_btm_features()`; when scales is not None, call `focal_mean_multiscale()` for each of the 7 derivatives (slope, vrm, surface_ratio, northness, eastness, max_curvature, complexity) and `compute_rdmv()` for depth; sample the scaled arrays at point locations; append all resulting columns to the output DataFrame
- [x] T016 [US1] Verify backward compatibility in `tests/unit/test_btm_features.py`: assert existing tests still pass when `scales=None` (default behaviour unchanged); run `pytest tests/unit/test_btm_features.py -v`

---

## Phase 4: User Story 4 — Backscatter GLCM Texture Features (P2)

**Story Goal**: Extract GLCM contrast and homogeneity from the backscatter raster at specified window sizes and add them to the feature DataFrame as `btm_glcm_contrast_{scale}` and `btm_glcm_homogeneity_{scale}` columns.

**Independent Test**: Call `compute_glcm_texture(backscatter_array, point_rows, point_cols, scales=[7,11])` and assert the output DataFrame contains columns `btm_glcm_contrast_7`, `btm_glcm_homogeneity_7`, `btm_glcm_contrast_11`, `btm_glcm_homogeneity_11` with no all-NaN columns and contrast ≥ 0.

### TDD: Tests First

- [x] T017 [US4] Write test `tests/unit/test_glcm.py`: `test_glcm_output_columns` — given a 40×40 float32 backscatter array and 5 point coordinates, assert `compute_glcm_texture(arr, rows, cols, scales=[7])` returns a DataFrame with columns `["btm_glcm_contrast_7","btm_glcm_homogeneity_7"]` and 5 rows
- [x] T018 [P] [US4] Write test `tests/unit/test_glcm.py`: `test_glcm_contrast_nonnegative` — assert all contrast values ≥ 0.0 for a random backscatter array
- [x] T019 [P] [US4] Write test `tests/unit/test_glcm.py`: `test_glcm_homogeneity_range` — assert all homogeneity values are in [0.0, 1.0] for a random backscatter array
- [x] T020 [P] [US4] Write test `tests/unit/test_glcm.py`: `test_glcm_rotation_invariance` — assert that transposing the backscatter array does not substantially change the mean homogeneity (direction-averaged GLCM is approximately rotation-invariant)
- [x] T021 [P] [US4] Write test `tests/unit/test_glcm.py`: `test_glcm_nodata_in_patch` — given a backscatter array with NaN values, assert `compute_glcm_texture` returns finite float values (not NaN) at the test point location (NaN pixels replaced by patch median before GLCM)

### Implementation

- [x] T022 [US4] Implement `compute_glcm_texture(backscatter_array: np.ndarray, point_rows: np.ndarray, point_cols: np.ndarray, scales: list[int], n_levels: int = 32) -> pd.DataFrame` in `btm/core/glcm.py`: for each point, extract a `scale × scale` patch centred on the point; replace NaN with patch median; quantise to `n_levels` grey levels; call `skimage.feature.graycomatrix` with distances=[1] and angles=[0, π/4, π/2, 3π/4]; compute `graycoprops(glcm, "contrast")` and `graycoprops(glcm, "homogeneity")`; average over 4 directions; cite Haralick (1973) and Nemani et al. (2022) in docstring; promote pattern from `scripts/experiment_v2.py`
- [x] T023 [US4] Modify `btm/features/extract.py`: add `include_glcm: bool = False` and `backscatter_tif: str | None = None` parameters to `extract_btm_features()`; when `include_glcm=True`, open backscatter raster with rasterio, convert point coordinates to pixel indices, call `compute_glcm_texture()` at each requested scale, append GLCM columns to output DataFrame; raise `ValueError` if `include_glcm=True` and `backscatter_tif` is None

---

## Phase 5: User Story 2 — Train and Evaluate Model (P1)

**Story Goal**: Train a classification model on the full multi-scale feature set (US1 + US4 features), evaluate with 5-fold spatial CV, compare weighted-F1 to baseline (0.8024), log per-scale feature importances, and produce a Kaggle submission CSV.

**Independent Test**: Run `scripts/experiment_v10.py --dry-run` (5 training points, 2 scales) and assert it completes without error, a submission CSV is written, and CV result is logged.

- [x] T024a [US2] Write test `tests/unit/test_experiment_v10.py`: `test_feature_importance_output_schema` — given a mock trained model and feature list, assert `_write_feature_importance(importances, feature_names, path)` writes a CSV to the specified path with columns `["feature_name", "importance_mean", "importance_std", "rank", "selected"]` and one row per feature (FR-006)
- [x] T024b [P] [US2] Write test `tests/unit/test_experiment_v10.py`: `test_submission_csv_format` — given a mock DataFrame of predictions, assert `_write_submission(predictions, path)` writes a CSV with column names matching `data/sample_submission.csv` exactly (ID column + class column), all class values drawn from the known class set `{NVB, FMAT, SGZ, ALG, SGAM}`, and row count matching the test set length (FR-009)
- [x] T024 [US2] Write `scripts/experiment_v10.py` with private helpers `_write_feature_importance()` and `_write_submission()` (tested in T024a/T024b): load `data/train.csv` and `data/test.csv`; call `extract_btm_features(..., scales=[3,7,11,15,21], include_glcm=True, backscatter_tif="data/MBES/backscatter.tif")`; train CatBoost or RF classifier with existing 5-fold spatial CV strategy (re-use pattern from `scripts/experiment_v9.py`); log mean and std weighted-F1 and compare to 0.8024 baseline; call `_write_feature_importance()` to `reports/metrics/feature_importance_v10.csv`; call `_write_submission()` to `data/submission_v10.csv`
- [x] T025 [US2] Run feature extraction on full training set (`data/train.csv`, 6256 points): `python scripts/experiment_v10.py --extract-only`; confirm total extraction time is under 10 minutes (SC-005); log elapsed time and feature shape
- [x] T026 [US2] Run full CV training in `scripts/experiment_v10.py` on the complete feature set; log cv_weighted_f1 and compare against SC-001 (≥ 0.8024) and SC-002 (≥ 0.79518 Kaggle OR CV improvement ≥ 0.005); compute `sklearn.metrics.classification_report` per fold, log per-class recall to console, assert `recall['SGAM'] >= baseline_recall` (SC-003, where baseline_recall is the SGAM recall from the best prior run documented in run-registry or runsheet), and write per-class metrics to `reports/metrics/cv_per_class_v10.csv`
- [x] T027 [US2] Generate Kaggle submission CSV at `data/submission_v10.csv`; verify the file has the correct format (ID column + predicted class column matching `data/sample_submission.csv` structure)

---

## Phase 6: User Story 3 — Feature Selection (P2)

**Story Goal**: Apply permutation importance-based feature selection to reduce the ~50-column multi-scale feature set to ≤ 25 features, retrain on the reduced set, and verify CV performance does not degrade by more than 0.01 (SC-004).

**Independent Test**: Call `select_by_permutation_importance(X_train, y_train, model, feature_names)` and assert the returned `FeatureSelectionResult` has `selected_count ≤ 25` and all `importance_mean ≥ 0`.

### TDD: Tests First

- [x] T028 [US3] Write test `tests/unit/test_feature_selection.py`: `test_select_returns_result` — given a small synthetic DataFrame (50 samples, 10 features), assert `select_by_permutation_importance(X, y, model, feature_names)` returns an object with attributes `selected_features` (list), `dropped_features` (list), and `importance_df` (DataFrame with columns feature_name, importance_mean, importance_std, selected, rank)
- [x] T029 [P] [US3] Write test `tests/unit/test_feature_selection.py`: `test_select_reduces_features` — given 20 features of which half are pure noise (random), assert the selection retains ≤ 15 features
- [x] T030 [P] [US3] Write test `tests/unit/test_feature_selection.py`: `test_select_logs_dropped` — mock the logger and assert a log.info call contains the name of at least one dropped feature and its importance score

### Implementation

- [x] T031 [US3] Implement `select_by_permutation_importance(X: pd.DataFrame, y: pd.Series, model, n_repeats: int = 10, importance_threshold: float = 0.0, corr_threshold: float = 0.95) -> FeatureSelectionResult` in `src/benthic_model/features/selection.py`: (1) apply Spearman correlation pre-filter to drop one of any pair with |r| > 0.95 (keep higher-variance); (2) call `sklearn.inspection.permutation_importance(model, X, y, n_repeats=n_repeats)`; (3) drop features with `importance_mean ≤ importance_threshold`; (4) log retained and dropped features with their importance scores (FR-010); cite Breiman (2001) in docstring
- [x] T032 [US3] Integrate feature selection into `scripts/experiment_v10.py`: after initial CV training, call `select_by_permutation_importance()` using the trained model; retrain on `selected_features` only; log CV weighted-F1 before and after selection; assert compliance with SC-004 (≤ 25 features, CV degradation ≤ 0.01)
- [x] T033 [US3] Generate new submission CSV at `data/submission_v10_selected.csv` using the selected-feature model; compare predicted classes against `data/submission_v10.csv` to quantify prediction changes

---

## Phase 7: Integration Test & Polish

**Purpose**: Validate end-to-end pipeline, confirm existing test suite still passes, document results.

- [x] T034 Write integration test `tests/integration/test_multiscale_pipeline.py`: using the committed fixture rasters at `tests/data/016/bathymetry.tif` and `tests/data/016/backscatter.tif` (created in T005), call `extract_btm_features(sample_points, bathy_tif="tests/data/016/bathymetry.tif", scales=[3,7], include_glcm=True, backscatter_tif="tests/data/016/backscatter.tif")`; assert output shape, all expected column names present (`btm_slope_3`, `btm_slope_7`, `btm_rdmv_7`, `btm_glcm_contrast_7`, etc.), no all-NaN columns — do NOT fall back to synthetic rasters (constitution Principle I)
- [x] T035 [P] Run full test suite: `pytest tests/ -m "not arcgis and not qgis" -v`; assert all 192 pre-existing tests still pass plus all new tests pass; log final test count
- [x] T036 [P] Write `docs/run-016-multiscale-terrain.md`: document approach (multi-scale terrain + GLCM), CV weighted-F1 results, per-scale feature importances, top selected features, Kaggle score, comparison to baseline runs
- [x] T037 Update `docs/runsheet-hybrid-kaggle.md`: add row(s) for experiment_v10 with CV score, Kaggle score (if submitted), feature count, and link to `docs/run-016-multiscale-terrain.md`

---

## Dependencies (Story Completion Order)

```
T001–T005 (Setup)
    │
    ├── T006–T016 (US1: multi-scale terrain) ─────────────────────────────┐
    │                                                                       │
    ├── T017–T023 (US4: GLCM texture) ─────────────────────────────────────┤
    │                                                                       │
    └──────────────────────────────────────────────► T024–T027 (US2: train/evaluate)
                                                            │
                                                            ▼
                                                   T028–T033 (US3: feature selection)
                                                            │
                                                            ▼
                                                   T034–T037 (Polish)
```

**US1 and US4 are independent** — can be implemented in parallel after Phase 1 setup.

---

## Parallel Execution Examples

### Stream A: Multi-Scale Terrain (US1)

```
T001 → T006, T007, T008, T009, T010, T011 (parallel tests) → T012, T013, T014 (parallel impls) → T015 → T016
```

### Stream B: GLCM Texture (US4) — concurrent with Stream A

```
T002 → T017, T018, T019, T020, T021 (parallel tests) → T022 → T023
```

### Stream C: Feature Selection stubs (can be prepared in parallel with A+B)

```
T003, T004, T005 → T028, T029, T030 (parallel tests) → T031 (await US2 completion for T032)
```

---

## Implementation Strategy

**MVP Scope**: US1 only (Tasks T001–T016) — delivers multi-scale terrain features and validates core hypothesis with minimal risk. Train a quick model after T016 to get an early CV signal before committing to GLCM (US4).

**Increment 2**: Add US4 (T017–T023) — GLCM features from backscatter. Retrain with combined feature set.

**Increment 3**: US2 (T024–T027) — Full experimental driver with CV evaluation and Kaggle submission.

**Increment 4**: US3 (T028–T033) — Feature selection to reduce dimensionality and potentially improve score.

---

## Format Validation

All tasks follow the required format: `- [ ] [TaskID] [P?] [Story?] Description with file path`

- Total tasks: 37
- Phase 1 (Setup): 5 tasks
- Phase 2 (Foundational): 0 tasks (existing project, dependencies already declared)
- Phase 3 (US1 — multi-scale terrain, P1): 11 tasks (6 tests, 5 implementation)
- Phase 4 (US4 — GLCM texture, P2): 7 tasks (5 tests, 2 implementation)
- Phase 5 (US2 — train/evaluate, P1): 4 tasks
- Phase 6 (US3 — feature selection, P2): 6 tasks (3 tests, 3 implementation)
- Phase 7 (Polish): 4 tasks

**Parallel opportunities**: T002, T003, T004, T005 with T001; T007–T011 with T006; T013–T014 with T012; T018–T021 with T017; T029–T030 with T028; T035–T036 with T034.

**Suggested MVP**: T001–T016 (US1 only) — delivers testable multi-scale terrain features and an early CV signal in ~13 tasks.
