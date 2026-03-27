# Tasks: GIS-Derived Feature Engineering

**Branch**: `008-gis-feature-engineering`
**Input**: Design documents from `specs/008-gis-feature-engineering/`
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Data model**: [data-model.md](data-model.md)

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (independent files, no blocking deps)
- **[Story]**: User story label (US1, US2, US3 as per spec.md priorities)
- Exact file paths included in every task description

---

## Phase 1: Setup

**Purpose**: Repository and test infrastructure ready before any implementation.

- [X] T001 Create `scripts/experiment_v5.py` with module docstring, top-level constants (WEIGHT_SLOPE=80, WEIGHT_BZ=20, WEIGHT_AF=20, N_BZ_CLUSTERS=5, N_AF_CLUSTERS=8, SEED=42, N_SPATIAL_BLOCKS=10, KRIGING_MIN_ZONE_PTS=10), and `if __name__ == "__main__": main()` stub
- [X] T002 [P] Create `tests/unit/test_experiment_v5_features.py` with import stubs and five empty test function skeletons matching the plan test plan (test_add_gis_features_raw_shape, test_gis_feature_weights, test_krige_gis_features_no_nan, test_write_submission_format, test_run_report_contains_all_runs)
- [X] T003 [P] Verify `outputs/gis_layers/rasters/slope.tif`, `backscatter_zones.tif`, and `acoustic_facies.tif` exist and are readable; confirm CRS == EPSG:28355 and shapes match `data/MBES/bathymetry.tif` (manual check or small validation script)

**Checkpoint**: `experiment_v5.py` stub importable; test file present with failing stubs; derived rasters confirmed on disk.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core helpers that all three user-story phases depend on. MUST be complete before US1 work begins.

**CRITICAL**: No user story implementation can begin until this phase is complete.

- [X] T004 Implement `_load_rasters()` in `scripts/experiment_v5.py` -- loads `data/MBES/bathymetry.tif` and `data/MBES/backscatter.tif` into NumPy float32 arrays, returns `(bathy, back, bathy_f, back_f, bt, bkt, cell_size)`; mirrors `load_rasters()` in `experiment_v2.py`
- [X] T005 [P] Implement `_ensure_derived_rasters()` in `scripts/experiment_v5.py` -- checks existence of `outputs/gis_layers/rasters/slope.tif`, `backscatter_zones.tif`, `acoustic_facies.tif`; prints a clear message if any are missing with instruction to run `step1_prepare_gis_layers.py`; raises `FileNotFoundError` if missing (does not silently continue)
- [X] T006 [P] Implement `_load_derived_rasters()` in `scripts/experiment_v5.py` -- loads the three derived rasters from disk into NumPy arrays; returns `(slope_arr, bz_arr, af_arr, derived_transform)` where `derived_transform` is the affine transform from `slope.tif`
- [X] T007 Implement `extract_all_features(xs, ys, bathy, back, bathy_f, back_f, bt, bkt, cell_size)` in `scripts/experiment_v5.py` -- verbatim copy of the same function from `scripts/experiment_v2.py` (multi-scale focal stats, curvature, GLCM, patch stats, interactions); returns `pd.DataFrame` of ~114 base features
- [X] T008 [P] Implement `get_models(seed)` in `scripts/experiment_v5.py` -- verbatim copy from `scripts/experiment_v2.py`; returns dict of four ensemble model factory lambdas (lgbm, xgb, catboost, rf) with identical hyperparameters
- [X] T009 Implement `_run_spatial_block_cv(X, y_labels, enc, train_coords, seed, n_blocks)` in `scripts/experiment_v5.py` -- parameterised spatial-block CV loop adapted from `experiment_v2.py`; `train_coords` is a `(n, 2)` array of `[x, y]` values used for KMeans block assignment; returns `(weighted_f1, per_class_f1_dict, mean_importances_series, elapsed_seconds)`

**Checkpoint**: Foundation helpers importable; `test_experiment_v5_features.py` can import from the module; base feature extraction produces ~114-column DataFrame (spot-check manually).

---

## Phase 3: User Story 1 - GIS-Derived Feature Engineering and Baseline Prediction (Priority: P1)

**Goal**: Augment the v2 feature set with raw point-sampled slope/backscatter-zones/acoustic-facies features (80/20/20 weighted), train the ensemble, compute baseline vs Run A weighted-F1, and produce `data/submission_v5_gis_features.csv`.

**Independent Test**: `python scripts/experiment_v5.py --phase baseline,run_a` (or full run) completes without error; `data/submission_v5_gis_features.csv` exists with correct `ID,class` format and same row count as `data/test.csv`.

### Tests for User Story 1

- [X] T010 [P] [US1] Implement `test_add_gis_features_raw_shape` in `tests/unit/test_experiment_v5_features.py`: create a 10-row synthetic base DataFrame + fake slope/bz/af arrays; call `_add_gis_features_raw()`; assert output has `n_base + 14` columns and contains `gis_slope`, `gis_bz_0`, `gis_bz_4`, `gis_af_0`, `gis_af_7`
- [X] T011 [P] [US1] Implement `test_gis_feature_weights` in `tests/unit/test_experiment_v5_features.py`: single point, slope=1.0 -> `gis_slope == 80.0`; bz=2 -> `gis_bz_2 == 20.0` and all other `gis_bz_*` columns == 0.0; af=5 -> `gis_af_5 == 20.0`
- [X] T012 [P] [US1] Implement `test_write_submission_format` in `tests/unit/test_experiment_v5_features.py`: call `_write_submission(tmp_path, pred_labels, test_ids, "test_run")`; assert CSV has columns `["ID","class"]`; row count equals `len(test_ids)`; no duplicate IDs
- [X] T013 [US1] Run `python -m pytest tests/unit/test_experiment_v5_features.py::test_add_gis_features_raw_shape tests/unit/test_experiment_v5_features.py::test_gis_feature_weights tests/unit/test_experiment_v5_features.py::test_write_submission_format` -- confirm tests FAIL (red) before implementation

### Implementation for User Story 1

- [X] T014 [US1] Implement `_add_gis_features_raw(base_df, slope_arr, bz_arr, af_arr, derived_transform, xs, ys)` in `scripts/experiment_v5.py` -- samples each derived raster at `(xs, ys)` using `sample_raster_at_points()`; applies global-median fallback for NaN; scales slope by WEIGHT_SLOPE; OHE-encodes bz (5 cols) and af (8 cols) via `pd.get_dummies`; scales OHE columns by WEIGHT_BZ and WEIGHT_AF respectively; appends all 14 new columns to `base_df`; returns augmented DataFrame
- [X] T015 [US1] Implement `_write_submission(out_path, pred_labels, test_ids, run_name)` in `scripts/experiment_v5.py` -- writes `ID,class` CSV; asserts row count == len(test_ids) before writing
- [ ] T016 [US1] Implement `baseline` and `run_a` phases in `main()` in `scripts/experiment_v5.py`:
  - baseline: build v2 feature matrix (no GIS); call `_run_spatial_block_cv(X, y, enc, train_df[["x","y"]].values, SEED, N_SPATIAL_BLOCKS)`; store `baseline_f1` and `baseline_importances`
  - run_a: build augmented feature matrix via `_add_gis_features_raw()`; call `_run_spatial_block_cv(X_aug, y, enc, train_df[["x","y"]].values, SEED, N_SPATIAL_BLOCKS)`; store `run_a_f1` and `run_a_importances`; fit final model on full train set; predict test; write `data/submission_v5_gis_features.csv`
- [ ] T017 [US1] Run tests T010-T012 again -- confirm they now PASS (green); also manually verify `data/submission_v5_gis_features.csv` row count matches `data/test.csv`

**Checkpoint**: `data/submission_v5_gis_features.csv` exists; US1 tests green; `baseline_f1` and `run_a_f1` printed to stdout.

---

## Phase 4: User Story 2 - Kriging-Smoothed Feature Improvement and Comparative Validation (Priority: P2)

**Goal**: Replace raw point-sampled GIS features with zone-stratified ordinary kriging predictions; train the same ensemble; produce `data/submission_v5_gis_features_kriging.csv`; compare with Run A.

**Independent Test**: `python scripts/experiment_v5.py` (full run) completes; `data/submission_v5_gis_features_kriging.csv` exists with correct format; run report shows both run_a and run_b F1 values.

### Tests for User Story 2

- [X] T018 [P] [US2] Implement `test_krige_gis_features_no_nan` in `tests/unit/test_experiment_v5_features.py`: create 30 synthetic train points with random coords + zone values (bz: 0-4, af: 0-7, slope: float) and 10 test points; call `_krige_gis_features()`; assert result shape == (10, 3); assert no NaN values in result
- [X] T019 [US2] Run `python -m pytest tests/unit/test_experiment_v5_features.py::test_krige_gis_features_no_nan` -- confirm test FAILS (red) before implementation

### Implementation for User Story 2

- [X] T020 [US2] Implement `_krige_gis_features(train_df, test_df, slope_tr, bz_tr, af_tr, slope_te_raw, bz_te_raw, af_te_raw)` in `scripts/experiment_v5.py`:
  - Uses `bz_tr` as stratification zones (5 zones)
  - For each zone and each of the three GIS features (slope, bz value, af value):
    - If zone has >= KRIGING_MIN_ZONE_PTS training points: fit `OrdinaryKriging(xs_zone, ys_zone, values_zone, variogram_model=KRIGING_VARIOGRAM, nlags=KRIGING_NLAGS, weight=True)` from pykrige; predict at test points in zone
    - Else: fit `KNeighborsRegressor(n_neighbors=min(5, n_pts))` on zone train coords; predict at test coords; emit `warnings.warn(f"Zone {z}: KNN fallback ({n_pts} pts)")`
  - Returns DataFrame with columns `krige_slope`, `krige_bz`, `krige_af` for all test points (no NaN)
- [X] T021 [US2] Implement `_add_gis_features_kriged(base_df, kriged_df)` in `scripts/experiment_v5.py`:
  - Rounds `krige_bz` to nearest int, clips to [0, N_BZ_CLUSTERS-1]; OHE; scale by WEIGHT_BZ
  - Rounds `krige_af` to nearest int, clips to [0, N_AF_CLUSTERS-1]; OHE; scale by WEIGHT_AF
  - Scales `krige_slope` by WEIGHT_SLOPE
  - Appends 14 `gis_*` columns to `base_df`; returns augmented DataFrame
- [ ] T022 [US2] Implement `run_b` phase in `main()`: training uses the same raw-sampled GIS feature matrix as Run A (kriging is NOT applied during CV — it only applies to test-point inference); call `_run_spatial_block_cv(X_aug, y, enc, train_df[["x","y"]].values, SEED, N_SPATIAL_BLOCKS)` to produce `run_b_f1` (will equal `run_a_f1` — this is expected and documented); for the final test predictions: call `_krige_gis_features()` then `_add_gis_features_kriged()` to build kriged test feature matrix; predict with final model trained on full raw-feature train set; write `data/submission_v5_gis_features_kriging.csv`. Note: Run A vs Run B differ only in test predictions, not in CV F1 — document this in the run report.
- [ ] T023 [US2] Run test T018 again -- confirm it now PASSES (green); spot-check that `data/submission_v5_gis_features_kriging.csv` has correct row count

**Checkpoint**: `data/submission_v5_gis_features_kriging.csv` exists; T018 green; `run_b_f1` printed to stdout alongside `baseline_f1` and `run_a_f1`.

---

## Phase 5: User Story 3 - Best Submission Selection and Pull Request (Priority: P3)

**Goal**: Compare Run A vs Run B, designate best submission, write run report, create PR.

**Independent Test**: `data/submission_v5_best.csv` exists and its content matches either Run A or Run B submission; `docs/run-008-gis-feature-engineering.md` exists and contains the comparison table with all three run F1 values.

### Tests for User Story 3

- [X] T024 [P] [US3] Implement `test_run_report_contains_all_runs` in `tests/unit/test_experiment_v5_features.py`: create a synthetic results dict (`{"baseline": 0.70, "run_a": 0.73, "run_b": 0.75, ...}`); call `_write_run_report(out_dir=tmp_path, results=results)`; read output Markdown at `tmp_path / "run-008-gis-feature-engineering.md"`; assert all three run names and all three F1 values appear in the text
- [X] T025 [US3] Run `python -m pytest tests/unit/test_experiment_v5_features.py::test_run_report_contains_all_runs` -- confirm test FAILS (red) before implementation

### Implementation for User Story 3

- [X] T026 [US3] Implement `_write_run_report(out_dir, results)` in `scripts/experiment_v5.py` -- `out_dir` defaults to `_REPO / "docs"`; writes `out_dir / "run-008-gis-feature-engineering.md"`; report sections: (1) parameters table (constants, seed, n_blocks, kriging config), (2) metrics comparison table (baseline / run_a / run_b: weighted-F1, per-class F1, top-3 features by gain, wall-clock seconds), (3) note that Run A and Run B CV F1 values are equal by design (kriging applies only to test-point inference -- see spec US2/AC2), (4) best-run identification paragraph, (5) brief interpretation of whether GIS features helped vs baseline
- [ ] T027 [US3] Implement best-run selection and `submission_v5_best.csv` copy in `main()` -- after both runs: compare `run_a_f1` and `run_b_f1`; copy the higher-scoring submission file to `data/submission_v5_best.csv`; print which run was selected and the winning F1 score
- [ ] T028 [US3] Run full experiment: `python scripts/experiment_v5.py` -- verify all three submission CSVs exist with correct row counts; verify `docs/run-008-gis-feature-engineering.md` exists and contains the metrics table
- [ ] T029 [US3] Run test T024 again -- confirm it now PASSES (green)
- [ ] T030 [US3] Run full test suite: `python -m pytest tests/unit/test_experiment_v5_features.py -v` -- all 5 tests green; no ruff lint errors (`python -m ruff check scripts/experiment_v5.py tests/unit/test_experiment_v5_features.py`)
- [ ] T031 [P] [US3] git add all new and modified files; commit with message `feat(008): GIS feature engineering experiment -- slope/bz/af 80/20/20 weighting, run A+B comparison, best submission v5`
- [ ] T032 [US3] Create PR from `008-gis-feature-engineering` -> `main`; PR description MUST reference: (a) the weighted-F1 improvement over v4 baseline, (b) link to `docs/run-008-gis-feature-engineering.md`, (c) which of Run A or Run B was selected as best

**Checkpoint**: All 5 unit tests green; lint passes; 3 submission CSVs and run report present; PR created.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T035 [US3] Update `CHANGELOG` -- add entry for branch 008 under the appropriate version header: new files `scripts/experiment_v5.py`, `tests/unit/test_experiment_v5_features.py`, `data/submission_v5_*.csv`, `docs/run-008-gis-feature-engineering.md`; describe the GIS feature engineering experiment and weighted-F1 result
- [ ] T033 [P] Verify `data/submission_v5_best.csv` row count matches `data/test.csv` and has no duplicate IDs: `python -c "import pandas as pd; s=pd.read_csv('data/submission_v5_best.csv'); t=pd.read_csv('data/test.csv'); assert len(s)==len(t) and s['ID'].nunique()==len(s), 'Submission mismatch'"`
- [ ] T034 [P] Follow the quickstart in `specs/008-gis-feature-engineering/quickstart.md` from a clean terminal (no cached state) to verify the end-to-end reproduce instructions are accurate; fix any discrepancies in `quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies -- start immediately
- **Foundational (Phase 2)**: Depends on setup; BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 completion
- **US2 (Phase 4)**: Depends on Phase 2 completion; independent from US1 except it reads `run_a_f1` for comparison -- implement US1 first for the comparison number, but US2 code can be written in parallel
- **US3 (Phase 5)**: Depends on US1 AND US2 (needs both run outputs and F1 scores)
- **Polish (Phase 6)**: Depends on US3 completion

### User Story Dependencies

- **US1 (P1)**: Can start after Phase 2; no dependencies on US2/US3
- **US2 (P2)**: Can start after Phase 2; uses US1 raw-feature helpers (`_load_derived_rasters`, `_add_gis_features_raw`) but code is additive
- **US3 (P3)**: Strictly depends on both US1 and US2 being complete (needs all F1 values)

### Parallel Opportunities

| Task group       | Parallelisable?      | Notes                                                           |
| ---------------- | -------------------- | --------------------------------------------------------------- |
| T002, T003       | Yes (with T001)      | Test stubs and raster check independent of script stub          |
| T005, T006, T008 | Yes (within Phase 2) | Each helper function in isolation                               |
| T010, T011, T012 | Yes                  | Independent test functions                                      |
| T018             | Yes (with T010-T012) | US2 test can be written while US1 implementation is in progress |
| T024             | Yes (with T018)      | US3 test can be written while US2 implementation is in progress |
| T031, T033, T034 | Yes                  | Final checks independent of each other                          |

---

## Parallel Execution Example: User Story 1

```bash
# Terminal A: write tests (T010, T011, T012)
python -m pytest tests/unit/test_experiment_v5_features.py -k "shape or weights or submission" -v
# Expected: FAIL (red) until T013 done

# Terminal B: implement _add_gis_features_raw (T014)
# edit scripts/experiment_v5.py

# Terminal C: implement _write_submission (T015)
# edit scripts/experiment_v5.py

# After T014+T015: implement main() baseline+run_a phases (T016)
# Then T017: re-run tests -> green
```

---

## Implementation Strategy

### MVP (User Story 1 only)

Completing Phase 1 + Phase 2 + Phase 3 delivers:

- A working augmented-feature experiment script
- A valid `data/submission_v5_gis_features.csv`
- Documented baseline vs Run A F1 comparison (printed to stdout)
- 3 of 5 unit tests passing

This is the minimum deliverable to validate the GIS feature engineering hypothesis.

### Full Delivery

All 5 phases deliver both run variants, the run report, and the PR -- matching all success criteria (SC-001 through SC-005) in the spec.

---

## Format Validation

All tasks follow the required checklist format:

- [x] Every task starts with `- [ ]`
- [x] Every task has a sequential Task ID (T001-T034)
- [x] `[P]` marker present only on parallelisable tasks
- [x] `[USn]` label present on all user-story phase tasks (Phases 3-5)
- [x] Every task includes an exact file path
