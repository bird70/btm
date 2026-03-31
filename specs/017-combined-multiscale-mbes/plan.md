# Implementation Plan: Combined Multi-Scale BTM + MBES-8 Features

**Branch**: `017-combined-multiscale-mbes` | **Date**: 2026-03-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/017-combined-multiscale-mbes/spec.md`

## Summary

Create `experiment_v11.py` that combines the 33 selected multi-scale BTM features (from spec-016) with 8 MBES point-sample features (depth, backscatter, slope, VRM, complexity, max_curvature, northness, eastness), adds Random Forest as a third model alongside CatBoost and LightGBM, and evaluates via 5-fold spatial-blocked CV (matching R04/R06's scheme). This directly addresses both documented causes of the spec-016 performance gap vs the R04/R06 baseline (CV F1=0.8024).

## Technical Context

**Language/Version**: Python 3.11+ (current: 3.13.5)
**Primary Dependencies**: NumPy, SciPy, rasterio, scikit-learn (RandomForestClassifier), CatBoost, LightGBM, pandas
**Storage**: GeoTIFF rasters (`data/MBES/`), CSV point data (`data/train.csv`, `data/test.csv`), CSV feature cache
**Testing**: pytest (229 tests currently passing)
**Target Platform**: Windows (dev), cross-platform core
**Project Type**: Experiment script (standalone) reusing library code from `btm` and `benthic_model`
**Performance Goals**: Combined feature extraction for 6256 training points within 15 minutes; CV training within 15 minutes
**Constraints**: Reuse spec-016 cached BTM features (CSV); MBES-8 extraction adds ~30 s overhead
**Scale/Scope**: ~41 combined features (33 BTM-selected + 8 MBES); reduced to ≤ 25 after feature selection

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

- [x] **I. TDD** — Tests identified: `test_experiment_v11.py` for public helpers (`_write_feature_importance`, `_write_submission`, `_extract_mbes8_features`). Tests written before implementation as in spec-016 pattern.
- [x] **II. Code Quality** — Ruff linting configured; CI blocks on lint errors. The new script follows the same style and logging conventions as `experiment_v10.py`.
- [x] **III. Performance** — MBES-8 extraction uses existing vectorised rasterio point-sampling and NumPy/SciPy kernels (Horn slope, Sappington VRM). BTM-33 features loaded from CSV cache — no raster re-computation.
- [x] **IV. Scientific Accuracy** — MBES-8 features use the same algorithms as spec-015 and spec-016 (Horn 1981 slope, Sappington 2007 VRM, Evans 1980 max_curvature). RF hyperparameters match run-014 R04 exactly for reproducibility.
- [x] **V. Platform Portability** — The experiment script uses only standard Python, NumPy, SciPy, scikit-learn, CatBoost, LightGBM, rasterio, and pandas. No arcpy/QGIS dependency.
- [x] **VI. Simplicity** — Single new script (`experiment_v11.py`) following the established experiment pattern. MBES-8 extraction is a ~40-line function lifted from `experiment_v9.py`. No new library modules needed.

## Project Structure

### Documentation (this feature)

```text
specs/017-combined-multiscale-mbes/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
scripts/
└── experiment_v11.py      # NEW: combined BTM-33 + MBES-8 experiment driver

tests/
└── unit/
    └── test_experiment_v11.py  # NEW: tests for v11 public helpers

reports/
└── metrics/
    ├── cache_train_feats_v10.csv    # REUSED: spec-016 cached BTM features
    ├── cache_test_feats_v10.csv     # REUSED: spec-016 cached test features
    ├── feature_importance_v11.csv   # OUTPUT: combined-set feature importances
    ├── feature_selection_v11.csv    # OUTPUT: permutation importance results
    └── cv_per_class_v11.csv         # OUTPUT: per-class recall per fold

data/
├── submission_v11.csv               # OUTPUT: full-feature submission
└── submission_v11_selected.csv      # OUTPUT: selected-feature submission

docs/
├── run-017-combined-multiscale-mbes.md  # OUTPUT: run documentation
└── runsheet-hybrid-kaggle.md            # MODIFIED: append v11 rows
```

**Structure Decision**: Single new experiment script in `scripts/` following the established `experiment_v{N}.py` convention. No new library modules — MBES-8 extraction is a private helper within the script (same as v9). Feature selection reuses `src/benthic_model/features/selection.py` from spec-016.

## Complexity Tracking

No constitution violations. No complexity justifications needed.

## Post-Design Constitution Re-Check

_Re-evaluated after Phase 1 design completion:_

- [x] **I. TDD** — Test file identified: `tests/unit/test_experiment_v11.py` covering MBES-8 extraction helper, feature importance writer, submission writer. Tests written before implementation.
- [x] **II. Code Quality** — Script follows existing experiment conventions (logging, argparse, cache, dry-run). Ruff configured.
- [x] **III. Performance** — BTM-33 from CSV cache (~0.1 s). MBES-8 extraction: one rasterio open + point-sample + 6 derived arrays (~30 s). Total extraction < 1 min. CV training ~10 min.
- [x] **IV. Scientific Accuracy** — MBES-8 algorithms: Horn 1981 (slope, aspect → northness/eastness), Sappington 2007 (VRM), Evans 1980 (max_curvature), Wilson 2007 (complexity). All from `experiment_v9.py` proven code.
- [x] **V. Platform Portability** — Pure Python + standard scientific stack. No GIS runtime.
- [x] **VI. Simplicity** — One script, one test file, three output files. Reuses existing BTM cache, selection module, and experiment pattern.
