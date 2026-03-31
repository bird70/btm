# Implementation Plan: Multi-Scale Terrain Features

**Branch**: `016-multiscale-terrain-features` | **Date**: 2026-03-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/016-multiscale-terrain-features/spec.md`

## Summary

Add multi-scale terrain feature computation (calculate-then-average at 5 window sizes) and backscatter GLCM texture features to the BTM feature extraction pipeline, with permutation importance-based feature selection, to improve benthic habitat classification accuracy beyond the current 0.795 Kaggle / 0.802 CV baselines.

## Technical Context

**Language/Version**: Python 3.11+ (current: 3.13.5)  
**Primary Dependencies**: NumPy, SciPy (`scipy.ndimage.uniform_filter`), scikit-image (`graycomatrix`/`graycoprops`), rasterio, scikit-learn, pandas  
**Storage**: GeoTIFF rasters (`data/MBES/`), CSV point data (`data/train.csv`, `data/test.csv`)  
**Testing**: pytest (192 tests currently passing)  
**Target Platform**: Windows (dev), cross-platform core  
**Project Type**: Library (BTM core) + ML pipeline (benthic_model)  
**Performance Goals**: Feature extraction for 6256 training points within 10 minutes  
**Constraints**: Rasters < 100 MB (survey-scale); existing `extract_btm_features` reads full band  
**Scale/Scope**: 5 scales × ~7 derivatives = ~35 multi-scale columns + ~10 GLCM columns ≈ 45–60 total new features; feature selection reduces to ≤ 25

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

- [x] **I. TDD** — Tests will be written before implementation for: focal_mean_multiscale, RDMV computation, GLCM extraction, feature selection, column naming conventions.
- [x] **II. Code Quality** — Ruff linting configured; CI blocks on lint errors. All new functions will have docstrings with cited references.
- [x] **III. Performance** — `scipy.ndimage.uniform_filter` is the vectorised focal mean operation (NumPy/SciPy, no Python loops). Raster < 100 MB guard already enforced in `extract_eco_raster_features`. GLCM uses `skimage.feature.graycomatrix` (C-optimised).
- [x] **IV. Scientific Accuracy** — Multi-scale aggregation follows Misiuk et al. (2021) "calculate-average" method. RDMV follows Lecours et al. (2017). GLCM follows Haralick (1973) via scikit-image. All references cited in spec.
- [x] **V. Platform Portability** — All new code goes in `btm/core/` (pure Python + NumPy/SciPy) and `btm/features/` (pandas integration). No arcpy/QGIS dependency. No ArcGIS-only fallback needed.
- [x] **VI. Simplicity** — Each new computation is a standalone function: `focal_mean_multiscale()`, `compute_rdmv()`, `compute_glcm_texture()`. Single responsibility preserved. No speculative abstractions.

## Project Structure

### Documentation (this feature)

```text
specs/016-multiscale-terrain-features/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
btm/
├── core/
│   ├── multiscale.py      # NEW: focal_mean_multiscale(), compute_rdmv()
│   └── glcm.py            # NEW: compute_glcm_texture()
├── features/
│   └── extract.py         # MODIFIED: add scales param, integrate multiscale + GLCM

src/benthic_model/
├── features/
│   └── selection.py       # NEW: permutation importance feature selection
├── cli.py                 # MODIFIED: add --multiscale flag

scripts/
└── experiment_v10.py      # NEW: multi-scale experiment driver

tests/
├── unit/
│   ├── test_multiscale.py        # NEW: focal mean at multiple scales, RDMV
│   ├── test_glcm.py              # NEW: GLCM contrast/homogeneity
│   └── test_feature_selection.py # NEW: permutation importance selection
└── integration/
    └── test_multiscale_pipeline.py  # NEW: end-to-end multi-scale extraction
```

**Structure Decision**: Two new core modules (`multiscale.py`, `glcm.py`) follow the existing pattern of one-algorithm-per-file in `btm/core/`. Feature selection lives in `src/benthic_model/features/` since it's ML-pipeline-specific, not a terrain algorithm.

## Complexity Tracking

No constitution violations. No complexity justifications needed.

## Post-Design Constitution Re-Check

_Re-evaluated after Phase 1 design completion:_

- [x] **I. TDD** — Test files identified: `test_multiscale.py`, `test_glcm.py`, `test_feature_selection.py`, `test_multiscale_pipeline.py`. Tests will cover: focal mean correctness at each scale, RDMV formula, GLCM contrast/homogeneity values, column naming convention, boundary handling, feature selection output shape.
- [x] **II. Code Quality** — All new functions have cited peer-reviewed references (Misiuk 2021, Lecours 2017, Haralick 1973, Sappington 2007, Horn 1981). Ruff configured.
- [x] **III. Performance** — `uniform_filter` is O(n) per pixel (separable filter). Full raster stays under 100 MB guard. GLCM extraction at point locations only (not full raster) keeps it tractable.
- [x] **IV. Scientific Accuracy** — Each algorithm traced to a reference: focal mean multi-scale (Misiuk et al. 2021), RDMV (Lecours et al. 2017), GLCM (Haralick 1973 via scikit-image), permutation importance (Breiman 2001).
- [x] **V. Platform Portability** — All new code in `btm/core/` (pure Python + NumPy/SciPy) and `btm/features/` (pandas). No arcpy dependency. Tests run without any GIS runtime.
- [x] **VI. Simplicity** — Two new single-responsibility core modules: `multiscale.py` (focal averaging + RDMV), `glcm.py` (texture features). One new ML module: `selection.py` (permutation importance). No speculative abstractions.
