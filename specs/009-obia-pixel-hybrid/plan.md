# Implementation Plan: OBIA + Pixel-Based Hybrid Classification

**Branch**: `009-obia-pixel-hybrid` | **Date**: 2026-03-27 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/009-obia-pixel-hybrid/spec.md`

**Reference**: Ierodiaconou et al. (2018) Marine Geodesy DOI: 10.1007/s11001-017-9338-z

## Summary

Implement a combined OBIA + pixel-based (PB) hybrid habitat classification experiment (`scripts/experiment_v6.py`) following Ierodiaconou et al. (2018). The experiment derives 8 pixel-level terrain/acoustic features from MBES rasters, performs image segmentation using SLIC on the same dataset, computes 10 per-segment statistics, and trains a 4-model ensemble (LightGBM, XGBoost, CatBoost, RandomForest) with spatial block CV. The target is a spatial CV weighted F1 ≥ 0.72 on the Refuge Cove 5-class benthic dataset, surpassing the current v2 baseline. The paper's combined model achieved 83.6% OA (κ=0.78) versus 72.5% PB-only and 78.5% OB-only.

## Technical Context

**Language/Version**: Python 3.13.5 (`.venv\Scripts\python.exe`)  
**Primary Dependencies**: rasterio, numpy, scipy, scikit-image (0.26), lightgbm, xgboost, catboost, scikit-learn (1.8), pandas  
**Storage**: File-based — GeoTIFF rasters in `data/MBES/`, CSV in `data/`, Markdown in `docs/`  
**Testing**: pytest; unit tests in `tests/unit/test_experiment_v6_features.py`  
**Target Platform**: Standalone Python script; no GIS runtime required (arcpy-free)  
**Project Type**: Single experiment script (data-science / ML pipeline)  
**Performance Goals**: Full raster derivation + segmentation completes in < 10 minutes on a modern workstation  
**Constraints**: 18-feature input only (per spec clarification Q1); same spatial CV as v2 for comparability  
**Scale/Scope**: 4040×4743 raster grid; 566 training points; 98 test points; 5 target classes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **I. TDD** — Failing unit tests for all 8 pixel-based features and segment-stat extraction will be written before implementation. See `tests/unit/test_experiment_v6_features.py` in tasks.md.
- [x] **II. Code Quality** — `ruff` is already configured in `pyproject.toml`; existing CI lint check applies. New code will be linted before commit.
- [x] **III. Performance** — All raster operations use vectorised NumPy/SciPy (ndimage, gradient). No pixel-by-pixel Python loops. SLIC segmentation uses scikit-image's C-optimised implementation.
- [x] **IV. Scientific Accuracy** — Every algorithm has a peer-reviewed citation: slope=Horn (1981), VRM=Sappington et al. (2007), complexity=Jenness (2004), curvature=Zevenbergen & Thorne (1987), aspect=Roberts (1986), segmentation=Achanta et al. (2012), classifiers=vendor papers. All references in `research.md`.
- [x] **V. Platform Portability** — `experiment_v6.py` is a standalone Python script with no arcpy/ArcGIS/QGIS dependency. Reads rasters via rasterio. Existing `btm.core.vrm` and `btm.core.slope` are already arcpy-free.
- [x] **VI. Simplicity** — Single experiment script + single test file. No new library modules. No CLI wrappers. All logic in one file for reproducibility and auditability.

**Post-design re-check**: All checks still pass. No architecture changes introduced during Phase 1 design.

## Project Structure

### Documentation (this feature)

```text
specs/009-obia-pixel-hybrid/
├── plan.md              # This file
├── research.md          # Phase 0 — technical decisions (DONE)
├── data-model.md        # Phase 1 — entities and data flow (DONE)
├── quickstart.md        # Phase 1 — how to run (DONE)
└── tasks.md             # Phase 2 — task breakdown (/speckit.tasks command)
```

### Source Code (repository root)

```text
scripts/
└── experiment_v6.py     # Main experiment script (new)

tests/
└── unit/
    └── test_experiment_v6_features.py  # Unit tests (new)

data/
└── submission_v6_best.csv              # Output: Kaggle submission (generated)

docs/
└── run-009-obia-pixel-hybrid.md        # Output: run report (generated)
```

**Reused without modification**:
- `btm/core/slope.py` — `compute_slope(array, cell_size, nodata)`
- `btm/core/vrm.py` — `compute_vrm(array, neighborhood_size, cell_size, nodata)`
- `btm/features/extract.py` — `sample_raster_at_points(arr, transform, xs, ys)`

---

## Phase 0 — Research Findings

Full details: [`research.md`](research.md)

| Decision | Outcome | Key Rationale |
|----------|---------|---------------|
| Segmentation algorithm | SLIC (`skimage.segmentation.slic`, n_segments≈4000, compactness=0.01) | Direct n_segments control, multi-channel, C-optimised |
| Rugosity | `btm.core.vrm.compute_vrm()` — Vector Ruggedness Measure | Already implemented; marine habitat mapping standard |
| Complexity | `1/cos(slope_rad)` filtered with `ndimage.uniform_filter(..., size=3)` | Jenness (2004) surface-to-planar area approximation |
| Max curvature | `abs(scipy.ndimage.laplace(bathy)) / cell_size²` | Fast Laplace approximation of profile+plan curvature |
| Northness / Eastness | `cos/sin(arctan2(∂bathy/∂x, ∂bathy/∂y))` | Circular-safe aspect decomposition; values in [-1, 1] |
| Classifier ensemble | Reuse `get_models()` from `experiment_v2.py` | Apples-to-apples baseline comparison |
| CV strategy | KMeans spatial block (n_clusters=10, seed=42) | Identical to v2; avoids spatial autocorrelation leakage |
| Seg input channels | Normalised [bathymetry, backscatter, vrm] | Matches paper's eCognition channel selection (Section 2.2) |

All NEEDS CLARIFICATION items from spec and template are resolved. No blocking unknowns.

---

## Phase 1 — Design Artefacts

### Data Model

Full details: [`data-model.md`](data-model.md)

**Key entities**:
- **E1** Raw rasters (bathy, back, transform, cell_size)
- **E2** Pixel-based derivatives (slope, vrm, complexity, max_curvature, northness, eastness)
- **E3** Segment label array (H×W int32, from SLIC)
- **E4** Segment statistics DataFrame (n_segs × 10)
- **E5** Point feature tables (X_train shape 566×18, X_test shape 98×18)
- **E6** Spatial CV block assignments (574 ints in [0, 9])
- **E7** OOF probability arrays (dict[str → ndarray(566,5)])
- **E8** Outputs (submission CSV + run report Markdown)

**Data flow**:
```
E1 → E2 (vectorised derivation)
E1 + E2 → E3 (SLIC segmentation on normalised stack)
E3 + E1 + E2 → E4 (per-segment aggregation)
E2 (sampled at points) + E4 (joined) → E5
E5 + E6 → E7 (spatial CV loop) → E8
```

### Feature List (18 total, 3 configurations)

| Config | Features | Source Entity |
|--------|----------|---------------|
| **PB-only** (8) | depth, backscatter, slope, rugosity, complexity, max_curvature, northness, eastness | E2 |
| **OB-only** (10) | seg_bathy_{mean,std,skew}, seg_back_{mean,std,skew}, seg_vrm_{mean,std,skew}, seg_pixel_count | E4 |
| **Combined** (18) | all PB + all OB | E2 + E4 |

All three configurations will be evaluated in CV to reproduce the paper's three-way comparison.

### Segmentation Implementation Notes

```python
from skimage.segmentation import slic

# Normalise channels to zero mean unit variance
seg_stack = np.stack([
    (bathy - bathy.mean()) / bathy.std(),
    (back  - back.mean())  / back.std(),
    (vrm   - vrm.mean())   / vrm.std(),
], axis=-1)

# NaN → 0 before passing to SLIC (nodata cells will be ignored post-hoc)
seg_stack = np.nan_to_num(seg_stack, nan=0.0)

labels = slic(
    seg_stack,
    n_segments=4000,       # target ~4800 px/segment ≈ 306 m²
    compactness=0.01,       # low compactness → irregular natural boundaries
    enforce_connectivity=True,
    start_label=0,
)
```

### Segment Statistics Aggregation

```python
from scipy.stats import skew as scipy_skew

seg_stats = {}
for seg_id in np.unique(labels):
    mask = labels == seg_id
    for arr, name in [(bathy, 'bathy'), (back, 'back'), (vrm, 'vrm')]:
        vals = arr[mask]
        valid = vals[np.isfinite(vals)]
        seg_stats[seg_id] = {
            f'seg_{name}_mean': valid.mean(),
            f'seg_{name}_std':  valid.std(),
            f'seg_{name}_skew': scipy_skew(valid) if len(valid) >= 3 else 0.0,
        }
    seg_stats[seg_id]['seg_pixel_count'] = mask.sum()
```

*Note*: This loop over unique segment IDs is acceptable since n_segments ≈ 4000 (not pixel-level).

### Spatial CV Pattern (reused from v2)

```python
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_sample_weight

km = KMeans(n_clusters=10, random_state=42, n_init=10)
blocks = km.fit_predict(train[['x', 'y']].values)

for name, make_model in models.items():
    oof_p = np.zeros((len(y), n_classes))
    for b in range(10):
        va = np.where(blocks == b)[0]
        tr = np.where(blocks != b)[0]
        m = make_model()
        if name == 'xgb':
            sw = compute_sample_weight('balanced', y[tr])
            m.fit(X[tr], y[tr], sample_weight=sw)
        else:
            m.fit(X[tr], y[tr])
        oof_p[va] = m.predict_proba(X[va])
    oof_proba[name] = oof_p
```

### Unit Tests Planned

Tests in `tests/unit/test_experiment_v6_features.py`:

1. `test_pb_features_all_finite` — all 8 PB columns non-NaN for training points
2. `test_northness_eastness_range` — values in [-1.001, 1.001]
3. `test_segment_labels_shape` — `labels.shape == bathy.shape`
4. `test_segment_labels_non_negative` — all label values ≥ 0
5. `test_segment_mean_size_range` — mean pixel count in [2400, 7200]
6. `test_ob_features_all_finite` — 10 OB columns non-NaN for training points
7. `test_combined_feature_count` — combined DataFrame has exactly 18 feature columns

### Run Report Template

`docs/run-009-obia-pixel-hybrid.md` will include:
- Paper reference and motivation
- Feature derivation notes and any parameter tuning
- CV F1 per model: PB-only / OB-only / Combined
- Ensemble CV F1 comparison (3 configs)
- Standard classification report (5 classes)
- Top 10 feature importances (from LightGBM)
- Mean segment size vs target
- Submission file location

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Combined CV F1 < OB-only or PB-only | Medium | Low | SC-002 is a SHOULD; submit best-CV config regardless (per clarification Q5) |
| SLIC mean segment size far from 300 m² | Medium | Low | Adjust `n_segments` empirically; log actual mean in run report |
| NaN in segment stats (near-edge or nodata segments) | Low | Medium | Clip nodata before aggregation; fallback `scipy_skew` with `len < 3` guard |
| CatBoost install missing in new environment | Low | Low | `pip install catboost` already in requirements; checked in session |
| Skewness undefined for segments with < 3 pixels | Low | Low | Guard with `len(valid) >= 3` else 0.0 |

---

## Next Step

Run `/speckit.tasks` to generate `specs/009-obia-pixel-hybrid/tasks.md` with dependency-ordered implementation tasks.

