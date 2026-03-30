# Implementation Plan: Ecology-Informed Habitat Feature Engineering

**Branch**: `015-ecology-habitat-features` | **Date**: 2026-03-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/015-ecology-habitat-features/spec.md`

## Summary

Add ecology-informed bathymetric spatial derivatives (northness, eastness, max curvature, complexity) and an in-pipeline SGAM niche indicator to the benthic habitat classification pipeline to improve SGAM class recall (current per-class F1 ≤ 0.043). Raster derivatives are computed in `btm/features/extract.py` and sampled at point locations via the existing `btm-export-features` CLI. At training time, `src/benthic_model/features/eco_features.py` computes depth zone (4 quantile bins, ordinal) and the `btm_sgam_niche` binary indicator using data-driven p25 percentile cutoffs. A new `include_eco_features` flag in `FeatureFlags` gates these columns. Four new configs drive the Kaggle experiments; the GPU CatBoost holdover uses the existing artifact. Optional: expose RF hyperparameter overrides via `model_params` in `PipelineConfig`.

## Technical Context

**Language/Version**: Python 3.13.5  
**Primary Dependencies**: NumPy, SciPy, rasterio, pandas, scikit-learn, pytest  
**Storage**: CSV files (`data/train_btm.csv`, `data/train_btm_eco.csv`, `data/test.csv`); MBES raster (`data/MBES/bathymetry.tif`)  
**Testing**: pytest; suite passes `pytest tests/` from repo root without any GIS runtime  
**Target Platform**: Windows development machine; core logic is cross-platform  
**Project Type**: Scientific ML pipeline / analysis CLI  
**Performance Goals**: Eco-feature raster extraction completes in < 120 s for ~6,256 training rows and ~2,600 test rows  
**Constraints**: NaN rate ≤ 10% per eco-feature column; must not break existing test suite (154 pass); all new code covered by unit tests written first  
**Scale/Scope**: ~6,256 training + ~2,600 test rows; bathymetry raster ~25 cm resolution over Fagatele Bay study area

## Constitution Check

- [x] **I. TDD** — Failing tests for `eco_features.py` and new `FeatureFlags.include_eco_features` will be written in `tests/unit/test_eco_features.py` before any implementation code. Red-Green-Refactor cycle enforced.
- [x] **II. Code Quality** — ruff linting already configured; new code follows existing module conventions (PEP 8, docstrings on public functions, named constants for magic numbers like `N_DEPTH_BINS = 4`).
- [x] **III. Performance** — Raster derivatives use SciPy `ndimage.convolve` with vectorised NumPy kernels; no Python-level pixel loops. Raster cell-size is read from the GeoTIFF metadata (not hardcoded).
- [x] **IV. Scientific Accuracy** — Northness/eastness traced to Wilson et al. (2007) Table 1 and Horn (1981) gradient method. Curvature traced to Schmidt et al. (2003). Complexity traced to Wilson et al. (2007). References cited in `research.md` and module docstrings.
- [x] **V. Platform Portability** — All eco-feature computation is pure Python (NumPy/SciPy/rasterio). No arcpy or QGIS dependency. Raster paths passed at call time; no hardcoded paths.
- [x] **VI. Simplicity** — Each new module has a single responsibility: `eco_features.py` (in-pipeline transformations; depth zone + niche indicator only), raster derivative code added to `btm/features/extract.py` (consistent with where all other raster sampling lives).

## Project Structure

### Documentation (this feature)

```text
specs/015-ecology-habitat-features/
├── plan.md              # This file
├── research.md          # Phase 0: algorithm decisions + references
├── data-model.md        # Phase 1: entity definitions + data flow
├── contracts/
│   └── eco-feature-schema.md   # Feature column names, dtypes, value ranges
├── quickstart.md        # How to run the eco-feature experiments
└── tasks.md             # Phase 2 output (from /speckit.tasks)
```

### Source Code (repository root)

```text
btm/features/
├── extract.py           # MODIFIED: add extract_eco_raster_features() + include_eco_features param

src/benthic_model/
├── config.py            # MODIFIED: add FeatureFlags.include_eco_features + PipelineConfig.model_params
├── features/
│   ├── engineering.py   # MODIFIED: call add_eco_features() when include_eco_features flag is set
│   └── eco_features.py  # NEW: EcoFeatureTransformer (depth zone + SGAM niche indicator)

configs/
├── rf-btm-eco-depth.yaml         # NEW: RF + BTM + depth zones only
├── rf-btm-eco-full.yaml          # NEW: RF + BTM + all eco derivatives
├── rf-btm-interactions.yaml      # NEW: RF + BTM + pairwise interactions
├── rf-btm-tuned.yaml             # NEW: RF + BTM + tuned hyperparameters

tests/unit/
└── test_eco_features.py          # NEW: unit tests for EcoFeatureTransformer
```

## Complexity Tracking

No constitution violations. All additions follow existing module patterns without introducing new abstractions beyond what is required.

---

## Phase 0: Research

See [research.md](research.md) for full findings. Summary:

| Question | Decision | Rationale |
|---|---|---|
| Northness/eastness formula | `northness = sin(atan2(dy, dx))`, `eastness = cos(atan2(dy, dx))` using Horn (1981) gradients | Consistent with paper Table 1 "Sinus/Cosinus of azimuthal direction"; Horn method standard in ArcGIS Spatial Analyst |
| Max curvature computation | `max(|plan_curvature|, |profile_curvature|)` from 2nd-order polynomial fit | Schmidt et al. 2003 definition; maximum of the two principal curvatures |
| Complexity computation | Second derivative of slope (apply slope kernel twice, or finite-diff the slope surface) | Wilson et al. 2007; Laplacian of slope |
| Raster derivatives platform | `scipy.ndimage.convolve` with Sobel/Laplacian kernels on NumPy arrays | No arcpy dependency; reproducible on any machine with rasterio + scipy |
| Depth zone boundaries | 4 bins at training-set quantiles (p25, p50, p75 of `bathymetry` column) computed at fit time | Equal-frequency ensures populated bins even with skewed depth distribution |
| SGAM niche indicator threshold | p25 of `btm_fine_bpi` AND p25 of `btm_slope` in training set + depth within observed SGAM depth min/max | Data-driven; thresholds stored in run artifact for reproducibility |
| RF hyperparameter config | Add `model_params: dict` to `PipelineConfig`; passed to RF constructor via `**kwargs` | Minimal change; avoids proliferating specialized model builder subclasses |
| Code location for raster derivatives | Extend `btm/features/extract.py`; add `include_eco_features` param to `extract_btm_features()` | Consistent with all other raster-level BTM derivative code |

---

## Phase 1: Design & Contracts

### Data Model

See [data-model.md](data-model.md) for entity definitions.

**Data flow**:

```
data/MBES/bathymetry.tif
        │
        ▼ [btm-export-features --include-eco-features]
        │  (btm/features/extract.py::extract_btm_features + extract_eco_raster_features)
        │
data/train_btm_eco.csv
  columns added: btm_northness, btm_eastness, btm_max_curvature, btm_complexity
  (depth values already present from bathymetry column)
        │
        ▼ [benthic-model train --config configs/rf-btm-eco-*.yaml]
        │  (src/benthic_model/features/eco_features.py::EcoFeatureTransformer.fit_transform)
        │  fit: compute p25(bpi), p25(slope), depth_bins, sgam_depth_range from training set
        │  transform: add btm_depth_zone (int 1-4), btm_sgam_niche (int 0/1)
        │
 model training → artifacts/runs/<run_id>/
   model.joblib    (includes fitted EcoFeatureTransformer params)
   metrics.json    (per-class F1 for all 5 classes)
   eco_thresholds.json  (p25_bpi, p25_slope, depth_bin_edges, sgam_depth_[min|max])
        │
        ▼ [benthic-model predict]
        │  transform only (no re-fit)
        │
 data/submission_<run_id>.csv  →  kaggle competitions submit
```

### Contracts

See [contracts/eco-feature-schema.md](contracts/eco-feature-schema.md).

**New columns added by this feature** (all numeric, NaN-filled to 0.0 on transform):

| Column | Type | Range | Source |
|---|---|---|---|
| `btm_northness` | float64 | [−1, 1] | Raster extraction (Horn gradient → atan2) |
| `btm_eastness` | float64 | [−1, 1] | Raster extraction (Horn gradient → atan2) |
| `btm_max_curvature` | float64 | unbounded | Raster extraction (2nd-order polynomial) |
| `btm_complexity` | float64 | ≥ 0 | Raster extraction (slope of slope) |
| `btm_depth_zone` | int64 | {1, 2, 3, 4} | Training-time fit (quantile bins of bathymetry) |
| `btm_sgam_niche` | int64 | {0, 1} | Training-time fit (p25 BPI + slope + SGAM depth range) |

### RF Hyperparameter Config

`PipelineConfig` gains an optional `model_params: dict[str, Any] | None = None` field. When present, its contents are forwarded as keyword arguments to the RF (or other model) constructor. Example config:

```yaml
model_type: rf
model_params:
  n_estimators: 500
  max_features: sqrt
  min_samples_leaf: 1
feature_flags:
  include_btm_features: true
  include_eco_features: true
```

### Agent Context Update

Run after Phase 1 artifacts are written:

```powershell
.specify/scripts/powershell/update-agent-context.ps1 -AgentType copilot
```

---

## Experiment Plan (Today's Kaggle Batch)

| Priority | Run | Config | Feature Set | Notes |
|---|---|---|---|---|
| 1 | GPU CatBoost (existing) | `candidate.yaml` + GPU | BTM winner flags | Artifact already exists — predict only |
| 2 | RF + BTM + eco-depth | `rf-btm-eco-depth.yaml` | BTM + depth_zone + sgam_niche | Test depth signal first |
| 3 | RF + BTM + all eco | `rf-btm-eco-full.yaml` | BTM + all 6 eco columns | Full eco-feature set |
| 4 | RF + BTM + interactions | `rf-btm-interactions.yaml` | BTM + pairwise interactions | Revisit interactions with BTM |
| 5 | RF + BTM tuned | `rf-btm-tuned.yaml` | BTM + tuned n_estimators/max_features | Hyperparameter sweep |

---

## Gate Results

| Gate | Condition | Status |
|---|---|---|
| Constitution I (TDD) | Tests written before implementation | ✓ PLANNED — see tasks.md |
| Constitution III (Performance) | SciPy vectorised ops, no pixel loops | ✓ PLANNED |
| Constitution IV (Scientific accuracy) | All algorithms traced to peer-reviewed source | ✓ — see research.md |
| SC-005 | Eco-feature NaN rate ≤ 10% | Verify after extraction |
| SC-006 | Reproducibility — same output given same inputs | ✓ PLANNED — threshold storage in artifact |
