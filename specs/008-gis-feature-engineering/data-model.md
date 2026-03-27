# Data Model: GIS Feature Engineering Experiment

**Feature**: `008-gis-feature-engineering`  
**Date**: 2026-03-27

---

## Input Data

### Raster Inputs

| File        | Path                        | Type            | Description                                     |
| ----------- | --------------------------- | --------------- | ----------------------------------------------- |
| Bathymetry  | `data/MBES/bathymetry.tif`  | float32, 1-band | Water depth in metres (negative), EPSG:28355    |
| Backscatter | `data/MBES/backscatter.tif` | float32, 1-band | Acoustic backscatter intensity (dB), EPSG:28355 |

### Pre-Derived Rasters (Phase 1 inputs, already on disk)

| File              | Path                                               | Type    | No. Classes | Algorithm                                                | Reference            |
| ----------------- | -------------------------------------------------- | ------- | ----------- | -------------------------------------------------------- | -------------------- |
| Slope             | `outputs/gis_layers/rasters/slope.tif`             | float32 | —           | Horn (1981)                                              | Horn 1981            |
| Backscatter zones | `outputs/gis_layers/rasters/backscatter_zones.tif` | int32   | 5           | K-means on Gaussian-smoothed backscatter (σ=10, seed 42) | Hartigan & Wong 1979 |
| Acoustic facies   | `outputs/gis_layers/rasters/acoustic_facies.tif`   | int32   | 8           | K-means on standardised (depth, backscatter), seed 42    | Hartigan & Wong 1979 |

### Tabular Inputs

| File             | Columns           | Rows   | Notes                            |
| ---------------- | ----------------- | ------ | -------------------------------- |
| `data/train.csv` | `ID, x, y, class` | ~2 000 | Labelled points in EPSG:28355    |
| `data/test.csv`  | `ID, x, y`        | ~1 000 | Unlabelled points for submission |

---

## Feature Matrix

### Base Features (v2 feature set — unchanged)

Produced by `extract_all_features()` cloned from `scripts/experiment_v2.py`:

| Group                  | Feature names                                                                | Count    | Notes                             |
| ---------------------- | ---------------------------------------------------------------------------- | -------- | --------------------------------- |
| Coordinates            | `x`, `y`                                                                     | 2        | Raw projected coords              |
| Raw raster             | `bathy`, `back`                                                              | 2        | Sampled at point                  |
| BTM derivatives        | `slope`, `vrm`                                                               | 2        | Horn 1981; Sappington et al. 2007 |
| Multi-scale focal mean | `b_fm_{3,5,…}`, `k_fm_{3,5,…}`                                               | 16       | 8 radii × 2 rasters               |
| Multi-scale focal std  | `b_std_{…}`, `k_std_{…}`                                                     | 16       | 8 radii × 2 rasters               |
| Multi-scale TPI        | `b_tpi_{…}`, `k_tpi_{…}`                                                     | 16       | 8 radii × 2 rasters               |
| Multi-scale curvature  | `mcurv_s{σ}`, `gcurv_s{σ}`, `mslope_s{σ}`                                    | 15       | 5 σ × 3 types                     |
| Aspect                 | `asp_sin`, `asp_cos`                                                         | 2        | Circular encoding                 |
| Relative backscatter   | `rel_back_{size}`                                                            | 4        | 4 radii                           |
| Backscatter gradient   | `back_grad`                                                                  | 1        | Magnitude of ∇backscatter         |
| Patch statistics       | `b_range`, `b_iqr`, `k_range`, `k_iqr`, `k_skew`, `s_max`, `s_std`           | 28       | 4 half-sizes × 7 stats            |
| GLCM texture           | `glcm_{contrast,dissimilarity,homogeneity,energy,correlation}`               | 5        | 21×21 backscatter patches         |
| Interaction terms      | `depth_x_back`, `slope_x_back`, `acoustic_hard`, `vrm_x_back`, `slope_x_vrm` | 5        | Domain-derived                    |
| **Total base**         |                                                                              | **~114** |                                   |

### New GIS-Derived Features (appended, weighted)

| Column name             | Source                        | Weight applied | Notes                                         |
| ----------------------- | ----------------------------- | -------------- | --------------------------------------------- |
| `gis_slope`             | `slope.tif` sampled at point  | × 80           | Continuous; same values as `slope` but scaled |
| `gis_bz_0` … `gis_bz_4` | `backscatter_zones.tif` → OHE | × 20 each      | 5 binary columns                              |
| `gis_af_0` … `gis_af_7` | `acoustic_facies.tif` → OHE   | × 20 each      | 8 binary columns                              |
| **Total new**           |                               |                | **14 columns**                                |

**Total feature matrix width**: ~128 columns (Run A and Run B).

> **Note on weighting invariance**: GBDT/RF models are invariant to monotonic feature scaling (split thresholds shift proportionally). The weights serve as a documented prior and retain meaning for any downstream linear baseline. See `research.md §2` for full discussion.

---

## Run Variants

### Run A — Raw GIS Features

- GIS columns populated by direct point sampling (`sample_raster_at_points()`) of the three derived rasters.
- OHE applied; weight factors applied.
- Ensemble trained on full augmented matrix.

### Run B — Kriging-Smoothed GIS Features

- `gis_slope` values at test points are kriged from training point observations (ordinary kriging, spherical variogram, zone-stratified).
- `gis_bz_*` OHE columns: kriging is applied to the **raw zone integer** at train points; the predicted float value at test points is then used as a soft zone membership before OHE (or, alternatively, the OHE columns are each kriged as a binary field — chosen approach: krige raw integer, then OHE at inference).
- `gis_af_*` treated identically.
- Fallback: KNeighborsRegressor(n_neighbors=min(5, n_pts)) for zones with < 10 training points.
- All v2 base features pass through unchanged.

---

## Outputs

### Submission CSVs

| File                                          | Produced by     | Format                                          |
| --------------------------------------------- | --------------- | ----------------------------------------------- |
| `data/submission_v5_gis_features.csv`         | Run A           | `ID,class` one row per test point               |
| `data/submission_v5_gis_features_kriging.csv` | Run B           | `ID,class` one row per test point               |
| `data/submission_v5_best.csv`                 | Post-comparison | Copy of whichever run has higher CV weighted-F1 |

### Run Report

`docs/run-008-gis-feature-engineering.md` — Markdown table comparing:

| Column           | Content                                                 |
| ---------------- | ------------------------------------------------------- |
| Run              | baseline (v2 only), Run A (raw GIS), Run B (kriged GIS) |
| Weighted-F1 (CV) | 10-block spatial CV, macro-weighted                     |
| Per-class F1     | ALG, NVB, … (all classes)                               |
| Top-3 features   | By mean GBDT feature gain                               |
| Wall-clock (s)   | Training + inference time                               |

---

## State Transitions

```
MBES rasters (bathymetry, backscatter)
    │
    ▼  [step1_prepare_gis_layers.py OR _ensure_derived_rasters()]
Derived rasters (slope, backscatter_zones, acoustic_facies)
    │
    ▼  [experiment_v5.py — phase: baseline]
v2 feature matrix → spatial-block CV → baseline_f1
    │
    ▼  [experiment_v5.py — phase: run_a]
Augmented feature matrix (v2 + raw GIS) → CV → run_a_f1 → submission_v5_gis_features.csv
    │
    ▼  [experiment_v5.py — phase: run_b]
Augmented feature matrix (v2 + kriged GIS) → CV → run_b_f1 → submission_v5_gis_features_kriging.csv
    │
    ▼  [experiment_v5.py — post-comparison]
submission_v5_best.csv + docs/run-008-gis-feature-engineering.md
```
