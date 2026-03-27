# Data Model: OBIA + Pixel-Based Hybrid Classification

**Branch**: `009-obia-pixel-hybrid`  
**Phase**: 1 — Design  
**Reference**: Ierodiaconou et al. (2018) Marine Geodesy DOI: 10.1007/s11001-017-9338-z

---

## Entities

### E1 — Raw Rasters (Inputs)

| Field         | Type              | Shape          | Notes                                    |
|---------------|-------------------|----------------|------------------------------------------|
| `bathy`       | `np.ndarray[f32]` | (H, W) = (4040, 4743) | Raw bathymetry from `data/MBES/bathymetry.tif`; NoData → `NaN` |
| `back`        | `np.ndarray[f32]` | (H, W)         | Raw backscatter from `data/MBES/backscatter.tif`; NoData → `NaN` |
| `transform`   | `Affine`          | —              | Rasterio affine transform (bathy CRS)    |
| `cell_size`   | `float`           | —              | 0.25 (metres per pixel)                  |

### E2 — Pixel-Based Derivative Rasters

All arrays have shape (H, W) and dtype `float32`. Computed from E1 before any sub-sampling.

| Field            | Formula / Algorithm                          | Reference                |
|------------------|----------------------------------------------|--------------------------|
| `slope`          | Horn (1981) 3×3 weighted kernel              | Ierodiaconou Table 1     |
| `vrm`            | Sappington et al. (2007) VRM, 3×3 window     | ibid.                    |
| `complexity`     | `1/cos(slope_rad)` uniform-filtered 3×3      | Jenness (2004)           |
| `max_curvature`  | `\|scipy.ndimage.laplace(bathy)\| / cell²`   | Zevenbergen & Thorne (1987) |
| `northness`      | `cos(arctan2(gx, gy))` where g=`np.gradient` | Roberts (1986)           |
| `eastness`       | `sin(arctan2(gx, gy))`                       | Roberts (1986)           |

### E3 — Segment Labels

| Field     | Type              | Shape   | Notes                                      |
|-----------|-------------------|---------|--------------------------------------------|
| `labels`  | `np.ndarray[i32]` | (H, W)  | Integer segment IDs ≥ 0; no background    |

Produced by `skimage.segmentation.slic()` on 3-channel normalised stack `[bathy, back, vrm]`.  
Target: `n_segments ≈ 4000` → mean object ≈ 4800 px ≈ 300 m².

### E4 — Segment Statistics DataFrame

Computed by aggregating pixel-level values per segment label.

| Column                  | Dtype   | Notes                               |
|-------------------------|---------|-------------------------------------|
| `seg_bathy_mean`        | float32 | Mean bathymetry per segment         |
| `seg_bathy_std`         | float32 | Std dev bathymetry per segment      |
| `seg_bathy_skew`        | float32 | Skewness bathymetry per segment     |
| `seg_back_mean`         | float32 | Mean backscatter per segment        |
| `seg_back_std`          | float32 | Std dev backscatter per segment     |
| `seg_back_skew`         | float32 | Skewness backscatter per segment    |
| `seg_vrm_mean`          | float32 | Mean VRM (rugosity) per segment     |
| `seg_vrm_std`           | float32 | Std dev VRM per segment             |
| `seg_vrm_skew`          | float32 | Skewness VRM per segment            |
| `seg_pixel_count`       | int32   | Number of pixels in segment (shape proxy) |

Index: integer segment label. Shape: `(n_unique_labels, 10)`.

### E5 — Point Feature Tables

One row per training / test point. Produced by sampling E2 rasters and joining E4 segment stats.

#### Training (`X_train`, `y_train`)

| Column            | Type    | Source                          |
|-------------------|---------|---------------------------------|
| `x`, `y`          | float64 | From `data/train.csv`           |
| `depth`           | float32 | Sampled from `bathy`            |
| `backscatter`     | float32 | Sampled from `back`             |
| `slope`           | float32 | Sampled from `slope` raster     |
| `rugosity`        | float32 | Sampled from `vrm` raster       |
| `complexity`      | float32 | Sampled from `complexity` raster|
| `max_curvature`   | float32 | Sampled from `max_curvature`    |
| `northness`       | float32 | Sampled from `northness` raster |
| `eastness`        | float32 | Sampled from `eastness` raster  |
| `seg_bathy_mean`  | float32 | Joined from E4                  |
| `seg_bathy_std`   | float32 | Joined from E4                  |
| `seg_bathy_skew`  | float32 | Joined from E4                  |
| `seg_back_mean`   | float32 | Joined from E4                  |
| `seg_back_std`    | float32 | Joined from E4                  |
| `seg_back_skew`   | float32 | Joined from E4                  |
| `seg_vrm_mean`    | float32 | Joined from E4                  |
| `seg_vrm_std`     | float32 | Joined from E4                  |
| `seg_vrm_skew`    | float32 | Joined from E4                  |
| `seg_pixel_count` | int32   | Joined from E4                  |
| `class`           | str     | Target label (ALG/FMAT/NVB/SGAM/SGZ) |

Shape: `(n_train, 19+1)` where n_train = 566 (from `data/train.csv`).

#### Test (`X_test`)

Same 18 feature columns (no `class`). Shape: `(98, 18)`.

### E6 — CV Block Assignment

| Column   | Type   | Notes                                              |
|----------|--------|----------------------------------------------------|
| `block`  | int32  | KMeans cluster ID in [0, 9]; one value per training point |

Produced by `KMeans(n_clusters=10, random_state=42).fit_predict(train[['x','y']].values)`.

### E7 — OOF Probability Arrays

| Variable      | Shape            | Notes                                             |
|---------------|------------------|---------------------------------------------------|
| `oof_proba`   | dict[str → ndarray(n_train, 5)] | Keyed by model name; columns = 5 class probabilities |
| `avg_proba`   | ndarray(n_train, 5) | Mean across all 4 models                        |

### E8 — Outputs

| Artifact                          | Format  | Notes                                        |
|-----------------------------------|---------|----------------------------------------------|
| `data/submission_v6_best.csv`     | CSV     | 98 rows × 2 cols (`ID`, `class`)             |
| `docs/run-009-obia-pixel-hybrid.md` | Markdown | CV metrics, class report, feature importance top 10 |

---

## Data Flow

```
E1 (Raw Rasters)
   ├─→ E2 (PB Derivatives: slope, vrm, complexity, curvature, northness, eastness)
   └─→ E3 (Segment Labels via SLIC on [bathy, back, vrm])
          └─→ E4 (Segment Statistics: per-label aggregations of E1+E2)
                 ┌──────────────────────────────────┘
                 ↓
E1 coords + E2 point samples + E4 joined stats
   ├─→ E5 X_train / y_train
   └─→ E5 X_test

E5 + E6 blocks
   └─→ E7 (OOF probas per model; spatial CV loop)
          └─→ E8 submission_v6_best.csv + run report
```

---

## Validation Rules

| Rule | Entity | Constraint |
|------|--------|------------|
| VR-01 | E2 | All derivative rasters same shape as `bathy` (H, W) |
| VR-02 | E3 | `labels.shape == bathy.shape`; all values ≥ 0 |
| VR-03 | E3 | Mean segment size in [2400, 7200] px (300 m² ± 50%) |
| VR-04 | E4 | No NaN in stats for any segment with ≥ 3 pixels |
| VR-05 | E5 | All 18 feature columns finite for all 566+98 points |
| VR-06 | E8 | `submission_v6_best.csv` has exactly 98 rows |
| VR-07 | E5 | `northness`, `eastness` values in [-1, 1] |
