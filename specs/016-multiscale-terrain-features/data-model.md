# Data Model: Multi-Scale Terrain Features

## Entities

### MultiScaleDerivative

A terrain derivative computed at multiple spatial scales via the calculate-then-average method.

| Field         | Type                  | Description                                                   |
| ------------- | --------------------- | ------------------------------------------------------------- |
| name          | str                   | Derivative identifier (e.g., "slope", "vrm", "surface_ratio") |
| base_array    | np.ndarray (float32)  | The 3×3 native-resolution derivative raster                   |
| scales        | list[int]             | Window sizes for focal averaging (e.g., [3, 7, 11, 15, 21])   |
| scaled_arrays | dict[int, np.ndarray] | Scale → focal-averaged raster                                 |

**Column naming convention**: `btm_{derivative}_{scale}` (e.g., `btm_slope_3`, `btm_slope_7`, `btm_vrm_21`)

**Derivatives subject to multi-scaling** (7 total):

- `slope` — Horn (1981) slope in degrees
- `vrm` — Sappington (2007) Vector Ruggedness Measure
- `surface_ratio` — Jenness (2002) surface-to-planar area ratio
- `northness` — cos(aspect), Wilson et al. (2007)
- `eastness` — sin(aspect), Wilson et al. (2007)
- `max_curvature` — Evans (1980) / Schmidt et al. (2003)
- `complexity` — Wilson et al. (2007) slope-of-slope

### RDMV (Relative Difference to Mean Value)

A new terrain derivative computed at each scale.

| Field       | Type                 | Description                      |
| ----------- | -------------------- | -------------------------------- |
| depth_array | np.ndarray (float64) | Source bathymetry raster         |
| scale       | int                  | Focal window size                |
| rdmv_array  | np.ndarray (float64) | (depth − focal_mean) / focal_std |

**Column naming**: `btm_rdmv_{scale}` (e.g., `btm_rdmv_7`, `btm_rdmv_21`)

**Validation**: RDMV is undefined where focal_std = 0 (flat areas); set to 0.0 in those cells.

### GLCMTexture

GLCM texture features extracted from the backscatter raster.

| Field             | Type                 | Description                                              |
| ----------------- | -------------------- | -------------------------------------------------------- |
| backscatter_array | np.ndarray (float32) | Source backscatter raster band                           |
| window_size       | int                  | Patch extraction size for GLCM computation               |
| n_levels          | int                  | Grey-level quantisation depth (default: 32)              |
| contrast          | np.ndarray (float64) | GLCM contrast at each pixel (averaged over 4 directions) |
| homogeneity       | np.ndarray (float64) | GLCM homogeneity at each pixel                           |

**Column naming**: `btm_glcm_contrast_{scale}`, `btm_glcm_homogeneity_{scale}`

**Note**: GLCM is computed per-pixel by extracting a patch of `window_size × window_size` centred on each pixel. For point-sampled extraction, extract patches at point locations only (not full raster) for performance.

### FeatureSelectionResult

Record of which features were retained after permutation importance selection.

| Field           | Type  | Description                             |
| --------------- | ----- | --------------------------------------- |
| feature_name    | str   | Column name in the DataFrame            |
| importance_mean | float | Mean permutation importance score       |
| importance_std  | float | Standard deviation across folds/repeats |
| selected        | bool  | Whether the feature was retained        |
| rank            | int   | Importance rank (1 = most important)    |

## Relationships

```
Bathymetry Raster
  └─→ 7 native derivatives (slope, vrm, surface_ratio, northness, eastness, max_curvature, complexity)
       └─→ MultiScaleDerivative at 5 scales each → 35 columns
  └─→ RDMV at 5 scales → 5 columns

Backscatter Raster
  └─→ GLCMTexture (contrast + homogeneity) at 5 scales → 10 columns

Existing BTM features (unchanged)
  └─→ broad_bpi, fine_bpi, broad_std, fine_std (4 columns)
  └─→ Interaction features: bpi_magnitude, broad_x_fine_std, rough_total (3 columns)

Total new columns: ~50
After feature selection: ≤ 25
```

## State Transitions

```
Raw rasters → Compute native derivatives (3×3) → Apply focal mean at scales
  → Sample at point locations → Merge with existing features → Train model
  → Compute permutation importance → Select top features → Retrain → Predict → Submit
```
