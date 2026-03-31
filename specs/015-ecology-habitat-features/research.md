# Research: Ecology-Informed Habitat Feature Engineering

**Branch**: `015-ecology-habitat-features` | **Date**: 2026-03-31

---

## 1. Northness & Eastness

### Decision

Implement using the **Horn (1981) gradient method** — the same algorithm used by ArcGIS Spatial Analyst (referenced in the paper) — applied to the raw DEM:

```
For a 3×3 elevation window:
  [a b c]
  [d e f]
  [g h i]

dx = ((c + 2f + i) - (a + 2d + g)) / (8 × cell_size)   [east–west gradient]
dy = ((g + 2h + i) - (a + 2b + c)) / (8 × cell_size)   [south–north gradient; note direction]

azimuth = atan2(dy, dx)   [from east, counterclockwise — standard math convention]

northness = sin(azimuth) = dy / sqrt(dx² + dy²)
eastness  = cos(azimuth) = dx / sqrt(dx² + dy²)
```

Range of both outputs: [−1, 1]. Flat areas (dx=dy=0) produce NaN → filled with 0.0.

### Rationale

- Wilson et al. (2007) Table 1 describes Northness as the "Sinus component" and Eastness as the "Cosinus component" of the azimuthal direction of steepest slope, consistent with the above formulae using the math (east-origin, counterclockwise) convention.
- The Horn (1981) operator is the standard implementation in ArcGIS Spatial Analyst, matching the paper's software column.
- Implementation uses `scipy.ndimage.convolve` with pre-defined 3×3 kernels — no GIS runtime required.

### Alternatives Considered

- ArcGIS `arcpy.sa.aspect()`: rejected — requires ArcGIS runtime
- GDAL `gdaldem aspect`: rejected — adds a subprocess dependency when SciPy can do this natively
- Pre-computing a separate aspect raster and sampling it: rejected — unnecessary intermediate file; inline computation is simpler and reproducible

### References

- Wilson, M.F.J., O'Connell, B., Brown, C., Guinan, J.C., Grehan, A.J. (2007). Multiscale Terrain Analysis of Multibeam Bathymetry Data for Habitat Mapping on the Irish Continental Slope. _IEEE Journal of Oceanic Engineering_, 32(3), 679–699.
- Horn, B.K.P. (1981). Hill shading and the reflectance map. _Proceedings of the IEEE_, 69(1), 14–47.

---

## 2. Maximum Curvature

### Decision

Compute **plan curvature** and **profile curvature** independently using second-order polynomial fitting on the 3×3 elevation window (Evans, 1980; Schmidt et al., 2003), then take the maximum of their **absolute values**:

```
max_curvature = max(|plan_curvature|, |profile_curvature|)
```

The second-order polynomial coefficients for a 3×3 window are:

```
A = ((z1 + z3 + z4 + z6 + z7 + z9) / 2 - (z2 + z5 + z8)) / (3 × L²)
B = ((z1 + z2 + z3 + z7 + z8 + z9) / 2 - (z4 + z5 + z6)) / (3 × L²)
C = (-z1 + z3 - z7 + z9) / (4 × L²)
D = (-z1 - z2 - z3 + z7 + z8 + z9) / (6 × L)
E = (-z1 + z3 + z4 - z6 - z7 + z9) / (6 × L)

(using the Evans/Zevenbergen-Thorne notation, L = cell_size)

profile_curvature = -2 * (A*D² + B*E² + C*D*E) / (D² + E²)   [along slope direction]
plan_curvature    = 2  * (B*D² + A*E² - C*D*E) / (D² + E²)   [perpendicular to slope]
```

Where flat areas (D=E=0) produce NaN → filled with 0.0.

### Rationale

- Schmidt et al. (2003) definition: "steepest curve of either plan or profile convexity" — taking the maximum absolute value captures the dominant curvature signal regardless of slope direction.
- Standard approach in geomorphometry libraries (e.g., SAGA GIS, WhiteboxTools).
- Implementable with `scipy.ndimage.convolve` using pre-computed kernels.

### Alternatives Considered

- Only computing profile curvature: rejected — plan curvature captures lateral flow concentration, relevant for sediment accumulation zones where SGAM habitat occurs
- Using the curvature magnitude (quadratic sum): rejected — harder to interpret; max is simpler and maps directly to the paper's description

### References

- Schmidt, J., Evans, I.S., Brinkmann, J. (2003). Comparison of polynomial models for land surface curvature calculation. _International Journal of Geographical Information Science_, 17(8), 797–814.
- Evans, I.S. (1980). An integrated system of terrain analysis and slope mapping. _Z. Geomorphologie_, Suppl.-Bd. 36, 274–295.

---

## 3. Complexity (Rate of Change of Slope)

### Decision

Compute **slope** at each cell (Wilson et al. 2007 definition: degrees from horizontal), then apply the **same slope computation kernel to the slope array** — giving the second derivative of elevation, equivalent to the Laplacian of the DEM in degrees/pixel units:

```
slope_array = slope_from_dem(dem_array, cell_size)           # degrees, Horn method
complexity  = slope_from_dem(slope_array, cell_size)         # degrees/cell, unitless rate
complexity  = abs(complexity)                                  # always non-negative
```

### Rationale

- Wilson et al. (2007): "Second derivative of slope (or rate of change of slope) — a measure of the terrain's local variability."
- This is the simplest faithful implementation: applying the slope operator twice.
- No GIS runtime required; the same Horn gradient kernel is reused.
- The result is conceptually a Laplacian-of-DEM (measure of terrain convexity/concavity rate), capturing transition zones between flat sediment areas and adjacent rugose reef — ecologically meaningful for SGAM/ALG boundary detection.

### Alternatives Considered

- Discrete Laplacian kernel directly on DEM: equivalent mathematically but loses the Horn averaging; slope-of-slope is more consistent with the paper's description
- Computing rate of change of the pre-existing `btm_slope` column without raster: possible but the btm_slope column is a point sample, not a neighbourhood operation — would lose the spatial window context

### References

- Wilson et al. (2007), ibid.

---

## 4. Depth Zone Boundaries

### Decision

**4 quantile bins** (equal-frequency) computed from the `bathymetry` column in the training CSV at fit time. Bin boundaries are the 25th, 50th, and 75th percentiles of the training bathymetry distribution. Ordinal integer encoding: 1 = very shallow, 2 = shallow, 3 = mid, 4 = deeper.

### Expected depth distribution (from data/train.csv description)

The Fagatele Bay study area is a shallow bay, approximately 0–50 m depth. Equal-frequency binning will place roughly equal numbers of training samples in each bin, which is better for RF than equal-width bins (which could leave rare depth ranges in a nearly-empty bin).

### Rationale

- Data-driven: no hardcoded depth thresholds that would be survey-specific
- Reproducible: quantile thresholds logged in `eco_thresholds.json` artifact per run
- Equal-frequency ensures RF has enough samples in each bin to learn depth-zone patterns

### Alternatives Considered

- Equal-width bins: rejected — depth distributions in bathymetric surveys are often non-uniform; equal-width bins may produce very sparse bins
- Fixed ecological thresholds (0–5 m, 5–15 m, > 15 m): rejected — would need domain expert validation; quantile approach lets the data speak first; ecological interpretation can be added after

---

## 5. SGAM Niche Indicator Thresholds

### Decision

Binary indicator `btm_sgam_niche = 1` if all three conditions hold:

1. `btm_fine_bpi < p25(btm_fine_bpi)` in training set
2. `btm_slope < p25(btm_slope)` in training set
3. `depth_min_sgam ≤ bathymetry ≤ depth_max_sgam` where bounds are the min/max of `bathymetry` for training points with class == "SGAM"

If the SGAM class has fewer than 5 training points, condition 3 is skipped (indicator = 1 when conditions 1 + 2 hold) and a warning is logged.

### Rationale

- The ecological hypothesis is: SGAM occupies the shallower, flatter, lower-BPI portion of the study area — a depositional niche
- Using p25 as threshold tests the hypothesis that SGAM is in the bottom quartile of both BPI and slope distributions — a conservative threshold that avoids false negatives
- SGAM depth range from training labels is the most direct ecological signal — if SGAM training points are observed at depths 3–18 m, this becomes the depth constraint
- All thresholds stored in `eco_thresholds.json` for full reproducibility

### Alternatives Considered

- Continuous product score: rejected — binary is easier to inspect and debug; importance analysis will reveal if the signal is weak
- Fixed absolute thresholds: rejected — too survey-specific

---

## 6. Code Location for Raster Derivatives

### Decision

Extend `btm/features/extract.py` with a new function `extract_eco_raster_features()` and add `include_eco_features: bool = False` to `extract_btm_features()`. The CLI in `btm/cli/export_features.py` gains a `--include-eco-features` flag.

### Rationale

- `btm/features/extract.py` already contains all raster sampling and derivative computation for the BTM pipeline — this is the correct home for new spatial derivatives
- Keeping raster extraction in `btm/` and training-time transformations in `src/benthic_model/features/eco_features.py` maintains the existing architectural separation
- A single new `--include-eco-features` flag in the CLI is the minimal addition to produce `train_btm_eco.csv`

---

## 7. RF Hyperparameter Configuration

### Decision

Add `model_params: dict[str, Any] | None = None` to `PipelineConfig`. In the `build_rf_candidate()` / `build_baseline_model()` builders, unpack `model_params` as `**kwargs` to the `RandomForestClassifier` constructor. Non-RF model builders ignore it.

```yaml
# Example rf-btm-tuned.yaml
model_type: rf
model_params:
  n_estimators: 500
  max_features: sqrt
  min_samples_leaf: 1
feature_flags:
  include_btm_features: true
  include_eco_features: false
```

### Rationale

- Generic `model_params` dict avoids adding one config field per hyperparameter
- RF constructor already accepts kwargs; unpacking is safe
- Future models (LGBM, CatBoost) can also use `model_params` for their own kwargs without further config changes

### Alternatives Considered

- Dedicated `rf_params` dataclass: rejected — YAGNI; dict is sufficient for one experiment config
- Grid search / RandomizedSearchCV in the pipeline: out of scope for this spec — the goal is 1–2 targeted configs, not automated tuning
