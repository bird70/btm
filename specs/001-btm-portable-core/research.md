# Research: BTM Portable Core

**Feature**: `001-btm-portable-core`  
**Date**: 2026-03-25  
**Status**: Complete — no NEEDS CLARIFICATION items remain

---

## 1. Algorithm References & Numerical Parity

### 1.1 Bathymetric Position Index (BPI)

- **Decision**: Implement as annular focal mean subtraction, integer-rounded.
- **Formula**: `BPI = round(bathy − mean_annulus(inner_radius, outer_radius))`
- **Reference**: Wright, D.J. et al. 2005. ArcGIS Benthic Terrain Modeler. BTM v1/v2 documentation. Also codified in `Install/toolbox/scripts/bpi.py` (original).
- **NumPy implementation**: `scipy.ndimage.generic_filter` or manual vectorised sliding-window with a precomputed annulus mask. The annulus mask is a boolean 2-D array where `inner_radius ≤ distance_from_centre ≤ outer_radius`; mean is computed over True cells for each neighbourhood.
- **Rationale**: `scipy.ndimage.uniform_filter` computes rectangular means; annular means require a custom mask. Using `scipy.ndimage.generic_filter` with a precomputed footprint boolean array is the canonical SciPy approach and produces identical results to ArcGIS `FocalStatistics(NbrAnnulus(...), "MEAN")`.
- **Alternatives considered**: Manually sliding a NumPy view (pure NumPy, faster for small arrays); rejected for large rasters due to memory. `sklearn.feature_extraction.image.extract_patches_2d` — rejected, not in required dependencies.

### 1.2 BPI Standardization

- **Decision**: Implement as integer-rounded z-score × 100.
- **Formula**: `stdBPI = round((bpi − mean(bpi)) / std(bpi) × 100)`
- **Reference**: BTM v3.0 `standardize_bpi_grids.py`. No external citation needed — statistical standardization.
- **NumPy implementation**: `np.round((arr - arr_mean) / arr_std * 100).astype(np.int32)`. Mean and standard deviation are computed over all valid (non-NoData) cells.
- **Rationale**: Exact match to original `Int(Plus(Times(Divide(Minus(bpi, mean), std), 100), 0.5))` formula. `arcpy.sa.Int(x + 0.5)` is equivalent to `round(x)` for positive numbers; for full generality `np.round` (half-to-even) is used, consistent with IEEE 754 rounding used by ArcGIS.

### 1.3 Slope

- **Decision**: Implement Horn (1981) 3×3 finite-difference gradient.
- **Formula**: `slope_deg = degrees(arctan(sqrt((dz/dx)² + (dz/dy)²)))` where dz/dx and dz/dy are computed with the Horn weighted finite difference kernel.
- **Reference**: Horn, B.K.P. 1981. Hill shading and the reflectance map. _Proceedings of the IEEE_, 69(1): 14–47. This is the exact algorithm used by ArcGIS Pro `arcpy.sa.Slope(..., "DEGREE", 1)` with `z_factor=1`.
- **NumPy implementation**:
  - Pad array by 1 cell (reflect mode) to handle edges.
  - `dz_dx = ((e + 2f + g) - (a + 2b + c)) / (8 * cell_size)` using the 3×3 neighbourhood labels a–i (top-left to bottom-right, Horn convention).
  - `dz_dy = ((g + 2h + i) - (a + 2d + g)) / (8 * cell_size)`
  - Use `scipy.ndimage` shifted arrays for efficiency; or vectorised array slicing.
- **Rationale**: `scipy.ndimage.sobel` computes unweighted Sobel (Horn without the ×2 centre weighting); rejected because it does not match ArcGIS output. The explicit Horn kernel is required for SC-002 numerical parity.
- **Alternatives considered**: `richdem` library (Horn slope included) — rejected to minimize dependencies. `skimage.filters.sobel` — same rejection reason. GDAL `DEMProcessing` — adds GDAL CLI dependency; rasterio already provides GDAL access but its Python API doesn't expose DEM processing; rejected for clean Python layer.

### 1.4 Vector Ruggedness Measure (VRM)

- **Decision**: Implement exactly as per Sappington et al. 2007 with square neighbourhood.
- **Formula**:
  1. `slope_rad = slope_deg × π / 180`; `aspect_rad = aspect_deg × π / 180`
  2. `xy = sin(slope_rad)` — controls horizontal component magnitude
  3. `x = sin(aspect_rad) × xy`; `y = cos(aspect_rad) × xy`; `z = cos(slope_rad)` — but flat cells (aspect == -1) set x=y=0
  4. Focal sum of x, y, z over n×n square neighbourhood
  5. `resultant = sqrt(x_sum² + y_sum² + z_sum²)`
  6. `VRM = 1 − resultant / n²`
- **Reference**: Sappington, J.M., K.M. Longshore, D.B. Thompson. 2007. Quantifying Landscape Ruggedness for Animal Habitat Analysis: A Case Study Using Bighorn Sheep in the Mojave Desert. _Journal of Wildlife Management_, 71(5): 1419–1426.
- **NumPy implementation**: Slope and aspect derived from the Horn kernel (reuses `compute_slope`). Aspect computed separately from the same Horn gradient as `degrees(arctan2(-dz_dy, dz_dx))` with ArcGIS convention (0=N, clockwise). `scipy.ndimage.uniform_filter(x, size=n)` for the focal sum (scaled by n²). Since `uniform_filter` gives mean (not sum), multiply result by n² to get sum.
- **Neighbourhood shape**: Fixed square (rectangle). Per clarification Q4 and Sappington et al. 2007.

### 1.5 Surface-Area-to-Planar-Area Ratio (Jenness Rugosity)

- **Decision**: Preserve Jenness (2002) triangle-based method exactly.
- **Formula**: Eight planar triangles formed by the central cell and its 8 neighbours. Surface area = sum of triangle areas. Planar area = cell_size². Ratio = surface_area / planar_area.
- **Reference**: Jenness, J. 2002. Surface Areas and Ratios from Elevation Grid (surfgrids.avx) extension for ArcView 3.x, v1.2. Jenness Enterprises.
- **NumPy implementation**: Eight shifted arrays (neighbours), vectorised triangle area computation using the Jenness formula (already implemented in `surface_area_to_planar_area.py`; extract the NumPy logic and remove arcpy coupling).
- **Note**: This is the secondary rugosity method. VRM (Sappington) is considered superior for most analyses.

### 1.6 Terrain Classification

- **Decision**: Preserve the sequential conditional CON cascade exactly.
- **Logic**: For each class in dictionary order: apply thresholds on depth, slope, fine BPI, broad BPI using half-open intervals. First matching class wins. Output is integer class code array.
- **NumPy implementation**:
  ```
  result = np.zeros(shape, dtype=np.int32)  # 0 = unclassified
  for cls in classes:
      mask = np.ones(shape, dtype=bool)
      if cls.depth_lower is not None:  mask &= (bathy >= cls.depth_lower)
      if cls.depth_upper is not None:  mask &= (bathy < cls.depth_upper)
      if cls.slope_lower is not None:  mask &= (slope >= cls.slope_lower)
      # ... etc for fine/broad BPI
      result = np.where(mask & (result == 0), cls.code, result)
  ```
- **Rationale**: Exact functional equivalent of the ArcGIS CON cascade (first matching class wins because `result == 0` guard). This matches the original BTM 3.0 behaviour (CHG-007).

### 1.7 Depth Statistics

- **Decision**: Implement mean, std, variance via `scipy.ndimage`; IQR and kurtosis via per-neighbourhood array operations.
- **Focal mean**: `scipy.ndimage.uniform_filter(arr, size=n)`
- **Focal std/variance**: `sqrt(focal_mean(arr²) - focal_mean(arr)²)` or `scipy.ndimage.generic_filter(arr, np.std, size=n)` (slower but correct for window overlap).
- **IQR**: Preserve original stacked-neighbourhood NumPy approach from `depth_statistics.py`; remove arcpy/netCDF4 dependency, use rasterio windowed I/O instead.
- **Kurtosis**: Same; use `scipy.stats.kurtosis` on the neighbourhood stack.
- **Reference**: No single external citation for focal statistics; standard descriptive statistics.

---

## 2. Dependency Decisions

### 2.1 Raster I/O: rasterio vs alternatives

- **Decision**: rasterio ≥ 1.3 for all raster reads/writes.
- **Rationale**: rasterio is the de-facto standard Python raster I/O library (wraps GDAL), is `pip`-installable, works cross-platform, and supports windowed (block-based) reads/writes natively via `rasterio.windows.Window`. It also supports all GDAL compression options (LZW, DEFLATE) and preserves CRS/transform metadata transparently.
- **Alternatives considered**:
  - `GDAL Python bindings` directly — rejected: lower-level, more verbose, same GDAL dependency.
  - `xarray + rioxarray` — rejected: unnecessary abstraction overhead; xarray is a large dependency not needed for array-at-a-time processing.
  - `fiona + GDAL` — not applicable (vector, not raster).

### 2.2 Block-based processing

- **Decision**: Use `rasterio.windows.Window` with `dataset.read(window=...)` and `dataset.write(window=...)` for tiled processing.
- **Rationale**: Replaces the original NetCDF4 intermediate file approach (CHG-008). rasterio windowed I/O is idiomatic, efficient, and removes the `netCDF4` dependency entirely.
- **Overlap handling**: Each tile is read with an overlap border of `radius` cells; the overlap border pixels in the output are discarded (only the inner tile is written), preventing edge artifacts in neighbourhood operations.
- **Rationale for overlap size**: For BPI with outer_radius `r`, the overlap must be ≥ `r` cells. For VRM with neighbourhood size `n`, overlap must be ≥ `n//2` cells.

### 2.3 Classification I/O: openpyxl vs xlrd

- **Decision**: openpyxl ≥ 3.1 for `.xlsx`; `csv` stdlib for `.csv`; `xml.dom.minidom` stdlib for `.xml`.
- **Rationale**: `xlrd` ≥ 2.0 dropped `.xlsx` support; openpyxl is the maintained replacement. Preserves CSV and XML reading that are already pure stdlib (CHG-009).

### 2.4 Linting: ruff vs flake8/black

- **Decision**: `ruff` (replaces both flake8 and black).
- **Rationale**: ruff is significantly faster than the flake8+black combination, is a single dependency, and is the current community standard for new Python projects. Configured in `pyproject.toml` `[tool.ruff]`.

### 2.5 Testing: pytest configuration

- **Decision**: pytest ≥ 8 with `pytest-cov` and custom marks.
- **Marks registered in `pyproject.toml`**:
  - `arcgis` — tests requiring ArcGIS Pro conda environment
  - `qgis` — tests requiring PyQGIS
- **CI behaviour**: `pytest -m "not arcgis and not qgis"` for the default (no-GIS) CI run.
- **Coverage target**: ≥ 90% line coverage on `btm/core/` and `btm/classification/`.

---

## 3. ArcGIS Pro Adapter: Best Practices

- **Python Toolbox (`.pyt`)**: All tool classes live in `Install/toolbox/btm.pyt`. Each class has `getParameterInfo()`, `execute()`, and optionally `updateParameters()` / `updateMessages()`.
- **Import isolation**: The `.pyt` imports `btm.adapters.arcgis.tools` at tool execution time (not at module import time) to avoid import errors when `btm` is not on `sys.path`.
- **arcpy import guard**: `btm.adapters.arcgis` checks `importlib.util.find_spec("arcpy")` at the top; raises `ImportError` with a clear message if not found.
- **Spatial Analyst fallback**: For `compute_slope`, the ArcGIS adapter tries `arcpy.sa.Slope` first; catches `arcpy.ExecuteError` or `RuntimeError` from failed licence checkout, then falls back to `btm.core.slope.compute_slope`. A warning is added via `arcpy.AddWarning`.
- **Parameter naming**: Match original BTM 3.0 `.pyt.xml` parameter names exactly (FR-018) to preserve Model Builder backward compatibility.

---

## 4. QGIS Provider: Best Practices

- **Provider registration**: Subclass `QgsProcessingProvider`; register via a QGIS plugin's `initProcessing()` hook or as a standalone provider loaded by `QgsApplication.processingRegistry().addProvider(BtmProvider())`.
- **Algorithm parameter schemas**: Derive from shared Python `dataclass` or `TypedDict` definitions in `btm/core/params.py` (FR-022). Both CLI and QGIS algorithm constructors read the same param schemas to prevent drift.
- **Raster input/output**: Use `QgsRasterLayer` for display; internally extract the file path and pass to `btm.core` via rasterio.
- **Defer to P4**: QGIS provider scaffolding is created in this feature but algorithm implementations can be stubbed; full implementation follows after P1–P3 pass.

---

## 5. pyproject.toml Structure

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "btm"
version = "4.0.0.dev0"
requires-python = ">=3.11"
dependencies = [
    "rasterio>=1.3",
    "numpy>=1.24",
    "scipy>=1.11",
    "openpyxl>=3.1",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff"]

[project.scripts]
btm-bpi              = "btm.cli.bpi:main"
btm-standardize-bpi  = "btm.cli.standardize_bpi:main"
btm-slope            = "btm.cli.slope:main"
btm-vrm              = "btm.cli.vrm:main"
btm-surface-ratio    = "btm.cli.surface_ratio:main"
btm-classify         = "btm.cli.classify:main"
btm-run-model        = "btm.cli.run_model:main"
btm-depth-stats      = "btm.cli.depth_statistics:main"
btm-scale-compare    = "btm.cli.scale_comparison:main"

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "UP"]

[tool.pytest.ini_options]
markers = [
    "arcgis: requires ArcGIS Pro conda environment",
    "qgis: requires PyQGIS / QGIS LTR 3.34+",
]
testpaths = ["tests"]

[tool.coverage.run]
source = ["btm/core", "btm/classification"]
omit = ["btm/adapters/*", "btm/cli/*"]
```

---

## 6. Numerical Parity Verification Strategy (SC-002)

To verify that new outputs match BTM 3.0 (±1 int / ±0.01°):

1. **Reference outputs**: Generated once from working BTM 3.0 on ArcGIS 10.x or ArcGIS Pro (pre-upgrade) against `tests/data/bathy5m_clip.tif` and saved as `.npy` arrays in `tests/data/reference/`.
2. **Parity test**: `tests/integration/test_numerical_parity.py` loads each reference array and compares it against the new `btm.core` output using `np.testing.assert_allclose(actual, expected, atol=1.0)` for integer rasters and `atol=0.01` for slope.
3. **Tolerance rationale**: ±1 integer unit accounts for floating-point rounding differences between NumPy's IEEE 754 and ArcGIS's internal rounding. ±0.01° accounts for float32 vs float64 precision differences in slope.

---

## 7. All NEEDS CLARIFICATION resolved

| Item                       | Resolution                                                         |
| -------------------------- | ------------------------------------------------------------------ |
| Package distribution       | Local/git install only; `pyproject.toml` with `pip install .`      |
| Intermediate output policy | Keep all by default; `--no-intermediates` flag suppresses them     |
| Overwrite policy           | Always overwrite silently (idempotent)                             |
| VRM neighbourhood shape    | Rectangle (square) only — Sappington et al. 2007                   |
| Logging strategy           | Python `logging` module; NullHandler in core; StreamHandler in CLI |
