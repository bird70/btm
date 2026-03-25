# Data Model: BTM Portable Core

**Feature**: `001-btm-portable-core`  
**Date**: 2026-03-25

---

## Core Entities

### RasterDataset

Internal abstraction passed between all core functions and I/O adapters.

| Field         | Type                                   | Description                                                    |
| ------------- | -------------------------------------- | -------------------------------------------------------------- |
| `path`        | `str \| None`                          | Absolute path to source/destination file; `None` for in-memory |
| `array`       | `numpy.ndarray` (float32/float64, 2-D) | Pixel data; NoData cells hold the `nodata` sentinel            |
| `crs`         | `rasterio.crs.CRS \| None`             | Coordinate Reference System                                    |
| `transform`   | `affine.Affine`                        | Affine transform (top-left corner + cell size)                 |
| `nodata`      | `float \| int \| None`                 | NoData sentinel value                                          |
| `cell_width`  | `float`                                | Horizontal cell size in CRS units                              |
| `cell_height` | `float`                                | Vertical cell size in CRS units (positive)                     |

**Validation rules**:

- `array.ndim == 2` (single band only)
- `cell_width > 0` and `cell_height > 0`
- When `nodata` is set, `array` must not be masked — NoData is represented as the sentinel value

**Methods** (on `RasterDataset` or companion I/O helpers):

- `cell_size() → float`: `(cell_width + cell_height) / 2`
- `from_file(path) → RasterDataset`: read via rasterio
- `to_file(path, dtype, compress='lzw')`: write via rasterio, always overwrites

---

### ClassEntry

One row from a classification dictionary.

| Field             | Type            | Description                                         |
| ----------------- | --------------- | --------------------------------------------------- |
| `code`            | `int`           | Integer class identifier (written to output raster) |
| `name`            | `str`           | Human-readable zone/class name                      |
| `depth_lower`     | `float \| None` | Minimum depth threshold (inclusive)                 |
| `depth_upper`     | `float \| None` | Maximum depth threshold (exclusive)                 |
| `slope_lower`     | `float \| None` | Minimum slope threshold                             |
| `slope_upper`     | `float \| None` | Maximum slope threshold                             |
| `broad_bpi_lower` | `float \| None` | Minimum broad standardised BPI threshold            |
| `broad_bpi_upper` | `float \| None` | Maximum broad standardised BPI threshold            |
| `fine_bpi_lower`  | `float \| None` | Minimum fine standardised BPI threshold             |
| `fine_bpi_upper`  | `float \| None` | Maximum fine standardised BPI threshold             |

**Validation rules**:

- `code` must be unique within a `ClassificationDictionary`
- `name` must be non-empty
- Any pair `(lower, upper)` where both are set must satisfy `lower < upper`

**Source column mapping** (maintained from BTM 3.0 CSV schema):

| CSV column          | `ClassEntry` field |
| ------------------- | ------------------ |
| `Class`             | `code`             |
| `Zone`              | `name`             |
| `Depth_LowerBounds` | `depth_lower`      |
| `Depth_UpperBounds` | `depth_upper`      |
| `Slope_LowerBounds` | `slope_lower`      |
| `Slope_UpperBounds` | `slope_upper`      |
| `SSB_LowerBounds`   | `broad_bpi_lower`  |
| `SSB_UpperBounds`   | `broad_bpi_upper`  |
| `LSB_LowerBounds`   | `fine_bpi_lower`   |
| `LSB_UpperBounds`   | `fine_bpi_upper`   |

---

### ClassificationDictionary

Container for a validated list of `ClassEntry` objects.

| Field           | Type               | Description                                                   |
| --------------- | ------------------ | ------------------------------------------------------------- |
| `classes`       | `list[ClassEntry]` | Ordered list; classification order matters (first match wins) |
| `source_format` | `str`              | `'csv'`, `'xml'`, or `'xlsx'`                                 |
| `source_path`   | `str`              | Path of the original file                                     |

**Validation rules**:

- `len(classes) > 0`
- All `code` values must be unique

---

### BpiParams

Parameters for a single BPI computation.

| Field          | Type  | Description                                      |
| -------------- | ----- | ------------------------------------------------ |
| `inner_radius` | `int` | Inner ring radius, in cells (exclusive boundary) |
| `outer_radius` | `int` | Outer ring radius, in cells (inclusive boundary) |
| `scale`        | `str` | `'broad'` or `'fine'` — labels the output only   |

**Validation rules**:

- `inner_radius >= 1`
- `outer_radius > inner_radius`

---

### ModelParams

All parameters needed by `run_full_model`.

| Field                 | Type        | Description                                          |
| --------------------- | ----------- | ---------------------------------------------------- |
| `bathy_path`          | `str`       | Path to input bathymetric raster                     |
| `broad_bpi`           | `BpiParams` | Parameters for broad-scale BPI                       |
| `fine_bpi`            | `BpiParams` | Parameters for fine-scale BPI                        |
| `classification_file` | `str`       | Path to classification dictionary                    |
| `outdir`              | `str`       | Output directory path                                |
| `keep_intermediates`  | `bool`      | Default `True`; `False` writes only classified zones |

---

## State Transitions

```
Input bathymetric raster (BathymetricRaster)
        │
        ├─── compute_bpi(broad params)   ──► BpiRaster (broad)
        │         │
        │         └── standardize_bpi()  ──► StandardizedBpiRaster (broad_std)
        │
        ├─── compute_bpi(fine params)    ──► BpiRaster (fine)
        │         │
        │         └── standardize_bpi()  ──► StandardizedBpiRaster (fine_std)
        │
        ├─── compute_slope()             ──► SlopeRaster
        │
        └─[all above + ClassificationDictionary]
                  │
                  └── classify_terrain() ──► ClassifiedTerrainRaster
```

---

## File Naming Convention (produced by `run_full_model`)

| File                            | Description                          |
| ------------------------------- | ------------------------------------ |
| `{outdir}/broad_bpi.tif`        | Broad-scale BPI (integer)            |
| `{outdir}/fine_bpi.tif`         | Fine-scale BPI (integer)             |
| `{outdir}/broad_std.tif`        | Standardised broad BPI (int32, ×100) |
| `{outdir}/fine_std.tif`         | Standardised fine BPI (int32, ×100)  |
| `{outdir}/slope.tif`            | Slope in degrees (float32)           |
| `{outdir}/classified_zones.tif` | Integer class codes (int32)          |

All outputs: LZW-compressed GeoTIFF, matching CRS and extent of input.
