# Feature Specification: BTM Portable Core — Platform-Agnostic Modernization

**Feature Branch**: `001-btm-portable-core`  
**Created**: 2026-03-25  
**Status**: Draft  
**Input**: User description: "Modernize the Benthic Terrain Modeler (BTM) as a platform-portable Python package for deriving BTM outputs from bathymetric raster data, runnable as standalone Python, QGIS Processing provider, or ArcGIS Pro geoprocessing tool, with no hard GIS runtime dependency in core algorithm logic."

## Clarifications

### Session 2026-03-25

- Q: What is the package distribution model for v1? → A: Local/git install only — `pyproject.toml` supports `pip install .` / `pip install -e .` from a clone; no PyPI publishing in scope for this feature.
- Q: How should `run_full_model` handle intermediate outputs? → A: Write all intermediates by default (broad_bpi, fine_bpi, broad_std, fine_std, slope alongside classified_zones); provide a `--no-intermediates` CLI flag (and `keep_intermediates=False` API kwarg) to suppress writing intermediates and retain only the final classified raster.
- Q: What is the overwrite policy when output files already exist? → A: Always overwrite silently (idempotent re-runs, matching original BTM 3.0 behaviour); no flag required.
- Q: What neighbourhood shape should VRM support? → A: Rectangle (square) only, matching the Sappington et al. 2007 reference algorithm. No circular window option for VRM.
- Q: What is the logging/observability strategy? → A: Python `logging` module in `btm.core` with a `NullHandler` by default; CLI entry points attach a `StreamHandler` to stdout (INFO by default, DEBUG with `--verbose`, WARNING with `--quiet`); optional `--log-file <path>` flag writes a copy to disk.

## Background

The existing BTM codebase (v3.0) is an ArcGIS-centric Python toolbox that is no longer
reliable in ArcGIS Pro ≥ 3.2. All algorithm logic is interleaved with direct `arcpy` calls,
making it impossible to run or test outside an ArcGIS installation.

This feature replaces the core computational layer with a platform-agnostic Python package
while preserving scientific fidelity to the original algorithms. Platform-specific entry
points (CLI, ArcGIS Pro `.pyt`, QGIS Processing provider) wrap that core as thin adapters.

## User Scenarios & Testing _(mandatory)_

### User Story 1 — Standalone Full BTM Workflow (Priority: P1)

A geospatial analyst with Python installed (but no GIS desktop application) wants to derive
a complete set of BTM outputs — broad-scale BPI, fine-scale BPI, standardised BPI grids,
slope, VRM ruggedness, and a classified benthic terrain raster — from a single input
bathymetric raster and a classification dictionary, using only the command line.

**Why this priority**: Delivering a working, ArcGIS-free pipeline is the primary motivation
for this modernization. It unlocks all users who cannot or do not use ArcGIS Pro, and
provides a reproducible, scriptable workflow for scientific analysis.

**Independent Test**: Run `python -m btm.run_model` against `tests/data/bathy5m_clip.tif`
with the canonical Fagatele Bay classification dictionary. Verify all six output rasters are
produced and that the classified output matches the expected zone distribution within a
defined tolerance.

**Acceptance Scenarios**:

1. **Given** a valid single-band bathymetric GeoTIFF and a classification dictionary (CSV),
   **When** the analyst runs `python -m btm.run_model --bathy <path> --broad-inner 2 --broad-outer 20 --fine-inner 1 --fine-outer 5 --classdict <path> --outdir <path>`,
   **Then** six output GeoTIFFs are produced (broad_bpi, fine_bpi, broad_std, fine_std,
   slope, classified_zones), all with LZW compression, matching the spatial extent and CRS
   of the input.
2. **Given** a bathymetric raster larger than available RAM,
   **When** the run_model command is executed,
   **Then** all tools complete without memory errors by using block-based processing.
3. **Given** a malformed or missing classification dictionary,
   **When** the tool is run,
   **Then** a clear, actionable error message is printed to stderr and no partial output is
   written to disk.

---

### User Story 2 — Individual Algorithm as Importable Function (Priority: P2)

A researcher wants to compute only the broad-scale BPI for a custom analysis by importing
the BTM core library in a Jupyter notebook or research script, passing in a NumPy array
with accompanying metadata, and receiving a NumPy array result — no files written unless
explicitly requested.

**Why this priority**: An importable, pure-Python API is the foundation for testing and for
integration into larger geospatial workflows. Every subsequent integration (CLI, QGIS,
ArcGIS) depends on this layer existing correctly.

**Independent Test**: Import `from btm.core.bpi import compute_bpi` in a plain Python 3
environment (no arcpy, no QGIS), call it with a NumPy array and radius parameters, and
assert the output array shape and dtype match expectations.

**Acceptance Scenarios**:

1. **Given** a valid 2-D NumPy float array representing bathymetry,
   **When** `compute_bpi(array, inner_radius, outer_radius, cell_size)` is called,
   **Then** a NumPy array of identical shape is returned containing BPI values computed as
   `round(bathy - focal_mean_annulus)` (matching the original algorithm).
2. **Given** `inner_radius >= outer_radius`,
   **When** any core algorithm is called,
   **Then** a `ValueError` with a descriptive message is raised before any computation.
3. **Given** an array containing NoData sentinel values,
   **When** any core algorithm is called,
   **Then** NoData cells propagate correctly to the output without corrupting valid cells.

---

### User Story 3 — ArcGIS Pro Geoprocessing Toolbox (Priority: P3)

An existing BTM user working in ArcGIS Pro ≥ 3.2 wants to run the BTM tools from the
Geoprocessing pane or Model Builder, using the same parameters as before, and get results
consistent with the original toolbox.

**Why this priority**: Maintaining ArcGIS Pro compatibility ensures continuity for current
users migrating from the broken v3.0 toolbox. This is the third priority because it
requires the P1/P2 core to exist first, and the user base is a subset of all potential users.

**Independent Test**: In ArcGIS Pro ≥ 3.2, add `Install/toolbox/btm.pyt` to the
Geoprocessing pane. Run the "Run Full BTM Model" tool with the Fagatele Bay test data.
Verify the classified output matches the canonical result within tolerance.

**Acceptance Scenarios**:

1. **Given** ArcGIS Pro ≥ 3.2 with Spatial Analyst licensed,
   **When** the user opens `btm.pyt` and runs any BTM tool,
   **Then** the tool executes using the platform-agnostic core and produces correct output.
2. **Given** Spatial Analyst is NOT licensed,
   **When** the user tries to run a tool that requires it (e.g., slope via arcpy.sa.Slope),
   **Then** the tool falls back to the pure-Python slope implementation and displays a
   warning to the user.
3. **Given** an ArcGIS-managed geodatabase workspace,
   **When** output paths are specified,
   **Then** the tool writes outputs to that geodatabase with valid table names.

---

### User Story 4 — QGIS Processing Provider (Priority: P4)

A QGIS LTR ≥ 3.34 user wants to access individual BTM algorithms from the QGIS Processing
Toolbox, chain them in a QGIS Model, or call them from a PyQGIS script.

**Why this priority**: QGIS is the preferred open-source path per the constitution. This
delivers BTM to a new audience without ArcGIS, but depends on the P2 core being stable.

**Independent Test**: In QGIS 3.34+, activate the BTM provider plugin. Open the Processing
Toolbox, find "BTM" group, and run "Compute BPI" against the test raster. Verify the output
raster is correct.

**Acceptance Scenarios**:

1. **Given** the BTM QGIS provider is installed,
   **When** the user runs the "Compute BPI" algorithm via the Processing Toolbox,
   **Then** the tool delegates to `btm.core.bpi.compute_bpi` and writes a valid GeoTIFF.
2. **Given** invalid parameter values (e.g., negative radii),
   **When** the algorithm is executed from QGIS,
   **Then** QGIS displays a validation error before any processing occurs.

---

### Edge Cases

- **Raster with NoData border cells**: BPI annulus neighbourhood extends outside the raster
  extent; border cells MUST be set to NoData rather than computed with partial neighbourhoods.
- **Very small rasters**: Inner radius larger than raster dimensions; tool MUST raise an
  informative error.
- **Non-square pixels**: Cell width ≠ cell height; cell size used in VRM and surface area
  calculations MUST use the mean of the two values, matching the original BTM v3 approach.
- **Classification with overlapping bounds**: Two classes share the same range on a
  dimension; the class appearing first in the dictionary MUST take precedence (same as
  original CON cascade behaviour).
- **Classification dictionary missing a column**: Tool MUST report the missing column name
  with a descriptive error, not a generic KeyError.
- **Raster CRS undefined**: Tool MUST process and write output, preserving whatever
  metadata exists, and emit a warning — not an error — about missing CRS.

## Requirements _(mandatory)_

### Functional Requirements

#### Core Algorithm Layer (`btm/core/`)

- **FR-001**: The package MUST provide a `compute_bpi(array, inner_radius, outer_radius, cell_size, nodata)` function that returns a NumPy array. The algorithm is: `round(bathy − mean_of_annulus_neighbourhood)`, where the annulus excludes cells within `inner_radius` and beyond `outer_radius` (measured in cells). Reference: Wright et al. 2005; BTM v3.0.
- **FR-002**: The package MUST provide a `standardize_bpi(array, nodata)` function that internally computes the masked mean and standard deviation of the array (excluding NoData cells), then returns a NumPy array with values `round((array − mean) / std × 100)`. This standardizes BPI to a dimensionless z-score scaled by 100. A `ValueError` MUST be raised if `std == 0` and the array is not entirely NoData.
- **FR-003**: The package MUST provide a `compute_slope(array, cell_size)` function that returns slope in degrees, using the Horn (1981) eight-direction finite-difference method — the same method used by the original `arcpy.sa.Slope`.
- **FR-004**: The package MUST provide a `compute_vrm(array, neighborhood_size, cell_size)` function implementing the Vector Ruggedness Measure (Sappington et al. 2007): `VRM = 1 − (resultant_vector / n²)` where `n` = neighbourhood side length in cells. The neighbourhood shape is fixed as a square (rectangle); no circular window option is provided, in strict accordance with the reference algorithm.
- **FR-005**: The package MUST provide a `compute_surface_planar_ratio(array, cell_size)` function implementing the Jenness (2002) surface-area-to-planar-area rugosity method as a secondary rugosity option.
- **FR-006**: The package MUST provide a `classify_terrain(broad_std, fine_std, slope, bathy, classes)` function that applies a sequential conditional evaluation matching the original CON cascade: `depth → slope → fine_BPI → broad_BPI`, returning an integer-coded class array.
- **FR-007**: The package MUST provide functions to read classification dictionaries from CSV (with the original column schema), XML (original BTM XML format), and XLSX formats, returning a uniform list-of-dicts structure.
- **FR-008**: The package MUST provide a `run_full_model(bathy_path, broad_inner, broad_outer, fine_inner, fine_outer, classification_file, outdir, keep_intermediates=True)` orchestrator that calls FR-001 through FR-007 in sequence. By default (`keep_intermediates=True`) all six outputs (broad_bpi, fine_bpi, broad_std, fine_std, slope, classified_zones) are written as GeoTIFF to `outdir`. When `keep_intermediates=False`, only `classified_zones` is written; intermediates are computed in memory and discarded. The CLI flag `--no-intermediates` maps to `keep_intermediates=False`. Existing output files are always silently overwritten (idempotent re-runs).
- **FR-009**: All outputs written to disk MUST use LZW lossless compression.
- **FR-010**: All core functions MUST operate on NumPy arrays and MUST NOT import arcpy, qgis, or any other GIS runtime.
- **FR-011**: Block-based processing MUST be available for BPI, slope, and VRM to handle rasters larger than available RAM, splitting the raster into overlapping tiles, computing per-tile, and reassembling without edge artefacts.
- **FR-012**: Core modules MUST expose depth statistics: focal mean, standard deviation, variance, interquartile range (IQR), and kurtosis over a rectangular or circular neighbourhood.

#### Raster I/O (`btm/io/`)

- **FR-013**: The package MUST read and write GeoTIFF rasters (and any GDAL-supported format) without requiring a GIS desktop runtime.
- **FR-014**: The I/O layer MUST expose a common `RasterDataset` abstraction (path, array, CRS, transform, nodata, cell_width, cell_height) used by all core functions. `RasterDataset` MUST expose a `cell_size() -> float` method returning the arithmetic mean of `cell_width` and `cell_height`. All algorithms that accept a cell size parameter MUST call `cell_size()` rather than assuming square pixels; this ensures correct behaviour when cell width ≠ cell height.

#### CLI Entry Points (`python -m btm.<tool>`)

- **FR-015**: Each algorithm MUST be runnable as `python -m btm.<tool> [args]` with `--help` support. Tools: `bpi`, `standardize_bpi`, `slope`, `vrm`, `surface_ratio`, `classify`, `run_model`, `depth_statistics`, `scale_comparison`.
- **FR-016**: All CLI tools MUST accept `--verbose` / `--quiet` flags and an optional `--log-file <path>` flag. `--verbose` sets log level to DEBUG; `--quiet` sets it to WARNING; default is INFO. When `--log-file` is provided, all log records are also written to that file. Log output goes to stdout by default.
- **FR-023**: All `btm.core` modules MUST use Python's standard `logging` module for all diagnostic output. Each module MUST add only a `NullHandler` at import time, following the Python library best-practice pattern (PEP 3147 / logging HOWTO). Callers (CLI, ArcGIS wrapper, QGIS provider) are responsible for configuring handlers and levels; the core library MUST NOT configure logging globally.
- **FR-017**: All CLI tools MUST return exit code 0 on success and non-zero on failure.

#### ArcGIS Pro Wrapper (`Install/toolbox/btm.pyt`)

- **FR-018**: The `.pyt` toolbox MUST expose all tools present in the original BTM 3.0 toolbox with the same parameter names, enabling backward compatibility for any saved Model Builder definitions.
- **FR-019**: Each `.pyt` tool MUST delegate all computation to the corresponding `btm.core` function; the wrapper MUST only convert between ArcGIS parameter types and `btm.core` input types.
- **FR-020**: When arcpy Spatial Analyst is available, the ArcGIS wrapper MAY use `arcpy.sa.Slope` for slope; if unavailable, it MUST fall back silently to the `btm.core.slope` implementation.

#### QGIS Provider (`btm/adapters/qgis/`)

- **FR-021**: A QGIS Processing provider MUST register all core algorithms (BPI, Standardize BPI, Slope, VRM, Classify, Run Full Model) in the QGIS Geoprocessing framework. The `surface_ratio` (FR-005) and `depth_statistics` (FR-012) tools are **CLI-only** for this feature version and are NOT included in the QGIS provider; this is a deliberate scope boundary, not an oversight.
- **FR-022**: Provider parameter definitions MUST be derived from the same parameter schemas used by the CLI to avoid drift.

#### Changes from Original BTM 3.0

- **CHG-001 (Improvement)**: All algorithm logic extracted from arcpy-coupled scripts into pure-Python `btm/core/` modules. Original code had arcpy calls mixed into computation; new code separates I/O, algorithm, and platform concerns.
- **CHG-002 (Improvement)**: Raster I/O now uses rasterio/GDAL instead of `arcpy.CopyRaster_management`, enabling cross-platform support and removing licence requirements for simple file operations.
- **CHG-003 (Improvement)**: Slope calculation mirrors the Horn (1981) method used by ArcGIS `Slope` under the hood; output should be numerically identical but now runs without the Spatial Analyst extension.
- **CHG-004 (Preserved)**: BPI formula preserved exactly: `round(bathy − focal_annulus_mean)`. No algorithmic change.
- **CHG-005 (Preserved)**: BPI standardization formula preserved: `round((bpi − mean) / std × 100)`. No change.
- **CHG-006 (Preserved)**: VRM formula preserved exactly from Sappington et al. 2007.
- **CHG-007 (Preserved)**: Classification CON cascade logic preserved: depth → slope → fine BPI → broad BPI, first matching class wins.
- **CHG-008 (Improvement)**: Block-based processor rewritten without the NetCDF4 intermediate file (original used arcpy's `RasterToNetCDF_md`); the new approach uses rasterio windowed reads/writes directly, removing the netCDF4 dependency.
- **CHG-009 (Improvement)**: `xlrd` dependency dropped; Excel classification files now read with `openpyxl` only.
- **CHG-010 (Preserved)**: All three classification dictionary formats (CSV, XML, XLSX) continue to be supported with no format changes.
- **CHG-011 (Deprecation)**: The `.esriaddin` format is frozen. No new functionality is added to the Add-in; existing Add-in code moves to `legacy/` unchanged.
- **CHG-012 (Improvement)**: ArcGIS Pro `.pyt` rewritten to delegate to `btm.core`; the Spatial Analyst extension is only checked out when specifically needed (not at module import time).

### Key Entities

- **BathymetricRaster**: Input single-band raster representing water depth (typically negative values). Key attributes: file path, CRS, spatial transform, cell size, NoData value, NumPy array representation.
- **BpiRaster**: Output of BPI computation. Attributes: scale type (broad/fine), inner radius, outer radius, same spatial envelope as input.
- **StandardizedBpiRaster**: Output of BPI standardization. Dimensionless integer values centred near 0, scaled by 100.
- **SlopeRaster**: Output in degrees (0–90). Derived from bathymetry using an 8-direction finite-difference kernel.
- **VrmRaster**: Vector Ruggedness Measure output. Values in range [0, 1]; 0 = flat, approaching 1 = extremely rugged.
- **ClassificationDictionary**: A tabular structure where each row defines one terrain class via threshold ranges on depth, slope, broad BPI, and fine BPI. Supported formats: CSV, XML, XLSX.
- **ClassifiedTerrainRaster**: Integer-coded raster where each cell value maps to a class in the ClassificationDictionary. Includes a raster attribute table with class names.
- **RasterDataset**: Internal abstraction holding (array, crs, transform, nodata, path); used by all core functions and I/O adapters to decouple format from algorithm.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: The full BTM model pipeline (`run_model`) completes successfully against `tests/data/bathy5m_clip.tif` on a machine with no ArcGIS installation.
- **SC-002**: All six BTM outputs (broad_bpi, fine_bpi, broad_std, fine_std, slope, classified_zones) produced by the new implementation match the outputs of the original BTM 3.0 (run in a working ArcGIS environment) cell-by-cell within a tolerance of ±1 integer unit for integer rasters and ±0.01 degrees for slope — covering both the Fagatele Bay zone and structure test cases.
- **SC-003**: The automated test suite (`pytest`) passes in full with zero `arcpy` imports outside of explicitly marked `@pytest.mark.arcgis` tests; CI runs green on a machine with only Python 3.11 and rasterio installed.
- **SC-004**: Block-based processing handles the full-resolution Fagatele Bay 5 m raster (≥ 100 MB) without exhausting available system memory (measured via process peak RSS).
- **SC-005**: Running `python -m btm.run_model --help` produces usage documentation without any GIS runtime installed.
- **SC-006 (ArcGIS Pro)**: When run inside ArcGIS Pro ≥ 3.2, all tools complete without errors and results match SC-002 tolerances; this is validated by the `@pytest.mark.arcgis` test suite.
- **SC-007**: Test coverage for `btm/core/` is ≥ 90% line coverage.

## Assumptions

- The input bathymetric raster is a single-band floating-point raster. Multi-band inputs are
  out of scope.
- Depth values may be positive or negative; the tools make no assumption about sign convention
  — results depend entirely on the provided classification dictionary thresholds.
- The canonical reference dataset (`tests/data/bathy5m_clip.tif`, `fagatelebay*.xml`, and
  `fagatelebay*.csv`) remains in the repository and is used as the numerical ground truth for
  output validation (SC-002).
- The legacy `Install/toolbox/scripts/` directory is preserved unchanged in `legacy/`; it is
  the reference implementation for numerical comparison but is NOT executed by the new pipeline.
- Arc-Chord Ratio (ACR) tool requires 3D Analyst and GeoStatistical Analyst extensions;
  it is out of scope for the standalone/QGIS paths in this feature (ArcGIS-only fallback
  per Constitution Principle V rule 4).
- QGIS provider is treated as "value-add" scope; the P4 user story is implemented only after
  P1–P3 are complete and passing.
- The `scale_comparison` visualization tool (matplotlib output) is included in the CLI path
  only; no QGIS or ArcGIS Pro GUI wrapper is required for v1.
- **Distribution**: The package is distributed as a local/git install only for this feature.
  Users install via `pip install .` or `pip install -e .` from a clone of this repository.
  A `pyproject.toml` with `[project]` metadata and `[project.scripts]` entry points MUST be
  provided. PyPI publication is out of scope.
