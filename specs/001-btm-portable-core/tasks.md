---
description: "Task list for BTM Portable Core — Platform-Agnostic Modernization"
---

# Tasks: BTM Portable Core — Platform-Agnostic Modernization

**Input**: Design documents from `specs/001-btm-portable-core/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: TDD — tests are written FIRST, must fail, then implementation follows (Constitution Principle I).

**Organization**: Tasks are grouped by user story. US1 (standalone CLI pipeline) is the MVP — all
other stories depend on the US1/US2 core being complete.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependencies)
- **[Story]**: User story label (US1–US4)
- Exact file paths in all descriptions

---

## Phase 1: Setup

**Purpose**: Repository scaffold, package metadata, tooling configuration.

- [X] T001 Move `Install/toolbox/scripts/` verbatim to `legacy/Install/toolbox/scripts/` (freeze original code as READ-ONLY reference)
- [X] T002 Create `btm/` package root with `btm/__init__.py` (version `4.0.0.dev0`), `btm/core/__init__.py`, `btm/io/__init__.py`, `btm/classification/__init__.py`, `btm/cli/__init__.py`, `btm/adapters/__init__.py`, `btm/adapters/arcgis/__init__.py`, `btm/adapters/qgis/__init__.py`
- [X] T003 [P] Create `pyproject.toml` at repository root with `[build-system]` (setuptools ≥ 68), `[project]` (name=btm, version=4.0.0.dev0, requires-python=≥3.11, dependencies: rasterio≥1.3, numpy≥1.24, scipy≥1.11, openpyxl≥3.1), `[project.optional-dependencies]` dev group (pytest≥8, pytest-cov, ruff), `[project.scripts]` (btm-bpi, btm-standardize-bpi, btm-slope, btm-vrm, btm-surface-ratio, btm-classify, btm-run-model, btm-depth-stats, btm-scale-compare), `[tool.ruff]` (line-length=100, select E/F/W/I/UP), `[tool.pytest.ini_options]` (markers: arcgis, qgis; testpaths: tests)
- [X] T004 [P] Create `tests/conftest.py` with pytest fixture declarations: `bathy_raster_path`, `fagatelebay_csv_path`, `fagatelebay_xml_path`, `tmp_outdir`; register `arcgis` and `qgis` markers; add `collect_ignore` guard for `tests/arcgis/` when arcpy unavailable
- [X] T005 [P] Create `btm/_logging.py` with `get_logger(name)` returning a module-level logger with `NullHandler`; and `configure_cli_logging(verbose, quiet, log_file)` that attaches a `StreamHandler` (stdout) and optional `FileHandler`

**Checkpoint**: `pip install -e ".[dev]"` succeeds; `pytest --collect-only` runs without import errors.

---

## Phase 2: Foundational — Raster I/O & Classification Readers

**Purpose**: Shared infrastructure required by ALL user story implementations.

- [X] T006 Write failing tests in `tests/unit/test_raster_io.py` covering: `RasterDataset.from_file` reads bathy5m_clip.tif (correct shape, CRS, transform, nodata); `RasterDataset.to_file` writes a GeoTIFF with LZW compression; `cell_size()` returns mean of width/height; reading a non-existent file raises `FileNotFoundError`
- [X] T007 Implement `btm/io/raster.py`: `RasterDataset` dataclass (fields: path, array, crs, transform, nodata, cell_width, cell_height); `from_file(path) -> RasterDataset` using rasterio; `to_file(path, dtype, compress='lzw')` using rasterio; `cell_size() -> float` = mean of width/height; always overwrites output
- [X] T008 [P] Write failing tests in `tests/unit/test_classification_readers.py` covering: CSV reader returns correct list of `ClassEntry` from `tests/data/fagatelebay.csv`; XML reader returns correct list from `tests/data/fagatelebay_zone.xml`; XLSX reader returns correct list from a test .xlsx fixture; missing column raises `ValueError` with column name; malformed CSV raises `ValueError` with clear message
- [X] T009 [P] Implement `btm/classification/reader.py`: `ClassEntry` dataclass; `read_classification(path) -> list[ClassEntry]` that dispatches to CSV/XML/XLSX reader by extension; `btm/classification/csv_reader.py`; `btm/classification/xml_reader.py` (port from original `utils.py` BtmDocument XML logic, remove arcpy); `btm/classification/xlsx_reader.py` (openpyxl, replaces xlrd)
- [X] T010 Write failing tests in `tests/unit/test_block_processor.py` covering: windowed read of a small test raster in 2×2 tiles with overlap; output array matches full-raster read; overlap border pixels are correctly discarded
- [X] T011 Implement `btm/io/block.py`: `BlockProcessor` class with `process(func, src_path, dst_path, block_size, overlap)` using `rasterio.windows.Window` tiled reads/writes; overlap padding applied on read; inner tile written to output; handles edge tiles (partial blocks near raster boundary)

**Checkpoint**: `pytest tests/unit/test_raster_io.py tests/unit/test_classification_readers.py tests/unit/test_block_processor.py` all GREEN. No arcpy imported anywhere in `btm/`.

---

## Phase 3: User Story 2 — Importable Core Algorithm API (Priority: P2) 🎯

**Goal**: Each algorithm available as a pure-Python function; importable with no GIS runtime.

**Independent Test**: `from btm.core.bpi import compute_bpi` works in a plain Python 3.11 environment with only rasterio/numpy/scipy installed.

### Tests for User Story 2

> **Write tests FIRST — all must FAIL before implementation begins.**

- [X] T012 [P] [US2] Write failing tests in `tests/unit/test_bpi.py`: `compute_bpi` returns array of identical shape; annulus mean is correct for a synthetic 5×5 array with known values; integer rounding matches original formula (`round(bathy − focal_mean)`); inner≥outer radius raises `ValueError`; NoData sentinel propagates through output
- [X] T013 [P] [US2] Write failing tests in `tests/unit/test_standardize.py`: `standardize_bpi` returns int32 array; formula `round((bpi − mean) / std × 100)` matches for known input; all-constant input (std=0) raises `ZeroDivisionError` or returns zeros with warning
- [X] T014 [P] [US2] Write failing tests in `tests/unit/test_slope.py`: `compute_slope` returns float32 array in degrees [0, 90]; flat array (all zeros) → all-zero slope; 45° analytical ramp → slope ≈ 45°; Horn kernel coefficients verified against reference formula; NoData border cells handled
- [X] T015 [P] [US2] Write failing tests in `tests/unit/test_vrm.py`: `compute_vrm` returns float32 array in [0, 1]; flat surface → VRM ≈ 0; highly varied surface → VRM approaches 1; neighbourhood_size=1 raises `ValueError`; output shape matches input
- [X] T016 [P] [US2] Write failing tests in `tests/unit/test_surface_ratio.py`: `compute_surface_planar_ratio` returns float32; flat surface → ratio ≈ 1.0; sloped surface → ratio > 1.0; cell_size used correctly; output shape matches input
- [X] T017 [P] [US2] Write failing tests in `tests/unit/test_classify.py`: `classify_terrain` returns int32 array; correct class assigned for cells matching a single class; class priority order (first-match wins); all-zero when no class matches; NoData cells in any input → NoData in output

### Implementation for User Story 2

- [X] T018 [P] [US2] Implement `btm/core/bpi.py`: `compute_bpi(array, inner_radius, outer_radius, cell_size, nodata) -> np.ndarray`. Build boolean annulus footprint mask; use `scipy.ndimage.generic_filter` with the footprint for focal mean; subtract from input; `np.round(...).astype(np.int32)`; validate `inner_radius < outer_radius ≥ 1`; propagate NoData
- [X] T019 [P] [US2] Implement `btm/core/standardize.py`: `standardize_bpi(array, nodata) -> np.ndarray`. Compute masked mean and std (excluding NoData); `np.round((array − mean) / std * 100).astype(np.int32)`; raise `ValueError` if std == 0 and array is not all-NoData
- [X] T020 [P] [US2] Implement `btm/core/slope.py`: `compute_slope(array, cell_size, nodata) -> np.ndarray`. Pad array (reflect); apply Horn (1981) 3×3 weighted finite-difference kernel for dz/dx and dz/dy; `np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))`; output float32; propagate NoData
- [X] T021 [P] [US2] Implement `btm/core/vrm.py`: `compute_vrm(array, neighborhood_size, cell_size, nodata) -> np.ndarray`. Derive slope and aspect from Horn kernel (reuse slope.py helpers); compute x/y/z unit vector components; `scipy.ndimage.uniform_filter` for focal sums (scale by n²); `1 − resultant / n²`; validate `neighborhood_size ≥ 3`; output float32
- [X] T022 [P] [US2] Implement `btm/core/surface_ratio.py`: `compute_surface_planar_ratio(array, cell_size, nodata) -> np.ndarray`. Port Jenness (2002) triangle area method from `legacy/Install/toolbox/scripts/surface_area_to_planar_area.py`; remove all arcpy imports; use vectorised NumPy shifted-array approach; output float32
- [X] T023 [US2] Implement `btm/core/classify.py`: `classify_terrain(broad_std, fine_std, slope, bathy, classes, nodata) -> np.ndarray`. Sequential mask loop over `classes` (ClassEntry list); `np.where(mask & (result == 0), cls.code, result)` cascade; output int32 with 0 = unclassified; NoData in any input layer → NoData in output

**Checkpoint**: `pytest tests/unit/ -m "not arcgis and not qgis"` all GREEN. `python -c "from btm.core.bpi import compute_bpi; print('ok')"` works with no GIS runtime.

---

## Phase 4: User Story 1 — Standalone Full BTM Workflow (Priority: P1) 🎯 MVP

**Goal**: Complete end-to-end pipeline runnable with `python -m btm.run_model` and no GIS installation.

**Independent Test**: `python -m btm.run_model --bathy tests/data/bathy5m_clip.tif --broad-inner 2 --broad-outer 20 --fine-inner 1 --fine-outer 5 --classdict tests/data/fagatelebay.csv --outdir /tmp/btm_out` produces 6 GeoTIFF outputs.

### Tests for User Story 1

- [X] T024 [US1] Write failing tests in `tests/unit/test_run_model.py`: `run_full_model` returns paths to 6 output files; `keep_intermediates=False` produces only 1 file; all outputs are valid GeoTIFFs (rasterio can open them); output spatial extent and CRS match input; overwrite=True: re-running with same outdir succeeds
- [X] T025 [P] [US1] Write failing integration tests in `tests/integration/test_pipeline_integration.py`: full pipeline against `tests/data/bathy5m_clip.tif` + `fagatelebay.csv` produces 6 non-empty output files; classified_zones raster contains at least 2 distinct class codes; all output rasters are LZW-compressed
- [X] T032 [P] [US1] Generate reference numpy output arrays from BTM 3.0: document the command used in `tests/data/reference/README.md`; save `broad_bpi_ref.npy`, `fine_bpi_ref.npy`, `broad_std_ref.npy`, `fine_std_ref.npy`, `slope_ref.npy`, `classified_zones_ref.npy`
- [X] T026 [P] [US1] Write failing numerical parity tests in `tests/integration/test_numerical_parity.py`: BPI output matches reference `.npy` within `atol=1`; slope output matches reference within `atol=0.01`; classified_zones matches reference exactly (same integer codes, ≥95% cell agreement); (references generated from BTM 3.0 and saved to `tests/data/reference/`)

### Implementation for User Story 1

- [X] T028 [US1] Implement `btm/core/depth_statistics.py`: `compute_focal_stats(array, n_size, stats, window_type, nodata) -> dict[str, np.ndarray]`. Focal mean/std/variance via `scipy.ndimage.uniform_filter`; IQR via stacked-neighbourhood percentile approach (port from legacy `depth_statistics.py`); kurtosis via `scipy.stats.kurtosis` on neighbourhood stack; remove arcpy/netCDF4 dependency; use rasterio windowed I/O for large rasters
- [X] T027 [P] [US1] Write failing tests in `tests/unit/test_depth_statistics.py` covering: focal mean on a 5×5 synthetic array; IQR and kurtosis on known distributions; unsupported stat name raises `ValueError`

- [X] T029 [US1] Implement `btm/core/model.py`: `run_full_model(bathy_path, broad_bpi_inner, broad_bpi_outer, fine_bpi_inner, fine_bpi_outer, classification_file, outdir, keep_intermediates=True, block_size=None) -> dict[str, str]`. Orchestrate: read bathy → BPI broad → BPI fine → standardize both → slope → classify → write outputs; `keep_intermediates` controls which files are written; `block_size` passed to `BlockProcessor` when set; always overwrites; return dict of output name → path
- [X] T030 [P] [US1] Implement `btm/cli/bpi.py`, `btm/cli/standardize_bpi.py`, `btm/cli/slope.py`, `btm/cli/vrm.py`, `btm/cli/surface_ratio.py`, `btm/cli/depth_statistics.py`, `btm/cli/scale_comparison.py`: each uses argparse; calls corresponding `btm.core` function; wires `btm._logging.configure_cli_logging(verbose, quiet, log_file)`; returns exit codes per CLI contracts; `if __name__ == '__main__': sys.exit(main())`
- [X] T031 [US1] Implement `btm/cli/classify.py` and `btm/cli/run_model.py`: `run_model.py` parses all `ModelParams` fields + `--no-intermediates` and `--block-size`; calls `btm.core.model.run_full_model`; prints each output path to stdout on success; validates all input file paths exist before starting; returns exit codes per `contracts/cli-run-model.md`

**Checkpoint**: `pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis"` all GREEN. `python -m btm.run_model --help` works. Full pipeline produces 6 GeoTIFFs. Parity tests pass (T026).

---

## Phase 5: User Story 3 — ArcGIS Pro Geoprocessing Toolbox (Priority: P3)

**Goal**: All BTM tools accessible from ArcGIS Pro ≥ 3.2 Geoprocessing pane, delegating to `btm.core`.

**Independent Test**: In ArcGIS Pro ≥ 3.2, open `Install/toolbox/btm.pyt` and run "Run Full BTM Model" with Fagatele Bay test data. Output matches SC-002 tolerances.

### Tests for User Story 3

- [X] T033 [US3] Write failing tests in `tests/arcgis/test_pyt_tools.py` (marked `@pytest.mark.arcgis`): each tool class in `btm.pyt` has correct `getParameterInfo()` returning expected parameter names; `execute()` calls the correct `btm.core` function; spatial analyst fallback for slope: when `arcpy.sa.Slope` raises, `btm.core.slope.compute_slope` is called instead

### Implementation for User Story 3

- [X] T034 [US3] Implement `btm/adapters/arcgis/tools.py`: one class per BTM tool (BpiTool, StandardizeBpiTool, SlopeTool, VrmTool, SurfaceRatioTool, ClassifyTool, RunFullModelTool, DepthStatsTool); each class: `getParameterInfo()` returns parameter list matching original BTM 3.0 `.pyt.xml` names; `execute(parameters, messages)` extracts `.valueAsText` / `.value` from each param, calls corresponding `btm.core` function, writes output via `btm.io.raster`; spatial analyst fallback in SlopeTool
- [X] T035 [US3] Rewrite `Install/toolbox/btm.pyt`: import and expose all tool classes from `btm.adapters.arcgis.tools`; `Toolbox.tools` list matches original; toolbox name/alias preserved (`btm`, `BenthicTerrainModeler`); arcpy import guarded with `importlib.util.find_spec`
- [X] T036 [P] [US3] Update `Install/toolbox/btm.pyt.xml` tool metadata to reflect the rewritten `.pyt` parameter names; verify all tool descriptions and parameter help text are preserved from BTM 3.0

**Checkpoint**: `pytest tests/arcgis/ -m arcgis` GREEN (requires ArcGIS Pro). Opening `btm.pyt` in ArcGIS Pro shows all original tool names with no errors.

---

## Phase 6: User Story 4 — QGIS Processing Provider (Priority: P4)

**Goal**: Core BTM algorithms accessible from QGIS LTR ≥ 3.34 Processing Toolbox.

**Independent Test**: In QGIS 3.34+, activate `BtmProvider`. Run "Compute BPI" from Processing Toolbox on `bathy5m_clip.tif`. Output matches `btm.core.bpi` result.

### Tests for User Story 4

- [X] T037 [US4] Write failing tests in `tests/unit/test_qgis_provider.py` (marked `@pytest.mark.qgis`): `BtmProvider.id()` returns `'btm'`; `BtmProvider.algorithms()` contains expected algorithm names; `BpiBtmAlgorithm.parameterDefinitions()` returns correct param count and types; algorithm `processAlgorithm()` calls `btm.core.bpi.compute_bpi`

### Implementation for User Story 4

- [X] T038 [US4] Implement `btm/adapters/qgis/provider.py`: `BtmProvider(QgsProcessingProvider)` with `id()='btm'`, `name()='Benthic Terrain Modeler'`; `loadAlgorithms()` registers all algorithm classes
- [X] T039 [P] [US4] Implement `btm/adapters/qgis/algorithms/bpi_algorithm.py`, `standardize_bpi_algorithm.py`, `slope_algorithm.py`, `vrm_algorithm.py`, `classify_algorithm.py`, `run_model_algorithm.py`: each subclasses `QgsProcessingAlgorithm`; parameter definitions derived from shared parameter schema (matching CLI contracts); `processAlgorithm()` extracts raster paths from `QgsRasterLayer` context, delegates to `btm.core`, writes output via rasterio, returns result dict

**Checkpoint**: `pytest tests/unit/test_qgis_provider.py -m qgis` GREEN (requires PyQGIS). QGIS Processing Toolbox shows "Benthic Terrain Modeler" group with 6 algorithms.

---

## Final Phase: Polish & Cross-Cutting Concerns

- [X] T040 [P] Verify `ruff check btm/ tests/` passes with zero errors; fix any violations
- [X] T041 [P] Run `pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis" --cov=btm/core --cov=btm/classification --cov-report=term-missing --cov-fail-under=90`; fix any gaps to reach ≥ 90% coverage
- [X] T042 [P] Add `CHANGELOG` entry for v4.0.0.dev0: list CHG-001 through CHG-012 as described in spec.md
- [X] T043 [P] Verify all 6 output filenames in `run_full_model` match the naming convention in `data-model.md` (`broad_bpi.tif`, `fine_bpi.tif`, `broad_std.tif`, `fine_std.tif`, `slope.tif`, `classified_zones.tif`)
- [X] T044 Update `README.md` to document the new portable installation path (`pip install -e .`) and CLI usage (reference `specs/001-btm-portable-core/quickstart.md`)

---

## Dependencies

| Phase                  | Depends on | Notes                                        |
| ---------------------- | ---------- | -------------------------------------------- |
| Phase 1 (Setup)        | —          | No dependencies                              |
| Phase 2 (Foundations)  | Phase 1    | I/O and readers needed by all stories        |
| Phase 3 (US2 core API) | Phase 2    | Core algorithms need RasterDataset + readers |
| Phase 4 (US1 CLI)      | Phase 3    | CLI needs all core algorithms complete       |
| Phase 5 (US3 ArcGIS)   | Phase 3    | ArcGIS adapter wraps core; needs US2 done    |
| Phase 6 (US4 QGIS)     | Phase 3    | QGIS adapter wraps core; needs US2 done      |
| Final (polish)         | Phases 4–6 | All implementations done before polish       |

Phases 5 and 6 can be worked in parallel after Phase 3 is complete.

---

## Parallel Execution Examples

**After Phase 2 completes** — all of these can be parallelised across team members:

```
# US2 tests (parallel — different files)
T012 test_bpi.py  |  T013 test_standardize.py  |  T014 test_slope.py  |  T015 test_vrm.py
T016 test_surface_ratio.py  |  T017 test_classify.py

# US2 implementations (after their respective tests fail)
T018 bpi.py  |  T019 standardize.py  |  T020 slope.py  |  T021 vrm.py
T022 surface_ratio.py

# US3 + US4 after Phase 3 complete
T033-T036 (ArcGIS)  ||  T037-T039 (QGIS)
```

---

## Implementation Strategy

**MVP = Phase 1 + Phase 2 + Phase 3 + Phase 4 (T001–T031)**

This delivers User Story 1 (standalone CLI pipeline) and User Story 2 (importable API),
which together satisfy SC-001 through SC-005 and SC-007.

Work order within each phase strictly follows TDD:

1. Write tests (they must FAIL)
2. Get user/reviewer confirmation tests capture the intent
3. Implement until tests pass (Green)
4. Refactor while keeping tests green

Do not skip step 1. Do not write implementation before tests exist.
