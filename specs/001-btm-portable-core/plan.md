# Implementation Plan: BTM Portable Core — Platform-Agnostic Modernization

**Branch**: `001-btm-portable-core` | **Date**: 2026-03-25 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/001-btm-portable-core/spec.md`

## Summary

Replace the ArcGIS-only BTM v3.0 toolbox with a platform-agnostic Python package (`btm`)
whose core algorithm layer (`btm/core/`) has zero GIS runtime dependencies. All scientific
algorithms (BPI, standardize BPI, slope, VRM, surface-to-planar ratio, classification,
depth statistics) are implemented as pure NumPy/SciPy functions with rasterio-based I/O.
Three thin adapter layers expose the core to users: a CLI (`python -m btm.<tool>`), a
rewritten ArcGIS Pro `.pyt` toolbox (targeting Pro ≥ 3.2), and a QGIS Processing provider
(LTR ≥ 3.34). Numerical output must match the original BTM 3.0 within ±1 integer unit /
±0.01° slope. Distributed as a local/git install via `pyproject.toml`.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: rasterio ≥ 1.3, GDAL (via rasterio), NumPy ≥ 1.24, SciPy ≥ 1.11, openpyxl ≥ 3.1  
**Optional (platform adapters)**: arcpy (ArcGIS Pro ≥ 3.2 conda env), PyQGIS / qgis.core (QGIS LTR ≥ 3.34)  
**Storage**: GeoTIFF files (rasterio/GDAL); classification dictionaries as CSV / XML / XLSX  
**Testing**: pytest ≥ 8; `@pytest.mark.arcgis` / `@pytest.mark.qgis` markers for runtime-gated tests; pytest-cov for coverage  
**Target Platform**: Cross-platform (Windows/macOS/Linux) for core + CLI; Windows-only for ArcGIS adapter  
**Project Type**: Python library + CLI tools  
**Performance Goals**: Full pipeline on ≥ 1 GB raster completes without OOM; block-based windowed I/O (rasterio) used for large inputs  
**Constraints**: Core imports zero GIS runtime (`arcpy`, `qgis`); `pyproject.toml` installable via `pip install .`; no PyPI publish  
**Scale/Scope**: 9 algorithm modules, 3 adapter layers, ~12 CLI entry points, test suite achieving ≥ 90% coverage on `btm/core/`

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

- [x] **I. TDD** — Failing tests for each algorithm are scoped in research.md and written before implementation. Test suite targets ≥ 90% coverage on `btm/core/`.
- [x] **II. Code Quality** — `ruff` (linting + formatting) configured in `pyproject.toml`; CI step fails on lint errors before tests run.
- [x] **III. Performance** — rasterio windowed reads/writes used for block-based processing; NumPy/SciPy vectorised operations throughout; no pixel-level Python loops.
- [x] **IV. Scientific Accuracy** — Each algorithm cites its peer-reviewed source: BPI (Wright et al. 2005), slope (Horn 1981), VRM (Sappington et al. 2007), surface ratio (Jenness 2002). Classification logic preserved from BTM 3.0 CON cascade exactly.
- [x] **V. Platform Portability** — `btm/core/` contains zero GIS runtime imports. ArcGIS `.pyt`, QGIS provider, and CLI are independent thin adapters. arcpy-gated tests use `@pytest.mark.arcgis`; CI runs clean without ArcGIS installed.
- [x] **VI. Simplicity** — One module per algorithm; `btm/io/` for raster I/O; `btm/classification/` for dictionary parsing. QGIS provider and `.pyt` parameter schemas derived from the same shared parameter definitions.

**Gate result: PASS** — All six gates satisfied. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-btm-portable-core/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI schemas)
│   ├── cli-bpi.md
│   ├── cli-standardize-bpi.md
│   ├── cli-slope.md
│   ├── cli-vrm.md
│   ├── cli-classify.md
│   ├── cli-run-model.md
│   └── cli-depth-statistics.md
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
btm/                          # New portable package (pip install .)
├── __init__.py
├── core/                     # Pure-Python algorithm layer — ZERO GIS runtime deps
│   ├── __init__.py
│   ├── bpi.py                # compute_bpi()
│   ├── standardize.py        # standardize_bpi()
│   ├── slope.py              # compute_slope()  [Horn 1981]
│   ├── vrm.py                # compute_vrm()    [Sappington 2007]
│   ├── surface_ratio.py      # compute_surface_planar_ratio()  [Jenness 2002]
│   ├── classify.py           # classify_terrain()
│   ├── depth_statistics.py   # focal mean/std/var/IQR/kurtosis
│   └── model.py              # run_full_model() orchestrator
├── classification/           # Dictionary I/O — CSV, XML, XLSX
│   ├── __init__.py
│   ├── reader.py             # BtmDocument-equivalent; returns list[ClassEntry]
│   ├── csv_reader.py
│   ├── xml_reader.py
│   └── xlsx_reader.py
├── io/                       # Raster I/O via rasterio
│   ├── __init__.py
│   ├── raster.py             # RasterDataset dataclass + read/write helpers
│   └── block.py              # Windowed block processor
├── adapters/
│   ├── arcgis/               # ArcGIS Pro wrapper (arcpy optional)
│   │   ├── __init__.py
│   │   └── tools.py          # BTM tool classes (delegates to btm.core)
│   └── qgis/                 # QGIS Processing provider (PyQGIS optional)
│       ├── __init__.py
│       ├── provider.py
│       └── algorithms/
│           ├── bpi_algorithm.py
│           ├── slope_algorithm.py
│           ├── vrm_algorithm.py
│           ├── classify_algorithm.py
│           └── run_model_algorithm.py
├── cli/                      # CLI entry points
│   ├── __init__.py
│   ├── bpi.py
│   ├── standardize_bpi.py
│   ├── slope.py
│   ├── vrm.py
│   ├── surface_ratio.py
│   ├── classify.py
│   ├── run_model.py
│   ├── depth_statistics.py
│   └── scale_comparison.py
└── _logging.py               # Shared logging setup helpers

Install/
└── toolbox/
    ├── btm.pyt               # Rewritten .pyt — thin wrapper over btm.adapters.arcgis
    └── btm.pyt.xml           # Toolbox metadata (unchanged structure)

legacy/                       # Frozen original scripts (READ-ONLY reference)
└── Install/
    └── toolbox/
        └── scripts/          # Original arcpy-coupled scripts preserved here

tests/
├── conftest.py               # pytest fixtures + marker declarations
├── unit/                     # Pure-Python; no GIS runtime required
│   ├── test_bpi.py
│   ├── test_standardize.py
│   ├── test_slope.py
│   ├── test_vrm.py
│   ├── test_surface_ratio.py
│   ├── test_classify.py
│   ├── test_depth_statistics.py
│   ├── test_raster_io.py
│   ├── test_classification_readers.py
│   └── test_run_model.py
├── integration/              # Require actual test rasters in tests/data/
│   ├── test_pipeline_integration.py
│   └── test_numerical_parity.py   # SC-002 parity checks vs reference outputs
└── arcgis/                   # @pytest.mark.arcgis — requires ArcGIS Pro
    └── test_pyt_tools.py

pyproject.toml                # Build system, deps, entry points, ruff config
```

**Structure Decision**: Single-package layout rooted at `btm/` at the repository root.
The existing `Install/toolbox/scripts/` is moved verbatim to `legacy/Install/toolbox/scripts/`
and kept as the frozen reference. The new `.pyt` at `Install/toolbox/btm.pyt` imports from
`btm.adapters.arcgis` rather than the legacy scripts.

## Complexity Tracking

No constitution violations. The multi-adapter structure (CLI + ArcGIS + QGIS) is required
by Constitution Principle V and is explicitly scoped in the feature spec (US1–US4). Each
adapter is a thin delegating layer, not an independent re-implementation.
