# Tests

## Running the test suite

```bash
# All tests (no ArcGIS required) — ~10 seconds
pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis"

# With coverage
pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis" \
    --cov=btm/core --cov=btm/classification --cov-report=term-missing

# ArcGIS Pro adapter tests (requires ArcGIS Pro + arcpy on PATH)
pytest tests/arcgis/ -m arcgis
```

## Layout

```
tests/
  unit/        Fast unit tests for individual modules (no file I/O where possible)
  integration/ Full pipeline tests against the Fagatele Bay reference raster
  arcgis/      ArcGIS Pro toolbox tests (testMain.py, test_pyt_tools.py)
               — skipped automatically when arcpy is unavailable
  data/        Fagatele Bay test dataset
               bathy5m_clip.tif  — 5 m bathymetry (input)
               fagatelebay.csv   — classification dictionary (11 classes)
               fagatelebay.xml   — same dictionary in XML format
               fagatelebay.xlsx  — same dictionary in XLSX format
               broad_bpi.tif     — pre-computed broad BPI (BTM 3.0 reference)
               fine_bpi.tif      — pre-computed fine BPI
               broad_std.tif     — pre-computed standardised broad BPI
               fine_std.tif      — pre-computed standardised fine BPI
               slope.tif         — pre-computed slope
               kurtosis.npy      — pre-computed kurtosis array
               reference/        — .npy files for numerical parity tests
                                   (see reference/README.md to generate)
```

## Numerical parity tests

`tests/integration/test_numerical_parity.py` checks that BTM v4 outputs agree
with BTM 3.0 (ArcGIS Pro) to within rounding tolerance. These tests are
automatically skipped until the reference `.npy` files are present.
See `tests/data/reference/README.md` for how to generate them.

## ArcGIS tests

`tests/arcgis/testMain.py` contains the original BTM 3.0 nose-style tests that
cover the arcpy-coupled scripts in `Install/toolbox/scripts/`. They require
ArcGIS Pro to be installed and `arcpy` to be importable. They are collected
only when arcpy is available.
