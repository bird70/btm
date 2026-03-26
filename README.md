# Benthic Terrain Modeler (BTM) v4

A platform-agnostic Python library and command-line toolset for analysis and
classification of benthic (seafloor) terrain.

BTM v4 decouples the scientific algorithms from ArcGIS, making them available
as a pure-Python package that runs anywhere with Python 3.11+. The original
ArcGIS Pro toolbox (`Install/toolbox/btm.pyt`) continues to work for users
who need it; the new core library powers both.

## What it does

| Algorithm        | CLI command           | Description                                             |
| ---------------- | --------------------- | ------------------------------------------------------- |
| BPI              | `btm-bpi`             | Bathymetric Position Index — crests, flats, depressions |
| Standardise BPI  | `btm-standardize-bpi` | Z-score rescaling × 100                                 |
| Slope            | `btm-slope`           | Horn (1981) 3×3 finite-difference, degrees              |
| VRM              | `btm-vrm`             | Sappington (2007) Vector Ruggedness Measure             |
| Surface ratio    | `btm-surface-ratio`   | Jenness (2002) surface-to-planar area                   |
| Depth statistics | `btm-depth-stats`     | Focal mean, std, IQR, kurtosis                          |
| Scale comparison | `btm-scale-compare`   | BPI across a range of scales                            |
| Classify         | `btm-classify`        | Rule-based terrain classification                       |
| Full pipeline    | `btm-run-model`       | All of the above in one command                         |

## Requirements

- Python 3.11+
- rasterio ≥ 1.3
- numpy ≥ 1.24
- scipy ≥ 1.11
- openpyxl ≥ 3.1

No ArcGIS installation required for the core library and CLI.

## Installation

```bash
# Clone (include the datatype submodule)
git clone https://github.com/EsriOceans/btm
git submodule update --init --recursive

# Create a virtual environment and install
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

## Quick start

Run the complete BTM pipeline on the bundled Fagatele Bay demo data:

```bash
btm-run-model \
  --bathy     tests/data/bathy5m_clip.tif \
  --classdict tests/data/fagatelebay.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner   1 --fine-outer   5 \
  --outdir    outputs/fagatelebay
```

Inspect the outputs and class distribution:

```bash
python scripts/run_fagatelebay.py --outdir outputs/fagatelebay
```

See [TESTING.md](TESTING.md) for a full walkthrough with expected values and
tips on visualising outputs in QGIS or ArcGIS Pro.

For combining BTM with machine-learning classification (XGBoost / LightGBM),
see [docs/classification-methods.md](docs/classification-methods.md) and the
step-by-step [docs/runsheet-hybrid-kaggle.md](docs/runsheet-hybrid-kaggle.md).

## Running the tests

```bash
# Unit + integration tests — no ArcGIS required, completes in ~10 s
pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis"

# With coverage report
pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis" \
    --cov=btm/core --cov=btm/classification --cov-report=term-missing

# ArcGIS Pro adapter tests (requires ArcGIS Pro + arcpy)
pytest tests/arcgis/ -m arcgis
```

## Repository layout

```
btm/
  core/            Scientific algorithms — zero GIS runtime imports
  io/              Raster I/O via rasterio
  classification/  Class dictionary readers (CSV, XML, XLSX)
  cli/             Argparse entry points (one per algorithm)
  adapters/
    arcgis/        ArcGIS Pro .pyt adapter (requires arcpy)
    qgis/          QGIS Processing provider (requires PyQGIS)

Install/
  toolbox/
    btm.pyt        ArcGIS Pro Python toolbox (legacy scripts)
    scripts/       Original arcpy-coupled scripts (BTM 3.0 reference)

legacy/
  10.0/            ArcGIS 10.0 era scripts (historical archive)
  build.bat        ArcGIS addin builder (no longer maintained)

tests/
  unit/            Fast unit tests, no file I/O
  integration/     Full pipeline tests against Fagatele Bay data
  arcgis/          ArcGIS Pro / arcpy tests (skipped without arcpy)
  data/            Fagatele Bay test dataset + classification files

scripts/           Helper scripts for development and demos
specs/             Feature specifications (speckit artefacts)
```

## Background

BTM was originally developed by ESRI Oceans as an ArcGIS toolbox (v1–v3,
2010–2017). v4 extracts the core algorithms into a standalone Python package
while preserving ArcGIS compatibility through a thin adapter layer. The
scientific algorithms are identical; they are simply decoupled from arcpy.

## License

See [LICENSE](LICENSE) for details.

## Citing

We ask that you use the following citation for this software:

> Wright, D.J., Pendleton, M., Boulware, J., Walbridge, S., Gerlt, B., Eslinger, D., Sampson, D., and Huntley, E. 2012. ArcGIS Benthic Terrain Modeler (BTM), v. 3.0, Environmental Systems Research Institute, NOAA Coastal Services Center, Massachusetts Office of Coastal Zone Management. Available online at [http://esriurl.com/5754](http://esriurl.com/5754).
