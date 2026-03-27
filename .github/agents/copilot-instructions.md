# btm Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-27

## Active Technologies
- Python 3.13 (`.venv\Scripts\python.exe`; `requires-python = ">=3.11"`) + rasterio, numpy, scipy, pandas, scikit-learn, lightgbm, xgboost, catboost, pykrige, scikit-image (all confirmed installed in venv) (008-gis-feature-engineering)
- File-based — GeoTIFF rasters under `outputs/gis_layers/rasters/`; CSV under `data/`; Markdown report under `docs/` (008-gis-feature-engineering)
- Python 3.13.5 (`.venv\Scripts\python.exe`) + rasterio, numpy, scipy, scikit-image (0.26), lightgbm, xgboost, catboost, scikit-learn (1.8), pandas (009-obia-pixel-hybrid)
- File-based — GeoTIFF rasters in `data/MBES/`, CSV in `data/`, Markdown in `docs/` (009-obia-pixel-hybrid)

- Python 3.11+ + rasterio ≥ 1.3, GDAL (via rasterio), NumPy ≥ 1.24, SciPy ≥ 1.11, openpyxl ≥ 3.1 (001-btm-portable-core)

## Project Structure

```text
src/
tests/
```

## Commands

cd src; pytest; ruff check .

## Code Style

Python 3.11+: Follow standard conventions

## Recent Changes
- 009-obia-pixel-hybrid: Added Python 3.13.5 (`.venv\Scripts\python.exe`) + rasterio, numpy, scipy, scikit-image (0.26), lightgbm, xgboost, catboost, scikit-learn (1.8), pandas
- 008-gis-feature-engineering: Added Python 3.13 (`.venv\Scripts\python.exe`; `requires-python = ">=3.11"`) + rasterio, numpy, scipy, pandas, scikit-learn, lightgbm, xgboost, catboost, pykrige, scikit-image (all confirmed installed in venv)

- 001-btm-portable-core: Added Python 3.11+ + rasterio ≥ 1.3, GDAL (via rasterio), NumPy ≥ 1.24, SciPy ≥ 1.11, openpyxl ≥ 3.1

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
