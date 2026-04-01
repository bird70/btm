# btm Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-01

## Active Technologies
- Python 3.13 (`.venv\Scripts\python.exe`; `requires-python = ">=3.11"`) + rasterio, numpy, scipy, pandas, scikit-learn, lightgbm, xgboost, catboost, pykrige, scikit-image (all confirmed installed in venv) (008-gis-feature-engineering)
- File-based — GeoTIFF rasters under `outputs/gis_layers/rasters/`; CSV under `data/`; Markdown report under `docs/` (008-gis-feature-engineering)
- Python 3.13.5 (`.venv\Scripts\python.exe`) + rasterio, numpy, scipy, scikit-image (0.26), lightgbm, xgboost, catboost, scikit-learn (1.8), pandas (009-obia-pixel-hybrid)
- File-based — GeoTIFF rasters in `data/MBES/`, CSV in `data/`, Markdown in `docs/` (009-obia-pixel-hybrid)
- Python 3.13.5 + scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.2.0, (014-btm-model-sweep)
- JSONL run registry (`artifacts/experiments/run_registry.jsonl`), (014-btm-model-sweep)
- Python 3.13.5 + NumPy, SciPy, rasterio, pandas, scikit-learn, pytest (015-ecology-habitat-features)
- CSV files (`data/train_btm.csv`, `data/train_btm_eco.csv`, `data/test.csv`); MBES raster (`data/MBES/bathymetry.tif`) (015-ecology-habitat-features)
- Python 3.11+ (current: 3.13.5) + NumPy, SciPy (`scipy.ndimage.uniform_filter`), scikit-image (`graycomatrix`/`graycoprops`), rasterio, scikit-learn, pandas (016-multiscale-terrain-features)
- GeoTIFF rasters (`data/MBES/`), CSV point data (`data/train.csv`, `data/test.csv`) (016-multiscale-terrain-features)
- Python 3.11+ (current: 3.13.5) + NumPy, SciPy, rasterio, scikit-learn (RandomForestClassifier), CatBoost, LightGBM, pandas (017-combined-multiscale-mbes)
- GeoTIFF rasters (`data/MBES/`), CSV point data (`data/train.csv`, `data/test.csv`), CSV feature cache (017-combined-multiscale-mbes)
- Python 3.11+ (3.13.5 in workspace) + scikit-learn (RF), lightgbm (LightGBM), pandas, numpy, rasterio, PyYAML, joblib (019-pipeline-cv-improvement)
- Flat files — CSV (data/submissions), JSON (metrics/provenance), joblib (models), YAML (configs) (019-pipeline-cv-improvement)

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
- 019-pipeline-cv-improvement: Added Python 3.11+ (3.13.5 in workspace) + scikit-learn (RF), lightgbm (LightGBM), pandas, numpy, rasterio, PyYAML, joblib
- 017-combined-multiscale-mbes: Added Python 3.11+ (current: 3.13.5) + NumPy, SciPy, rasterio, scikit-learn (RandomForestClassifier), CatBoost, LightGBM, pandas
- 016-multiscale-terrain-features: Added Python 3.11+ (current: 3.13.5) + NumPy, SciPy (`scipy.ndimage.uniform_filter`), scikit-image (`graycomatrix`/`graycoprops`), rasterio, scikit-learn, pandas


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
