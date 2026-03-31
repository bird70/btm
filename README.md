# BTM Monorepo — Benthic Terrain Modeler + Kaggle ML Pipeline

This repository is a **unified monorepo** that combines two complementary
projects for seafloor habitat analysis:

| Component         | Package         | Purpose                                                                                                                                 |
| ----------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **BTM v4**        | `btm`           | Platform-agnostic scientific library for benthic terrain analysis (BPI, Slope, VRM, …)                                                  |
| **benthic_model** | `benthic_model` | Reproducible ML pipeline for the [NIWA Kaggle Benthic Habitat competition](https://github.com/niwacolours/benthic-terrain-model-kaggle) |

Both components share the same `data/` directory (MBES bathymetry + backscatter
rasters and train/test CSVs from the Refugio Cove competition dataset) and a
single Python virtual environment.

---

## BTM v4 — Terrain Analysis Library

BTM v4 decouples the Esri BTM scientific algorithms from ArcGIS, making them
available as a pure-Python package that runs anywhere with Python 3.11+. The
original ArcGIS Pro toolbox (`Install/toolbox/btm.pyt`) continues to work for
users who need it.

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
| Export features  | `btm-export-features` | Sample BTM derivatives at point locations for ML input  |

---

## benthic_model — Kaggle ML Pipeline

`benthic_model` (`src/benthic_model/`) is a structured, reproducible supervised
ML pipeline that:

- Extracts predictor variables from MBES bathymetry and backscatter GeoTIFFs at
  sample coordinates.
- Engineers neighbourhood and interaction features (texture, TPI, focal stats).
- Trains baseline (Random Forest) and candidate (XGBoost / LightGBM / CatBoost)
  classifiers using **spatial blocked k-fold** cross-validation.
- Compares runs against a declared baseline using weighted F1.
- Generates `ID,class` submission files for the Kaggle competition.

Competition classes: `ALG`, `FMAT`, `NVB`, `SGAM`, `SGZ`
Best CV result: weighted F1 **0.8024** (candidate-20260330064219, rf-btm-fine, spatial blocked CV)
Best Kaggle public F1: **0.79518** (R04/R06 RF+BTM, run 014-btm-model-sweep) — previous best was 0.76413

### Pipeline commands

```bash
# Train baseline
PYTHONPATH=src python -m benthic_model.cli train \
  --train-csv data/train.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif \
  --config configs/baseline.yaml --run-type baseline --seed 42

# Train candidate
PYTHONPATH=src python -m benthic_model.cli train \
  --train-csv data/train.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif \
  --config configs/candidate.yaml --run-type candidate --seed 42

# Evaluate (spatial blocked CV)
PYTHONPATH=src python -m benthic_model.cli evaluate \
  --run-id <run_id> --fold-scheme spatial_blocked

# Predict test labels
PYTHONPATH=src python -m benthic_model.cli predict \
  --run-id <run_id> --test-csv data/test.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif

# Build Kaggle submission
PYTHONPATH=src python -m benthic_model.cli make-submission \
  --predictions artifacts/predictions/<run_id>_test_predictions.csv \
  --sample-submission data/sample_submission.csv \
  --output submissions/submission.csv
```

### Hybrid BTM + ML workflow

BTM terrain derivatives can be injected as additional features:

```bash
# Extract BTM features at all sample points
btm-export-features \
  --bathy data/MBES/bathymetry.tif \
  --points data/train.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner   1 --fine-outer   5 \
  --outdir data/btm_rasters \
  --output data/train_btm.csv
```

See [docs/runsheet-hybrid-kaggle.md](docs/runsheet-hybrid-kaggle.md) for the
full end-to-end hybrid workflow,
[docs/classification-methods.md](docs/classification-methods.md) for a
comparison of rule-based vs. ML classification approaches, and
[docs/modelling-overview.md](docs/modelling-overview.md) for a concise
overview of all modelling strategies and their results.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/niwacolours/btm
cd btm

# Create a virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# BTM scientific library only
pip install -e ".[dev]"

# BTM + ML pipeline (benthic_model) — recommended for Kaggle work
pip install -e ".[dev,benthic]"

# Everything
pip install -e ".[all]"
```

Verify both packages are importable:

```bash
python -c "import btm; import benthic_model; print('Both packages OK')"
btm-export-features --help
```

---

## Quick start — BTM terrain analysis

Run the complete BTM pipeline on the bundled Fagatele Bay demo data:

```bash
btm-run-model \
  --bathy     tests/data/bathy5m_clip.tif \
  --classdict tests/data/fagatelebay.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner   1 --fine-outer   5 \
  --outdir    outputs/fagatelebay
```

See [TESTING.md](TESTING.md) for a full walkthrough with expected values.

---

## Running the tests

```bash
# BTM unit + integration tests (no ArcGIS required, ~10 s)
pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis and not benthic"

# benthic_model ML pipeline tests (requires benthic extras)
pytest tests/ -m benthic
# or all tests from src/benthic_model:
pytest tests/unit/test_config_and_metadata.py tests/unit/test_cv_protocols.py \
       tests/contract/ tests/integration/test_training_evaluation_pipeline.py

# Full test suite (all non-ArcGIS tests)
pytest tests/ -m "not arcgis and not qgis"

# With coverage
pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis" \
    --cov=btm/core --cov=btm/classification --cov=src/benthic_model \
    --cov-report=term-missing

# ArcGIS Pro adapter tests (requires ArcGIS Pro + arcpy)
pytest tests/arcgis/ -m arcgis
```

---

## Repository layout

```
btm/                        BTM scientific library
  core/                     Algorithms — no GIS runtime required
  io/                       Raster I/O (rasterio)
  classification/           Class dictionary readers (CSV, XML, XLSX)
  cli/                      Argparse CLI entry points
  features/                 Point-sampled BTM feature extraction for ML
  adapters/
    arcgis/                 ArcGIS Pro .pyt adapter (requires arcpy)
    qgis/                   QGIS Processing provider (requires PyQGIS)

src/
  benthic_model/            Kaggle ML pipeline package
    cli.py                  Train / evaluate / predict / make-submission commands
    config.py               Pipeline configuration schema (YAML)
    data/                   Ingest, raster extraction, validation utilities
    features/               Feature engineering and spatial context
    models/                 Baseline (RF) and candidate (XGBoost) trainers
    evaluation/             Metrics, spatial CV, baseline-vs-candidate comparison
    inference/              Test-set prediction
    submission/             Kaggle submission file writer and validator
    experiment/             Run metadata and registry

data/                       Competition dataset (shared by both pipelines)
  train.csv                 Labelled sample points (ID, x, y, class)
  test.csv                  Unlabelled test points (ID, x, y)
  sample_submission.csv     Competition submission template
  METADATA.MD               Survey metadata (CRS, resolution, collection details)
  MBES/
    bathymetry.tif          Single-band depth raster (UTM Zone 55S, metres)
    backscatter.tif         Single-band acoustic backscatter intensity raster

configs/                    benthic_model run configurations
  baseline.yaml             Baseline model settings
  candidate.yaml            Candidate model settings

notebooks/                  Jupyter notebooks for experimentation
  01_data_validation.ipynb
  02_feature_exploration.ipynb
  03_model_experiments.ipynb

scripts/                    Standalone experiment scripts (run history)
  experiment_v2.py          Early LightGBM baseline
  experiment_v5.py          Spatial ensemble (KNN + CatBoost + LGB)
  experiment_v6.py          OBIA + pixel-based hybrid (Kaggle 0.764)
  experiment_v7.py          Enhanced focal + SMOTE (identified spatial leakage)
  experiment_v8.py          Clean KNN/RF/LGB, no leakage (RF generalisation failure)
  experiment_v9.py          CatBoost + LGB + texture (current best CV 0.626)

reports/                    Metrics, quality, and reproducibility evidence
  metrics/
    final_model_summary.md  Selected model summary (baseline F1 = 0.8026)
    baseline-20260324181702_*.{md,json,csv}
  quality/
    ci_gate_report.md       Lint + test gate status
  reproducibility/
    baseline-20260324181702_reproducibility.md

submissions/                Final Kaggle submission files
  submission.csv            Current submission (baseline-20260324181702)

artifacts/                  Model artefacts and intermediate outputs
specs/                      Feature specifications (speckit)
  001-btm-portable-core/    BTM platform-portability spec
  007-gis-expert-annotation/
  008-gis-feature-engineering/
  009-obia-pixel-hybrid/
  001-build-benthic-model/  benthic_model full pipeline spec
tests/                      Automated tests
  unit/                     BTM algorithm unit tests + benthic_model unit tests
  integration/              BTM pipeline + benthic_model integration tests
  contract/                 benthic_model CLI and output format contracts
  arcgis/                   ArcGIS Pro adapter tests (requires arcpy)
  data/                     Raster fixtures (Fagatele Bay clip)
docs/                       Run reports and method documentation
  classification-methods.md Rule-based vs ML classification comparison
  runsheet-hybrid-kaggle.md End-to-end BTM + ML hybrid workflow
  run-008-gis-feature-engineering.md
  run-009-obia-pixel-hybrid.md
  run-010-enhanced-features-lgb.md
  run-011-clean-knn-rf-lgb.md
  run-012-v9-catboost-lgb-texture.md
chat-initial-prompts/       Initial AI-assisted research chat logs
```

---

## Experiment / branch history

Each numbered branch corresponds to a Kaggle experiment iteration:

| Branch                       | Experiment             | Key change                 | Best Kaggle score     |
| ---------------------------- | ---------------------- | -------------------------- | --------------------- |
| 001-btm-portable-core        | BTM library extraction | ArcGIS → pure Python       | —                     |
| 003-btm-ml-hybrid            | BTM + LightGBM         | First ML submission        | —                     |
| 004-kaggle-hybrid-submission | BTM + LGB hybrid       | v4 submission              | —                     |
| 005-kaggle-spatial-ensemble  | Spatial ensemble       | KNN + CatBoost + LGB       | —                     |
| 006-kaggle-kriging-ensemble  | Kriging ensemble       | Depth-stratified kriging   | —                     |
| 007-gis-expert-annotation    | Human-in-the-loop      | Expert annotation pass     | —                     |
| 008-gis-feature-engineering  | GIS features           | Slope/BZ/AF weighted       | 0.6971 CV             |
| 009-obia-pixel-hybrid        | OBIA + pixel hybrid    | SLIC segmentation          | **0.7639 Kaggle**     |
| 010-enhanced-features-lgb    | Enhanced + SMOTE       | Identified spatial leakage | —                     |
| 011-clean-knn-rf-lgb         | Clean no-leakage       | KNN/RF/LGB, no coords      | 0.47 Kaggle (RF fail) |
| 012-v9-catboost-lgb-texture  | CatBoost + texture     | +bathy_std_9, tpi_9        | 0.6256 CV             |
| 013-consolidate-monorepo     | **This branch**        | Merged btm + btm-k         | —                     |

Submission CSVs are in `data/submission_v*.csv` and `submissions/submission.csv`.

---

## Key documentation

| Document                                                                         | Description                               |
| -------------------------------------------------------------------------------- | ----------------------------------------- |
| [docs/classification-methods.md](docs/classification-methods.md)                 | Rule-based BTM vs ML classification       |
| [docs/runsheet-hybrid-kaggle.md](docs/runsheet-hybrid-kaggle.md)                 | End-to-end hybrid BTM + benthic_model     |
| [TESTING.md](TESTING.md)                                                         | BTM test walkthrough with expected values |
| [specs/001-build-benthic-model/spec.md](specs/001-build-benthic-model/spec.md)   | benthic_model feature specification       |
| [specs/001-build-benthic-model/plan.md](specs/001-build-benthic-model/plan.md)   | Implementation plan                       |
| [reports/metrics/final_model_summary.md](reports/metrics/final_model_summary.md) | Best model summary                        |
| [data/METADATA.MD](data/METADATA.MD)                                             | Competition dataset metadata              |

---

## Requirements at a glance

| Layer                  | Key packages                                         |
| ---------------------- | ---------------------------------------------------- |
| BTM core               | rasterio, numpy, scipy, scikit-image, openpyxl       |
| BTM + ML bridge        | + pandas                                             |
| benthic_model pipeline | + scikit-learn, xgboost, pyyaml, matplotlib, seaborn |
| Development            | + pytest, ruff                                       |

Python 3.11+ required; 3.12 used for benthic_model development.

---

## License

See [LICENSE](LICENSE) for details.

## Citing

We ask that you use the following citation for the BTM scientific library:

> Wright, D.J., Pendleton, M., Boulware, J., Walbridge, S., Gerlt, B., Eslinger, D., Sampson, D., and Huntley, E. 2012. ArcGIS Benthic Terrain Modeler (BTM), v. 3.0, Environmental Systems Research Institute, NOAA Coastal Services Center, Massachusetts Office of Coastal Zone Management. Available online at [http://esriurl.com/5754](http://esriurl.com/5754).
