# Quickstart: Benthic Habitat Classification Pipeline

## 1. Create and Use Python 3.12 Virtual Environment

```bash
cd /Users/tilmann/Documents/GitHub/benthic-terrain-model-kaggle
python3.12 -m venv .venv312
source .venv312/bin/activate
python --version
```

Expected version: Python 3.12.x

## 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install pandas numpy rasterio scikit-learn xgboost scipy jupyter matplotlib seaborn pytest ruff
```

## 3. Prepare Data Layout

Place competition files in a local data directory:

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── METADATA.MD
└── MBES/
    ├── bathymetry.tif
    └── backscatter.tif
```

## 4. Register Notebook Kernel (Local Experimentation)

```bash
pip install ipykernel
python -m ipykernel install --user --name benthic-312 --display-name "Python 3.12 (benthic)"
```

Then select the `Python 3.12 (benthic)` kernel in Jupyter notebooks.

## 5. Run Baseline Training

```bash
PYTHONPATH=src python -m benthic_model.cli train \
  --train-csv data/train.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif \
  --config configs/baseline.yaml \
  --seed 42 \
  --run-type baseline
```

## 6. Run Candidate Training and Evaluation

```bash
PYTHONPATH=src python -m benthic_model.cli train \
  --train-csv data/train.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif \
  --config configs/candidate.yaml \
  --seed 42 \
  --run-type candidate

PYTHONPATH=src python -m benthic_model.cli evaluate \
  --run-id <candidate_run_id> \
  --fold-scheme spatial_blocked
```

## 7. Generate and Validate Submission

```bash
PYTHONPATH=src python -m benthic_model.cli predict \
  --run-id <selected_run_id> \
  --test-csv data/test.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif

PYTHONPATH=src python -m benthic_model.cli make-submission \
  --predictions artifacts/predictions/<selected_run_id>_test_predictions.csv \
  --sample-submission data/sample_submission.csv \
  --output submissions/submission.csv
```

## 8. Validated Run Notes (2026-03-24)

- Baseline run (`baseline-20260324181702`) weighted F1: `0.802627`
- Candidate run (`candidate-20260324181959`) weighted F1: `0.786368`
- Candidate was rejected by degradation policy (`0.016259 > 0.005`)
- Submission was generated from baseline run and validated:
  - Rows: `98`
  - Columns: `ID,class`
  - IDs unique: `true`

See final interpretation in `reports/metrics/final_model_summary.md`.

## 9. Constitution-Aligned Validation Checklist

- Code Quality: `ruff check .`, `ruff format --check .`, `pytest`
- Model Performance: compare baseline vs candidate weighted F1 + per-class F1 report
- UX Consistency: verify metric names/units/class-label formatting are consistent across CLI, reports, and notebook outputs
- Reproducibility: ensure run metadata includes code revision, config, seed, fold scheme, and artifact paths
