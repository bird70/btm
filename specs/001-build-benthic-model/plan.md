# Implementation Plan: Benthic Habitat Classification Pipeline

**Branch**: `001-build-benthic-model` | **Date**: 2026-03-24 | **Spec**: `/specs/001-build-benthic-model/spec.md`
**Input**: Feature specification from `/specs/001-build-benthic-model/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a reproducible Python 3.12 ML pipeline that extracts predictors from MBES GeoTIFF rasters at
sample coordinates, engineers local geospatial context features, trains habitat-classification models,
evaluates candidates with weighted F1 using spatial blocked cross-validation, and generates valid Kaggle
submission files (`ID,class`). Development will use a notebook-first experimentation workflow backed by
scriptable pipeline components for repeatability.

## Technical Context

**Language/Version**: Python 3.12 (local virtual environment at `.venv312`)  
**Primary Dependencies**: pandas, numpy, rasterio, scikit-learn, xgboost, scipy, jupyter, matplotlib, seaborn  
**Storage**: Local filesystem artifacts (`data/`, `artifacts/`, `reports/`, `submissions/`)  
**Testing**: pytest for unit/integration; schema and contract checks for submission outputs  
**Target Platform**: Local macOS development with notebook experimentation; script execution in CI-compatible environment
**Project Type**: Single Python ML project (notebook-assisted, script-first reproducibility)  
**Performance Goals**: Improve candidate weighted F1 over baseline; keep full training/eval cycle reproducible and trackable  
**Constraints**: Use only competition-provided data; spatial blocked k-fold as authoritative model selection; deterministic reruns within ±0.01 weighted F1  
**Scale/Scope**: One competition dataset (train/test tables + 2 GeoTIFF layers), multiclass habitat prediction, iterative experiment comparison

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Code Quality Gate**: PASS (planned). Enforce `ruff check`, `ruff format --check`, `pytest` for
  changed modules, plus pinned-dependency validation for critical packages; merge is blocked on any failing quality job.
- **Model Performance Gate**: PASS (planned). Baseline and candidate compared using weighted F1 and
  per-class F1 under spatial blocked k-fold, with machine-readable metric artifacts and a 0.005 degradation threshold policy.
- **UX Consistency Gate**: PASS (planned). Standardized naming for class labels and metrics across notebooks,
  CLI output, reports, and submission files; user-facing message checklist captured in quickstart.
- **Reproducibility Gate**: PASS (planned). Every promoted run records code revision, config hash, data split
  reference, seed, command, and output artifact paths.
- **Evidence Plan**: PASS (planned). Persist evidence in `reports/metrics/`, `artifacts/experiments/`, and
  `submissions/`; include machine-readable metrics in `reports/metrics/*.json` and dependency pinning output in
  `reports/quality/dependency_pinning_report.md`; reviewer verifies gates at PR time.

## Project Structure

### Documentation (this feature)

```text
specs/001-build-benthic-model/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── pipeline-cli.md
└── tasks.md
```

### Source Code (repository root)
```text
notebooks/
├── 01_data_validation.ipynb
├── 02_feature_exploration.ipynb
└── 03_model_experiments.ipynb

src/
├── benthic_model/
│   ├── data/
│   │   ├── ingest.py
│   │   ├── raster_extract.py
│   │   └── validation.py
│   ├── features/
│   │   ├── engineering.py
│   │   └── spatial_context.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── candidate.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── cv.py
│   │   └── compare.py
│   ├── inference/
│   │   └── predict.py
│   ├── submission/
│   │   └── writer.py
│   └── cli.py

tests/
├── unit/
├── integration/
└── contract/

artifacts/
reports/
submissions/
```

**Structure Decision**: Single-project Python ML structure with notebook-assisted experimentation. Core
logic is implemented in `src/benthic_model` for reproducibility, while notebooks in `notebooks/` remain
consumer interfaces for exploratory analysis.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
