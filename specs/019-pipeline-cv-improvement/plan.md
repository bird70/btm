# Implementation Plan: Pipeline CV Improvement Investigation

**Branch**: `019-pipeline-cv-improvement` | **Date**: 2026-04-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/019-pipeline-cv-improvement/spec.md`

## Summary

Systematically investigate whether the R04 pipeline baseline (RF + BTM-10, spatial blocked 5-fold CV=0.8024, Kaggle=0.7952) can be meaningfully improved through hyperparameter tuning, feature flag variations, LightGBM as an alternative model type, and seed-ensemble techniques. All experiments are evaluated against a statistically-quantified noise floor (seed variance). The investigation produces a comprehensive experiment report and, if warranted, a validated Kaggle submission.

## Technical Context

**Language/Version**: Python 3.11+ (3.13.5 in workspace)  
**Primary Dependencies**: scikit-learn (RF), lightgbm (LightGBM), pandas, numpy, rasterio, PyYAML, joblib  
**Storage**: Flat files — CSV (data/submissions), JSON (metrics/provenance), joblib (models), YAML (configs)  
**Testing**: pytest (236 tests passing; markers `arcgis`/`qgis` for platform-gated tests)  
**Target Platform**: Windows (development), cross-platform Python  
**Project Type**: CLI pipeline (`python -m benthic_model.cli train/predict/make-submission`)  
**Performance Goals**: Each pipeline train run completes in <60s on the small competition dataset (590 train, 98 test samples)  
**Constraints**: Maximum 2 Kaggle submissions; only RF and LightGBM model types in scope; large-window BTM features (≥15 cells) excluded  
**Scale/Scope**: 590 training samples, 98 test samples, 5 habitat classes (ALG, FMAT, NVB, SGAM, SGZ), 10-14 features per sample

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

- [x] **I. TDD** — Existing test suite (236 tests) covers pipeline components. New experiment scripts are validated by running with --dry-run and verifying CV outputs match expectations. No new library code is introduced (only configs and scripts).
- [x] **II. Code Quality** — Linting via ruff; CI blocks on lint errors. Experiment scripts follow established patterns (experiment_v10-v12.py).
- [x] **III. Performance** — Not applicable: this feature does not introduce raster processing. Pipeline runs use pre-extracted CSV features. NumPy/scikit-learn vectorized operations used throughout.
- [x] **IV. Scientific Accuracy** — RF and LightGBM are standard ML classifiers. Spatial blocked CV methodology follows geospatial ML best practices (spatial autocorrelation addressed). Seed variance quantification follows standard statistical methodology.
- [x] **V. Platform Portability** — All experiment code uses the existing `benthic_model.cli` interface. No ArcGIS/QGIS dependencies. Core logic in `src/benthic_model/`. CLI entry point via `python -m benthic_model.cli`.
- [x] **VI. Simplicity** — No new abstractions introduced. Experiments use existing pipeline CLI with different YAML configs. Majority-vote ensemble is a standalone script, not a new library module.

## Project Structure

### Documentation (this feature)

```text
specs/019-pipeline-cv-improvement/
├── plan.md              # This file
├── research.md          # Phase 0: Prior experiment findings + decisions
├── data-model.md        # Phase 1: Experiment run entity model
├── quickstart.md        # Phase 1: How to reproduce experiments
├── contracts/           # Phase 1: CLI interface contracts
└── tasks.md             # Phase 2: Implementation tasks (via /speckit.tasks)
```

### Source Code (repository root)

```text
configs/                          # YAML experiment configurations
├── rf-btm-fine.yaml              # R04 baseline (existing)
├── lgbm-candidate.yaml           # LightGBM baseline (existing)
├── rf-btm-s*.yaml                # RF hyperparameter variants (existing from prior experiments)
└── lgbm-btm-*.yaml               # NEW: LightGBM hyperparameter variants

scripts/
├── experiment_v10.py             # Prior experiment (existing)
├── experiment_v11.py             # Prior experiment (existing)
├── experiment_v12.py             # Prior experiment (existing)
├── experiment_v13.py             # Multi-seed CV stability (existing, not yet run)
└── experiment_v14_seed_ensemble.py  # NEW: Seed ensemble majority vote

artifacts/
├── runs/                         # Pipeline run outputs (model, metrics, provenance)
└── predictions/                  # Test prediction CSVs per run

data/
├── train_btm.csv                 # Primary training data (existing)
├── test.csv                      # Test data (existing)
└── submission_v13_seed_ensemble.csv  # Seed ensemble output (existing)

docs/
└── run-019-pipeline-cv-improvement.md  # Experiment run report

reports/metrics/                  # Per-run comparison reports
```

**Structure Decision**: Uses the existing single-project layout. No new source modules — only configuration files, experiment scripts, and documentation artifacts.

## Complexity Tracking

> No constitution violations. All gates pass.
