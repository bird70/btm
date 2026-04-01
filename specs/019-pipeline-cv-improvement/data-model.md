# Data Model: Pipeline CV Improvement Investigation

**Feature**: 019-pipeline-cv-improvement  
**Date**: 2026-04-01

## Entities

### ExperimentRun

A single pipeline execution producing a trained model and evaluation metrics.

| Field          | Type            | Description                                                |
| -------------- | --------------- | ---------------------------------------------------------- |
| run_id         | string          | Unique ID: `{run_type}-{YYYYMMDDHHMMSS}`                   |
| run_type       | enum            | `baseline` or `candidate`                                  |
| config_ref     | string          | Path to YAML configuration file                            |
| config_hash    | string          | SHA-256 prefix (12 chars) of config file content           |
| seed           | int             | Random seed used for CV fold assignment and model training |
| model_type     | enum            | `rf`, `lgbm`, `rf_lgbm_ensemble`                           |
| fold_scheme    | string          | `spatial_blocked` (only valid scheme)                      |
| data_split_ref | string          | Path to training CSV                                       |
| weighted_f1    | float           | Cross-validated weighted F1 score (primary metric)         |
| per_class_f1   | dict[str→float] | Per-class F1 scores: ALG, FMAT, NVB, SGAM, SGZ             |
| code_revision  | string          | Git short SHA at time of run                               |

**Relationships**: Each ExperimentRun produces one Model artifact and one Predictions artifact.

**Validation rules**:

- `fold_scheme` MUST be `spatial_blocked` (FR-007)
- `model_type` MUST be `rf` or `lgbm` (scope constraint)
- `weighted_f1` MUST be in range [0.0, 1.0]

### Configuration

A YAML file defining model type, hyperparameters, feature flags, and CV settings.

| Field         | Type   | Description                                                               |
| ------------- | ------ | ------------------------------------------------------------------------- |
| model_type    | enum   | `rf` or `lgbm`                                                            |
| seed          | int    | Default random seed                                                       |
| cv            | object | CrossValidationConfig (n_splits, fold_scheme, random_state, spatial_bins) |
| feature_flags | object | FeatureFlags (include_focal_stats, include_interactions, etc.)            |
| model_params  | dict   | Model-specific hyperparameter overrides                                   |

**Validation rules**:

- `cv.fold_scheme` MUST be `spatial_blocked`
- `feature_flags.include_eco_features` and `feature_flags.include_spatial_z_scores` are optional toggles
- `model_params` keys must be valid for the target model type

### Submission

Competition-format CSV mapping test sample IDs to predicted habitat classes.

| Field | Type | Description                                |
| ----- | ---- | ------------------------------------------ |
| ID    | int  | Test sample identifier (1-98)              |
| class | enum | Predicted class: ALG, FMAT, NVB, SGAM, SGZ |

**Validation rules**:

- Exactly 98 rows
- All IDs from sample_submission.csv present
- All class values in allowed set {ALG, FMAT, NVB, SGAM, SGZ}

### SeedEnsemble

A collection of predictions from multiple seed runs combined via majority voting.

| Field                | Type                 | Description                             |
| -------------------- | -------------------- | --------------------------------------- |
| seed_runs            | list[ExperimentRun]  | Source runs (≥5)                        |
| vote_matrix          | DataFrame            | Per-sample votes from each seed         |
| ensemble_predictions | list[Submission row] | Final majority-vote predictions         |
| confidence           | dict[int→int]        | Per-sample vote count for winning class |

**State transitions**:

1. Individual seed runs complete → predictions generated
2. All seed predictions loaded → vote matrix constructed
3. Per-sample mode computed → ensemble predictions produced
4. Ensemble compared to R04 baseline → delta report generated

## Relationships

```
Configuration (YAML) ─1:N──▶ ExperimentRun
ExperimentRun ─1:1──▶ Model (joblib)
ExperimentRun ─1:1──▶ Predictions (CSV)
ExperimentRun ─N:1──▶ SeedEnsemble
SeedEnsemble ─1:1──▶ Submission (CSV)
```
