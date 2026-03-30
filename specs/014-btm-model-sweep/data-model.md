# Data Model: Systematic BTM Model Sweep

**Branch**: `014-btm-model-sweep` | **Date**: 2026-03-30

---

## Entities

### 1. `PipelineConfig` (extended)

Extends the existing `src/benthic_model/config.py` dataclass.

| Field           | Type                    | Default       | Description                                                                                              |
| --------------- | ----------------------- | ------------- | -------------------------------------------------------------------------------------------------------- |
| `seed`          | `int`                   | `42`          | Global random seed                                                                                       |
| `cv`            | `CrossValidationConfig` | default       | CV strategy params                                                                                       |
| `model_type`    | `str \| None`           | `None`        | One of `rf`, `xgb`, `lgbm`, `catboost`, `rf_lgbm_ensemble`. `None` falls back to `run_type` CLI argument |
| `feature_flags` | `FeatureFlags \| None`  | `None`        | Optional; `None` means all features enabled (backwards-compatible)                                       |
| `data_dir`      | `str`                   | `"data"`      | Path to data directory                                                                                   |
| `artifacts_dir` | `str`                   | `"artifacts"` | Path to artifacts directory                                                                              |

### 2. `FeatureFlags` (new dataclass)

Lives in `src/benthic_model/config.py` alongside `PipelineConfig`.

| Field                      | Type   | Default | Description                                                                                                           |
| -------------------------- | ------ | ------- | --------------------------------------------------------------------------------------------------------------------- |
| `include_focal_stats`      | `bool` | `True`  | Include multi-scale focal statistics (bathy_std_3/5/9, back_std_3/5/9)                                                |
| `include_interactions`     | `bool` | `True`  | Include interaction terms (`bathymetry_x_backscatter`, `acoustic_hardness_proxy`, `relief_index`)                     |
| `include_spatial_z_scores` | `bool` | `True`  | Include `bathymetry_z`, `backscatter_z`, `bathymetry_backscatter_rank` from `spatial_context.py`                      |
| `include_btm_features`     | `bool` | `True`  | Informational flag — BTM columns (`btm_*`) are auto-included when present in CSV; set `False` to document intent only |

**Validation rules**:

- All fields are boolean; invalid types raise `ValueError` on `PipelineConfig.from_yaml()`.
- A config with `include_focal_stats: false` AND `include_interactions: false`
  AND `include_spatial_z_scores: false` produces only the 8 raw MBES core
  columns plus any `btm_*` columns present.

### 3. `ExperimentMetadata` (extended)

Existing dataclass in `src/benthic_model/experiment/metadata.py`. Two new fields
are added with `field(default=...)` to preserve backwards compatibility — records
written by older code that lack these fields can still be parsed via `from_dict()`.

| Key field             | Type    | Default | Notes                                                                                                      |
| --------------------- | ------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `run_id`              | `str`   | —       | Timestamp + config hash — unique identifier                                                                |
| `run_type`            | `str`   | —       | CLI `--run-type` argument (e.g., `baseline`, `candidate`)                                                  |
| `metric_weighted_f1`  | `float` | —       | Mean CV weighted-F1                                                                                        |
| `metric_per_class_f1` | `dict`  | —       | Per-class F1 dict                                                                                          |
| `config_ref`          | `str`   | —       | Path to config YAML                                                                                        |
| `config_hash`         | `str`   | —       | SHA-256 of config file content                                                                             |
| `model_type_used`     | `str`   | `""`    | Resolved model type string (e.g., `rf`, `lgbm`); populated by `train_and_register_run()`                   |
| `feature_flags_used`  | `dict`  | `{}`    | Snapshot of `FeatureFlags` as a dict; populated by `train_and_register_run()`; empty dict for pre-014 runs |

### 4. `KaggleScoreRecord` (new CSV row schema)

File: `artifacts/experiments/kaggle_scores.csv`

| Column             | Type             | Example                                       | Description                         |
| ------------------ | ---------------- | --------------------------------------------- | ----------------------------------- |
| `run_id`           | `str`            | `baseline-20260330142311`                     | Foreign key into run_registry.jsonl |
| `submission_file`  | `str`            | `data/submission_baseline-20260330142311.csv` | Path to the submitted CSV           |
| `kaggle_public_f1` | `float`          | `0.76394`                                     | Public leaderboard score            |
| `submitted_at`     | `str` (ISO date) | `2026-03-30`                                  | Date of Kaggle submission           |
| `message`          | `str`            | `R01 baseline RF: CV=0.8026`                  | Submit message used in Kaggle CLI   |

### 5. Config YAML schema (all 14 runs)

All configs share the same root schema. Only the differing fields are listed.

```yaml
# Shared structure
model_type: <rf|xgb|lgbm|catboost|rf_lgbm_ensemble> # optional; absent = run-type-driven
seed: 42
cv:
  n_splits: <5 or 6> # R14 uses 6; all others use 5
  fold_scheme: spatial_blocked
  random_state: 42
  spatial_bins: 4
feature_flags: # optional; absent = all flags True
  include_focal_stats: <bool>
  include_interactions: <bool>
  include_spatial_z_scores: <bool>
  include_btm_features: <bool>
spatial_coords: <bool> # informational; True only on R13
```

---

## State Transitions

```
Config YAML
   │
   ▼  benthic_model.cli train
ExperimentMetadata in run_registry.jsonl
   │
   ▼  benthic_model.cli evaluate
ExperimentMetadata.metric_weighted_f1 populated
   │
   ├─ CV < 0.75 → no submission generated
   │
   └─ CV ≥ 0.75 → benthic_model.cli predict
                        │
                        ▼
              data/submission_<run_id>.csv
                        │
                        ▼  kaggle competitions submit
              KaggleScoreRecord appended to kaggle_scores.csv
```

---

## Run Matrix — Config File Summary

| Run | File                      | model_type         |   include_focal   | include_interactions | include_spatial_z | train_csv                |
| --- | ------------------------- | ------------------ | :---------------: | :------------------: | :---------------: | ------------------------ |
| R01 | `baseline.yaml`           | _(absent→rf)_      |         ✓         |          ✓           |         ✓         | `train.csv`              |
| R02 | `rf-core-only.yaml`       | `rf`               |         ✗         |          ✗           |         ✗         | `train.csv`              |
| R03 | `rf-no-interactions.yaml` | `rf`               |         ✓         |          ✗           |         ✓         | `train.csv`              |
| R04 | `rf-btm-fine.yaml`        | `rf`               |         ✗         |          ✗           |         ✗         | `train_btm.csv`          |
| R05 | `rf-btm-broad.yaml`       | `rf`               |         ✗         |          ✗           |         ✗         | `train_btm.csv`          |
| R06 | `rf-btm-full.yaml`        | `rf`               |         ✗         |          ✗           |         ✗         | `train_btm.csv`          |
| R07 | `rf-texture.yaml`         | `rf`               |         ✓         |          ✗           |         ✗         | `train.csv`              |
| R08 | `rf-btm-texture.yaml`     | `rf`               |         ✓         |          ✗           |         ✗         | `train_btm.csv`          |
| R09 | `lgbm-candidate.yaml`     | `lgbm`             | TBD Phase2 winner |          →           |         →         | TBD                      |
| R10 | `xgb-candidate.yaml`      | `xgb`              | TBD Phase2 winner |          →           |         →         | TBD                      |
| R11 | `catboost-candidate.yaml` | `catboost`         | TBD Phase2 winner |          →           |         →         | TBD                      |
| R12 | `rf-lgbm-ensemble.yaml`   | `rf_lgbm_ensemble` | TBD Phase2 winner |          →           |         →         | TBD                      |
| R13 | `rf-spatial-diag.yaml`    | `rf`               |         ✗         |          ✗           |         ✓         | `train.csv`              |
| R14 | `rf-6fold-baseline.yaml`  | `rf`               |         ✓         |          ✓           |         ✓         | `train.csv` (n_splits=6) |

_R04–R06 use `train_btm.csv` (BTM columns auto-included). The `include_btm_features` flag is `true` implicitly._  
_R09–R12 YAML `feature_flags` to be copied from Phase 2 winner config after gate passes._
