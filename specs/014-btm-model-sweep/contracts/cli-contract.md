# CLI Contract: benthic_model Pipeline Extension

**Branch**: `014-btm-model-sweep` | **Date**: 2026-03-30  
**Scope**: Contracts for new config schema and new model-type values introduced
by this feature. Existing CLI arguments are unchanged.

---

## 1. Config YAML Contract

### 1.1 New optional keys

The following keys MAY be added to any config YAML. Absence of a key is
equivalent to its default value. Existing configs without these keys continue
to work identically.

```yaml
# model_type — selects the model family
# Type: string | absent
# Allowed values: "rf", "xgb", "lgbm", "catboost", "rf_lgbm_ensemble"
# Default when absent: derived from --run-type CLI argument
#   --run-type baseline  → "rf"
#   --run-type candidate → "xgb"
model_type: rf

# feature_flags — controls which engineered feature groups are added
# Type: mapping | absent
# Default when absent: all flags are True (full backwards-compatible behaviour)
feature_flags:
  include_focal_stats: true # bathy_std_3/5/9, back_std_3/5/9, tpi_*
  include_interactions: true # bathymetry_x_backscatter, acoustic_hardness_proxy, relief_index
  include_spatial_z_scores: true # bathymetry_z, backscatter_z, bathymetry_backscatter_rank
  include_btm_features: true # informational; btm_* columns included when present in CSV
  spatial_coords: false # informational; True documents intentional spatial-coord inclusion
```

### 1.2 Schema validation rules

| Rule                                  | Behaviour on violation                       |
| ------------------------------------- | -------------------------------------------- |
| `model_type` value not in allowed set | `ValueError` raised at train time            |
| `feature_flags` contains unknown key  | `ValueError` raised at train time            |
| `feature_flags` value not boolean     | `ValueError` raised at train time            |
| All flags `False`                     | Allowed; produces raw MBES core columns only |

### 1.3 Unchanged keys

`seed`, `cv.n_splits`, `cv.fold_scheme`, `cv.random_state`, `cv.spatial_bins`
— unchanged semantics and defaults.

---

## 2. `benthic_model.cli train` contract extension

**No new CLI arguments.** The `--config` argument now supports YAML files
containing `model_type` and `feature_flags`. Existing positional/keyword
arguments are unchanged:

```
python -m benthic_model.cli train
  --train-csv       <path>          required
  --bathymetry-tif  <path>          required
  --backscatter-tif <path>          required
  --config          <path>          required
  --run-type        baseline|candidate  required
  --seed            <int>           optional, default 42
```

**Output contract** (unchanged):

- Prints `Training completed. run_id=<ID>` on success.
- Appends one JSONL line to `artifacts/experiments/run_registry.jsonl`.
- Exit code 0 on success, 1 on failure.

---

## 3. Run registry entry contract extension

New fields added to `ExperimentMetadata` (written to `run_registry.jsonl`):

| Field                | Type   | Notes                                                          |
| -------------------- | ------ | -------------------------------------------------------------- |
| `model_type_used`    | `str`  | Resolved model type (e.g., `"lgbm"` even if config key absent) |
| `feature_flags_used` | `dict` | Resolved flags after applying defaults                         |

Existing fields (`run_id`, `run_type`, `metric_weighted_f1`, `metric_per_class_f1`,
`config_ref`, `config_hash`, `seed`, `code_revision`, `timestamp`) — unchanged.

---

## 4. `kaggle_scores.csv` contract

File location: `artifacts/experiments/kaggle_scores.csv`  
Created by: first Kaggle CLI submit + score-fetch operation (or manually
bootstrapped as an empty file with headers).

```csv
run_id,submission_file,kaggle_public_f1,submitted_at,message
```

| Column             | Type   | Constraints                                            |
| ------------------ | ------ | ------------------------------------------------------ |
| `run_id`           | string | Must match a `run_id` in `run_registry.jsonl`          |
| `submission_file`  | string | Relative path from repo root                           |
| `kaggle_public_f1` | float  | 0.0–1.0; `-1.0` if scoring failed or pending           |
| `submitted_at`     | string | ISO date `YYYY-MM-DD`                                  |
| `message`          | string | The `-m` message used in the Kaggle CLI submit command |

---

## 5. New model builder function signatures

Added to `src/benthic_model/models/candidate.py`:

```python
def build_lgbm_model(seed: int = 42) -> CandidateLGBMModel: ...
def build_catboost_model(seed: int = 42) -> CandidateCatBoostModel: ...
def build_rf_lgbm_ensemble_model(seed: int = 42) -> CandidateEnsembleModel: ...
```

Each class implements the same `.fit(X, y) / .predict(X)` interface as
`BaselineRandomForestModel` and `CandidateXGBoostModel`. No other public API
changes.

Updated in `src/benthic_model/models/train.py`:

```python
def _build_model(run_type: str, seed: int, model_type: str | None = None):
    # model_type from config takes precedence over run_type
    ...
```
