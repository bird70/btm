# Research: Systematic BTM Model Sweep

**Branch**: `014-btm-model-sweep` | **Date**: 2026-03-30  
**Phase**: 0 — resolves all NEEDS CLARIFICATION items from Technical Context

---

## R1 — Config-driven feature selection: design pattern

**Question**: How to allow 14 distinct feature sets without 14 bespoke training
scripts, while keeping the change minimal and backwards-compatible?

**Decision**: Add an optional `feature_flags` mapping to `PipelineConfig` +
config YAML. `engineer_features()` receives the flags and conditionally skips
interaction-term and spatial-z-score generation. The existing `baseline.yaml`
omits `feature_flags` entirely — absence means "all features on" — so current
behaviour is unchanged.

```yaml
# Example: rf-core-only.yaml — turns everything off except raw MBES columns
model_type: rf
feature_flags:
  include_focal_stats: false
  include_interactions: false
  include_spatial_z_scores: false
  include_btm_features: false # ignored if btm_* columns not in CSV
seed: 42
cv:
  n_splits: 5
  fold_scheme: spatial_blocked
  random_state: 42
  spatial_bins: 4
```

**Alternatives considered**:

- Separate feature engineering modules per run type. Rejected — too much
  duplication; breaks existing test coverage.
- CLI `--feature-flags` argument. Rejected — flags belong in the reproducible
  config artifact, not in the command invocation.
- Column whitelist in YAML. Rejected — brittle against column-name changes;
  flag semantics are more stable than exact column names.

**Rationale**: Backwards-compatible boolean flags in YAML are the simplest
possible extension. `select_model_feature_columns()` already picks all numeric
columns — flags control what gets _added_ to the DataFrame before that
selection step.

---

## R2 — Model type dispatch: how to select RF / LightGBM / CatBoost / ensemble

**Question**: The current `--run-type` CLI flag selects only `baseline` or
`candidate`. How do we add LightGBM, CatBoost, and ensemble without breaking
the existing dispatch?

**Decision**: Add a `model_type` key to `PipelineConfig`. When absent, the
existing `run_type` CLI argument drives dispatch exactly as today
(`baseline` → RF, `candidate` → XGBoost). When present, `model_type`
overrides the type selection inside `_build_model()`.

| `model_type` value | Builder called                             |
| ------------------ | ------------------------------------------ |
| `rf`               | `build_baseline_model(seed)`               |
| `xgb`              | `build_candidate_model(seed)` (existing)   |
| `lgbm`             | `build_lgbm_model(seed)` (new)             |
| `catboost`         | `build_catboost_model(seed)` (new)         |
| `rf_lgbm_ensemble` | `build_rf_lgbm_ensemble_model(seed)` (new) |

**Alternatives considered**:

- New `--run-type` values (`lgbm`, `catboost`). Rejected — `run_type` is
  stored in the registry as a category label; conflating it with model family
  pollutes lineage queries.
- Subclassed pipeline objects per model. Rejected — unnecessary abstraction for
  five variants.

**Rationale**: Config YAML already owns all other pipeline parameters; adding
`model_type` keeps the config file as the single reproducible description of a
run.

---

## R3 — LightGBM, CatBoost, and ensemble hyperparameters for Phase 3

**Question**: What hyperparameters should the Phase 3 candidate configs use?

**Decision**: Use conservative defaults derived from the v9 experiment run
reports, which are known to generalise well on this dataset:

| Model                | Key params                                                                         | Source                                    |
| -------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| LightGBM             | `n_estimators=600`, `learning_rate=0.03`, `num_leaves=63`, `class_weight=balanced` | `docs/run-012-v9-catboost-lgb-texture.md` |
| CatBoost             | `iterations=800`, `depth=7`, `learning_rate=0.05`, `auto_class_weights=Balanced`   | `docs/run-012-v9-catboost-lgb-texture.md` |
| RF+LightGBM ensemble | soft-vote, equal weights                                                           | v9 approach                               |

**Why not tune further?** The primary hypothesis is that feature selection
matters more than hyperparameter tuning. Phase 3 demonstrates model-family
differences on a fixed, well-chosen feature set. Tuning should follow Phase 3
only if a clear gap between CV and Kaggle is improved.

---

## R4 — Kaggle score fetch: how to automate recording after submission

**Question**: After `kaggle competitions submit`, how do we reliably fetch the
public score?

**Decision**: After each submit, wait 90 seconds (Kaggle scoring latency), then
call:

```powershell
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submissions `
  -c geohab-mlwg-competition-2026 --csv
```

The first row of the CSV output is the most recent submission and contains the
`publicScore` field. Parse it and append to `artifacts/experiments/kaggle_scores.csv`.

**Authentication**: Kaggle CLI v2 supports token-based auth via the
`KAGGLE_API_TOKEN` environment variable (a single token string, no username
required). The token was set in the terminal session as
`$env:KAGGLE_API_TOKEN`. For persistent auth, create
`%USERPROFILE%\.kaggle\kaggle.json` with `{"token": "<value>"}`. The
Kaggle CLI v2 docs confirm this schema.

**Alternatives considered**:

- Manual score recording only. Rejected — per FR-007, scores must be recorded
  after each submission; manual recording introduces error and delays.
- Kaggle API Python client directly. Rejected — the CLI is already installed
  and tested; adding a Python API dependency is unnecessary.

---

## R5 — BTM feature column names and availability

**Question**: Which BTM columns are automatically included by
`select_model_feature_columns()`, and do they need explicit declaration?

**Decision**: `btm-export-features` writes columns with a `btm_` prefix:
`btm_bpi_fine`, `btm_bpi_broad`, `btm_vrm`, `btm_slope`, `btm_depth_mean`,
`btm_depth_std`, `btm_surface_ratio`. All are numeric. Because
`select_model_feature_columns()` includes all numeric columns not in the
exclusion list, they are automatically available when present in the CSV.

To suppress them in non-BTM runs (R01–R03, R07, R13, R14), simply point
`--train-csv` at `data/train.csv` (no BTM columns); the flag
`include_btm_features` is then informational only. For BTM runs (R04–R08),
point `--train-csv` at `data/train_btm.csv`.

---

## R6 — Kaggle submission budget allocation across runs

**Question**: Which 12 of 14 runs should be submitted, and in what order,
given the 4/day cap?

**Decision**:

| Day   | Runs to submit                 | Rationale                                                                 |
| ----- | ------------------------------ | ------------------------------------------------------------------------- |
| Day 1 | R01, R02, R03                  | Phase 1 ground truth; submit all 3 regardless of CV score                 |
| Day 2 | R07, R06, R05 (top CV Phase 2) | Texture-only first (smallest delta from baseline); broad+full BTM next    |
| Day 3 | R04, R08, R13 + 1 spare        | Fine BTM, max feature, diagnostic spatial coord                           |
| Day 4 | R11, R09, R10, R12             | Phase 3: CatBoost first (historically strongest), then LGB, XGB, ensemble |

R14 (6-fold baseline) is NOT submitted — its sole purpose is comparing its CV
score against R01's CV score to test whether fold count changes the gap.

**Rationale**: Submit the most informative runs first. Day 1 establishes the
true baseline Kaggle score. Day 2 tests the strongest Phase 2 hypothesis
(texture features) and the BTM full set. Day 3 completes Phase 2.
Day 4 is model-family sweep.
