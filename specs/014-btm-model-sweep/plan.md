# Implementation Plan: Systematic BTM Model Sweep

**Branch**: `014-btm-model-sweep` | **Date**: 2026-03-30 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/014-btm-model-sweep/spec.md`

## Summary

Implement a 14-run systematic sweep of BTM terrain feature sets and ML model
families to find the optimal configuration for the
`geohab-mlwg-competition-2026` Kaggle competition. The primary driver is
**closing the CV–Kaggle gap** that has caused sophisticated multi-model
pipelines (v7: 0.691, v9: 0.692) to underperform the simpler 0.76394 baseline.

The implementation has two distinct concerns:

1. **Pipeline extension** — add config-driven feature flags and multi-family
   model builders to `benthic_model` so all 14 run variants can be expressed
   as YAML config files without bespoke scripts.
2. **Experiment execution** — run all 14 configs through the existing
   `benthic_model.cli` train/evaluate/predict workflow, submit to Kaggle via
   `C:\DEVlocal\btm\.venv\Scripts\kaggle.exe`, and record all scores in
   `artifacts/experiments/kaggle_scores.csv`.

## Technical Context

**Language/Version**: Python 3.13.5  
**Primary Dependencies**: scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.2.0,
CatBoost 1.2.10, rasterio, pandas, numpy, btm (local), benthic_model (local),
PyYAML  
**Storage**: JSONL run registry (`artifacts/experiments/run_registry.jsonl`),
CSV Kaggle scores (`artifacts/experiments/kaggle_scores.csv`), joblib model
artifacts, YAML configs  
**Testing**: pytest — existing structure: `tests/unit/`, `tests/integration/`,
`tests/contract/`  
**Target Platform**: Windows local (PowerShell), no server deployment  
**Project Type**: CLI pipeline / ML experiment runner  
**Performance Goals**: Each RF training run ≤ 10 min on ~2 000 training points;
LightGBM/XGBoost/CatBoost runs ≤ 5 min  
**Constraints**: Kaggle daily submission budget: 4 submissions/day (max 5,
reserve 1). All experiments must be deterministic given `seed: 42`.  
**Scale/Scope**: 14 runs × (train + evaluate + predict + submit) operations;
~12 Kaggle submissions over 4 days

## Constitution Check

_GATE: Evaluated against `specs/014-btm-model-sweep` constitution._

- [x] **I. TDD** — New feature-flag logic in `engineer_features()` and new
      model builders (`lgbm`, `catboost`, `rf_lgbm_ensemble`) require failing unit
      tests written before implementation. Tests added to `tests/unit/test_feature_engineering.py`
      and new `tests/unit/test_model_builders.py`. Existing tests must stay green.
- [x] **II. Code Quality** — ruff/flake8 already configured; CI blocks on lint.
      New code follows existing patterns (dataclasses, type hints, docstrings on
      public functions).
- [x] **III. Performance** — No new raster block processing required (BTM
      feature export reuses existing `btm-export-features`). Training on ~2 000
      points is not memory-bound. N/A for raster compression.
- [x] **IV. Scientific Accuracy** — No new algorithms. Feature sets reference
      Ierodiaconou et al. (2018) and the BTM library (BPI: Lundblad et al. 2006,
      VRM: Sappington et al. 2007, Slope: Horn 1981). No deviations from references.
- [x] **V. Platform Portability** — Pure Python ML, no ArcGIS dependency.
      Config-driven feature flags live in `benthic_model` only. Kaggle CLI is an
      external tool accessed via absolute path; the pipeline itself has no Kaggle
      dependency.
- [x] **VI. Simplicity** — Config-driven feature flags replace 14 one-off
      script variants. The `model_type` YAML key extends the existing `PipelineConfig`
      dataclass pattern. No new abstraction layers introduced.

## Project Structure

### Documentation (this feature)

```text
specs/014-btm-model-sweep/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/
│   └── cli-contract.md  ← Phase 1 output
└── tasks.md             ← /speckit.tasks output (not created here)
```

### Source Code (affected paths)

```text
configs/                      ← 13 new YAML files (R02–R14)
├── baseline.yaml             ← existing, unchanged
├── rf-core-only.yaml         ← new R02
├── rf-no-interactions.yaml   ← new R03
├── rf-btm-fine.yaml          ← new R04
├── rf-btm-broad.yaml         ← new R05
├── rf-btm-full.yaml          ← new R06
├── rf-texture.yaml           ← new R07
├── rf-btm-texture.yaml       ← new R08
├── lgbm-candidate.yaml       ← new R09
├── xgb-candidate.yaml        ← new R10
├── catboost-candidate.yaml   ← new R11
├── rf-lgbm-ensemble.yaml     ← new R12
├── rf-spatial-diag.yaml      ← new R13
└── rf-6fold-baseline.yaml    ← new R14

src/benthic_model/
├── config.py                 ← extend PipelineConfig with feature_flags + model_type
├── features/
│   └── engineering.py        ← respect feature_flags from config
├── models/
│   ├── baseline.py           ← unchanged
│   ├── candidate.py          ← add build_lgbm_model(), build_catboost_model(),
│   │                            build_rf_lgbm_ensemble_model()
│   └── train.py              ← extend _build_model() dispatcher for new types

artifacts/experiments/
├── run_registry.jsonl        ← populated at runtime
└── kaggle_scores.csv         ← created at runtime (scaffold committed empty)

tests/
├── unit/
│   ├── test_feature_engineering.py   ← extend with feature_flag tests
│   └── test_model_builders.py        ← new: tests for lgbm/catboost/ensemble builders
└── contract/
    └── test_cli_train_evaluate_contract.py  ← extend with model_type coverage
```

**Structure Decision**: Single-project layout. Only `src/benthic_model/`,
`configs/`, and `artifacts/` are affected. No new packages or top-level dirs.

## Complexity Tracking

No constitution violations. Config-driven `feature_flags` and `model_type` are
the minimum extension needed to avoid writing 14 duplicate training scripts.

---

## Constitution Check — Post-Design Re-evaluation

_Re-checked after Phase 1 design (data-model, contracts, quickstart)._

All six gates remain satisfied. Key update from design phase:

- **I. TDD**: Specific failing tests identified — see Implementation Design §Tests.
- **II. Code Quality**: `FeatureFlags` and `model_type` integrate naturally into
  the existing `PipelineConfig` dataclass; no new abstraction layers.
- **VI. Simplicity**: The `trains_csv` variation (plain vs BTM-augmented) is driven
  by the `--train-csv` CLI arg, not a config flag, keeping the YAML minimal.

No violations. No complexity exceptions required.

---

## Implementation Design

### A. `FeatureFlags` dataclass and `PipelineConfig` extension

**File**: `src/benthic_model/config.py`

Add `FeatureFlags` as a new `@dataclass(slots=True)` with four boolean fields,
all defaulting to `True`. Add `feature_flags: FeatureFlags | None = None` and
`model_type: str | None = None` to `PipelineConfig`.

Extend `PipelineConfig.from_dict()` to parse `feature_flags` from the YAML
mapping, constructing a `FeatureFlags` instance. When the key is absent, leave
as `None` (treated as all-True).

Validation: add a `__post_init__` check on `PipelineConfig` that raises
`ValueError` when `model_type` is set to a value outside the allowed set.

### B. Feature engineering — honour flags

**File**: `src/benthic_model/features/engineering.py`

`engineer_features()` gains an optional `flags: FeatureFlags | None = None`
parameter. When `None`, behaviour is unchanged. When provided:

- `flags.include_interactions = False` → skip adding `bathymetry_x_backscatter`,
  `acoustic_hardness_proxy`, `relief_index`
- `flags.include_spatial_z_scores = False` → skip calling
  `add_spatial_context_features()`
- `flags.include_focal_stats = False` → skip adding any `*_std_*` or `tpi_*`
  columns (columns already present in the CSV from `btm-export-features` are
  unaffected; only dynamically computed focal stats are skipped)

`select_model_feature_columns()` is unchanged — it still picks all numeric
non-excluded columns. The flags control what gets _added_ before selection.

The `flags` are passed through from `train_and_register_run()` which reads
`config_data.get("feature_flags")` and constructs a `FeatureFlags` instance.

### C. New model builders

**File**: `src/benthic_model/models/candidate.py`

Add three new classes:

```python
class CandidateLGBMModel:
    # LGBMClassifier: n_estimators=600, learning_rate=0.03,
    # num_leaves=63, class_weight="balanced", random_state=seed
    def fit(self, X, y): ...
    def predict(self, X): ...

class CandidateCatBoostModel:
    # CatBoostClassifier: iterations=800, depth=7, learning_rate=0.05,
    # auto_class_weights="Balanced", verbose=0, random_seed=seed
    def fit(self, X, y): ...
    def predict(self, X): ...

class CandidateEnsembleModel:
    # Holds one RF + one LightGBM; predict() soft-votes (equal weights)
    def fit(self, X, y): ...
    def predict(self, X): ...
```

Add three corresponding `build_*` factory functions.

**File**: `src/benthic_model/models/train.py`

Extend `_build_model()`:

```python
def _build_model(run_type: str, seed: int, model_type: str | None = None):
    resolved = model_type or ("rf" if run_type == "baseline" else "xgb")
    dispatch = {
        "rf": build_baseline_model,
        "xgb": build_candidate_model,
        "lgbm": build_lgbm_model,
        "catboost": build_catboost_model,
        "rf_lgbm_ensemble": build_rf_lgbm_ensemble_model,
    }
    if resolved not in dispatch:
        raise ValueError(f"Unknown model_type: {resolved}")
    return dispatch[resolved](seed=seed)
```

`train_and_register_run()` reads `model_type` from the config YAML and passes
it to `_build_model()`.

### D. 13 new config YAML files

All in `configs/`. Full flag combinations per data-model.md run matrix.
Phase 3 configs (R09–R12) are created now with placeholder `feature_flags`
(all True); they MUST be updated by copy-pasting from the Phase 2 winner config
before Phase 3 runs execute.

### E. `kaggle_scores.csv` scaffold

Create `artifacts/experiments/kaggle_scores.csv` with headers only (committed
to repo so the file always exists):

```csv
run_id,submission_file,kaggle_public_f1,submitted_at,message
```

### F. Registry extension — `model_type_used` and `feature_flags_used`

Add `model_type_used: str` and `feature_flags_used: dict` to
`ExperimentMetadata`. Populate in `train_and_register_run()` from the resolved
values, so every registry entry is fully self-describing.

---

## Tests

| Test file                                            | New / extended | What it covers                                                                                                                            |
| ---------------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/test_feature_engineering.py`             | **Extended**   | `engineer_features()` with each flag False individually and all False; assert columns absent/present                                      |
| `tests/unit/test_model_builders.py`                  | **New**        | `build_lgbm_model`, `build_catboost_model`, `build_rf_lgbm_ensemble_model` — fit on tiny synthetic data, predict returns valid class list |
| `tests/unit/test_config_and_metadata.py`             | **Extended**   | `PipelineConfig.from_yaml()` with `model_type` and `feature_flags`; invalid `model_type` raises `ValueError`                              |
| `tests/contract/test_cli_train_evaluate_contract.py` | **Extended**   | Train with `rf-core-only.yaml` via CLI; assert run_id in registry and `feature_flags_used` present                                        |

All existing tests must pass without modification. Run before and after each
implementation step:

```powershell
.\.venv\Scripts\pytest.exe tests/ -x -q
```

---

## Execution Schedule

| Day             | Implementation tasks                               | Validation                               | Actual Outcome (2026-03-30)                                      |
| --------------- | -------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| Day 0 (pre-run) | A–F above (code + configs + tests)                 | All pytest tests green                   | ✓ 154 passed, 0 failed                                           |
| Day 1           | R01, R02, R03 train+evaluate+predict; submit all 3 | Phase 1 gate: CV ≥ 0.75                  | ✓ R01=0.731, R02/R03=0.763; gap R02=0.039 ≤ 0.04 PASS           |
| Day 1 pm        | `btm-export-features` for train+test BTM CSVs      | Verify `data/train_btm.csv` columns      | ✓ 10 BTM cols, 6256 rows; 2 all-NaN cols filled with 0           |
| Day 2           | R04–R08 train+evaluate+predict; submit top 3       | Phase 2 gate after Day 3                 | ✓ R04/R06=0.79518 **NEW BEST**; gap=0.007 PASS; daily limit hit  |
| Day 3           | Remaining Phase 2 submits + R13; analyse gaps      | Update Phase 3 configs with winner flags | ✓ configs updated; R13 confirmed no spatial leakage              |
| Day 4           | R09–R12 train+evaluate+predict; submit all 4       | Best Kaggle score recorded               | ✓ R09-R12 trained (CV: LGBM=0.795, XGB=0.775, CB=0.797, ENS=0.797) |
| Day 5           | Write `docs/run-014-btm-model-sweep.md`            | README updated if new best               | ✓ doc written; README updated with Kaggle=0.79518                |

---

## Risks and Mitigations

| Risk                                                                 | Likelihood | Mitigation                                                                                              |
| -------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| Kaggle auth token expires mid-sweep                                  | Low        | Keep `$env:KAGGLE_API_TOKEN` set; re-create token if needed                                             |
| Phase 2 configs all have similar CV scores; gap analysis ambiguous   | Medium     | Submit the top 2 and pick winner by Kaggle score directly                                               |
| LightGBM/CatBoost CV score much higher than Kaggle (spatial leakage) | Low        | Confirmed features have no spatial coords in Phase 3; check feature importance before submit            |
| `btm-export-features` produces NaN columns                           | Low        | `engineer_features()` fills NaN with median; check `train_btm.csv` column stats before Phase 2 training |
| `ExperimentMetadata` JSONL breaking change                           | Low        | Add fields with `field(default=...)` so existing `from_dict()` calls remain valid                       |
