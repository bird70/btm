# Implementation Plan: GIS-Derived Feature Engineering

**Branch**: `008-gis-feature-engineering` | **Date**: 2026-03-27 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/008-gis-feature-engineering/spec.md`

---

## Summary

Add three BTM-derived spatial features (slope, backscatter zones, acoustic facies) to the proven `experiment_v2.py` feature set. Apply 80/20/20 relative importance scaling (continuous column scaling for slope; one-hot-encode then scale for the categorical zone features). Train the existing 4-model ensemble (LightGBM + XGBoost + CatBoost + RandomForest) with 10-block spatial CV. Run twice -- once with raw point-sampled GIS features (Run A) and once with zone-stratified ordinary kriging applied to those same three features (Run B). Compare both runs against a freshly-computed v2 baseline, designate the best submission, and write a run report. Create a PR.

---

## Technical Context

**Language/Version**: Python 3.13 (`.venv\Scripts\python.exe`; `requires-python = ">=3.11"`)
**Primary Dependencies**: rasterio, numpy, scipy, pandas, scikit-learn, lightgbm, xgboost, catboost, pykrige, scikit-image (all confirmed installed in venv)
**Storage**: File-based — GeoTIFF rasters under `outputs/gis_layers/rasters/`; CSV under `data/`; Markdown report under `docs/`
**Testing**: pytest (`python -m pytest tests/`); test markers `arcgis` and `qgis` skip where runtime absent
**Target Platform**: Standalone Python script; Windows development machine; cross-platform compatible
**Project Type**: Research/experiment script (derives features, trains model, emits submission CSV)
**Performance Goals**: Each CV phase completes within 30 minutes on a modern 8-core CPU; kriging phase within 30 minutes (9 zone fits _x_ 3 features; bounded by per-zone n)
**Constraints**: Memory-bounded — rasters are loaded fully into NumPy arrays (~100 MB total for Fagatele Bay data); no block-based chunking needed at this dataset scale
**Scale/Scope**: ~2 000 train points; ~1 000 test points; ~128 model features; 3 experiment phases (baseline, run_a, run_b)

---

## Constitution Check

_GATE: Evaluated before Phase 0. Re-checked after Phase 1._

- [x] **I. TDD** — Tests for `experiment_v5.py` verify: (a) feature augmentation function output shape and column names, (b) weight scaling produces expected numeric values, (c) submission CSV has correct columns (`ID,class`) and row count matching `test.csv`, (d) kriged feature matrix has no NaN values. Tests use a small synthetic fixture, not live rasters, so they run in arcpy-free CI. _Exception documented below._
- [x] **II. Code Quality** — ruff configured in `pyproject.toml` (`line-length=100`, selectors E/F/W/I/UP). New script follows same conventions. Public helper functions have docstrings.
- [x] **III. Performance** — Rasters loaded once into NumPy arrays (`np.nan_to_num`). All feature extraction uses `uniform_filter`, `gaussian_filter`, vectorised operations (SciPy). K-means uses 50 000 pixel subsample. No pixel-level Python loops in hot paths. Dataset is <200 MB; block-based chunking is not required at this scale (documented).
- [x] **IV. Scientific Accuracy** — slope: Horn (1981) via `btm.core.slope` (already cited in codebase). K-means: Hartigan & Wong (1979). Ordinary kriging: Matheron (1963). All referenced in `research.md`.
- [x] **V. Platform Portability** — New script `scripts/experiment_v5.py` is a standalone Python script with no arcpy, QGIS, or GUI dependency. Imports only `btm.core.*` and `btm.features.extract` (both platform-agnostic). No wrappers needed (experiment script, not a packaged tool).
- [x] **VI. Simplicity** — Single script (`experiment_v5.py`) with clearly-named helper functions; one responsibility: _compare raw vs kriged GIS feature augmentation_. No speculative abstractions. Magic numbers replaced by named constants at top of file.

### Complexity Tracking

| Item                                                  | Why needed                                                                                    | Simpler alternative rejected because                                                                                                                  |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| TDD exception for experiment script                   | Experiment output (model accuracy) is non-deterministic and cannot be asserted in a unit test | Tests instead verify structural/format invariants (shape, columns, no NaN, row count) — this is the practical TDD interpretation for ML research code |
| 80/20/20 column scaling despite tree-model invariance | Spec FR-003 requires it; user requested it                                                    | Omitting scaling would deviate from spec; invariance is documented in research.md and run report                                                      |

---

## Project Structure

### Documentation (this feature)

```
specs/008-gis-feature-engineering/
+-- plan.md           <- This file
+-- spec.md           <- Feature specification
+-- research.md       <- Phase 0 research findings
+-- data-model.md     <- Phase 1 data model
+-- quickstart.md     <- How to run the experiment
+-- checklists/
|   +-- requirements.md
```

### Source Code Layout

```
scripts/
+-- experiment_v5.py       [NEW] Main experiment script (all 3 phases + report)

data/
+-- submission_v5_gis_features.csv         [NEW] Run A output
+-- submission_v5_gis_features_kriging.csv [NEW] Run B output
+-- submission_v5_best.csv                 [NEW] Best submission (copy)

docs/
+-- run-008-gis-feature-engineering.md    [NEW] Run report

tests/
+-- unit/
|   +-- test_experiment_v5_features.py    [NEW] Unit tests for feature engineering helpers
```

Existing files used (read-only):

```
btm/core/slope.py              -- compute_slope()
btm/core/vrm.py                -- compute_vrm()
btm/features/extract.py        -- sample_raster_at_points()
scripts/experiment_v2.py       -- extract_all_features() (cloned)
scripts/exp_backscatter_zones.py -- zone derivation + kriging patterns (adapted)
outputs/gis_layers/rasters/    -- slope.tif, backscatter_zones.tif, acoustic_facies.tif
data/MBES/                     -- bathymetry.tif, backscatter.tif
data/train.csv, data/test.csv
```

**Structure Decision**: Single standalone script with internal helper functions. No new package modules (YAGNI). The experiment script is intentionally self-contained so it is reproducible without the rest of the package structure.

---

## Phase 0: Research

All unknowns resolved. See [research.md](research.md) for full findings. Summary:

1. **Derived rasters**: All three confirmed on disk. Re-derivation logic added as a safety check.
2. **Weighting**: 80/20/20 column scaling implemented per spec; tree-model invariance documented.
3. **Kriging scope**: Zone-stratified ordinary kriging applied only to the 3 new GIS features.
4. **Cluster counts**: 5 (backscatter zones), 8 (acoustic facies) -- matching prior scripts.
5. **Baseline F1**: Computed inline from v2 feature set using identical CV folds.
6. **Code reuse**: `extract_all_features()` cloned; zone derivation + kriging adapted.

---

## Phase 1: Design

### Data Model

See [data-model.md](data-model.md). Key points:

- **Base feature matrix**: ~114 features from `extract_all_features()` (v2.py clone)
- **New GIS features**: 14 columns -- `gis_slope` (×80), `gis_bz_0..4` (OHE ×20), `gis_af_0..7` (OHE ×20)
- **Run A**: raw point-sampled GIS features
- **Run B**: kriged GIS features (raw int zone → krige → soft assignment → OHE)
- **Output**: 3 submission CSVs + 1 run report Markdown

### Interface Contracts

This is a purely internal experiment script. No external API, CLI entry point, or GIS adapter is exposed. Contract section not applicable per spec FR-009 (write to files, no interface boundary).

### Key Design Decisions

#### `experiment_v5.py` internal structure

```python
# Constants
WEIGHT_SLOPE = 80.0
WEIGHT_BZ    = 20.0
WEIGHT_AF    = 20.0
N_BZ_CLUSTERS = 5
N_AF_CLUSTERS = 8
SEED = 42
N_SPATIAL_BLOCKS = 10
KRIGING_MIN_ZONE_PTS = 10
KRIGING_VARIOGRAM = "spherical"
KRIGING_NLAGS = 6

# Main sections:
# 1. _load_rasters()           -- load bathy, backscatter
# 2. _ensure_derived_rasters() -- check slope/bz/af exist; derive if missing
# 3. _load_derived_rasters()   -- load slope, bz, af from disk
# 4. extract_all_features()    -- v2 feature set (cloned from experiment_v2.py)
# 5. _add_gis_features_raw()   -- sample + weight GIS features (Run A)
# 6. _krige_gis_features()     -- zone-stratified kriging (Run B)
# 7. _add_gis_features_kriged()-- apply kriging predictions + weight
# 8. get_models()              -- ensemble factory (cloned from experiment_v2.py)
# 9. _run_spatial_block_cv(X, y_labels, enc, train_coords, seed, n_blocks) -- CV loop; train_coords=(n,2) array
# 10. _write_submission()      -- write ID,class CSV
# 11. _write_run_report(out_dir, results) -- markdown comparison table; out_dir defaults to docs/
# 12. main()                   -- orchestrate baseline → run_a → run_b → report
```

#### Kriging for categorical zone features

The backscatter zone and acoustic facies values are integers (0–4 and 0–7). Kriging assumes a continuous numeric field. The approach:

1. At training points, use the raw integer zone assignment as the kriging target variable.
2. Fit `OrdinaryKriging(xs_train_zone, ys_train_zone, zone_values, variogram_model='spherical')` per spatial zone.
3. Predict at test points → float "soft zone" values (not constrained to integers).
4. Round to nearest integer, clip to [0, n_clusters-1], then one-hot encode.
5. Apply weight factor to OHE columns.

This treats the zone assignment as a spatially-autocorrelated field and lets kriging smooth it, which is a reasonable proxy for spatial interpolation of discrete habitat boundaries.

---

## Phase 1 Post-Design Constitution Re-Check

| Principle               | Status | Notes                                             |
| ----------------------- | ------ | ------------------------------------------------- |
| I. TDD                  | PASS   | Test file structure and fixtures identified below |
| II. Code Quality        | PASS   | ruff, docstrings, named constants planned         |
| III. Performance        | PASS   | Dataset < 200 MB; vectorised ops; no pixel loops  |
| IV. Scientific Accuracy | PASS   | All algorithms referenced; see research.md        |
| V. Platform Portability | PASS   | Standalone script; no GIS runtime                 |
| VI. Simplicity          | PASS   | Single script; single responsibility              |

### Test Plan (unit/test_experiment_v5_features.py)

```
test_add_gis_features_raw_shape
  - Given: small synthetic feature matrix + fake slope/bz/af arrays
  - When: _add_gis_features_raw() called
  - Then: output has n_base + 14 columns, correct column names

test_gis_feature_weights
  - Given: bz value = 2, weight = 20
  - When: OHE + scaling applied
  - Then: gis_bz_2 = 20.0, all others = 0.0; gis_slope = raw_slope * 80

test_krige_gis_features_no_nan
  - Given: synthetic train/test coords + zone values
  - When: _krige_gis_features() called
  - Then: result has no NaN entries; shape = (n_test, 3)

test_write_submission_format
  - Given: pred labels array + test IDs
  - When: _write_submission() called
  - Then: output CSV has exactly 2 columns (ID, class); row count matches test

test_run_report_contains_all_runs
  - Given: results dict with baseline, run_a, run_b f1 scores
  - When: _write_run_report() called
  - Then: report Markdown contains "baseline", "run_a", "run_b" and all f1 values
```

---

## Artifacts Summary

| Artifact               | Path                                              | Phase          |
| ---------------------- | ------------------------------------------------- | -------------- |
| Research findings      | `specs/008-gis-feature-engineering/research.md`   | Phase 0        |
| Data model             | `specs/008-gis-feature-engineering/data-model.md` | Phase 1        |
| Quickstart guide       | `specs/008-gis-feature-engineering/quickstart.md` | Phase 1        |
| Main experiment script | `scripts/experiment_v5.py`                        | Implementation |
| Unit tests             | `tests/unit/test_experiment_v5_features.py`       | Implementation |
| Run A submission       | `data/submission_v5_gis_features.csv`             | Run            |
| Run B submission       | `data/submission_v5_gis_features_kriging.csv`     | Run            |
| Best submission        | `data/submission_v5_best.csv`                     | Run            |
| Run report             | `docs/run-008-gis-feature-engineering.md`         | Run            |

Next step: run `/speckit.tasks` to generate `tasks.md`.
Next step: run `/speckit.tasks` to generate `tasks.md`.
