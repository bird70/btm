# Feature Specification: Systematic BTM Model Sweep

**Feature Branch**: `014-btm-model-sweep`  
**Created**: 2026-03-30  
**Status**: Draft  
**Input**: User description: "Systematic plan to run and compare best BTM model configurations, identify overfitting causes, and find the optimal terrain analysis and ML pipeline combination to exceed Kaggle submission F1 of 0.76394"

## Background

This specification addresses a critical gap that has emerged across ten-plus
experiment runs: local cross-validation scores consistently overestimate Kaggle
leaderboard performance. The clearest example is the best recorded CV score of
**0.8026** (baseline Random Forest, spatial blocked 5-fold) versus the best
Kaggle submission score of **0.76394** — a gap of ~0.04 points. More
sophisticated multi-model ensembles (v7: 0.69135, v9: 0.69219) performed
_worse_ on Kaggle despite higher or comparable CV scores, indicating that
complexity added overfitting rather than generalisation.

The hypothesis is that the baseline Random Forest on MBES-derived features, run
through the existing `benthic_model` pipeline, may already be at or near the
practical ceiling for this dataset. The goal of this sweep is to test that
hypothesis systematically, using the pipeline's built-in experiment registry for
reproducible tracking, and to produce at least one Kaggle submission for every
configuration that achieves a local CV ≥ 0.76.

---

## User Scenarios & Testing _(mandatory)_

### User Story 1 — Reproduce and Submit the Baseline (Priority: P1)

A researcher runs the existing baseline configuration exactly as documented and
obtains a Kaggle submission file. They submit it to establish the **true Kaggle
score** for the baseline — creating an honest starting point for all subsequent
comparisons.

**Why this priority**: The 0.8026 CV score has never been submitted to Kaggle.
Without a verified Kaggle score for the baseline, every comparison is built on
an unknown foundation. This run costs minimal compute and immediately answers
whether the 0.76394 ceiling came from OBIA complexity or from the baseline
features themselves.

**Independent Test**: Fully testable by running `benthic_model.cli` train /
evaluate / predict with `configs/baseline.yaml` and submitting the output CSV
via the Kaggle CLI.

**Acceptance Scenarios**:

1. **Given** the baseline config (`configs/baseline.yaml`, RF 300 trees, spatial
   blocked 5-fold CV), **When** the train command runs to completion, **Then** a
   run entry is written to `artifacts/experiments/run_registry.jsonl` with
   `run_type=baseline` and `weighted_f1` ≥ 0.75.
2. **Given** a successfully trained baseline run, **When** the predict command
   runs against `data/test.csv`, **Then** a valid `ID,class` submission CSV is
   produced with exactly the same row count as `data/test.csv` and all values
   in `{ALG, FMAT, NVB, SGAM, SGZ}`.
3. **Given** the submission CSV, **When** the Kaggle CLI submit command is
   executed, **Then** the public leaderboard score is automatically fetched and
   appended to `artifacts/experiments/kaggle_scores.csv`.

---

### User Story 2 — Sweep Feature Sets Against the RF Baseline (Priority: P2)

A researcher runs a matrix of configurations that vary the feature set while
holding the model family (Random Forest) fixed. Each run is registered in the
experiment registry. At the end, per-class F1 scores and CV–Kaggle gaps are
compared to identify which features generalise and which add noise.

**Why this priority**: The v7 failure (spatial memorisation via x_rel/y_rel)
and v10/v12 degradation both point to feature selection as the primary lever.
Identifying the minimal highest-performing feature set de-risks subsequent
model tuning.

**Independent Test**: Each configuration variant can be run and evaluated independently.

**Acceptance Scenarios**:

1. **Given** the run matrix (R04–R08) in Requirements FR-002, **When** each
   configuration is trained and evaluated, **Then** every run has a registry
   entry with `run_id`, `config_ref`, `weighted_f1`, and `per_class_f1`.
2. **Given** all RF runs in the matrix, **When** CV scores are ranked, **Then**
   the configuration with the highest CV weighted-F1 is identified as the
   feature-sweep winner and its submission CSV is produced.
3. **Given** any configuration that employs spatial coordinates (x, y, x_rel,
   y_rel, depth_z, backscatter_z), **When** its Kaggle score is ≥ 0.03 below
   its CV score, **Then** spatial memorisation is confirmed and those features
   are excluded from all Phase 3 candidates.

---

### User Story 3 — Sweep Model Families on the Winning Feature Set (Priority: P3)

A researcher takes the feature set that produced the best Phase 2 CV score with
the smallest CV–Kaggle gap, and re-trains it with LightGBM, XGBoost, and
CatBoost candidates, comparing Kaggle scores.

**Why this priority**: Model selection is only meaningful once the feature set
is stable. Running model sweeps before fixing features has been the source of
confounding results in previous branches.

**Independent Test**: Each model-family run is independently executable and
produces its own submission CSV.

**Acceptance Scenarios**:

1. **Given** the winning feature set from US2, **When** LightGBM, XGBoost, and
   CatBoost candidates are trained with the same CV scheme, **Then** each
   produces a registry entry and a Kaggle submission CSV.
2. **Given** all model-family runs on the winning feature set, **When** Kaggle
   scores are compared, **Then** the model family with the highest Kaggle score
   is declared the sweep winner.
3. **Given** the sweep winner, **When** a simple soft-vote ensemble of the best
   two models is evaluated, **Then** the ensemble's Kaggle score is recorded
   and compared against each individual model.

---

### User Story 4 — Document Findings as a Canonical Run Report (Priority: P4)

A researcher reviews all registered runs, annotates each with its Kaggle score,
and publishes a run report in `docs/` following the established `run-0NN-*.md`
format. The report summarises CV-vs-Kaggle gaps, names the best configuration,
and records lessons learned for future branches.

**Why this priority**: Without structured documentation, hard-won insights from
this sweep will be lost to future branches, repeating the v7/v8/v9 regression
cycle.

**Independent Test**: Completable by a single documentation pass after all
Kaggle scores are recorded.

**Acceptance Scenarios**:

1. **Given** all sweep run records and Kaggle scores, **When** the run report is
   written, **Then** `docs/run-013-btm-model-sweep.md` exists and follows the
   format of `docs/run-012-v9-catboost-lgb-texture.md`.
2. **Given** the run report, **Then** it contains a comparative table with CV
   score, Kaggle score, CV–Kaggle gap, and spatial-coordinate flag for every run.

---

### Edge Cases

- What if the baseline RF Kaggle score substantially exceeds 0.76394? That
  indicates the OBIA pipeline was introducing noise; the sweep should focus
  entirely on extending the baseline rather than trying alternative pipelines.
- What if all configurations show a CV–Kaggle gap > 0.05? The spatial-blocked
  CV scheme may be insufficient; a more aggressive spatial hold-out (entire
  survey quadrant as the test fold) should be trialled as a corrective action.
- What if the test set contains depth ranges not represented in training (OOD
  samples)? Per-class F1 for SGZ should be monitored; it historically scores
  near zero and may indicate distribution shift rather than model weakness.
- What if a Phase 2 config produces CV > 0.80 but Kaggle < 0.70? That config
  is disqualified regardless of CV score, and its feature columns are flagged
  as potential leakage sources.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: All runs MUST use the existing `benthic_model.cli` (train /
  evaluate / predict) commands; no standalone experiment scripts bypass the
  registry.
- **FR-002**: Each run in the sweep MUST correspond to one named, immutable
  config YAML in `configs/`. Configs must not be mutated after first use;
  create a new file for each variant.
- **FR-003**: The run registry (`artifacts/experiments/run_registry.jsonl`)
  MUST contain an entry for every executed run, recording at minimum:
  `run_id`, `run_type`, `config_ref`, `weighted_f1`, `per_class_f1`, `seed`,
  `git_rev`.
- **FR-004**: Every configuration achieving local CV weighted-F1 ≥ 0.75 in
  Phase 1 (or ≥ 0.76 in Phases 2–3) MUST have a submission CSV generated AND
  submitted to Kaggle via the CLI, subject to the daily submission budget
  defined in FR-009.
- **FR-005**: Configurations that include spatial coordinate features (x, y,
  x_rel, y_rel, depth_z, backscatter_z) MUST be labelled `spatial_coords: true`
  in their YAML and treated as a diagnostic track, not the primary track.
- **FR-006**: The sweep MUST be executable in phase order (Phase 1 → Phase 2 →
  Phase 3) with a phase gate check between each phase.
- **FR-007**: After each Kaggle CLI submission, the public score MUST be
  automatically fetched (via `kaggle competitions submissions`) and appended to
  `artifacts/experiments/kaggle_scores.csv` (columns: `run_id`,
  `submission_file`, `kaggle_public_f1`, `submitted_at`).
- **FR-008**: A final comparative table of all runs MUST appear in the run
  report (`docs/run-013-btm-model-sweep.md`).
- **FR-009**: The daily Kaggle submission budget is capped at **4 per calendar
  day** (leaving 1 emergency slot). Submissions are prioritised by:
  (1) phase gate run first, (2) highest-CV run in each phase group,
  (3) diagnostic runs last. Runs that are not submitted on a given day MUST
  be queued and submitted the following day before new runs are initiated.

### Run Matrix

The following fourteen runs form the sweep, grouped into three phases. Each
phase has a gate condition that must pass before advancing.

#### Phase 1 — Baseline Verification

(Gate: verify Kaggle ground truth before adding any new features)

| Run ID | Config                            | Model                 | Feature Set                                     | Purpose                                             |
| ------ | --------------------------------- | --------------------- | ----------------------------------------------- | --------------------------------------------------- |
| R01    | `configs/baseline.yaml`           | RF 300 trees balanced | MBES core + focal stats + interaction terms     | Reproduce 0.8026 CV; first honest Kaggle submission |
| R02    | `configs/rf-core-only.yaml`       | RF 300 trees balanced | MBES core 8 only (no focal, no interactions)    | Isolate whether engineered features help or hurt    |
| R03    | `configs/rf-no-interactions.yaml` | RF 300 trees balanced | MBES core 8 + focal stats, no interaction terms | Test interaction-term contribution in isolation     |

Phase 1 Gate: at least one run must achieve CV ≥ 0.75 before Phase 2.

#### Phase 2 — Feature Set Sweep

(Gate: choose winning feature set by smallest CV–Kaggle gap, not highest CV)

| Run ID | Config                        | Model | Feature Set                                       | Purpose                                               |
| ------ | ----------------------------- | ----- | ------------------------------------------------- | ----------------------------------------------------- |
| R04    | `configs/rf-btm-fine.yaml`    | RF    | Core 8 + BTM fine-scale (fine BPI, VRM, slope)    | Add BTM library terrain features at fine scale        |
| R05    | `configs/rf-btm-broad.yaml`   | RF    | Core 8 + BTM broad-scale (broad BPI, depth-stats) | Add BTM broad-scale terrain features                  |
| R06    | `configs/rf-btm-full.yaml`    | RF    | Core 8 + BTM fine + broad                         | Full BTM terrain feature set                          |
| R07    | `configs/rf-texture.yaml`     | RF    | Core 8 + texture (bathy_std_9, back_std_9, tpi_9) | Replicate v9 clean features in benthic_model pipeline |
| R08    | `configs/rf-btm-texture.yaml` | RF    | Core 8 + BTM full + texture                       | Maximum feature set; watch for overfitting signal     |

Phase 2 Gate: identify the single config with smallest CV–Kaggle gap; that
feature set advances to Phase 3.

#### Phase 3 — Model Family Sweep

(Gate: apply only to the Phase 2 winning feature set)

| Run ID | Config                            | Model                   | Feature Set             | Purpose                                   |
| ------ | --------------------------------- | ----------------------- | ----------------------- | ----------------------------------------- |
| R09    | `configs/lgbm-candidate.yaml`     | LightGBM                | Phase 2 winner features | LightGBM on optimal features              |
| R10    | `configs/xgb-candidate.yaml`      | XGBoost                 | Phase 2 winner features | XGBoost on optimal features               |
| R11    | `configs/catboost-candidate.yaml` | CatBoost                | Phase 2 winner features | CatBoost — historically best Kaggle model |
| R12    | `configs/rf-lgbm-ensemble.yaml`   | RF + LightGBM soft-vote | Phase 2 winner features | Soft-vote ensemble of best two families   |

#### Diagnostic Track (run alongside any phase, not gated)

| Run ID | Config                           | Model | Feature Set                  | Purpose                                                                         |
| ------ | -------------------------------- | ----- | ---------------------------- | ------------------------------------------------------------------------------- |
| R13    | `configs/rf-spatial-diag.yaml`   | RF    | Core 8 + x_rel/y_rel/depth_z | Intentionally reproduce v7-style spatial memorisation to quantify CV–Kaggle gap |
| R14    | `configs/rf-6fold-baseline.yaml` | RF    | Core 8 (same as R01)         | 6-fold CV on baseline features to test whether fold count changes the gap       |

### Key Entities

- **Config File**: A named, immutable YAML in `configs/` fully describing one
  experimental unit (model type, feature flags, CV parameters). One config
  per run.
- **Run**: One execution of `benthic_model.cli train` producing a unique
  `run_id` (timestamp + hash). Runs are deterministic given the same config
  and seed.
- **Registry Entry**: A JSONL line in `artifacts/experiments/run_registry.jsonl`
  recording full provenance (git rev, config hash, seed, CV scores).
- **Submission CSV**: An `ID,class` file from `benthic_model.cli predict`,
  named `data/submission_<run_id>.csv`, eligible for Kaggle upload.
- **Kaggle Score Record**: `artifacts/experiments/kaggle_scores.csv` with
  columns `run_id, submission_file, kaggle_public_f1, submitted_at`. Populated
  automatically via the Kaggle CLI after each submission.
- **Phase Gate**: Decision point requiring at least one CV score ≥ 0.75 (Phase
  1. or the identification of a feature set with acceptable CV–Kaggle gap
     (Phase 2) before proceeding.
- **Submission Budget**: A rolling count of Kaggle submissions per calendar day
  (UTC). Hard cap of 4 per day; the 5th slot is held in reserve. If the budget
  is exhausted, remaining submissions queue to the next day.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: At least one configuration achieves a Kaggle public leaderboard
  weighted-F1 > 0.76394, surpassing the historical best.
- **SC-002**: The CV–Kaggle gap for the best configuration is ≤ 0.04 points
  (down from the observed 0.04–0.11 in prior branches).
- **SC-003**: All 14 runs (R01–R14) are registered with full provenance in
  `run_registry.jsonl`; zero runs require manual score reconstruction.
- **SC-004**: Per-class F1 for the SGZ class achieves ≥ 0.30 in at least one
  submitted configuration (historically the worst-performing class, often 0.0).
- **SC-005**: A run report is published in `docs/` that allows any team member
  to reproduce the best submission from a clean checkout in under 30 minutes.
- **SC-006**: No submitted configuration regresses below a Kaggle score of 0.65
  (which would represent a worse result than the v8 regression at ~0.47).

## Kaggle CLI Setup

The Kaggle CLI is installed in a sibling virtual environment at
`C:\DEVlocal\btm\.venv\Scripts\kaggle.exe`. This path must be used explicitly
in all submission commands because the working virtual environment for model
training (`C:\DEVlocal\BTM-Hybrid\btm`) is separate.

### Authentication (one-time setup)

Kaggle CLI requires API credentials. These have **not** been configured yet.
To set this up:

1. Log in to [kaggle.com](https://www.kaggle.com) → **Account Settings** →
   scroll to **API** → click **Create New API Token**.
2. This downloads `kaggle.json` to your browser's default download location.
3. Place the file at `%USERPROFILE%\.kaggle\kaggle.json` (Windows):
   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
   Copy-Item <path-to-downloaded-kaggle.json> "$env:USERPROFILE\.kaggle\kaggle.json"
   ```
4. Verify authentication:
   ```powershell
   C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions list
   ```
   Expected: the `geohab-mlwg-competition-2026` competition appears in the list.

### Competition reference

| Property               | Value                                                                        |
| ---------------------- | ---------------------------------------------------------------------------- |
| Competition slug       | `geohab-mlwg-competition-2026`                                               |
| Competition URL        | https://www.kaggle.com/competitions/geohab-mlwg-competition-2026/submissions |
| Daily submission limit | 5 (agent uses max 4, reserving 1 emergency slot)                             |
| Kaggle CLI exe         | `C:\DEVlocal\btm\.venv\Scripts\kaggle.exe`                                   |

### Submission command pattern

```powershell
# Submit a prediction file
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit `
  -c geohab-mlwg-competition-2026 `
  -f data\submission_<RUN_ID>.csv `
  -m "<RUN_ID>: <CONFIG_NAME>, CV=<F1_SCORE>"

# Fetch latest submission scores (wait ~60s after submit for scoring)
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submissions `
  -c geohab-mlwg-competition-2026 --csv | Select-Object -First 5
```

Append the returned public score to `artifacts/experiments/kaggle_scores.csv`
immediately after each submission is scored.

---

## Runsheet

The following ordered steps constitute the complete execution sequence for this
sweep. Steps are designed to be executed in phase order because later steps
depend on results from earlier ones. Each step references Run IDs from the Run
Matrix in Requirements FR-002.

### Submission Budget Plan

| Day   | Planned Submissions | Runs                                                                    |
| ----- | ------------------- | ----------------------------------------------------------------------- |
| Day 1 | 3                   | R01, R02, R03 (all Phase 1 — fast RF, establish ground truth)           |
| Day 2 | 3–4                 | Top CV scorers from Phase 2 (R04–R08 trained same day, submit best 3–4) |
| Day 3 | Up to 4             | Remaining Phase 2 submittable runs + R13 diagnostic                     |
| Day 4 | 3–4                 | Phase 3 model family runs (R09–R12)                                     |

Rule: never submit two Phase 2 runs on the same day that use near-identical
feature sets unless their CV scores are ≥ 0.01 apart. This preserves budget for
meaningful comparisons.

### Step 1 — Environment and Auth Check

Verify both packages are importable, data files are present, and Kaggle CLI is
configured:

```powershell
# Package imports
python -c "import btm; import benthic_model; print('OK')"
python -c "import pandas as pd; df=pd.read_csv('data/train.csv'); print(len(df), 'training points')"
python -c "import rasterio; r=rasterio.open('data/MBES/bathymetry.tif'); print('CRS:', r.crs)"

# Kaggle CLI auth
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions list
```

Expected: no import errors; ≥ 2 000 training points; a valid projected CRS;
`geohab-mlwg-competition-2026` appears in the competition list. **Do not
proceed to Step 2 if Kaggle CLI auth fails** — all submission steps depend on it.

### Step 2 — Phase 1: Baseline Runs (R01, R02, R03)

Run each configuration in sequence. Record the `run_id` printed to stdout after
each train command.

```bash
# R01 — Baseline RF (reproduce 0.8026 CV)
PYTHONPATH=src python -m benthic_model.cli train \
  --train-csv data/train.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif \
  --config configs/baseline.yaml --run-type baseline --seed 42

# Then for each run_id:
PYTHONPATH=src python -m benthic_model.cli evaluate \
  --run-id <RUN_ID> --fold-scheme spatial_blocked

PYTHONPATH=src python -m benthic_model.cli predict \
  --run-id <RUN_ID> --test-csv data/test.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif

# Repeat for R02 (configs/rf-core-only.yaml) and R03 (configs/rf-no-interactions.yaml)
```

**Phase 1 Gate**: If best CV weighted-F1 < 0.75 for all three runs, review
feature engineering code before proceeding; do not advance to Phase 2.

### Step 3 — Submit Phase 1 Kaggle Predictions (Day 1 budget: max 3)

For each Phase 1 run where CV ≥ 0.75, submit via the Kaggle CLI and record
the score. All three Phase 1 runs should be submitted on Day 1 — they are the
ground-truth anchor for everything that follows.

```powershell
# Submit R01 baseline (always submit regardless of CV — it's the anchor)
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit `
  -c geohab-mlwg-competition-2026 `
  -f data\submission_<R01_RUN_ID>.csv `
  -m "R01 baseline RF: CV=<F1>"

# Wait ~90 seconds for scoring, then fetch:
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submissions `
  -c geohab-mlwg-competition-2026 --csv | Select-Object -First 3

# Repeat the submit+fetch pattern for R02 and R03
# (only if budget allows — all 3 should fit in Day 1's 4-submission cap)
```

After each fetch, append to `artifacts/experiments/kaggle_scores.csv`:

```
run_id,submission_file,kaggle_public_f1,submitted_at
<run_id>,data/submission_<run_id>.csv,<SCORE>,<DATE>
```

**Phase 1 Gate**: Proceed to Phase 2 only after at least one Kaggle score is
recorded. The submission with the smallest CV–Kaggle gap from Phase 1 sets the
benchmark feature set for comparison.

### Step 4 — Extract BTM Terrain Features (prerequisite for Phase 2 R04–R08)

```bash
# Fine-scale BTM features
btm-export-features \
  --bathy data/MBES/bathymetry.tif \
  --points data/train.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner 1 --fine-outer 5 \
  --outdir data/btm_rasters --output data/train_btm.csv

# Same for test points
btm-export-features \
  --bathy data/MBES/bathymetry.tif \
  --points data/test.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner 1 --fine-outer 5 \
  --output data/test_btm.csv
```

### Step 5 — Phase 2: Feature Set Sweep (R04–R08)

Point `--train-csv` at `data/train_btm.csv` (BTM columns auto-included as
numeric features). Train and evaluate all five configs; then apply the
submission budget.

```powershell
# Train all 5 Phase 2 configs (no submission budget consumed yet)
# Then rank by CV weighted-F1 and submit top 3-4 on Day 2:
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit `
  -c geohab-mlwg-competition-2026 `
  -f data\submission_<BEST_PHASE2_RUN_ID>.csv `
  -m "R0N <config-name>: CV=<F1>"
# Submit 2nd and 3rd best CV scorers similarly (Day 2: max 4 total)
# Remaining Phase 2 submissions carry over to Day 3
```

**Phase 2 Gate**: After all Phase 2 Kaggle scores are recorded, select the
config with the smallest CV–Kaggle gap (not highest CV alone). That config's
feature set — and its corresponding `--train-csv` file — advances to Phase 3.

### Step 6 — Phase 3: Model Family Sweep (R09–R12, Day 4: max 4 submissions)

Using only the Phase 2 winning feature set (and its corresponding merged CSV),
train all four model families. Submit all with CV ≥ 0.76 on Day 4:

```powershell
# Submit CatBoost first (historically best Kaggle model):
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit `
  -c geohab-mlwg-competition-2026 `
  -f data\submission_<R11_RUN_ID>.csv `
  -m "R11 CatBoost on Phase2-winner features: CV=<F1>"

# Then LightGBM, XGBoost, and ensemble — in descending CV order
# All 4 fit within the Day 4 budget cap of 4 submissions
```

### Step 7 — Diagnostic Track (R13, R14)

Run R13 (spatial coordinate config) and R14 (6-fold baseline) at any convenient
point alongside Phases 1–3. Submit **R13 only** to Kaggle — its leaderboard
score quantifies exactly how much spatial memorisation inflates CV. R14 does
not need a Kaggle submission; comparing its CV to R01's CV is sufficient to
answer the fold-count question.

```powershell
# R13 — submit on Day 3 alongside remaining Phase 2 submissions
C:\DEVlocal\btm\.venv\Scripts\kaggle.exe competitions submit `
  -c geohab-mlwg-competition-2026 `
  -f data\submission_<R13_RUN_ID>.csv `
  -m "R13 DIAGNOSTIC spatial-coords RF: CV=<F1> (expected Kaggle << CV)"
```

### Step 8 — Write Run Report

When all Kaggle scores are recorded:

1. Compare the CV–Kaggle gap table across all 14 runs.
2. Confirm or reject the hypothesis that simpler models generalise better.
3. Write `docs/run-013-btm-model-sweep.md` following the existing doc format.
4. Update `README.md` best result line if a new Kaggle high score was achieved.

---

## Assumptions

- The `benthic_model.cli` pipeline and `btm` library are already installed in
  the active virtual environment via `pip install -e ".[dev,ml]"`.
- `data/MBES/bathymetry.tif` and `data/MBES/backscatter.tif` are present and
  correctly projected; CRS matches the training point coordinates.
- `btm-export-features` CLI is functional for generating BTM terrain columns
  into merged CSVs (required for Phase 2 R04–R08).
- The Kaggle CLI is installed at `C:\DEVlocal\btm\.venv\Scripts\kaggle.exe`
  (a sibling virtual environment, separate from the training venv).
- Kaggle API credentials have **not** yet been configured. The one-time setup
  described in the Kaggle CLI Setup section must be completed before Step 1.
- The competition slug is `geohab-mlwg-competition-2026` and the account is
  already enrolled in the competition.
- Config YAML files for R02–R14 do not yet exist; they must be created in the
  implementation (plan phase). Creating them is an implementation task.
- LightGBM, XGBoost, and CatBoost must be installed before Phase 3 (R09–R12);
  installation is a prerequisite step, not part of this specification.
- All runs use `seed: 42` and `cv.n_splits: 5` with `fold_scheme:
spatial_blocked` unless explicitly noted; deviations must be recorded in
  the config YAML.
- `artifacts/experiments/run_registry.jsonl` is the single source of truth for
  run provenance; Kaggle scores are appended to
  `artifacts/experiments/kaggle_scores.csv` automatically after each CLI
  submission and score fetch.
