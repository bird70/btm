# Contract: Benthic Pipeline CLI

## Purpose
Define the user-facing command contract for reproducible training, evaluation, prediction, and submission generation.

## Command Surface

### `train`
- Description: Train a baseline or candidate model run.
- Required Inputs:
  - `--train-csv <path>`
  - `--bathymetry-tif <path>`
  - `--backscatter-tif <path>`
  - `--config <path>`
- Optional Inputs:
  - `--seed <int>`
  - `--run-type baseline|candidate`
- Outputs:
  - Model artifact path
  - Run metadata record
- Exit Conditions:
  - Non-zero on missing files, CRS mismatch, or invalid config.

### `evaluate`
- Description: Evaluate trained run(s) and emit baseline-vs-candidate report.
- Required Inputs:
  - `--run-id <id>`
  - `--fold-scheme spatial_blocked|stratified_random`
- Outputs:
  - Weighted F1
  - Per-class F1
  - Comparison report artifact
- Rules:
  - Model selection must rely on `spatial_blocked` evaluation results.

### `predict`
- Description: Generate test predictions using a selected run.
- Required Inputs:
  - `--run-id <id>`
  - `--test-csv <path>`
  - `--bathymetry-tif <path>`
  - `--backscatter-tif <path>`
- Outputs:
  - Prediction table with columns `ID,class`

### `make-submission`
- Description: Validate and write submission file.
- Required Inputs:
  - `--predictions <path>`
  - `--sample-submission <path>`
  - `--output <path>`
- Validation Contract:
  - Header exactly `ID,class`
  - Exactly one row per test ID
  - No duplicate IDs
  - Class values subset of training label vocabulary
- Outputs:
  - Submission CSV at output path
  - Validation summary

## UX and Logging Contract
- Metrics must use consistent labels: `weighted_f1`, `per_class_f1`.
- Class names must preserve canonical training vocabulary formatting.
- User-facing errors must include actionable next step and failing input path.

## Reproducibility Contract
Each command that mutates artifacts must emit run metadata including:
- code revision
- config reference
- seed
- fold scheme
- command invocation
- output artifact paths
