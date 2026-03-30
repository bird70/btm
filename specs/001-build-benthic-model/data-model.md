# Data Model: Benthic Habitat Classification Pipeline

## Entity: SamplePoint
- Description: A single observation location from `train.csv` or `test.csv`.
- Fields:
  - `id` (string/integer, required, unique within dataset)
  - `x` (float, required, UTM Zone 55S easting)
  - `y` (float, required, UTM Zone 55S northing)
  - `dataset_split` (enum: `train` | `test`, required)
- Validation Rules:
  - `id` MUST be unique per split.
  - `x`/`y` MUST be finite numeric values.

## Entity: HabitatLabel
- Description: The target class for train observations and prediction output for test observations.
- Fields:
  - `class_code` (string, required, categorical; e.g., ALG, FMAT, NVB)
- Validation Rules:
  - Training labels define the allowed class vocabulary for submission output.

## Entity: RasterLayer
- Description: MBES raster source for feature extraction.
- Fields:
  - `name` (enum: `bathymetry` | `backscatter`, required)
  - `path` (string, required)
  - `crs` (string, required)
  - `nodata_value` (numeric/null)
  - `pixel_resolution` (tuple[float, float], required)
- Validation Rules:
  - CRS MUST match coordinate interpretation for `SamplePoint` or trigger fail-fast.
  - File MUST be readable and georeferenced.

## Entity: FeatureVector
- Description: Model input representation for a single `SamplePoint`.
- Fields:
  - `sample_id` (foreign key -> SamplePoint.id)
  - `raw_bathymetry` (float/null)
  - `raw_backscatter` (float/null)
  - `window_stats_*` (float features from multiple neighborhood windows)
  - `gradient_*` (float features for local slope/contrast)
  - `interaction_*` (float engineered features)
  - `quality_flags` (array[string], optional)
- Validation Rules:
  - One feature vector per sample point.
  - Missing raster values MUST set deterministic fallback and quality flag.

## Entity: ValidationFold
- Description: Fold assignment artifact for evaluation.
- Fields:
  - `sample_id` (foreign key -> SamplePoint.id)
  - `fold_id` (integer, required)
  - `fold_scheme` (enum: `spatial_blocked` | `stratified_random`)
- Validation Rules:
  - Primary fold scheme for model selection MUST be `spatial_blocked`.

## Entity: ExperimentRun
- Description: A single baseline/candidate training-evaluation execution with provenance.
- Fields:
  - `run_id` (string, required, unique)
  - `run_type` (enum: `baseline` | `candidate`)
  - `timestamp` (datetime, required)
  - `code_revision` (string, required)
  - `config_ref` (string, required)
  - `seed` (integer, required)
  - `metric_weighted_f1` (float, required)
  - `metric_per_class_f1` (map[string, float], required)
  - `artifacts` (map[string, string], required)
- Validation Rules:
  - Candidate comparisons MUST reference at least one baseline run.
  - Re-run of same config+seed MUST remain within tolerance policy.

## Entity: SubmissionFile
- Description: Competition output artifact for test predictions.
- Fields:
  - `path` (string, required)
  - `row_count` (integer, required)
  - `header` (fixed string sequence: `ID,class`)
  - `generated_from_run_id` (foreign key -> ExperimentRun.run_id)
- Validation Rules:
  - Must contain exactly one row per test ID.
  - IDs MUST be unique.
  - Class values MUST exist in training class vocabulary.

## Relationships
- `SamplePoint (train)` optionally has one `HabitatLabel`.
- `SamplePoint` has one `FeatureVector`.
- `FeatureVector` participates in one or more `ValidationFold` assignments.
- `ExperimentRun` consumes many `FeatureVector` records and produces metrics/artifacts.
- `SubmissionFile` is generated from one `ExperimentRun` and includes predictions for all `SamplePoint (test)` records.
