# Contract: Hybrid Segmentation CLI and Artifact Interface

## Scope

This contract defines reproducible command inputs/outputs for the hybrid segmentation benchmark and stacking flow while preserving the existing `benthic_model` CLI baseline pipeline.

## Command Contract A: Baseline Probabilities Export

- Purpose: Produce RF/MLP validation/test probabilities for stacking.
- Invocation pattern:
  - `PYTHONPATH=src python -m benthic_model.cli train ...`
  - `PYTHONPATH=src python -m benthic_model.cli evaluate --run-id <run_id> ...`
  - `PYTHONPATH=src python -m benthic_model.cli predict --run-id <run_id> ...`
- Required inputs:
  - train CSV, test CSV, bathymetry raster, backscatter raster, config YAML.
- Required outputs:
  - `artifacts/runs/<run_id>/metrics.json`
  - `artifacts/predictions/<run_id>_val_predictions.csv`
  - `artifacts/predictions/<run_id>_test_predictions.csv`

## Command Contract B: Segmentation Benchmark Runner

- Purpose: Run exactly 3 segmentation candidates and export aligned outputs.
- Invocation pattern (planned script entrypoint):
  - `PYTHONPATH=src python -m benthic_model.cli segmentation-benchmark --config configs/<seg-config>.yaml`
  - Notebook equivalent is acceptable if it writes the same artifacts.
- Required inputs:
  - point labels dataset
  - raster features/images
  - candidate set fixed to `{segformer_ft, segformer_ft_crf, deeplabv3_ft}`
  - class-weight strategy `inverse_freq_capped`
- Required outputs:
  - `artifacts/segmentation/<run_id>/candidate_metrics.json`
  - `artifacts/segmentation/<run_id>/val_location_predictions.csv`
  - `artifacts/segmentation/<run_id>/test_location_predictions.csv`
- Output schema (`*_location_predictions.csv`):
  - `location_id` (string)
  - `predicted_class` (string)
  - `confidence` (float [0,1])
  - `prob_ALG`, `prob_FMAT`, `prob_NVB`, `prob_SGAM`, `prob_SGZ` (float [0,1])

## Command Contract C: Stacking and Promotion Gate

- Purpose: Combine baseline and segmentation features and make promotion decision.
- Invocation pattern (planned script entrypoint):
  - `PYTHONPATH=src python -m benthic_model.cli hybrid-stack --baseline-run <id> --seg-run <id>`
- Required behavior:
  - Use multinomial logistic regression as default meta-learner.
  - Evaluate gate: weighted F1 delta >= 0.02 and SGAM recall delta >= 0.0.
- Required outputs:
  - `artifacts/runs/<hybrid_run_id>/hybrid_metrics.json`
  - `artifacts/runs/<hybrid_run_id>/promotion_decision.json`
  - Optional submission candidate CSV if gate passes.

## Error Contract

- If segmentation artifact schema is missing required columns, command fails with clear schema validation message.
- If gate conditions fail, process exits successfully with `gate_passed=false` and baseline retained.
- If a candidate run fails, benchmark report must include failure reason and continue with remaining candidates when possible.
