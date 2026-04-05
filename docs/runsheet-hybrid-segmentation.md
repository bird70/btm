# Runsheet: Hybrid Segmentation Workflow (021)

## Scope

This runbook captures the practical command sequence for the hybrid segmentation workflow introduced in feature `021-hybrid-segmentation`.

## Prerequisites

- Python virtual environment at `.venv/`
- Source execution with `PYTHONPATH=src`
- Competition data present in `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`

## Command Sequence

```bash
# 1) Build 5x5 masks from labeled points
PYTHONPATH=src .venv/bin/python -m benthic_model.cli segmentation-build-masks \
  --train-csv data/train.csv \
  --output-dir artifacts/segmentation/mask-run-20260405

# 2) Benchmark 3 segmentation candidates
PYTHONPATH=src .venv/bin/python -m benthic_model.cli segmentation-benchmark \
  --config configs/segmentation-hybrid.yaml

# 3) Stack segmentation + baseline and evaluate promotion gate
PYTHONPATH=src .venv/bin/python -m benthic_model.cli hybrid-stack \
  --baseline-run candidate-20260330064219 \
  --seg-run seg-20260405042554 \
  --train-csv data/train.csv \
  --test-csv data/test.csv \
  --seg-base-dir artifacts/segmentation

# 4) Build competition-format CSV
PYTHONPATH=src .venv/bin/python -m benthic_model.cli make-submission \
  --predictions artifacts/predictions/hybrid-20260405042701_test_predictions.csv \
  --sample-submission data/sample_submission.csv \
  --output submissions/submission_hybrid_segmentation_20260405.csv
```

## Expected Artifacts

- `artifacts/segmentation/seg-*/candidate_metrics.json`
- `artifacts/segmentation/seg-*/val_location_predictions.csv`
- `artifacts/segmentation/seg-*/test_location_predictions.csv`
- `artifacts/runs/hybrid-*/hybrid_metrics.json`
- `artifacts/runs/hybrid-*/promotion_decision.json`
- `artifacts/predictions/hybrid-*_test_predictions.csv`
- `submissions/submission_hybrid_segmentation_20260405.csv`

## Submission Safety Guidance

- Daily Kaggle limit: 5 submissions.
- Submit only one final CSV after checking gate and metric summary.
- If gate fails, baseline fallback should be used for test predictions.

## Known Environment Caveat

If Kaggle CLI fails with missing username/key config, repair `~/.kaggle/kaggle.json` to valid JSON form:

```json
{"username": "<kaggle_username>", "key": "<kaggle_api_key>"}
```

Then run:

```bash
chmod 600 ~/.kaggle/kaggle.json
.venv/bin/python -m kaggle competitions submissions -c geohab-mlwg-competition-2026 --csv
```
