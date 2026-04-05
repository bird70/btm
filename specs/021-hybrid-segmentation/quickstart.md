# Quickstart: Hybrid Segmentation Ensemble (Python CLI + Kaggle/HF/GH)

## 1. Environment Setup

```bash
cd /Users/tilmann/Documents/GitHub/btm

# Create venv if absent
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev,benthic]"
pip install torch transformers datasets evaluate huggingface_hub pydensecrf kaggle
```

## 2. Authenticate Tooling

```bash
# Kaggle API
kaggle config view
# If needed: place kaggle.json under ~/.kaggle and chmod 600 ~/.kaggle/kaggle.json

# Hugging Face
hf auth login

# GitHub CLI
gh auth status
# If needed: gh auth login
```

## 3. Run Baseline RF/MLP Pipeline

```bash
PYTHONPATH=src python -m benthic_model.cli train \
  --train-csv data/train.csv \
  --bathymetry-tif data/MBES/bathymetry.tif \
  --backscatter-tif data/MBES/backscatter.tif \
  --config configs/rf-btm-fine.yaml \
  --run-type candidate \
  --seed 42
```

Capture the generated `<baseline_run_id>` for downstream stacking.

## 4. Create Segmentation Masks (5x5 expansion)

- Convert point labels to pixel-aligned supervision masks.
- Apply centered 5x5 expansion per labeled point.
- Persist mask artifacts with deterministic overlap policy.

Expected output (planned):

- `artifacts/segmentation/<seg_run_id>/masks/`
- `artifacts/segmentation/<seg_run_id>/mask_metadata.json`

## 5. Benchmark 3 Segmentation Candidates

Run and compare exactly:

1. `segformer_ft`
2. `segformer_ft_crf`
3. `deeplabv3_ft`

Notebook or CLI execution is allowed as long as contract artifacts are produced.

Expected output:

- `artifacts/segmentation/<seg_run_id>/candidate_metrics.json`
- `artifacts/segmentation/<seg_run_id>/val_location_predictions.csv`
- `artifacts/segmentation/<seg_run_id>/test_location_predictions.csv`

## 6. Stack with RF/MLP and Evaluate Gate

- Build meta-features: RF probabilities + MLP probabilities + segmentation probabilities + segmentation confidence.
- Train multinomial logistic regression meta-learner on validation split.
- Compute promotion gate:
  - weighted F1 delta >= 0.02
  - SGAM recall delta >= 0.0

Expected output:

- `artifacts/runs/<hybrid_run_id>/hybrid_metrics.json`
- `artifacts/runs/<hybrid_run_id>/promotion_decision.json`

## 7. Submission Path

If gate passes:

```bash
PYTHONPATH=src python -m benthic_model.cli make-submission \
  --predictions artifacts/predictions/<hybrid_run_id>_test_predictions.csv \
  --sample-submission data/sample_submission.csv \
  --output submissions/submission_hybrid_segmentation.csv
```

If gate fails, keep RF/MLP baseline submission path unchanged.

## 8. Minimal Verification Checklist

- All three candidates reported in `candidate_metrics.json`
- Validation metrics include weighted F1 and SGAM recall
- Promotion decision artifact exists and is reproducible
- Baseline fallback remains runnable end-to-end
