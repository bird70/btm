# Run 021: Hybrid Segmentation Ensemble

**Date**: 2026-04-05  
**Branch**: `021-hybrid-segmentation`

## Summary

Implemented and executed a complete hybrid segmentation workflow with three candidate approaches, plus meta-stacking and promotion gating.

## Executed Runs

- Mask build run: `artifacts/segmentation/mask-run-20260405`
- Segmentation benchmark run: `seg-20260405042554`
- Hybrid stack run: `hybrid-20260405042701`

## Candidate Benchmark Results

From `artifacts/segmentation/seg-20260405042554/candidate_metrics.json`:

| Candidate | Weighted F1 | Macro F1 | SGAM Recall | Runtime (min) | Status |
|---|---:|---:|---:|---:|---|
| segformer_ft | 0.7447 | 0.6915 | 1.0000 | 0.0030 | success |
| segformer_ft_crf | 0.7433 | 0.6902 | 1.0000 | 0.0042 | success |
| deeplabv3_ft | 0.9904 | 0.9876 | 1.0000 | 0.0410 | success |

Recommended candidate: `deeplabv3_ft`.

## Hybrid Promotion Gate

From `artifacts/runs/hybrid-20260405042701/promotion_decision.json`:

- Baseline weighted F1: 0.9904
- Hybrid weighted F1: 0.9912
- Weighted F1 delta: 0.0008
- Baseline SGAM recall: 1.0000
- Hybrid SGAM recall: 1.0000
- SGAM recall delta: 0.0000
- Gate passed: **false**

### Interpretation

Gate criteria require weighted F1 delta >= 0.02 and non-decreasing SGAM recall. SGAM condition passed, weighted F1 delta did not. Baseline fallback retained for final predictions.

## Final Submission Artifact

- `submissions/submission_hybrid_segmentation_20260405.csv`

This CSV was produced from gate-governed final predictions and is ready for Kaggle upload once local API credentials are valid.

## Kaggle Submission Status

Automated upload was blocked locally by invalid Kaggle credential configuration (`~/.kaggle/kaggle.json` not valid JSON and missing username/key fields). No quota was consumed.

## Recommended Next Step

Repair Kaggle API credentials and submit one final CSV:

```bash
.venv/bin/python -m kaggle competitions submit \
  -c geohab-mlwg-competition-2026 \
  -f submissions/submission_hybrid_segmentation_20260405.csv \
  -m "021-hybrid-segmentation gate-failed baseline-retained"
```
