# Final Model Summary

## Selected Run
- submission_model_run_id: `baseline-20260324181702`
- rejected_candidate_run_id: `candidate-20260324181959`

## Validation Metrics (Spatial Blocked CV)
- baseline weighted_f1: 0.802627
- candidate weighted_f1: 0.786368
- candidate minus baseline: -0.016259
- threshold policy: candidate rejected (`degradation > 0.005`)

## Interpretation
- The baseline model generalizes better under the authoritative spatial-blocked protocol.
- Candidate underperforms baseline by ~1.63 percentage points weighted F1, which exceeds the allowed degradation margin.
- Baseline was therefore selected for submission generation to preserve model-selection policy compliance.

## Submission Output Validation
- output path: `submissions/submission.csv`
- row count: 98
- unique IDs: true
- columns: `ID,class`
- predicted class set: `ALG`, `FMAT`, `NVB`, `SGAM`, `SGZ`

## Predicted Class Distribution
- NVB: 46
- FMAT: 21
- ALG: 18
- SGZ: 8
- SGAM: 5

## Related Artifacts
- `reports/metrics/baseline-20260324181702_comparison.md`
- `reports/metrics/baseline-20260324181702_metrics.json`
- `reports/metrics/baseline-20260324181702_per_class.csv`
- `reports/reproducibility/baseline-20260324181702_reproducibility.md`
- `submissions/submission.csv`
