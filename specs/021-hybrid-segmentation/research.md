# Phase 0 Research: Hybrid Segmentation Ensemble

## Decision 1: Benchmark Set of Three Approaches

- Decision: Benchmark exactly three approaches: (1) HF SegFormer fine-tuned, (2) HF SegFormer fine-tuned + CRF smoothing, (3) local DeepLabV3+ fine-tuned.
- Rationale: This is the best balance of effort, reproducibility, and expected gain for benthic semantic segmentation while respecting notebook-first execution.
- Alternatives considered:
  - HF pretrained without fine-tune: rejected for domain mismatch and weak expected uplift.
  - Mask2Former panoptic path: rejected as unnecessary complexity for continuous habitat mapping.
  - GNN and contrastive pretraining: deferred due to implementation overhead.

## Decision 2: Point-to-Mask Conversion Rule

- Decision: Use centered 5x5 pixel windows for point-label expansion with deterministic overlap handling.
- Rationale: Produces dense-enough supervision quickly and aligns with low-effort constraints.
- Alternatives considered:
  - 3x3 windows: rejected due to higher sparsity and weaker contextual supervision.
  - Adaptive-radius buffers: rejected for added parameter/search complexity.

## Decision 3: Minority-Class Protection

- Decision: Use inverse-frequency class weights with capped maximum class weight during segmentation training.
- Rationale: Directly addresses SGAM under-representation without destabilizing optimization.
- Alternatives considered:
  - Focal loss only: deferred; useful but adds hyperparameter burden for initial low-effort phase.
  - Oversampling-only: rejected as primary mechanism due to spatial overfitting risk.

## Decision 4: Stacking Method

- Decision: Use multinomial logistic regression as default meta-learner for RF/MLP + segmentation signals.
- Rationale: Strong baseline for tabular meta-features, interpretable, low tuning overhead.
- Alternatives considered:
  - MLP meta-learner: deferred for second-pass optimization.
  - Gradient boosting meta-learner: deferred to avoid compounding model complexity early.

## Decision 5: Promotion Gate

- Decision: Promote hybrid only if weighted F1 improves by >=2 percentage points and SGAM recall is non-decreasing against RF/MLP baseline.
- Rationale: Ensures net gain with minority-class safety, preventing aggregate-metric regressions.
- Alternatives considered:
  - Any positive weighted F1 gain: rejected as too weak for adoption overhead.
  - SGAM recall must increase by fixed absolute threshold: rejected for brittleness at small-sample scale.

## Decision 6: Execution Environment Strategy

- Decision: Keep core pipeline in repo CLI and run segmentation benchmarking in Kaggle-style notebooks with export/import contracts.
- Rationale: Avoids full migration while maximizing practical use of available GPU runtimes and pretrained models.
- Alternatives considered:
  - Full CLI-native deep learning integration first: rejected due to higher initial delivery cost.
  - Notebook-only end-to-end pipeline: rejected due to reduced reproducibility against existing CLI workflows.
