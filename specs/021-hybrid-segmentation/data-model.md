# Phase 1 Data Model: Hybrid Segmentation Ensemble

## Entity: SegmentationMaskSet

- Description: Training supervision artifact generated from point labels and raster coordinates.
- Fields:
  - mask_set_id: string (unique)
  - source_points_path: string
  - raster_grid_id: string
  - expansion_rule: enum (`centered_5x5`)
  - overlap_policy: enum (`deterministic_priority`)
  - class_index_map: object (class -> integer id)
  - output_mask_path: string
  - created_at: datetime
- Validation rules:
  - expansion_rule must equal `centered_5x5` for this feature.
  - every labeled point must map to at least one positive mask pixel.
  - mask labels must be within known class index map.

## Entity: SegmentationCandidateRun

- Description: One trained/evaluated candidate in the 3-approach benchmark.
- Fields:
  - run_id: string (unique)
  - candidate_type: enum (`segformer_ft`, `segformer_ft_crf`, `deeplabv3_ft`)
  - train_split_id: string
  - val_split_id: string
  - class_weight_strategy: enum (`inverse_freq_capped`)
  - class_weight_cap: float
  - epochs: integer
  - checkpoint_uri: string
  - metrics_json_path: string
  - weighted_f1: float
  - macro_f1: float
  - sgam_recall: float
  - runtime_minutes: float
  - status: enum (`success`, `failed`, `rejected`)
- Validation rules:
  - candidate_type must be one of exactly three allowed values.
  - weighted_f1 and sgam_recall are required for status `success`.
  - runtime_minutes must be non-negative.

## Entity: PerLocationSegmentationOutput

- Description: Segmentation-derived features aligned to sample locations for stacking.
- Fields:
  - location_id: string
  - run_id: string (FK -> SegmentationCandidateRun)
  - predicted_class: string
  - confidence: float
  - logits_or_probs: array<float>
  - optional_backbone_embedding: array<float> | null
- Validation rules:
  - confidence range: [0.0, 1.0].
  - predicted_class must exist in class index map.
  - one output record per location_id per run_id.

## Entity: HybridStackingDataset

- Description: Meta-feature dataset combining RF, MLP, and segmentation signals.
- Fields:
  - dataset_id: string
  - split: enum (`train_meta`, `val_meta`, `test_meta`)
  - feature_matrix_path: string
  - label_vector_path: string | null
  - columns:
    - rf_prob_* (per class)
    - mlp_prob_* (per class)
    - seg_prob_* (per class)
    - seg_confidence
  - created_at: datetime
- Validation rules:
  - class probability columns must be complete and aligned across models.
  - no NaN values in required meta-features.
  - label_vector required for train_meta and val_meta.

## Entity: HybridPromotionDecision

- Description: Decision artifact for adopting or rejecting hybrid model.
- Fields:
  - decision_id: string
  - baseline_weighted_f1: float
  - hybrid_weighted_f1: float
  - baseline_sgam_recall: float
  - hybrid_sgam_recall: float
  - weighted_f1_delta: float
  - sgam_recall_delta: float
  - gate_passed: boolean
  - decision_reason: string
  - generated_at: datetime
- Validation rules:
  - gate_passed true only if weighted_f1_delta >= 0.02 and sgam_recall_delta >= 0.0.
  - decision_reason must be non-empty.

## Relationships

- SegmentationMaskSet 1 -> many SegmentationCandidateRun
- SegmentationCandidateRun 1 -> many PerLocationSegmentationOutput
- PerLocationSegmentationOutput many -> 1 HybridStackingDataset (by split)
- HybridStackingDataset 1 -> 1 HybridPromotionDecision (validation stage)

## State Transitions

- SegmentationCandidateRun: `failed` -> `success` not allowed; rerun creates new run_id.
- SegmentationCandidateRun: `success` -> `rejected` allowed when promotion gate fails.
- HybridPromotionDecision: `gate_passed=false` implies baseline retained; `gate_passed=true` implies hybrid eligible for submission path.
