# Feature Specification: Hybrid Segmentation Ensemble

**Feature Branch**: `021-hybrid-segmentation`  
**Created**: 2026-04-05  
**Status**: Draft  
**Input**: User description: "Augment the existing RF/MLP ensemble with a low-effort image-segmentation hybrid path that is easy to run in Kaggle-style notebooks and can reuse pretrained models where practical."

## Clarifications

### Session 2026-04-05

- Q: Which exact three low-effort approaches should be benchmarked? -> A: HF SegFormer fine-tuned, HF SegFormer + CRF, local DeepLabV3+ fine-tuned.
- Q: What fixed mask expansion rule should be used for point-to-mask conversion? -> A: 5x5 pixel window centered on each labeled point.
- Q: What imbalance handling policy should be mandatory for SGAM protection? -> A: Inverse-frequency class weights with capped maximum weight.
- Q: What default meta-learner should stack RF/MLP and segmentation outputs? -> A: Multinomial logistic regression.
- Q: What is the promotion gate from baseline to hybrid? -> A: Require >=2pp weighted F1 gain and non-decreasing SGAM recall.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build Segmentation Masks (Priority: P1)

As a modeler, I can turn sparse point labels into training masks so the segmentation approach has usable supervision without manual pixel annotation.

**Why this priority**: Mask creation is the prerequisite for every other segmentation step and is the lowest-effort way to unlock the hybrid path.

**Independent Test**: Given a labeled sample set, the mask generator produces a corresponding mask for every labeled point and the output can be inspected before any model training begins.

**Acceptance Scenarios**:

1. **Given** a set of point labels, **When** masks are generated, **Then** every labeled point is expanded into a bounded local training region.
2. **Given** overlapping or adjacent labels, **When** masks are created, **Then** the resulting mask set remains consistent and traceable back to the source labels.
3. **Given** any labeled point, **When** mask generation runs, **Then** the point is expanded to a centered 5x5 pixel training window.

---

### User Story 2 - Train and Compare Low-Effort Segmentation Paths (Priority: P2)

As a modeler, I can run a notebook-friendly segmentation candidate and compare the three lowest-effort paths so I can choose the best balance of effort and gain.

**Why this priority**: The user wants the simplest viable route, not a large platform migration, so the feature should favor a pretrained notebook workflow and make the trade-off visible.

**Independent Test**: A single validation split can be used to compare a pretrained segmentation path, a spatial-smoothing variant, and a local fallback path, with each run producing comparable metrics and effort notes.

**Acceptance Scenarios**:

1. **Given** the same training and validation split, **When** each candidate path is executed, **Then** the workflow reports the same evaluation metrics for every candidate.
2. **Given** a notebook-style execution environment, **When** the preferred candidate is trained, **Then** the run completes without requiring a full port of the existing pipeline.
3. **Given** the candidate benchmark run, **When** the report is generated, **Then** it includes results for exactly SegFormer, SegFormer + CRF, and DeepLabV3+.

---

### User Story 3 - Stack Segmentation with the Existing Ensemble (Priority: P3)

As a modeler, I can combine segmentation outputs with the current RF/MLP ensemble so the new signal augments rather than replaces the existing baseline.

**Why this priority**: The main business value comes from lifting the existing ensemble while preserving a trusted fallback path.

**Independent Test**: The hybrid predictor can be run against the validation set and returns a final prediction derived from both the segmentation signal and the current ensemble outputs.

**Acceptance Scenarios**:

1. **Given** predictions from the current RF/MLP ensemble and the selected segmentation path, **When** they are combined, **Then** a single final prediction is produced for each location.
2. **Given** a minority-class case such as SGAM, **When** the hybrid model is evaluated, **Then** the report shows whether the added segmentation signal improves or preserves minority-class recall.
3. **Given** validation metrics for baseline and hybrid, **When** release readiness is assessed, **Then** hybrid promotion is allowed only if weighted F1 improves by at least 2 points and SGAM recall does not decrease.

### Edge Cases

- Point labels that are very close together must keep deterministic ownership rules when 5x5 windows overlap.
- Locations with weak or missing imagery must fall back to the existing ensemble rather than blocking the full run.
- Low-confidence segmentation outputs must not override the baseline without an explicit improvement on validation.
- Minority classes with very few examples must still be tracked separately so they do not disappear inside aggregate accuracy.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST convert labeled points into masks using a centered 5x5 pixel expansion rule with deterministic overlap handling.
- **FR-002**: The system MUST support a notebook-friendly segmentation training path that can run without moving the entire existing pipeline into the notebook environment.
- **FR-003**: The system MUST evaluate exactly three approaches: HF SegFormer fine-tuned, HF SegFormer fine-tuned with CRF smoothing, and local DeepLabV3+ fine-tuned.
- **FR-004**: The system MUST produce per-location class predictions and confidence scores from the selected segmentation approach.
- **FR-005**: The system MUST combine segmentation outputs with RF/MLP outputs using multinomial logistic regression as the default meta-learner.
- **FR-006**: The system MUST apply inverse-frequency class weights with a capped maximum per-class weight to protect rare classes such as SGAM.
- **FR-007**: The system MUST record validation metrics, runtime effort, and the recommended deployment path for each candidate approach.
- **FR-008**: The system MUST preserve RF/MLP as baseline and reject hybrid promotion unless weighted F1 improves by >=2 percentage points and SGAM recall is non-decreasing.

### Key Entities *(include if feature involves data)*

- **Segmentation Mask**: A training target derived from point labels that defines the local area used for supervision.
- **Segmentation Candidate**: One of the evaluated low-effort segmentation paths, each producing predictions and a validation score.
- **Hybrid Prediction**: The final output created by combining segmentation outputs with the existing RF/MLP ensemble.
- **Minority-Class Signal**: The tracked performance for sparse classes, especially SGAM, used to guard against regressions hidden by aggregate metrics.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of labeled training points are converted into masks with no manual pixel annotation required.
- **SC-002**: Three candidate approaches are benchmarked on the same validation split and a single ranked recommendation is produced.
- **SC-003**: Hybrid promotion only occurs when weighted F1 improves by at least 2 percentage points over RF/MLP baseline.
- **SC-004**: SGAM recall for the promoted model is non-decreasing versus RF/MLP baseline.
- **SC-005**: The recommended path can be executed in a notebook-style environment without a full migration of the current pipeline.

## Assumptions

- The existing RF/MLP ensemble remains the trusted baseline and is not replaced by the new feature.
- The first implementation uses a pretrained segmentation model rather than training a model from scratch.
- The training data and baseline predictions can be exported into a notebook-friendly format without changing their meaning.
- Kaggle-style notebook execution is the preferred low-effort environment; any broader platform port is out of scope for this feature.
- Spatial smoothing is only kept if it improves validation quality over the raw segmentation output.
- Class-weight capping values are set during planning but must be documented and reproducible.

## Out of Scope

- Panoptic segmentation, graph neural networks, and custom model architecture design.
- A full rewrite of the existing RF/MLP pipeline for notebook execution.
- Multi-task learning and contrastive pre-training unless they are later approved as follow-on work.
