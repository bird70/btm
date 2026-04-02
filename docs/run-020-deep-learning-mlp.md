# Run 020: Deep Learning (MLP) Investigation

**Branch**: `020-deep-learning-mlp`  
**Date**: 2026-04-02  
**Baseline**: R04 (RF + BTM-14, seed=42) — CV=0.8024, Kaggle=0.7952  
**Prior investigation**: 019 established RF noise floor (5-seed std=0.0186) and that all RF/LightGBM variants fall below the noise bar.

---

## Summary

First application of deep learning to this benthic habitat classification pipeline. Used sklearn's `MLPClassifier` (multi-layer perceptron, backpropagation, Adam optimiser, L2 regularisation) integrated into the existing pipeline as a new `model_type: mlp`. A hybrid `rf_mlp_ensemble` (RF + MLP soft-vote) was also implemented.

**Results**: MLP achieves CV=0.8156 at seed=42 (+0.013 over R04), but 5-seed mean is only 0.7955 (+0.006 over R04 mean 0.7891), below the statistical noise floor. Notable finding: MLP learns to detect SGAM (rare seagrass class) in 4/5 CV seeds vs RF's 1/5, suggesting a different and more SGAM-sensitive decision surface. However, both MLP and RF+MLP ensemble predict 0 SGAM in the test set (vs R04's 5), indicating the SGAM boundary does not generalise out-of-distribution.

**Conclusion**: No Kaggle submission (improvement below noise floor at mean level; test set SGAM detection degrades). DL implementation complete and production-ready for future experiments.

---

## Design Decisions

### Why sklearn MLP instead of PyTorch/TensorFlow

- No external DL framework installed; sklearn's `MLPClassifier` is available and is a genuine DL model (backpropagation, Adam, multiple layers)
- 6,256 samples × 14 tabular features: this is the right scale for sklearn MLP
- PyTorch/TensorFlow would add 300MB+ dependency overhead with no architectural benefit for a 14-feature tabular problem
- TabNet, FT-Transformer (PyTorch) offer no advantage over a well-regularised MLP for input dimensions this small

### Why two DL approaches

1. **Standalone MLP** — tests whether DL alone can improve over RF on this feature set
2. **RF + MLP soft-vote ensemble** — tests whether model diversification (tree partitions + smooth boundaries) helps

### MLP architecture rationale

- `(128, 64, 32)` hidden units: narrow-to-wide funnel appropriate for 14 input features
- ReLU activation, Adam optimiser: standard efficient choice
- `alpha=0.001` L2 regularisation: prevents overfitting on ~6k samples
- Early stopping (validation_fraction=0.1, n_iter_no_change=15): auto-stops at generalisation optimum
- `StandardScaler` preprocessing: mandatory — MLP is sensitive to feature scale; RF is not
- Balanced sample weighting via `compute_sample_weight("balanced")`: compensates for SGAM imbalance (2.7% of data)
- `LabelEncoder` internally: sklearn 1.8 MLPClassifier requires numeric labels for early-stopping validation

---

## Model Type: `mlp`

New model type added to `benthic_model.models.candidate.CandidateMLPModel`.

**Architecture**:
```
Input(14)
→ Dense(128, relu) → Dense(64, relu) → Dense(32, relu)
→ Dense(5, softmax)
```

**Training**: Adam, lr=0.001, L2=0.001, max_iter=500, early stopping (val_fraction=0.1, patience=15)

**Preprocessing**: StandardScaler (fit on training fold, applied to both train and val/test)

**Class weighting**: `compute_sample_weight("balanced")` applied per-fold

---

## Model Type: `rf_mlp_ensemble`

New model type `CandidateRFMLPEnsembleModel`: soft-vote (50% RF + 50% MLP probability average).

Class probability columns are re-ordered to match RF class ordering before averaging.

---

## Experiment Results

### MLP standalone (mlp-btm.yaml — 128, 64, 32)

| Seed | Run ID | CV | SGAM F1 |
|------|--------|----|---------|
| 42   | candidate-20260402033032 | 0.8156 | 0.3289 |
| 123  | candidate-20260402033311 | 0.7827 | 0.1842 |
| 456  | candidate-20260402033331 | 0.8132 | 0.1646 |
| 789  | candidate-20260402033356 | 0.7959 | 0.0000 |
| 2026 | candidate-20260402033420 | 0.7703 | 0.0038 |

**mean=0.7955, std=0.0195, noise floor (2×std)=0.039**

### Wide MLP (mlp-btm-wide.yaml — 256, 128, 64, 32)

| Seed | Run ID | CV | SGAM F1 |
|------|--------|----|---------|
| 42   | candidate-20260402033142 | 0.8164 | 0.0000 |

Higher overall CV but SGAM collapses to 0. Wider network overfits SGAM boundary; less suitable.

### RF + MLP ensemble (rf-mlp-ensemble.yaml)

| Seed | Run ID | CV | SGAM F1 |
|------|--------|----|---------|
| 42   | candidate-20260402033213 | 0.8152 | 0.0891 |

Partial SGAM improvement (RF anchors SGAM at low recall, MLP boosts somewhat).

---

## Comparison: MLP vs R04

| Metric | R04 (RF) | MLP (128,64,32) | Delta |
|--------|----------|-----------------|-------|
| 5-seed mean CV | 0.7891 | 0.7955 | +0.0064 |
| 5-seed std | 0.0186 | 0.0195 | ≈ same |
| Noise floor (2×std) | 0.0373 | 0.0390 | — |
| SGAM detected (seeds) | 1/5 | 4/5 | ↑ robustly |
| Best single-seed CV | 0.8101 (789) | 0.8156 (42) | +0.0055 |

**Key finding**: Mean improvement (+0.006) is below the noise floor (~0.038). Not statistically distinguishable. But SGAM detection is qualitatively more robust across seeds.

---

## Test Set Prediction Analysis

| Model | Changed vs R04 | SGAM predictions |
|-------|----------------|-----------------|
| R04 (reference) | — | 5 |
| MLP standalone | 30/98 (30.6%) | 0 |
| RF+MLP ensemble | 25/98 (25.5%) | 0 |

**Critical observation**: Both DL models predict 0 SGAM in the test set, despite learning SGAM boundaries in training CV. The SGAM boundary learned by MLP is specific to the training fold distribution and does not generalise to the test spatial block. This is the same spatial non-stationarity pattern seen with btm_complexity_21 in run-018 (but milder — no overall degradation, just SGAM loss).

---

## Submission Decision

**No Kaggle submission** for this investigation:
1. Mean CV improvement (+0.006) is below the noise floor (0.038)
2. MLP predicts 0 SGAM on test set — likely to lose SGAM F1 that R04 captures
3. Risk profile: changing 30% of predictions with uncertain net effect is not justified with limited submission budget
4. **R04 remains the recommended submission candidate**

---

## Technical Notes

### Changes to pipeline codebase

| File | Change |
|------|--------|
| `src/benthic_model/models/candidate.py` | Added `CandidateMLPModel`, `CandidateRFMLPEnsembleModel`, `build_mlp_model`, `build_rf_mlp_ensemble_model` |
| `src/benthic_model/models/train.py` | Added `mlp`, `rf_mlp_ensemble` to dispatch + `model_params` forwarding for MLP |
| `src/benthic_model/config.py` | Added `mlp`, `rf_mlp_ensemble` to `_ALLOWED_MODEL_TYPES` |
| `configs/mlp-btm.yaml` | Standalone MLP config |
| `configs/mlp-btm-wide.yaml` | Wide MLP config |
| `configs/rf-mlp-ensemble.yaml` | RF + MLP ensemble config |
| `tests/unit/test_mlp_models.py` | 14 unit tests for new model types |

### Known limitations

- `MLPClassifier` with `early_stopping=True` requires numeric labels (sklearn 1.8 regression); solved with internal `LabelEncoder`
- `sample_weight` in `MLPClassifier.fit()` available since sklearn 1.3 (confirmed on sklearn 1.8.0 ✓)
- `StandardScaler` is fit per CV fold (correct); same scaler is reused for the final model in production

### Future DL directions (if PyTorch is installed)

If deeper DL exploration is warranted, install `torch` + `pytorch-tabnet`:
- **TabNet**: attention-based feature selection, interpretable, 5k–100k sample range
- **FT-Transformer**: feature tokenization + transformer, competitive with GBDT on tabular data
- **Semi-supervised MLP**: use the 98 test samples for self-supervised pre-training, then fine-tune on labelled data

Given the current evidence (+0.006 mean improvement), the expected gain from these more complex architectures is uncertain. The fundamental limit appears to be the spatial non-stationarity of SGAM detection across train/test blocks, not model capacity.
