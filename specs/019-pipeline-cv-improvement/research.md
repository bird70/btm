# Research: Pipeline CV Improvement Investigation

**Feature**: 019-pipeline-cv-improvement  
**Date**: 2026-04-01  
**Status**: Complete (based on prior experimental results from branches 017/018)

## Research Tasks

### RT-1: Seed variance magnitude for R04 configuration

**Decision**: Seed variance is ±0.02 (std ≈ 0.017) for the R04 config on this dataset.

**Rationale**: Five seeds tested with identical config (rf-btm-fine.yaml) and data (train_btm.csv):

| Seed | CV (weighted F1) | SGAM F1 |
| ---- | ---------------- | ------- |
| 42   | 0.8024           | 0.2667  |
| 123  | 0.7626           | 0.0000  |
| 456  | 0.7890           | 0.0000  |
| 789  | 0.8101           | 0.0000  |
| 2026 | 0.7815           | 0.0000  |

- **Mean**: 0.7891, **Std**: 0.0174, **Range**: 0.7626–0.8101
- **Noise floor** (2× std): 0.035 — a candidate must exceed R04 by >0.035 to be statistically distinguishable
- **SGAM class**: Only detected by seed=42; all other seeds score 0.0 for SGAM — this rare class (5/590 samples) is unstable

**Alternatives considered**: Bootstrap CV, repeated k-fold — rejected as computationally expensive and the 5-seed approach already demonstrates the variance magnitude clearly.

### RT-2: Hyperparameter sensitivity of RF on BTM-10 features

**Decision**: R04 hyperparameters are near-optimal; no single-parameter change exceeds the noise floor.

**Rationale**: Systematic sweep results (all seed=42, train_btm.csv):

| Experiment        | Change from R04                      | CV     | Delta vs R04 |
| ----------------- | ------------------------------------ | ------ | ------------ |
| S2 (500 trees)    | n_estimators=500                     | 0.8001 | -0.0023      |
| S3 (deep)         | max_depth=20, n_estimators=500       | 0.8003 | -0.0021      |
| S4 (leaf=1)       | min_samples_leaf=1, n_estimators=500 | 0.8014 | -0.0010      |
| S8 (balanced_sub) | class_weight=balanced_subsample      | 0.7971 | -0.0053      |
| S9 (log2)         | max_features=log2                    | 0.8024 | ±0.0000      |
| S12 (1000 trees)  | n_estimators=1000                    | 0.8005 | -0.0019      |

All deltas are well within the ±0.02 seed noise — none are meaningful improvements.

**Alternatives considered**: Bayesian hyperparameter optimization — rejected as the parameter space is small and exhaustive grid results show a flat loss surface.

### RT-3: Feature additions (small-window BTM, eco, z-scores)

**Decision**: No tested feature addition improves CV beyond the noise floor.

**Rationale**:

| Experiment         | Features Added                    | CV     | Delta vs R04 |
| ------------------ | --------------------------------- | ------ | ------------ |
| S1 (eco)           | include_eco_features=true         | 0.7978 | -0.0046      |
| S6 (z-scores)      | include_spatial_z_scores=true     | 0.7954 | -0.0070      |
| S5a (scale-3 safe) | 5 scale-3 BTM features            | 0.7950 | -0.0074      |
| S5b (n3+e3)        | btm_northness_3, btm_eastness_3   | 0.7969 | -0.0055      |
| S5c (rdmv_3)       | btm_rdmv_3                        | 0.7997 | -0.0027      |
| S5d (rdmv+n3+e3)   | btm_rdmv_3+northness_3+eastness_3 | 0.7922 | -0.0102      |

All additions degrade CV. Additional features introduce noise on this small dataset (590 samples, 5 classes).

**Alternatives considered**: Large-window BTM features (window ≥15) — **explicitly rejected** due to catastrophic spatial non-stationarity (btm_complexity_21: CV=0.8033 → Kaggle=0.6896, train mean=9.2 vs test mean=13.4).

### RT-4: Coordinate importance (x, y features)

**Decision**: x,y coordinates carry genuine spatial signal and MUST NOT be removed.

**Rationale**: S10 (exclude_coords=true) → CV=0.6323, a massive -0.1701 drop. The spatial coordinates encode geographic habitat structure that the model correctly leverages.

**Alternatives considered**: Spatial coordinate transformations (polar, distance-to-centroid) — not tested, unlikely to improve on raw coordinates given the small dataset.

### RT-5: LightGBM as alternative model type

**Decision**: LightGBM should be re-evaluated with the BTM-10 feature set. Prior R09 run using lgbm-candidate.yaml achieved CV=0.7664 (seed=42), but LightGBM hyperparameters were not optimized.

**Rationale**: The existing `CandidateLGBMModel` uses n_estimators=600, learning_rate=0.03, num_leaves=63. These defaults may not be optimal for a 14-feature, 590-sample dataset. A lighter configuration (fewer estimators, larger learning rate, fewer leaves) may generalize better.

**Alternatives considered**: XGBoost — already tested as the default candidate model but excluded from scope per clarification. CatBoost — excluded due to demonstrated overfitting (CV=0.8139, Kaggle=0.76153).

### RT-6: Seed ensemble prediction stability

**Decision**: Majority voting across 5 seed models produces predictions identical to any single seed's final model.

**Rationale**: The final model is trained on ALL data (not on CV folds), and RandomForest with 300 trees is sufficiently stable that the seed only affects CV fold assignments and per-fold training. The 5 seed predictions were compared: 97/98 samples were unanimous (5/5 agreement), 1 sample had 3/5 agreement. The ensemble matched R04's predictions 100% (98/98 identical).

**Key insight**: Seed variance affects CV score estimation but NOT the final trained model's predictions. The ensemble approach does not produce a different submission — it only confirms prediction stability.

**Alternatives considered**: Probability-weighted ensemble (averaging predicted probabilities) — would require modifying the predict pipeline to output probabilities instead of class labels. Deferred as the hard-vote ensemble already shows no prediction diversity.

### RT-7: Cross-validation scheme sensitivity

**Decision**: Spatial blocked CV is the only valid evaluation scheme. Stratified random CV massively overestimates performance (0.9732 vs 0.8024) due to spatial leakage.

**Rationale**:

| CV Scheme                  | CV Score | Notes                                       |
| -------------------------- | -------- | ------------------------------------------- |
| Spatial blocked (5-fold)   | 0.8024   | Validated against Kaggle=0.7952 (gap=0.007) |
| Spatial blocked (10-fold)  | 0.8300   | Same model, less pessimistic estimate       |
| Stratified random (5-fold) | 0.9732   | Massive spatial leakage — INVALID           |

10-fold gives a higher CV estimate but produces the same model. It cannot be used to claim improvement.

**Alternatives considered**: Leave-one-block-out, spatial buffer CV — not available in the current pipeline.

## Summary of Findings

1. **R04 is near-optimal** for the RF model type on BTM-10 features with this dataset size
2. **Seed variance (±0.02)** means no single-seed CV comparison under 0.035 improvement is meaningful
3. **All tested variations** (hyperparameters, features, class weighting) fall within the noise floor
4. **Seed ensembling** does not change predictions because the final model (trained on all data) is stable across seeds
5. **LightGBM** is the remaining unexplored avenue — requires hyperparameter tuning for the small dataset
6. **Large-window BTM features** are strictly prohibited for submission
7. **The path to improvement** likely requires either (a) optimized LightGBM achieving genuinely higher CV, or (b) a probability-level RF+LightGBM ensemble, or (c) accepting R04 as the ceiling for this feature/sample configuration
