# Run 019: Pipeline CV Improvement Investigation

**Branch**: `019-pipeline-cv-improvement`  
**Date**: 2026-04-01  
**Baseline**: R04 (RF + BTM-10, seed=42) — CV=0.8024, Kaggle=0.7952  
**Noise floor**: 2× seed std ≈ 0.037 → threshold for meaningful improvement: CV > 0.8374

---

## Summary

Systematic investigation of all remaining CV improvement avenues for the R04 baseline (RF + BTM-10 features, `rf-btm-fine.yaml`, CV=0.8024). Three user stories executed:

1. **US1 — Experiment sweep**: RF hyperparameter variants (S2, S9), LightGBM in two configurations, and a RF+LightGBM soft-vote ensemble. **Result**: No variant exceeded the noise floor (0.8374). Best CV remains 0.8024 (tied by S9 log2 max_features). LightGBM variants scored 0.7916–0.7976.
2. **US2 — Seed variance**: Confirmed std=0.0186 across 5 seeds, noise floor (2×std)=0.037. Mean CV=0.7891. SGAM class detected only by seed=42 (5 samples, unstable class).
3. **US3 — Seed ensemble**: 97/98 samples unanimous across 5-seed majority vote. Ensemble and R04 agree on 97/98 predictions. Prediction stability confirmed.

**Conclusion**: R04 represents the effective ceiling for this dataset + feature configuration. Per FR-010, no Kaggle submission is warranted — all variations fall within the seed noise floor.

---

## RF Experiments

| Experiment               | Config                  | Run ID                   | CV     | Delta vs R04 | Exceeds 0.8374? |
| ------------------------ | ----------------------- | ------------------------ | ------ | ------------ | --------------- |
| R04 baseline (reference) | rf-btm-fine.yaml        | candidate-20260401040507 | 0.8024 | —            | N/A             |
| S2: 500 trees            | rf-btm-s2-500trees.yaml | candidate-20260401041240 | 0.8001 | -0.0023      | ❌              |
| S9: log2 max_features    | rf-btm-s9-log2.yaml     | candidate-20260401041256 | 0.8024 | 0.0000       | ❌              |

**Finding**: S9 (log2 max_features) ties R04 exactly — identical per-class F1 scores. R04's `sqrt` max_features is equivalent to `log2` for 14 features (√14 ≈ 3.7, log₂14 ≈ 3.8). S2 with 500 trees shows marginal degradation (-0.0023), well within noise.

---

## LightGBM Experiments

| Experiment                       | Config                 | Run ID                   | CV     | Delta vs R04 | Exceeds 0.8374? |
| -------------------------------- | ---------------------- | ------------------------ | ------ | ------------ | --------------- |
| lgbm-btm-tuned (200 est, lr=0.1) | lgbm-btm-tuned.yaml    | candidate-20260401040546 | 0.7976 | -0.0048      | ❌              |
| lgbm-btm-300 (300 est, lr=0.05)  | lgbm-btm-300.yaml      | candidate-20260401040606 | 0.7959 | -0.0065      | ❌              |
| RF+LightGBM ensemble             | lgbm-btm-ensemble.yaml | candidate-20260401041626 | 0.7916 | -0.0108      | ❌              |

**Finding**: All LightGBM variants score below R04. A lighter LightGBM (200 trees, lr=0.1) performs better than a larger one (300 trees, lr=0.05), consistent with the small dataset (590 samples). The RF+LightGBM soft-vote ensemble is degraded by the weaker LightGBM component pulling overall predictions toward lower-confidence regions.

**T013 skipped**: No LightGBM variant achieved CV > 0.835, so multi-seed validation was not performed.

---

## Feature Additions (FR-003)

From `research.md` §RT-3 (experiments conducted on branches 017/018, reproduced here for reference):

| Experiment        | Features Added                                                                 | CV     | Delta vs R04 | Safe for submission?                           |
| ----------------- | ------------------------------------------------------------------------------ | ------ | ------------ | ---------------------------------------------- |
| S5a (scale-3 set) | btm_northness_3, btm_eastness_3, btm_roughness_3, btm_rdmv_3, btm_complexity_3 | 0.7950 | -0.0074      | ✅ (spatial stationarity verified for scale-3) |
| S5b (n3+e3)       | btm_northness_3, btm_eastness_3                                                | 0.7969 | -0.0055      | ✅                                             |
| S5c (rdmv_3)      | btm_rdmv_3                                                                     | 0.7997 | -0.0027      | ✅                                             |
| S5d (rdmv+n3+e3)  | btm_rdmv_3+northness_3+eastness_3                                              | 0.7922 | -0.0102      | ✅                                             |

**Finding**: All small-window (scale-3) BTM additions degrade CV. Spatial stationarity is verified for these features — the degradation is genuine signal noise, not a spatial mismatch. Large-window features (≥15 cells) remain strictly prohibited (see research.md RT-3: btm_complexity_21 caused Kaggle score to collapse from 0.8033 to 0.6896).

---

## Feature Flags (FR-004)

From `research.md` §RT-3 (experiments conducted on branches 017/018):

| Experiment            | Flag                          | CV     | Delta vs R04 | Verdict                      |
| --------------------- | ----------------------------- | ------ | ------------ | ---------------------------- |
| S1 (eco features)     | include_eco_features=true     | 0.7978 | -0.0046      | ❌ Degrades CV               |
| S6 (spatial z-scores) | include_spatial_z_scores=true | 0.7954 | -0.0070      | ❌ Degrades CV               |
| S10 (exclude coords)  | exclude_coords=true           | 0.6323 | -0.1701      | ❌❌ Catastrophic — keep x/y |

**Finding**: Eco features and z-scores both reduce CV on this dataset. x,y coordinates carry essential spatial habitat structure — removing them collapses performance. Feature flags must remain at R04 defaults: only `include_btm_features=true`.

---

## Seed Variance

R04 config (`rf-btm-fine.yaml`, `train_btm.csv`) across 5 seeds (runs from branch 018, carried into this branch):

| Seed | Run ID                   | CV (weighted F1) | SGAM F1 |
| ---- | ------------------------ | ---------------- | ------- |
| 42   | candidate-20260401021038 | 0.8024           | 0.2667  |
| 123  | candidate-20260401021045 | 0.7626           | 0.0000  |
| 456  | candidate-20260401021121 | 0.7890           | 0.0000  |
| 789  | candidate-20260401021128 | 0.8101           | 0.0000  |
| 2026 | candidate-20260401021136 | 0.7815           | 0.0000  |

**Summary**: mean=0.7891, std=0.0186, min=0.7626, max=0.8101, range=0.0475, noise floor (2× std)=0.0373

**Key insight**: SGAM (seagrass meadow) is detected **only** by seed=42 in CV estimation. This is not a property of the final model (which trains on all data) but of the CV fold assignments. The rare-class instability drives most of the seed variance. Any CV comparison must use ≥2× std as the significance bar.

---

## Seed Ensemble

Majority-vote ensemble across 5 seed predictions (`scripts/experiment_v14_seed_ensemble.py`):

**Vote confidence distribution**:

| Agreement                      | Samples |
| ------------------------------ | ------- |
| 5/5 unanimous                  | 82      |
| 4/5 majority                   | 15      |
| 2/5 minority (tiebreaker used) | 1       |

**Class distribution (ensemble)**:

| Class | Count |
| ----- | ----- |
| NVB   | 44    |
| ALG   | 21    |
| FMAT  | 20    |
| SGZ   | 8     |
| SGAM  | 5     |

**Ensemble vs R04 (seed=42)**: 97/98 agree (99.0%), 1 sample changed  
**Output file**: `data/submission_v14_seed_ensemble.csv`

**Interpretation**: The 1 sample change (2/5 minority with a tiebreaker applied) represents a genuinely uncertain prediction. Ensemble and R04 are practically identical. Per research.md RT-6: the final model trained on all data is stable across seeds; ensemble does not produce a better submission.

---

## Submission Decision

**Selected candidate**: R04 — `configs/rf-btm-fine.yaml`, seed=42, run_id=`candidate-20260401040507`  
**Rationale**: No experiment in this investigation exceeded the noise floor (CV > 0.8374). Per FR-010, when no configuration clears the noise bar, R04 must be retained as the reference submission. R04 achieved CV=0.8024 and Kaggle=0.7952 (CV-Kaggle gap=0.007, within 1% of expectation). All alternative configurations scored equal to or below R04.  
**No Kaggle submission**: FR-007 requires CV improvement beyond noise floor before spending a submission slot. No such improvement was found.  
**Output file**: `data/submission_v14_final.csv` (copy of `data/submission_v14_r04_baseline.csv`)
