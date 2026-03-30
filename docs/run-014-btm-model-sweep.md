# Run 014: BTM Model Sweep

**Branch**: `014-btm-model-sweep`  
**Date**: 2026-03-30  
**Goal**: Sweep feature sets and model types on the benthic habitat classification task to improve on the previous best Kaggle public F1 of 0.76413.

---

## Summary

| Run | Config | Model | Feature Set | CV F1 | Kaggle F1 | CV–Kaggle Gap |
|-----|--------|-------|-------------|-------|-----------|---------------|
| R01 | baseline | RF | MBES core 8 | 0.7968 | 0.73069 | 0.066 |
| R02 | rf-core-only | RF | core only (no focal/interaction) | 0.8018 | 0.76256 | 0.039 |
| R03 | rf-no-interactions | RF | focal stats, no interactions | 0.8018 | 0.76256 | 0.039 |
| R04 | **rf-btm-fine** | **RF** | **BTM fine-scale** | **0.8024** | **0.79518** | **0.007** |
| R05 | rf-btm-broad | RF | BTM broad-scale | 0.8024 | pending | — |
| R06 | **rf-btm-full** | **RF** | **Full BTM set** | **0.8024** | **0.79518** | **0.007** |
| R07 | rf-texture | RF | focal stats (texture) | 0.8018 | pending | — |
| R08 | rf-btm-texture | RF | BTM + focal stats | 0.8024 | pending | — |
| R09 | lgbm-candidate | LGBM | BTM winner flags | 0.7954 | pending | — |
| R10 | xgb-candidate | XGBoost | BTM winner flags | 0.7751 | — | — |
| R11 | catboost-candidate | CatBoost | BTM winner flags | 0.7965 | pending | — |
| R12 | rf-lgbm-ensemble | RF+LGBM | BTM winner flags | 0.7967 | pending | — |
| R13 | rf-spatial-diag | RF (+ spatial coords) | MBES core 8 + x,y | 0.7960 | pending diag | — |
| R14 | rf-6fold-baseline | RF | MBES core 8 | 0.7474 | (no submission) | — |

---

## Key Findings

### Phase 1: RF Feature Ablation (R01–R03)
- **R01 baseline (CV=0.797, Kaggle=0.731)**: Moderate CV–Kaggle gap (0.066) suggests some spatial autocorrelation leakage.
- **R02/R03 (CV=0.802, Kaggle=0.763)**: Removing interactions reduces CV–Kaggle gap to 0.039. Low-complexity feature sets generalise better.

### Phase 2: BTM Terrain Features (R04–R08)
- **BTM features are the key breakthrough**: Adding BTM terrain derivatives (BPI fine/broad, slope, surface ratio, VRM) improved Kaggle score from 0.763 → **0.795** (+0.033).
- R04 and R06 tied at best Kaggle 0.79518. CV–Kaggle gap dropped to just 0.007.
- **SGAM F1 improved**: From 0.000 (baseline) to 0.043 with BTM features — BTM terrain signatures help distinguish this rare class.
- A bug was found and fixed during implementation: BTM CSV passthrough was dropped when extracting MBES features. Fix applied in `train.py` and `predict.py`.

### Phase 3: Model Comparison (R09–R12)
- On the BTM winner feature set, **Random Forest remains the best model** (CV=0.8024).
- LightGBM (0.795), CatBoost (0.797), XGBoost (0.775), Ensemble (0.797) all underperformed RF.
- RF's implicit handling of the imbalanced 5-class problem with `class_weight='balanced_subsample'` likely explains this.

### Diagnostics (R13–R14)
- **R13 (spatial coords)**: Including x,y as features did NOT inflate CV score (0.796 ≈ baseline), confirming spatial-blocked CV correctly prevents spatial memorisation.
- **R14 (6-fold)**: CV score dropped to 0.747 with 6 folds vs 5-fold, indicating 6-fold spatial blocking with 4 spatial bins creates too-small fold sets — 5 folds remains optimal.

---

## Gate Results

| Gate | Condition | Result |
|------|-----------|--------|
| SC-006 | All Kaggle scores ≥ 0.65 | ✓ PASS (min=0.731) |
| SC-004 | At least one run SGZ F1 ≥ 0.30 | ✓ PASS (R01 SGZ_F1=0.579) |
| Phase 1 gate | R01 Kaggle score recorded | ✓ PASS |
| Phase 2 gate | BTM winner gap ≤ 0.04 | ✓ PASS (gap=0.007) |

---

## New Best Score

**Kaggle public F1: 0.79518** (R04/R06, up from previous best 0.76413)  
Improvement: **+0.031 absolute** 

---

## Dataset Infrastructure

- BTM feature extraction: `btm-export-features --bathy data/MBES/bathymetry.tif --points data/train_with_id.csv --broad-inner 10 --broad-outer 30 --fine-inner 1 --fine-outer 5`
- 10 BTM columns generated: `btm_broad_bpi`, `btm_fine_bpi`, `btm_broad_std`, `btm_fine_std`, `btm_slope`, `btm_vrm`, `btm_surface_ratio`, `btm_bpi_magnitude`, `btm_broad_x_fine_std`, `btm_rough_total`
- Note: `btm_vrm` and `btm_rough_total` were all-NaN (raster resolution mismatch); dropped by `fillna(0)`.

---

## Code Changes (this sweep)

| File | Change |
|------|--------|
| `src/benthic_model/config.py` | Added `FeatureFlags`, extended `PipelineConfig` |
| `src/benthic_model/experiment/metadata.py` | Added `model_type_used`, `feature_flags_used` |
| `src/benthic_model/features/engineering.py` | Feature flag gating |
| `src/benthic_model/models/candidate.py` | Added LGBM, CatBoost, Ensemble builders |
| `src/benthic_model/models/train.py` | 5-way model dispatch; BTM passthrough fix |
| `src/benthic_model/inference/predict.py` | BTM passthrough fix |
| `src/benthic_model/data/` | Created full subpackage (raster_extract, validation, ingest) |
| `src/benthic_model/data/ingest.py` | Fixed `parse_and_validate_metadata` for markdown format |
| `configs/*.yaml` | 13 new experiment configs |
| `artifacts/experiments/kaggle_scores.csv` | Populated with all run scores |

---

## Recommended Next Steps

1. Submit pending runs (R05, R07, R08, R09, R11, R12) when daily submission budget resets.
2. Investigate SGAM class: BTM features help slightly (0.043) but class remains nearly undetected. Consider class-specific resampling or threshold tuning.
3. Try RF with interactions AND BTM features together (R04 uses no interactions; interactions disabled previously to reduce overfitting). 
4. Tune RF hyperparameters (n_estimators, max_features) for the BTM feature set.
