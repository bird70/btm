---
date: 2026-03-27 17:36
branch: 011-clean-knn-rf-lgb
script: scripts/experiment_v8.py
builds_on: experiment_v6.py (OBIA+PB hybrid, Kaggle 0.76394)
root_cause_of_v7_failure: x_rel/y_rel geographic coords (36.6%+11.8% importance) caused spatial memorisation
---

# Run Report: Clean Features — KNN / RF / LightGBM (Experiment v8)

## What Changed vs v7

| Removed from v7 | Reason |
|-----------------|--------|
| x_rel, y_rel (spatial coords) | Caused spatial memorisation — LOWEST Kaggle score |
| depth_z, backscatter_z (z-scores) | Redundant duplicates of depth + backscatter |
| SMOTE | Creates non-physical samples for spatially autocorrelated data |
| 4 interaction terms | Collinear with inputs; added noise |
| 6 multi-scale focal stds | Replaced by single best-scale bathy_std_9, back_std_9 |
| XGBoost, CatBoost | Replaced by KNN (new) to explore acoustic similarity |

## Feature Set

| Group | Count | Features |
|-------|-------|----------|
| PB paper (Ierodiaconou 2018) | 8 | depth, backscatter, slope, vrm, complexity, max_curvature, northness, eastness |
| PB texture/TPI (new) | 3 | bathy_std_9, back_std_9, tpi_9 |
| OB SLIC segments (v6) | 10 | seg_{bathy,back,vrm}_{mean,std,skew}, seg_pixel_count |
| **Total** | **21** | no spatial coordinates |

## Segmentation

| n_segments (target) | 4000 |
| n_segments (actual) | 2654 |
| Mean size (px) | 7219.9 |
| Mean size (m²) | 451.2 |

## CV Results

Spatial GroupKFold (n_splits=10, KMeans n_clusters=10, seed=42).
No SMOTE. StandardScaler inside KNN pipeline; raw features for RF/LGB.

| Configuration | Model | CV F1 mean ± std |
|---------------|-------|-----------------|
| pb_only | rf | 0.6021 ± 0.2975 ← **BEST** |
| ob_only | rf | 0.3785 ± 0.2398 |
| combined_best_single | rf | 0.5996 ± 0.1677 |
| combined_ensemble | ensemble | 0.5992 ± 0.1532 |

## Output

- Submission: `data\submission_v8_best.csv`
- Report: `docs\run-011-clean-knn-rf-lgb.md`
