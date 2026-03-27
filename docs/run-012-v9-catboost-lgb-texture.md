---
date: 2026-03-27 18:07
branch: 012-v9-catboost-lgb-texture
script: scripts/experiment_v9.py
v6_kaggle: 0.76394  (CatBoost combined 18 features, CV=0.5758)
v8_kaggle: ~0.47    (RF pb-only, spatial memorisation failure)
fix: returned to CatBoost+LGB on combined features; added 3 texture features
---

# Run Report: CatBoost+LGB + Texture Features (Experiment v9)

## Changes vs v6

| Added | Reason |
|-------|--------|
| bathy_std_9 | Bathymetric roughness — distinguishes rocky vs sediment substrate |
| back_std_9  | Acoustic texture — local backscatter variance |
| tpi_9       | Topographic position — ridge vs hollow vs slope |
| CatBoost depth=7, iterations=800 | More capacity (was depth=6, iterations=400) |
| LGB: lr=0.03, n_est=600 | Companion repo tuned params (was lr=0.05, n_est=400) |
| Soft-vote CatBoost+LGB ensemble | New 4th candidate |

| Removed | Reason |
|---------|--------|
| XGBoost | Slowest; CatBoost consistently outperformed it on this data |
| x_rel, y_rel, z-scores | Caused spatial memorisation in v7 (0.47 Kaggle) |
| SMOTE | Non-physical samples for spatially autocorrelated acoustic data |

Feature set: 21 total (11 PB + 10 OB), no spatial coordinates.

## Segmentation

| n_segments target | 4000 |
| n_segments actual | 2654 |
| Mean segment size | 7219.9 px = 451.2 m² |

## CV Results

10-fold spatial GroupKFold (KMeans, k=10, seed=42). Same as v6.

| Configuration | Best Model | CV F1 mean ± std |
|---------------|------------|-----------------|
| pb_only | rf | 0.6021 ± 0.2975 |
| ob_only | cat | 0.3886 ± 0.2521 |
| combined | cat | 0.6256 ± 0.1664 ← **BEST** |

- Submission: `data\submission_v9_best.csv`
