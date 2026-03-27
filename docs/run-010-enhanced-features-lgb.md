---
date: 2026-03-27 16:58
branch: 010-enhanced-features-lgb
script: scripts/experiment_v7.py
builds_on: experiment_v6.py (009-obia-pixel-hybrid)
reference_companion: niwacolours/benthic-terrain-model-kaggle@002-boost-weighted-f1
---

# Run Report: Enhanced Features + SMOTE + LightGBM/CatBoost (Experiment v7)

## Approach Summary

Extends the v6 OBIA+PB hybrid with multi-scale focal texture, TPI,
interaction terms, spatial context, and fold-safe SMOTE.
Ensemble reduced to LightGBM + CatBoost (best models from both repos).

| Feature Group | Count | Source |
|---------------|-------|--------|
| PB paper features (depth/slope/VRM/etc.) | 8 | Ierodiaconou 2018 |
| Multi-scale focal std (3/5/9 cells) | 6 | companion repo |
| TPI at 9 and 25 cells | 2 | companion repo |
| Interaction terms | 4 | companion repo |
| Spatial context (x_rel/y_rel/z-score) | 4 | companion repo |
| OBIA segment stats (v6) | 10 | Ierodiaconou 2018 |
| **Total** | **34** | |

## Segmentation Parameters

| n_segments (target) | 4000 |
| n_segments (actual) | 2654 |
| Mean segment size (px) | 7219.9 |
| Mean segment size (m²) | 451.2 |

## CV Results

Spatial block GroupKFold (n_splits=6, KMeans n_clusters=6, seed=42).
SMOTE applied on training folds only (fold-safe). LightGBM + CatBoost ensemble.

| Configuration | Best Model | CV Weighted-F1 (mean ± std) |
|---------------|------------|----------------------------|
| pb_only | cat | 0.5147 ± 0.2475 |
| ob_only | cat | 0.3932 ± 0.2138 |
| combined | cat | 0.6075 ± 0.2643 ← **BEST** |

## Top Feature Importances (Combined Config, Best Model)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | y_rel | 36.6353 |
| 2 | x_rel | 11.8561 |
| 3 | bathy_std_9 | 7.7754 |
| 4 | depth_z | 7.7535 |
| 5 | depth | 6.6922 |
| 6 | seg_back_mean | 2.6411 |
| 7 | bathy_x_back | 2.3979 |
| 8 | bathy_std_5 | 2.0299 |
| 9 | backscatter_z | 1.9671 |
| 10 | backscatter | 1.8011 |
| 11 | seg_bathy_std | 1.5934 |
| 12 | back_std_9 | 1.5828 |
| 13 | vrm | 1.2544 |
| 14 | seg_bathy_skew | 1.1228 |
| 15 | eastness | 1.1112 |

## Spatial Autocorrelation — Global Moran's I (training points, k=8 NN)

Moran's I > 0 indicates spatial clustering (nearby points similar).
High values suggest spatial autocorrelation leakage risk in non-spatial CV.

| Feature | Moran's I |
|---------|-----------|
| depth | 0.9942 |
| bathy_std_9 | 0.8494 |
| backscatter | 0.8264 |
| bathy_std_5 | 0.7438 |
| slope | 0.6470 |
| bathy_std_3 | 0.5823 |
| complexity | 0.5688 |
| vrm | 0.5590 |

## Output Files

- Submission: `data\submission_v7_best.csv`
- Report: `docs\run-010-enhanced-features-lgb.md`

Report generation: 0.0s
