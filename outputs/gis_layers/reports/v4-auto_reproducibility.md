# Reproducibility Report: submission_v4-auto

**Date**: 2026-03-26 20:47 UTC
**Git revision**: 2e9d31d
**Branch**: 007-gis-expert-annotation
**Runtime**: 806s

## Method

**Ensemble**: Depth-stratified indicator kriging + global indicator kriging + CatBoost + LightGBM
**Blend weights**: kriging_depth=0.6, kriging_global=0.15, catboost=0.1, lgbm=0.15
**Features**: 88 (88 terrain + no expert zones)
**Expert annotations**: No (automated fallback)

### Kriging Configuration
- Variogram model: exponential
- Max training points per kriging call: 3000 (global), 1500 (per zone)
- Depth zones: 5 (quantile-based)
- Adjacent zone overlap: ±1

### GBDT Configuration
- CatBoost: 1000 iter, depth=6, lr=0.02, l2=10, balanced weights
- LightGBM: 1000 iter, max_depth=8, lr=0.02, num_leaves=63, balanced weights

### Cross-Validation
- Spatial-block CV: 10 folds (KMeans on coordinates)
- Metric: weighted F1 score
- kriging_depth: 0.7277
- kriging_global: 0.7305
- catboost: 0.6917
- lgbm: 0.6825
- blend: 0.7664

### Feature Set
- Coordinates (x, y)
- Raw bathymetry + backscatter
- Multi-scale focal statistics (8 radii × 2 rasters × 3 stats)
- Multi-scale curvature (5 Gaussian sigmas × 3)
- BTM terrain derivatives (slope, VRM)
- Aspect (sin, cos)
- Relative backscatter (4 scales)
- Backscatter gradient magnitude
- Interaction features (depth×back, slope×back, acoustic hardness, VRM×back)
- Zone features (depth zone, backscatter zone, acoustic facies)
- Isobath distances (5 quantile contours)


## Output
- Submission: submission_v4.csv
- Rows: 98
- Class distribution: {'NVB': np.int64(35), 'ALG': np.int64(30), 'FMAT': np.int64(17), 'SGZ': np.int64(9), 'SGAM': np.int64(7)}

## Reproduction Command
```
python scripts/step3_refine_with_annotations.py --no-annotations --output data/submission_v4.csv --blend 0.60,0.15,0.10,0.15 --seed 42
```
