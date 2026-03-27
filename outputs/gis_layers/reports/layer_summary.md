# GIS Layer Summary — Human-in-the-Loop Habitat Mapping

**Date**: 2026-03-26 20:31 UTC
**Git revision**: 2e9d31d
**Branch**: 007-gis-expert-annotation
**CRS**: EPSG:28355 (GDA94 MGA Zone 55)
**Raster resolution**: 0.25 m
**Raster shape**: 4040 rows × 4743 cols
**Study area bounds**: E [453346.9, 454532.6], N [5678395.1, 5679405.1]

## Raster Layers

| File | Description | Type |
|------|-------------|------|
| slope.tif | Horn (1981) slope in degrees | Continuous |
| vrm.tif | Vector Ruggedness Measure (3×3 window) | Continuous |
| mean_curvature.tif | Gaussian-smoothed mean curvature (σ=4) | Continuous |
| backscatter_smooth.tif | Gaussian-smoothed backscatter (σ=10, ~2.5m) | Continuous |
| depth_zones.tif | 5 quantile-based depth zones | Categorical (0–4) |
| backscatter_zones.tif | 5 k-means backscatter zones (sorted by intensity) | Categorical (0–4) |
| acoustic_facies.tif | 8 combined depth×backscatter k-means zones | Categorical (0–7) |
| kriging_dominant.tif | Dominant kriging class (20×20 subsample) | Categorical (0–4) |

### Depth Zones
  - Zone 0: depth [-23.3, -11.3] m
  - Zone 1: depth [-11.3, -9.6] m
  - Zone 2: depth [-9.6, -8.0] m
  - Zone 3: depth [-8.0, -5.5] m
  - Zone 4: depth [-5.5, 0.9] m

### Backscatter Zones (sorted: 0=softest/darkest → 4=hardest/brightest)
  - Zone 0: mean backscatter = -29.0 dB
  - Zone 1: mean backscatter = -26.0 dB
  - Zone 2: mean backscatter = -23.4 dB
  - Zone 3: mean backscatter = -21.0 dB
  - Zone 4: mean backscatter = -16.1 dB

### Acoustic Facies (depth × backscatter k-means)
  - Facies 0: mean depth = -4.4 m, mean backscatter = -23.1 dB, pixels = 846421
  - Facies 1: mean depth = -11.2 m, mean backscatter = -29.1 dB, pixels = 1067440
  - Facies 2: mean depth = -12.7 m, mean backscatter = -22.6 dB, pixels = 553798
  - Facies 3: mean depth = -5.9 m, mean backscatter = -17.5 dB, pixels = 549857
  - Facies 4: mean depth = -8.5 m, mean backscatter = -21.9 dB, pixels = 878449
  - Facies 5: mean depth = -15.7 m, mean backscatter = -26.5 dB, pixels = 767374
  - Facies 6: mean depth = -9.4 m, mean backscatter = -25.6 dB, pixels = 1125759
  - Facies 7: mean depth = -18.3 m, mean backscatter = -22.1 dB, pixels = 506140

### Kriging Class Legend
  - Class 0: ALG
  - Class 1: FMAT
  - Class 2: NVB
  - Class 3: SGAM
  - Class 4: SGZ

## Vector Layers (annotation_layers.gpkg)

| Layer | Features | Description |
|-------|----------|-------------|
| acoustic_facies_zones | 102 | Zone polygons with editable `expert_label` field |
| depth_zones | 20 | Depth zone polygons |
| backscatter_zones | 122 | Backscatter zone polygons |
| training_points | 6256 | Training samples with class, prediction, confidence |
| test_points | 98 | Test samples with predicted class |

## Annotation Workflow

### Step 2: Expert Visual Analysis (your turn)

1. Open `outputs/gis_layers/vectors/annotation_layers.gpkg` in QGIS
2. Add the raster layers from `outputs/gis_layers/rasters/` as base maps
3. Inspect the `acoustic_facies_zones` layer against the backscatter/bathymetry
4. For each zone polygon, edit the `expert_label` field:
   - Use habitat descriptors (e.g., "hard_reef", "sand_plain", "mixed_rubble", "algal_cover", "seagrass")
   - Or confirm the automated `dominant_class` by copying it into `expert_label`
   - Use the `notes` field for any uncertainty or observations
5. You may split or merge polygons as needed — the refinement script uses spatial joins
6. Save the GeoPackage when done

### Step 3: Return the annotated GeoPackage

Save the file and run the refinement script (Step 3, to be provided) which will:
- Read your expert labels and use them as additional features/stratification
- Produce a refined Kaggle submission with reproducibility report

## Training Data Summary

| Class | Count | Proportion |
|-------|-------|------------|
| ALG | 682 | 10.9% |
| FMAT | 1544 | 24.7% |
| NVB | 3036 | 48.5% |
| SGAM | 170 | 2.7% |
| SGZ | 824 | 13.2% |

## Previous Results (Baselines)

| Version | Method | Spatial CV | Kaggle F1 |
|---------|--------|-----------|-----------|
| v1 | LightGBM only (25 features) | ~0.48 | ~0.70 |
| v2 | 40% KNN + 40% CatBoost + 20% LightGBM (80 features) | 0.7521 | 0.764 |
| v3 | 60% Depth-kriging + 15% Global-kriging + 10% CB + 15% LGBM (87 feat) | 0.7668 | 0.764 |
