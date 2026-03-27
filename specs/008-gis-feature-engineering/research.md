# Research: GIS-Derived Feature Engineering

**Feature**: `008-gis-feature-engineering`  
**Date**: 2026-03-27  
**Status**: Complete — all unknowns resolved

---

## 1. Derived Raster Availability

**Decision**: Use rasters already on disk. No re-derivation required for this run.

**Rationale**: All three required rasters confirmed present at `outputs/gis_layers/rasters/`:

- `slope.tif` — Horn (1981) slope in degrees, float32, matches bathymetry CRS/extent
- `backscatter_zones.tif` — int32, 5-class k-means on Gaussian-smoothed backscatter (σ=10, 50k pixel subsample, seed 42)
- `acoustic_facies.tif` — int32, 8-class k-means on standardised (depth × backscatter), same subsample

**Reproducibility**: `step1_prepare_gis_layers.py` unconditionally re-derives these rasters. The new experiment script will also include a lightweight `_ensure_derived_rasters()` helper that calls into the derivation logic if any file is missing.

**Alternatives considered**: Running derivation inline in the experiment script. Rejected — derivation already encapsulated in `step1_prepare_gis_layers.py`; calling it as a subprocess or importing the relevant functions avoids code duplication.

---

## 2. 80/20/20 Feature Weighting in Tree Ensembles

**Decision**: Implement column scaling exactly as specified in FR-003. Document the GBT invariance in the run report.

**Rationale**: Decision-tree-based models (GBDT, Random Forest) split on ordered thresholds and are invariant to monotonic feature rescaling — multiplying a column by constant `k` shifts all split thresholds by `k` but does not change which feature is chosen or the resulting predictions. The 80/20/20 scaling will therefore have negligible direct effect on tree model outputs.

However, the scaling is retained because:

1. The spec explicitly requires it (FR-003) and the user requested it.
2. It serves as a documented prior — any downstream linear or regularised model trained on this feature matrix will respond to the scaling.
3. Feature importance rankings in GBDT use split counts and gain, not feature magnitude, so the experiment result is interpretable regardless.

**Implementation**: Scale applied after one-hot encoding, before `DataFrame` construction:

```python
WEIGHT_SLOPE = 80.0
WEIGHT_BZ    = 20.0
WEIGHT_AF    = 20.0

slope_weighted = slope_values * WEIGHT_SLOPE          # shape (n,)
bz_ohe = pd.get_dummies(bz_values, prefix='gis_bz')  # shape (n, 5)
af_ohe = pd.get_dummies(af_values, prefix='gis_af')  # shape (n, 8)
bz_ohe_w = bz_ohe * WEIGHT_BZ
af_ohe_w = af_ohe * WEIGHT_AF
```

**Alternatives considered**: Repeating the slope column 4× to simulate higher weight (feature bagging proxy). Rejected — unintuitive and does not generalise to the kriging run. Using `sample_weight` or `feature_fraction` overrides in the model — rejected, out of scope per spec.

---

## 3. Kriging Scope and Implementation

**Decision**: Apply ordinary kriging per spatial zone to ONLY the three new GIS-derived features. All v2 features pass through unchanged.

**Rationale**: Existing `exp_backscatter_zones.py` implements exactly this pattern (zone-stratified kriging using pykrige `OrdinaryKriging`). Scoping kriging to the three new features:

- Isolates the effect of spatial smoothing on the new signals.
- Avoids the compute cost of kriging 120+ v2 features.
- Maintains the comparison integrity — the only difference between Run A and Run B is how the GIS features are interpolated.

**Zone definition**: Use the `back_zone_raster_sorted` (5-cluster backscatter zones) as the stratification variable for kriging (same as `exp_backscatter_zones.py`). This gives spatially coherent zones that align with substrate character.

**Fallback**: When a zone has `< 10` training points, fall back to `KNeighborsRegressor(n_neighbors=min(5, n_pts))` using spatial coordinates only. Log count of affected zones.

**Algorithm parameters**: Variogram model `'spherical'`; `nlags=6`; `weight=True`. These are the defaults from `exp_backscatter_zones.py` and are well-established for marine terrain data.

**References**:

- Matheron, G. (1963). _Principles of geostatistics_. Economic Geology, 58(8), 1246–1266.
- pykrige documentation: ordinary kriging with spherical variogram.

**Alternatives considered**: Kriging all v2 features — rejected: prohibitive compute cost (120+ separate kriging fits per zone). Universal kriging using covariates — rejected: adds complexity without clear benefit for this experiment scope.

---

## 4. Cluster Counts: Backscatter Zones and Acoustic Facies

**Decision**: 5 clusters for backscatter zones, 8 clusters for acoustic facies.

**Rationale**: These values are hard-coded in `exp_backscatter_zones.py` and `step1_prepare_gis_layers.py` and were chosen empirically to match the visible zonation in the Fagatele Bay MBES data:

- 5-zone backscatter segmentation separates the five dominant substrate intensity classes visible in the imagery.
- 8-zone acoustic facies captures the combined depth × backscatter space and corresponds to the number of distinct benthic habitat types reported in the site survey literature.

Using the same values ensures that the rasters already on disk (`backscatter_zones.tif`, `acoustic_facies.tif`) are directly compatible with the experiment script without re-derivation.

**Alternatives considered**: Using silhouette score to auto-select k. Rejected — adds non-determinism and would require re-deriving rasters; also out of scope per spec (which explicitly refers to "prior work").

---

## 5. v4 Baseline F1 Computation

**Decision**: Compute v4 baseline weighted-F1 from the `experiment_v2.py` feature set (no GIS features) using the same 10-block spatial-block CV protocol at the start of `experiment_v5.py`, before adding GIS features.

**Rationale**: No saved CV score from the v4 run exists in the repository. Re-computing it in the same script:

- Guarantees apples-to-apples comparison (identical CV folds, identical ensemble config, identical seed).
- Documents the baseline as a numeric fact in the run report.
- Adds ~10–15 minutes of wall-clock time but eliminates ambiguity.

**Implementation**: The script runs `phase="baseline"` first (v2 features only → train ensemble CV → record `baseline_f1`), then `phase="gis_raw"` (v2 + raw GIS features → record `run_a_f1`), then `phase="gis_kriging"` (v2 + kriged GIS features → record `run_b_f1`). All three use identical spatial blocks (KMeans seed 42, k=10).

**Alternatives considered**: Loading `submission_v4.csv` and estimating F1 by re-predicting on train split. Rejected — CV F1 and hold-out submission F1 are not directly comparable; the CV number is more meaningful.

---

## 6. Code Reuse Strategy

| Component                 | Source                             | Action                                     |
| ------------------------- | ---------------------------------- | ------------------------------------------ |
| `compute_slope`           | `btm.core.slope`                   | Import directly                            |
| `compute_vrm`             | `btm.core.vrm`                     | Import directly                            |
| `sample_raster_at_points` | `btm.features.extract`             | Import directly                            |
| `extract_all_features()`  | `scripts/experiment_v2.py`         | Copy function verbatim into new script     |
| K-means zone derivation   | `scripts/exp_backscatter_zones.py` | Adapt into `_derive_gis_features()` helper |
| Zone-stratified kriging   | `scripts/exp_backscatter_zones.py` | Adapt into `_krige_gis_features()` helper  |
| Ensemble train/CV loop    | `scripts/experiment_v2.py`         | Copy and parameterise                      |
| Submission writer         | `scripts/make_submission_v2.py`    | Inline (trivially short)                   |

**Rationale**: The `scripts/` directory contains exploratory code, not a library. Direct copy-and-adapt is appropriate; the new file provides a self-contained, documented, reproducible record of this specific experiment.
