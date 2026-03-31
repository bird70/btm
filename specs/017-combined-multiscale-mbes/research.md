# Research: Combined Multi-Scale BTM + MBES-8 Features

## R1: MBES-8 Feature Extraction Method

**Decision**: Reuse the `_compute_pb_features()` pattern from `scripts/experiment_v9.py` as a private helper in `experiment_v11.py`. Extract the 8 MBES point-sample features directly from bathymetry and backscatter rasters using rasterio + NumPy/SciPy.

**Rationale**: `experiment_v9.py` contains a proven, tested implementation that computes all 8 features (depth, backscatter, slope, VRM, complexity, max_curvature, northness, eastness) from the raw rasters. This code runs in ~30 s for 6256 points. Lifting it directly avoids the `benthic_model` pipeline dependency and keeps the experiment self-contained — matching the `experiment_v10.py` pattern.

**Alternatives considered**:

- Use `benthic_model` pipeline (the full `train.py` → `engineering.py` path) — rejected because it adds unnecessary pipeline coupling and extra features (SLIC segments, interactions) that would confound the comparison
- Sample rasters via `btm.features.extract.extract_btm_features` with `scales=None` — rejected because that function doesn't compute raw depth/backscatter point samples, only terrain derivatives

## R2: Which MBES-8 Features to Include

**Decision**: The canonical 8 PB (pixel-based) features from the Ierodiaconou et al. (2018) reference paper: depth, backscatter, slope, VRM, complexity, max_curvature, northness, eastness. Exclude the 3 texture/TPI features from v9 (bathy_std_9, back_std_9, tpi_9) and the 10 OB (object-based) SLIC segment features.

**Rationale**: R04/R06 achieved 0.8024 using the `benthic_model` pipeline's base feature set (MBES-8 + BTM-10). The MBES-8 features are the missing ingredient from spec-016. Including texture (v9 additions) or SLIC features would add confounds — those should be evaluated in follow-up experiments. The MBES-8 are also non-redundant with the BTM multi-scale features: depth and backscatter are raw raster values; BTM features are all derived terrain metrics.

**Alternatives considered**:

- Include all 11 v9 PB features (MBES-8 + 3 texture/TPI) — deferred to follow-up; start with the minimal set that addresses Cause 1
- Include 10 OB segment features — deferred (Option C from spec-016); adds segmentation overhead and a different spatial scale

## R3: BTM Feature Source

**Decision**: Load the 33 selected BTM features directly from the spec-016 CSV cache (`reports/metrics/cache_train_feats_v10.csv` and `cache_test_feats_v10.csv`), filtering to just the 33 selected columns listed in `reports/metrics/feature_selection_v10.csv`.

**Rationale**: These features were already computed and validated in spec-016 (NaN-safe extraction, all 64 features valid, permutation importance selection to 33). Re-extracting them from rasters would waste ~10 min and risk introducing drift if any code changed. The CSV cache is the single source of truth.

**Alternatives considered**:

- Re-extract from rasters using `btm.features.extract.extract_btm_features` — rejected due to 10 min overhead and risk of inconsistency
- Use all 64 BTM features (before selection) — deferred; start with the 33-feature selected set as the primary experiment, add a second CV run with the full 64 as an ablation

## R4: RF Hyperparameters

**Decision**: Use exactly R04's configuration: `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42, n_jobs=-1)`.

**Rationale**: R04 achieved the best Kaggle score (0.79518) and strong CV (0.8024) with these exact parameters. Run-015 R18 showed that "tuning" RF (n_estimators=500, max_features='sqrt') _degraded_ Kaggle performance (0.764 vs 0.795). The goal of this experiment is to test feature combination, not hyperparameter search — using proven parameters isolates the feature effect.

**Alternatives considered**:

- n_estimators=500, max_features='sqrt' — explicitly rejected by R18 evidence
- Hyperparameter grid search — out of scope per spec assumptions; adds confounds

## R5: CatBoost and LightGBM Hyperparameters

**Decision**: Use exactly the same configurations as `experiment_v10.py` — CatBoost (iterations=800, lr=0.03, depth=7, auto_class_weights="Balanced", CPU) and LightGBM (n_estimators=600, lr=0.03, num_leaves=31, max_depth=6, class_weight="balanced").

**Rationale**: Direct comparability with spec-016 results. The only variable should be the feature set, not the model configuration.

## R6: Cross-Source Feature Overlap

**Decision**: Accept potential overlap between BTM eco features (btm_northness, btm_eastness, btm_max_curvature, btm_complexity) and MBES-8 (northness, eastness, max_curvature, complexity). The Spearman correlation pre-filter (|r| > 0.95) will automatically detect and remove one of any highly correlated pair.

**Rationale**: The BTM eco features and MBES-8 features use the same underlying algorithms but may differ slightly due to nodata handling and implementation details. Letting the correlation filter handle this automatically is simpler and more robust than manually deduplicating.

**Alternatives considered**:

- Manually exclude the 4 overlapping BTM eco features — rejected; let the data decide via correlation filter
- Use only the MBES-8 versions — rejected; BTM eco features may have slightly different values due to nodata masking differences

## R7: CV Scheme

**Decision**: 5-fold spatial-blocked CV via `iter_spatial_blocked_folds(n_splits=5, spatial_bins=4, random_state=42)` from `src/benthic_model/evaluation/cv.py` — identical to R04/R06.

**Rationale**: The primary goal of v11 is to match or exceed R04/R06's CV F1=0.8024. Using the same CV scheme makes this comparison exact. This differs from experiment_v10's 10-fold KMeans GroupKFold, so v10↔v11 CV scores are not directly comparable — but the feature-set and model-choice effects are still evaluable against the R04/R06 reference.

**Alternatives considered**:

- 10-fold KMeans GroupKFold (v10's scheme) — rejected; creates a scheme mismatch that makes the 0.8024 target ambiguous. v11's primary comparison is against R04/R06, not v10.

**Note**: The spatial-blocked scheme uses quantile-based 4×4 spatial bins (up to 16 blocks) assigned round-robin to 5 folds. This is deterministic with `random_state=42`.
