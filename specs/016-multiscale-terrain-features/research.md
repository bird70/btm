# Research: Multi-Scale Terrain Features

## R1: Multi-Scale Aggregation Method

**Decision**: Calculate-then-average (compute derivative at native 3×3, then focal mean over increasing window sizes)

**Rationale**: Nemani et al. (2022) and Misiuk et al. (2021) both use this approach. It preserves the scientifically validated 3×3 kernel computation (Horn slope, Sappington VRM, Jenness surface ratio) while capturing broader-scale terrain context through spatial averaging. Simple to implement: `scipy.ndimage.uniform_filter(derivative, size=window_size)`.

**Alternatives considered**:

- Direct large-kernel computation (e.g., slope from 21×21 window) — rejected because it changes the algorithm semantics and requires modifying validated core functions
- Both methods with feature selection — rejected as premature complexity

## R2: Which Derivatives to Scale

**Decision**: Slope, VRM, surface ratio, northness, eastness, max curvature, complexity + new RDMV. BPI excluded.

**Rationale**: BPI already uses configurable annular radii at two ecological scales (broad/fine). Multi-scale windowing of BPI would duplicate RDMV's role. The seven selected derivatives are all computed at a fixed 3×3 kernel and would benefit from multi-scale context. RDMV (Relative Difference to Mean Value) is a new derivative that directly measures topographic position at each scale.

**Alternatives considered**:

- Include BPI at additional scales — rejected because RDMV captures the same concept (position relative to local mean) without the annular parameterisation complexity

## R3: RDMV Algorithm

**Decision**: RDMV = (cell_value − focal_mean) / focal_std, per Lecours et al. (2017)

**Rationale**: RDMV measures how different a cell's depth is from its local neighbourhood mean, normalised by the neighbourhood standard deviation. It characterises topographic position (ridges positive, depressions negative, flat ~0). Lecours et al. (2017) identified it as capturing ~70% of surface structure when combined with slope and aspect.

**Alternatives considered**:

- TPI (Topographic Position Index = value − focal_mean without normalisation) — RDMV preferred because normalisation makes values comparable across scales

## R4: GLCM Texture Implementation

**Decision**: Use `skimage.feature.graycomatrix` + `graycoprops` from scikit-image (already a project dependency). Compute contrast and homogeneity at multiple patch sizes. Average over 4 directions (0°, 45°, 90°, 135°) for rotation invariance.

**Rationale**: scikit-image is already in `pyproject.toml` dependencies. The existing `scripts/experiment_v2.py` and `scripts/experiment_v5.py` contain working GLCM code using this exact approach. The approach matches Nemani et al. (2022) who derived GLCM contrast and homogeneity from backscatter.

**Alternatives considered**:

- Custom GLCM implementation — rejected; scikit-image's C-optimised implementation is battle-tested
- More GLCM properties (energy, correlation, dissimilarity) — start with contrast + homogeneity (top-ranked in literature); add others only if feature selection demands them

## R5: Feature Selection Method

**Decision**: Permutation importance from scikit-learn (`sklearn.inspection.permutation_importance`)

**Rationale**: Simpler than Boruta; built into scikit-learn; sufficient for the expected 50–80 feature range. Boruta is better suited for 100+ features with many redundant shadow features. Permutation importance directly measures each feature's impact on model accuracy.

**Alternatives considered**:

- Boruta wrapper (Kursa & Rudnicki, 2010) — as used by Nemani et al. (2022); more powerful but adds a dependency and complexity not justified for our feature count
- Correlation-based pre-filter — will use Spearman correlation > 0.95 as a pre-filter before permutation importance, not as the primary method

## R6: Scale Window Sizes

**Decision**: 5 scales: 3, 7, 11, 15, 21 (cells). At 0.25 m cell size → 0.75 m, 1.75 m, 2.75 m, 3.75 m, 5.25 m.

**Rationale**: Nemani et al. (2022) used 10 scales (3–21) at 10 m resolution. Our raster resolution is ~0.25 m, so 21 cells = 5.25 m covers a reasonable ecological scale range for this high-resolution survey data. Odd window sizes are required for symmetric focal operations. Five scales balance feature dimensionality against spatial coverage.

**Alternatives considered**:

- 10 scales (3–21 at every odd size) — produces 100+ features, increasing dimensionality risk without clear benefit at our fine resolution
- 3 scales (3, 11, 21) — too sparse; misses intermediate scales that may be optimal for some derivatives

## R7: Focal Mean at Raster Boundaries

**Decision**: Use `mode="reflect"` in `scipy.ndimage.uniform_filter` for boundary handling.

**Rationale**: This is the default and matches the existing VRM and depth_statistics implementations in `btm/core/`. Reflection at boundaries is appropriate for bathymetric data where the terrain continues beyond the survey extent.

**Alternatives considered**:

- `mode="constant"` (pad with value) — introduces artificial terrain discontinuities at edges
- NaN at boundaries — would require masking that complicates downstream feature extraction; most boundary effects are small for moderate window sizes
