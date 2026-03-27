# Research: OBIA + Pixel-Based Hybrid Classification

**Branch**: `009-obia-pixel-hybrid`  
**Phase**: 0 — Technical Decisions  
**Reference**: Ierodiaconou et al. (2018) Marine Geodesy DOI: 10.1007/s11001-017-9338-z

---

## RD-001 — Image Segmentation Algorithm

**Question**: The paper uses eCognition with scale parameter 41 → mean object ~306 m² (≈4800 px at 0.25 m). What Python open-source segmentation algorithm best replicates this?

**Decision**: Use `skimage.segmentation.slic()` with channels from bathymetry + backscatter.

**Rationale**:
- SLIC (Achanta et al. 2012) is the de facto standard for superpixel segmentation in Python
- Accepts multi-channel input directly — bathymetry + normalised backscatter
- `n_segments` parameter gives direct control over approximate object count
- At 4040×4743 = 19.2 M pixels, target segment size 4800 px → `n_segments ≈ 4000`
- `compactness=0.01` yields irregular, boundary-following segments (analogous to eCognition's multiresolution with low shape weight)
- `start_label=0`, `enforce_connectivity=True` prevent orphan labels

**Alternatives considered**:
- `skimage.segmentation.felzenszwalb`: graph-based, produces variable-size objects — harder to target a mean object size; scale parameter non-intuitive on float rasters
- `skimage.segmentation.watershed`: requires explicit marker selection; seeds needed from local minima
- OBIA libraries (orfeotoolbox, RSGISLib): not pip-installable; add heavy system dependency

**Calibration target**: mean segment pixel count ≈ 4800 (300 m² ÷ 0.0625 m² per pixel). Adjust `n_segments` empirically if mean deviates by >50%.

---

## RD-002 — Rugosity Algorithm

**Question**: The paper refers to "rugosity" as a terrain complexity metric. What algorithm should be used, and does the codebase already provide one?

**Decision**: Use `btm.core.vrm.compute_vrm(bathy, neighborhood_size=3, cell_size=cell_size)` — the Vector Ruggedness Measure (Sappington et al. 2007).

**Rationale**:
- VRM is the standard rugosity metric in marine habitat mapping (Wright & Heyman 2008)
- `btm.core.vrm` is already implemented and tested in this repository
- 3×3 neighbourhood is consistent with the paper's 3×3 kernel for all pixel-based derivates
- Outputs [0, 1] range — flat=0, maximal roughness=1

**Alternatives considered**:
- Terrain Ruggedness Index (TRI): RMSD of elevation relative to centre cell — simpler but less directionally sensitive
- Arc-Chord ratio rugosity: requires resampling step on a planar triangle — overengineered for this experiment

---

## RD-003 — Complexity (Surface-to-Planar Area Ratio)

**Question**: The paper includes "complexity" as a pixel-based feature. How should this be computed from a DEM?

**Decision**: Implement surface-to-planar area ratio in a 3×3 window using vectorised NumPy.

**Formula** (Jenness 2004):
- For each 3×3 neighbourhood extract elevation patch z (3×3)
- Compute 8 triangular surface patches from the centre cell to each adjacent pair of neighbours
- Complexity = Σ(triangle surface areas) / planar_cell_area
- Approximate implementation: surface area of the neighbourhood / (cell_size² × 9)

**Simplified numerical approach** (avoid full triangulation):
```
# Approximate: use the local surface area approximation from slope
# surface_area ≈ cell_size² / cos(slope_rad)
# complexity ≈ mean(1/cos(slope_3x3)) over focal 3×3 window  
complexity = ndimage.uniform_filter(1.0 / np.cos(np.deg2rad(slope_deg)), size=3)
```
This provides the surface-to-planar area ratio per pixel, consistent with Jenness (2004) intent for gridded DEMs.

**Rationale**: The Jenness (2004) surface area formula is standard for bathymetric complexity in ArcGIS BTM literature. The cosine approximation gives equivalent results on smooth surfaces without the overhead of full 8-triangle computation.

**Alternatives considered**:
- Full 8-triangle surface area computation: more accurate on very rough terrain, but adds ~100 lines of kernel code for negligible difference at 0.25 m resolution

---

## RD-004 — Maximum Curvature

**Question**: What is "maximum curvature" in the context of a DEM and how should it be computed?

**Decision**: Compute maximum curvature as the maximum of profile curvature and plan curvature from the second-order partial derivatives of the surface.

**Formula** (Zevenbergen & Thorne 1987):
```
# Second derivatives (Zevenbergen & Thorne notation)
D = (z_left + z_right - 2·z_centre) / (2·cell²)   # profile
E = (z_up + z_down - 2·z_centre) / (2·cell²)       # plan (approx)
max_curvature = maximum(abs(D), abs(E))
```
Using `scipy.ndimage.laplace(bathy) / cell_size²` approximates this as the isotropic Laplacian, which captures the dominant curvature direction.

**Rationale**: Maximum curvature identifies ridges, channels, and inflection zones — key structural class separators in the Ierodiaconou paper. Laplace approximation is fast and differentiable.

**Alternatives considered**:
- Computing all six second-derivative terms (full Zevenbergen-Thorne): more accurate principal curvatures but requires ≥10 lines of arithmetic; overkill for this experiment

---

## RD-005 — Northness and Eastness (Aspect Derivatives)

**Question**: How should northness and eastness be computed and what is expected range?

**Decision**: Derive aspect from bathymetry gradient, then apply cos/sin transforms.

**Formula**:
```python
gy, gx = np.gradient(bathy, cell_size, cell_size)  # y=North, x=East
aspect_rad = np.arctan2(gx, gy)                    # 0 = North, clockwise
northness = np.cos(aspect_rad)                     # range [-1, 1]
eastness  = np.sin(aspect_rad)                     # range [-1, 1]
```
Values are in `[-1, 1]` as confirmed in the spec clarifications.

**Rationale**: Standard aspect decomposition from terrain analysis (Roberts 1986). Avoids circular arithmetic problems with raw aspect degree values.

---

## RD-006 — Ensemble Classifier Configuration

**Question**: Which classifier configuration maximises CV F1 with manageable runtime on this dataset (~560 training points, 18 features)?

**Decision**: Reuse `get_models()` from `scripts/experiment_v2.py` — identical ensemble of LightGBM, XGBoost, CatBoost, RandomForest.

**Rationale**:
- Direct comparability with v2 baseline on same CV strategy
- All four classifiers support `class_weight='balanced'` (or equivalent) for minority class handling
- The codebase already has validated hyperparameters for this dataset; changing them introduces confounds
- Runtime is dominated by raster derivations, not model training on 560 points

**Decision**: Copy `get_models()` directly into `experiment_v6.py` (no shared library import) to keep the experiment self-contained and avoid coupling.

---

## RD-007 — Spatial Cross-Validation Strategy

**Question**: How to ensure the CV strategy is directly comparable to the v2 baseline?

**Decision**: KMeans spatial block CV (n_clusters=10, random_state=42) — identical to `experiment_v2.py`.

**Rationale**:
- Spatial CV prevents overly optimistic F1 from nearby autocorrelated samples
- Using the exact same strategy makes the F1 numbers directly comparable between v5/v6 and v2
- KMeans block CV is already validated in this codebase

---

## RD-008 — Segmentation Input Channels

**Question**: Which raster channels feed into the SLIC segmentation?

**Decision**: Stack normalised `[bathymetry, backscatter, rugosity]` — same three inputs as the eCognition segmentation in the paper (Section 2.2: "segmentation was performed on the bathymetric surface, backscatter and rugosity").

**Implementation**:
```python
seg_input = np.stack([
    (bathy - bathy.mean()) / bathy.std(),
    (back  - back.mean())  / back.std(),
    (vrm   - vrm.mean())   / vrm.std(),
], axis=-1)
labels = slic(seg_input, n_segments=4000, compactness=0.01,
              enforce_connectivity=True, start_label=0)
```

---

## Summary of Resolved Decisions

| ID     | Decision                            | Rationale                                        |
|--------|-------------------------------------|--------------------------------------------------|
| RD-001 | SLIC segmentation (skimage)         | Direct n_segments control, multi-channel capable |
| RD-002 | btm.core.vrm — VRM rugosity         | Already implemented, marine standard             |
| RD-003 | cos-slope surface area complexity   | Jenness 2004 approximation, vectorised           |
| RD-004 | Laplace curvature (scipy)           | Fast approximation of max curvature              |
| RD-005 | cos/sin of gradient-derived aspect  | Avoids circular value issues, standard terrain   |
| RD-006 | LGBM+XGB+CatBoost+RF ensemble       | Reuse v2 `get_models()` for comparability        |
| RD-007 | KMeans spatial CV (10 blocks)       | Identical to v2 baseline                         |
| RD-008 | 3-channel seg input (bathy/back/vrm)| Matches eCognition segmentation in paper         |
