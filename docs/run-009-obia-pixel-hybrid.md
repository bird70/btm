---
date: 2026-03-27 16:06
branch: 009-obia-pixel-hybrid
script: scripts/experiment_v6.py
reference: Ierodiaconou et al. (2018) DOI:10.1007/s11001-017-9338-z
---

# Run Report: OBIA + Pixel-Based Hybrid (Experiment v6)

## Approach

Combined pixel-based (PB) + object-based (OB) habitat classification following
Ierodiaconou et al. (2018) on the Refuge Cove MBES dataset (0.25 m, 5 classes).
Paper baseline: PB=72.5%, OB=78.5%, Combined=83.6% overall accuracy.

## Segmentation Parameters

| Parameter | Value |
|-----------|-------|
| Algorithm | SLIC (scikit-image 0.26, Achanta et al. 2012) |
| Input channels | bathymetry, backscatter, VRM (min-max normalised) |
| n_segments (target) | 4000 |
| n_segments (actual) | 2654 |
| compactness | 0.01 |
| Mean segment size (px) | 7219.9 |
| Mean segment size (m²) | 451.2 |
| Target object size (m²) | ~300 (paper scale=41 equivalent) |

## CV Results

Spatial block GroupKFold (n_splits=10, KMeans n_clusters=10, seed=42).
Identical CV setup to experiment_v2 for direct comparability.

| Configuration | Best Model | CV Weighted-F1 (mean ± std) |
|---------------|------------|----------------------------|
| pb_only | rf | 0.5665 ± 0.2982 |
| ob_only | cat | 0.3886 ± 0.2521 |
| combined | cat | 0.5758 ± 0.1623 ← **BEST** |

## Top Feature Importances (Combined Config, Best Model)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | depth | 32.4729 |
| 2 | backscatter | 8.3685 |
| 3 | complexity | 7.6852 |
| 4 | seg_back_mean | 6.2362 |
| 5 | seg_vrm_skew | 5.5714 |
| 6 | seg_vrm_mean | 5.2661 |
| 7 | seg_bathy_std | 4.8794 |
| 8 | vrm | 4.6340 |
| 9 | seg_vrm_std | 3.9155 |
| 10 | seg_pixel_count | 3.7358 |

## Output Files

- Submission: `data\submission_v6_best.csv`
- Report: `docs\run-009-obia-pixel-hybrid.md`

## Timing

| Phase | Duration |
|-------|----------|
| Report generation | 0.0s |
