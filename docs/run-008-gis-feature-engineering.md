# Run 008 – GIS Feature Engineering

**Branch**: `008-gis-feature-engineering`

**Spec**: `specs/008-gis-feature-engineering/spec.md`


## Parameters

| Parameter | Value |
|-----------|-------|
| WEIGHT_SLOPE | 80.0 |
| WEIGHT_BZ | 20.0 |
| WEIGHT_AF | 20.0 |
| N_BZ_CLUSTERS | 5 |
| N_AF_CLUSTERS | 8 |
| SEED | 42 |
| N_SPATIAL_BLOCKS | 10 |
| KRIGING_MIN_ZONE_PTS | 10 |
| KRIGING_VARIOGRAM | spherical |
| KRIGING_NLAGS | 6 |

## Metrics Comparison

| Run | Weighted-F1 | Wall-clock | Top-3 features |
|-----|-------------|------------|----------------|
| baseline | 0.6927 | 758s | y, k_fm_201, b_std_801 |
| run_a | 0.6971 | 832s | y, x, b_std_801 |
| run_b | 0.6971 | 755s | y, x, b_std_801 |

### Per-class F1

**baseline**: ALG: 0.704 | FMAT: 0.740 | NVB: 0.830 | SGAM: 0.000 | SGZ: 0.233
**run_a**: ALG: 0.706 | FMAT: 0.738 | NVB: 0.832 | SGAM: 0.000 | SGZ: 0.262
**run_b**: ALG: 0.706 | FMAT: 0.738 | NVB: 0.832 | SGAM: 0.000 | SGZ: 0.262

## Note on Run A vs Run B CV F1

Run A and Run B report the **same CV weighted-F1** by design.
Kriging applies only to test-point inference in Run B; it does not affect cross-validation scores (see spec US2/AC2).
Any difference in leaderboard performance can only be assessed via submission to the competition.

## Best Run

Best run: **run_b** (CV weighted-F1 = 0.6971)

## Interpretation

GIS features (slope/bz/af) produced a weighted-F1 delta of **+0.0044** vs baseline.
A positive delta indicates the weighted GIS-derived features improved ensemble discrimination.
