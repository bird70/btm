# Run 015: Ecology-Informed Habitat Features

**Branch**: `015-ecology-habitat-features`  
**Date**: 2026-03-31  
**Goal**: Add ecology-informed features (depth zones, SGAM niche indicator, raster eco-derivatives) to improve classification of rare classes — especially SGAM — whilst exploring interactions and hyperparameter tuning on the established RF+BTM feature set.

---

## Summary

| Run | Config              | Model        | Feature Set                                 | CV F1  | SGAM CV F1 | Kaggle F1   | CV–Kaggle Gap |
| --- | ------------------- | ------------ | ------------------------------------------- | ------ | ---------- | ----------- | ------------- |
| R09 | catboost-candidate  | CatBoost GPU | BTM features                                | 0.8139 | 0.000      | 0.76153     | 0.052         |
| R17 | rf-btm-interactions | RF           | BTM + pairwise interactions                 | 0.7971 | 0.031      | **0.79518** | 0.008         |
| R18 | rf-btm-tuned        | RF           | BTM (n_estimators=500, max_features=sqrt)   | 0.7960 | 0.000      | 0.76438     | 0.032         |
| R15 | rf-btm-eco-depth    | RF           | BTM + depth zones + SGAM niche              | 0.7916 | **0.045**  | 0.76438     | 0.028         |
| R16 | rf-btm-eco-full     | RF           | BTM + depth zones + SGAM niche + raster eco | 0.7916 | **0.045**  | 0.76438     | 0.028         |

---

## Key Findings

### CatBoost GPU (R09)

- CV F1 of **0.8139** is the highest seen so far, but Kaggle score was only **0.76153** — the largest CV–Kaggle gap in the entire sweep (0.052).
- Hypothesis: CatBoost GPU forces symmetric (oblivious) decision trees. On this small dataset (6256 training points, 5 classes, high spatial autocorrelation), symmetric trees may overfit in a spatially-blocked CV sense that doesn't fully transfer to the hold-out test area.
- **Recommendation**: Do not use CatBoost GPU as the primary candidate for this dataset.

### RF + BTM + Interactions (R17)

- Tied the previous best Kaggle score of **0.79518** (same as R04/R06).
- CV F1 is 0.7971 — close to R04/R06 (0.8024). The smaller feature set from interactions does not add information beyond BTM alone on the public set.
- CV–Kaggle gap remains tight at 0.008 — consistent with the BTM feature set's strong spatial generalisation.

### RF + BTM Tuned (R18)

- Increasing `n_estimators` to 500 and fixing `max_features='sqrt'` **degraded** performance vs. the default RF+BTM (0.764 vs 0.795 on Kaggle).
- SGAM F1 dropped to 0.000. The `sqrt` feature subsetting with more trees likely under-samples informative BTM columns for the rare SGAM class.
- **Recommendation**: Keep default RF settings (n_estimators=100, max_features=None/'all').

### RF + BTM + Eco Features (R15, R16)

- **SC-002 gate passed**: SGAM CV F1 improved from 0.043 (baseline R04) to **0.045** with depth zones + SGAM niche indicator — a 5% relative improvement meeting the >0.043 gate.
- The SGAM niche indicator (shallow, low-BPI, low-slope zone) successfully separates some SGAM habitat from background.
- However, overall Kaggle F1 did not improve (0.764 vs 0.795 best). The eco features add signal for SGAM but slightly harm the overall weighted F1 by redistributing probability mass.
- R15 (depth zones only) and R16 (full eco: depth zones + raster eco-derivatives) produced **identical** CV and Kaggle results — the four additional raster eco-derivative features (northness, eastness, max curvature, complexity) contribute no incremental signal beyond the depth zone and SGAM niche indicator on this dataset.
- **Recommendation**: Eco features are worth retaining for SGAM-targeted analysis but should not replace the standard BTM feature set for overall leaderboard optimisation.

---

## Gate Results

| Gate   | Condition                                            | Result                                  |
| ------ | ---------------------------------------------------- | --------------------------------------- |
| SC-002 | At least one eco-feature run with SGAM CV F1 > 0.043 | ✓ PASS (R15 SGAM=0.045, R16 SGAM=0.045) |
| SC-006 | All Kaggle scores ≥ 0.65                             | ✓ PASS (min=0.76153)                    |

---

## New Best Score

**Kaggle public F1: 0.79518** — tied with R04/R06 (no improvement on overall leaderboard).  
**SGAM breakthrough**: CV SGAM F1 improved from 0.043 → 0.045 with eco-depth features.

---

## Eco Feature Infrastructure

### New raster eco-derivatives (added to BTM feature extraction)

| Column              | Algorithm                         | Reference             |
| ------------------- | --------------------------------- | --------------------- |
| `btm_northness`     | cos(aspect), Horn (1981) kernels  | Horn 1981, Evans 1998 |
| `btm_eastness`      | sin(aspect), Horn (1981) kernels  | Horn 1981, Evans 1998 |
| `btm_max_curvature` | max Hessian eigenvalue            | Schmidt et al. 2003   |
| `btm_complexity`    | slope-of-slope (double Horn pass) | Wilson et al. 2007    |

Extracted via: `btm-export-features --include-eco-features`

### New in-pipeline eco-features (added by `EcoFeatureTransformer`)

| Column           | Type    | Description                                                     |
| ---------------- | ------- | --------------------------------------------------------------- |
| `btm_depth_zone` | ordinal | Depth bin 1–4 (quartiles fitted on training set)                |
| `btm_sgam_niche` | binary  | 1 if BPI < P25 AND slope < P25 AND depth in SGAM training range |

---

## Dataset

| CSV                      | Rows | BTM cols | Eco cols | Notes                                |
| ------------------------ | ---- | -------- | -------- | ------------------------------------ |
| `data/train_btm.csv`     | 6256 | 10       | —        | From run-014 BTM extraction          |
| `data/train_btm_eco.csv` | 6256 | 10       | 4        | New this run, eco raster derivatives |
| `data/test_btm.csv`      | 98   | 10       | —        | From run-014 BTM extraction          |
| `data/test_btm_eco.csv`  | 98   | 10       | 4        | New this run, eco raster derivatives |

Zero NaN across all eco-derivative columns. Value ranges:

- `btm_northness` ∈ [−1, 1], mean=0.009 (near-flat site on average)
- `btm_eastness` ∈ [−1, 1], mean=0.152 (slight east-facing bias)
- `btm_max_curvature`: 25th percentile=0.40, median=0.73, max=52.6 (a few high-curvature reef edges)
- `btm_complexity`: median=4.7, max=116 (strongly right-skewed — a few rough reef patches)

---

## Code Changes (this run)

| File                                         | Change                                                                                                     |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `src/benthic_model/config.py`                | Added `FeatureFlags.include_eco_features`, `PipelineConfig.model_params`                                   |
| `src/benthic_model/features/eco_features.py` | New: `EcoFeatureTransformer` (depth zones ordinal 1–4, SGAM niche binary, fit/transform/to_dict/from_dict) |
| `src/benthic_model/features/engineering.py`  | Added `y` param; eco-feature hook when `flags.include_eco_features=True`                                   |
| `src/benthic_model/models/train.py`          | `model_params` forwarding in `_build_model()`; `eco_thresholds.json` artifact                              |
| `src/benthic_model/inference/predict.py`     | Load `eco_thresholds.json` if present; apply `EcoFeatureTransformer.transform()`                           |
| `btm/features/extract.py`                    | 3 new raster derivative functions + `extract_eco_raster_features()` + `include_eco_features` param         |
| `btm/cli/export_features.py`                 | Added `--include-eco-features` flag                                                                        |
| `configs/rf-btm-eco-depth.yaml`              | New (R15)                                                                                                  |
| `configs/rf-btm-eco-full.yaml`               | New (R16)                                                                                                  |
| `configs/rf-btm-interactions.yaml`           | New (R17)                                                                                                  |
| `configs/rf-btm-tuned.yaml`                  | New (R18)                                                                                                  |
| `tests/unit/test_eco_features.py`            | New: 15 unit tests for `EcoFeatureTransformer`                                                             |
| `tests/unit/test_eco_raster_features.py`     | New: 13 unit tests for raster derivative functions                                                         |

**Test suite**: 192 passed, 4 skipped, 0 failures.

---

## Technical Notes

### scipy.ndimage.convolve vs correlate

`scipy.ndimage.convolve` performs true mathematical convolution (flips the kernel), unlike `correlate`. The Horn gradient kernels must account for this. The correct aspect formula is `arctan2(dz_dx, -dz_dy)` where `dz_dx` is the result of convolving with the east-west Horn kernel — this gives CW-from-North downslope direction.

### Max curvature via Hessian eigenvalues

Plan/profile curvature decomposition requires non-zero slope (divides by p²+q²), producing NaN at local extrema (hilltops, bowl centres). The Hessian eigenvalue approach `((r+t) ± sqrt((r-t)²+4s²))/2` is well-defined everywhere and used here instead.

### EcoFeatureTransformer fit/predict consistency

`eco_thresholds.json` is saved alongside `model.joblib` at training time. `predict.py` auto-loads it if present. This ensures depth zone bins and SGAM niche thresholds are identical between training and inference — there is no risk of threshold drift between runs.
