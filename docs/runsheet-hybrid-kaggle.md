# Runsheet: Hybrid BTM + ML Classification on Kaggle-style Benthic Data

This runsheet walks through a complete end-to-end workflow that combines
BTM terrain derivatives with the `benthic_model` ML pipeline to classify
seafloor habitats from labelled MBES survey data.

The data format matches the
[niwacolours/benthic-terrain-model-kaggle](https://github.com/niwacolours/benthic-terrain-model-kaggle)
competition structure:

```
data/
  train.csv              # labelled points: ID, x, y, class
  test.csv               # unlabelled points: ID, x, y
  bathymetry.tif         # single-band depth GeoTIFF (metres, negative)
  backscatter.tif        # single-band backscatter intensity GeoTIFF
  METADATA.MD            # competition metadata
```

---

## Prerequisites

### Install both packages

```bash
# 1. BTM (this repo) — run from the btm repo root
pip install -e ".[dev,ml]"

# 2. benthic_model — run from the Kaggle repo
git clone https://github.com/niwacolours/benthic-terrain-model-kaggle
cd benthic-terrain-model-kaggle
git checkout 002-boost-weighted-f1
pip install -e .
```

Verify:

```bash
python -c "import btm; import benthic_model; print('Both packages ok')"
btm-export-features --help
```

### Python environment

Both packages require Python 3.11+. If you are using the BTM venv:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install benthic_model into the same venv
pip install -e /path/to/benthic-terrain-model-kaggle
```

---

## Step 1 — Verify your survey data

```bash
python - <<'EOF'
import rasterio, pandas as pd

train = pd.read_csv("data/train.csv")
print(f"Training points: {len(train)}")
print(f"Classes: {sorted(train['class'].unique())}")
print(f"Columns: {list(train.columns)}")

with rasterio.open("data/bathymetry.tif") as src:
    print(f"Bathymetry: {src.width}×{src.height}  CRS={src.crs}  nodata={src.nodata}")

with rasterio.open("data/backscatter.tif") as src:
    print(f"Backscatter: {src.width}×{src.height}  CRS={src.crs}")
EOF
```

**Sanity checks:**

- Training CSV must have columns `ID`, `x`, `y`, `class`.
- Raster CRS and point coordinates must match (both projected or both
  geographic, same EPSG). Reproject points with GeoPandas if needed.
- Bathymetry values should be negative (depth below sea surface). Positive
  values will not affect algorithm correctness but BPI labels will be inverted.

---

## Step 2 — Extract BTM terrain features

```bash
btm-export-features \
  --bathy      data/bathymetry.tif \
  --points     data/train.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner   1 --fine-outer   5 \
  --outdir     data/btm_rasters \
  --output     data/train_btm.csv

# Same for test points
btm-export-features \
  --bathy      data/bathymetry.tif \
  --points     data/test.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner   1 --fine-outer   5 \
  --output     data/test_btm.csv
```

This writes:

- `data/train_btm.csv` — original columns + 10 `btm_*` columns
- `data/test_btm.csv` — same structure for test points
- `data/btm_rasters/btm_*.tif` — full-raster derivatives for inspection

**Expected runtime:** ~10–60 s depending on raster size.

### Choosing BPI radii

BPI radii are in **cells**, so they depend on your survey resolution:

| Raster resolution | Broad inner | Broad outer | Fine inner | Fine outer |
| ----------------- | ----------- | ----------- | ---------- | ---------- |
| 1 m               | 50          | 150         | 5          | 25         |
| 5 m               | 10          | 30          | 1          | 5          |
| 10 m              | 5           | 15          | 1          | 3          |
| 25 m              | 2           | 6           | 1          | 2          |

Use wider radii for detecting shelf-scale ridges/depressions; narrower radii
for detecting individual biogenic structures. Run `btm-scale-compare` to
explore the effect.

---

## Step 3 — Merge BTM features into the training CSV

```python
# merge_btm.py
import pandas as pd

train = pd.read_csv("data/train.csv")
btm   = pd.read_csv("data/train_btm.csv")

# btm contains ID + x + y + btm_* — drop duplicated geometry columns
btm_cols = [c for c in btm.columns if c.startswith("btm_")]
combined = train.merge(btm[["ID"] + btm_cols], on="ID", how="left")
combined.to_csv("data/train_combined.csv", index=False)
print(f"Combined: {combined.shape}  columns={list(combined.columns)}")
```

Run:

```bash
python merge_btm.py
```

Do the same for test data:

```python
test  = pd.read_csv("data/test.csv")
btm_t = pd.read_csv("data/test_btm.csv")
btm_cols = [c for c in btm_t.columns if c.startswith("btm_")]
test_combined = test.merge(btm_t[["ID"] + btm_cols], on="ID", how="left")
test_combined.to_csv("data/test_combined.csv", index=False)
```

---

## Step 4 — Train the hybrid ML model

Use `benthic_model`'s standard training CLI, pointing to the combined CSVs:

```bash
# From the benthic-terrain-model-kaggle repo root
benthic-model train \
  --train-csv    /path/to/btm/data/train_combined.csv \
  --bathymetry   /path/to/btm/data/bathymetry.tif \
  --backscatter  /path/to/btm/data/backscatter.tif \
  --config       config/lgbm_tuned.yaml \
  --run-type     lgbm_tuned \
  --seed         42
```

The `benthic_model` pipeline will:

1. Call `extract_mbes_features()` internally to add backscatter-derived and
   TPI-based features.
2. Call `engineer_features()` to add interaction terms.
3. The `btm_*` columns already present in `train_combined.csv` pass through
   as additional numeric features **automatically** — no code change needed in
   `benthic_model`.

> **How it works:** `engineer_features()` in `benthic_model` calls
> `select_model_feature_columns()` which includes _all numeric columns not in
> the exclusion list_. The `btm_*` prefixed columns are numeric and not
> excluded, so they are automatically included in the feature matrix `X`.

### Enabling interaction terms between BTM and MBES features

For the `btm_rule_class` column to also appear, re-run Step 2 with
`--include-rule-class --classdict data/your_classification.csv`.
The integer class code can give the model a significant lift when there is
reasonable class alignment.

---

## Step 5 — Evaluate

```bash
benthic-model evaluate \
  --run-id  <run_id_from_step_4> \
  --test-csv    data/test_combined.csv \
  --bathymetry  data/bathymetry.tif \
  --backscatter data/backscatter.tif
```

This writes a comparison report and per-class F1 scores to
`reports/metrics/<run_id>_comparison.md`.

### Comparing rule-based vs ML vs hybrid

```bash
# Rule-based (no training data needed)
btm-run-model \
  --bathy      data/bathymetry.tif \
  --classdict  data/classification.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner   1 --fine-outer   5 \
  --outdir     outputs/rule_based

# ML baseline (benthic_model features only, no BTM)
benthic-model train \
  --train-csv   data/train.csv \
  --bathymetry  data/bathymetry.tif \
  --backscatter data/backscatter.tif \
  --config      config/lgbm_tuned.yaml \
  --run-type    lgbm_tuned \
  --seed 42

# Hybrid (BTM features + benthic_model)
# ... as in Step 4 above
```

---

## Step 6 — Generate submission (Kaggle format)

```bash
benthic-model predict \
  --run-id     <run_id_from_step_4> \
  --test-csv   data/test_combined.csv \
  --bathymetry data/bathymetry.tif \
  --backscatter data/backscatter.tif

benthic-model submit \
  --run-id     <run_id_from_step_4> \
  --test-csv   data/test_combined.csv
```

---

## What to expect

### Feature importance

On typical coral-reef MBES surveys the BTM derivatives are expected to rank
highly in LightGBM feature importance:

| Expected rank | Feature                      | Reason                                    |
| ------------- | ---------------------------- | ----------------------------------------- |
| Top 5         | `btm_broad_std`              | Primary landscape-scale discriminator     |
| Top 5         | `btm_fine_std`               | Primary local-scale discriminator         |
| Top 10        | `btm_vrm`                    | Biogenic structure indicator              |
| Top 10        | `btm_slope`                  | Confirms depth-gradient class transitions |
| Mid           | `btm_bpi_magnitude`          | Combined relief signal                    |
| Mid           | `btm_rule_class` _(if used)_ | Expert prior — very site-specific         |

The backscatter and TPI features from `benthic_model` remain important; BTM
features reduce the number of features needed to explain transitions but do
not replace acoustic information.

### Classification performance

Based on the class structure and the nature of BTM features, expect:

| Scenario                         | Expected Δ weighted-F1                                 |
| -------------------------------- | ------------------------------------------------------ |
| Bathymetry only (no backscatter) | **+3 to +8 pp** vs TPI-only baseline                   |
| With backscatter                 | **+1 to +4 pp** (BTM adds independently)               |
| Transition / mixed zones         | Largest improvements — these are where                 |
|                                  | hard BTM thresholds fail but BPI signal is informative |
| Homogeneous flat areas           | Small or no improvement                                |

These are estimates; actual gains depend on training set size, CRS accuracy,
and class distribution. Always validate with the spatial blocked CV scores.

### Why might it _not_ help?

- If the habitat classes are primarily substrate-driven (hard vs soft bottom)
  rather than morphology-driven, backscatter dominates and BTM adds little.
- If the BPI radii are poorly matched to the survey scale, BTM derivatives
  will be noisy.
- With very few training samples (<50 per class), adding more features can
  hurt by increasing the effective dimensionality.

---

## Troubleshooting

### CRS mismatch

```
ValueError: sample points fall outside raster extent
```

Reproject points to match raster CRS:

```python
import geopandas as gpd

gdf = gpd.GeoDataFrame(
    train, geometry=gpd.points_from_xy(train.x, train.y), crs="EPSG:4326"
)
target_epsg = 32756  # replace with your raster EPSG
gdf = gdf.to_crs(epsg=target_epsg)
train["x"] = gdf.geometry.x
train["y"] = gdf.geometry.y
```

### NaN values in btm\_\* columns

All BTM columns default to NaN for points that land outside the raster extent
or on nodata cells. Check:

```python
btm = pd.read_csv("data/train_btm.csv")
print(btm[["btm_broad_std","btm_slope"]].isna().sum())
```

`benthic_model`'s `engineer_features()` fills NaN with column median, so a
small number of NaN points is handled automatically.

### Memory on large rasters

For rasters wider than ~10,000 cells, the VRM and BPI computations may
require several GB of RAM. A tiled block-processing mode for
`extract_btm_features()` is planned for a future release.

---

## Summary command sequence

```bash
# Install
pip install -e ".[dev,ml]"
pip install -e /path/to/benthic-terrain-model-kaggle

# Extract BTM features
btm-export-features --bathy data/bathymetry.tif --points data/train.csv \
  --broad-inner 10 --broad-outer 30 --fine-inner 1 --fine-outer 5 \
  --output data/train_btm.csv

# Merge
python merge_btm.py

# Train hybrid model
cd /path/to/benthic-terrain-model-kaggle
benthic-model train --train-csv /path/to/btm/data/train_combined.csv \
  --bathymetry /path/to/btm/data/bathymetry.tif \
  --backscatter /path/to/btm/data/backscatter.tif \
  --config config/lgbm_tuned.yaml --run-type lgbm_tuned --seed 42

# Evaluate
benthic-model evaluate --run-id <run_id> \
  --test-csv /path/to/btm/data/test_combined.csv \
  --bathymetry /path/to/btm/data/bathymetry.tif \
  --backscatter /path/to/btm/data/backscatter.tif
```

---

## Experiment Run Registry

Chronological log of all experiments run on the Kaggle-style benthic dataset.
See `docs/` for per-run detailed write-ups.

| Experiment                | Branch / Doc                                   | Feature Set                                      | Feature Count | CV F1  | Kaggle F1 | Notes                                         |
| ------------------------- | ---------------------------------------------- | ------------------------------------------------ | ------------- | ------ | --------- | --------------------------------------------- |
| R04                       | —                                              | BTM base (RF)                                    | ~10           | 0.8024 | 0.79518   | Best Kaggle score; reference RF+BTM baseline  |
| R06                       | —                                              | BTM base (RF)                                    | ~10           | 0.8024 | 0.79518   | Tied R04 on leaderboard                       |
| R09                       | run-015                                        | BTM features (CatBoost GPU)                      | ~10           | 0.8139 | 0.76153   | Highest CV but largest CV–Kaggle gap (0.052)  |
| R15                       | [run-015](run-015-ecology-habitat-features.md) | BTM + depth zones + SGAM niche (RF)              | ~12           | 0.7916 | 0.76438   | SGAM recall 0.045 (+5% vs R04)                |
| R16                       | [run-015](run-015-ecology-habitat-features.md) | BTM + depth zones + SGAM niche + raster eco (RF) | ~16           | 0.7916 | 0.76438   | Identical to R15; raster eco adds no signal   |
| R17                       | [run-015](run-015-ecology-habitat-features.md) | BTM + pairwise interactions (RF)                 | ~50           | 0.7971 | 0.79518   | Tied best Kaggle; interactions add no signal  |
| R18                       | [run-015](run-015-ecology-habitat-features.md) | BTM (RF, n_est=500, sqrt features)               | ~10           | 0.7960 | 0.76438   | Tuning degraded Kaggle score                  |
| experiment_v10 (full)     | [run-016](run-016-multiscale-terrain.md)       | BTM multi-scale (35) + RDMV (5) + GLCM (10)      | 64            | 0.6377 | TBD       | CatBoost CPU; +11 pp vs pure BTM base         |
| experiment_v10 (selected) | [run-016](run-016-multiscale-terrain.md)       | As above, correlation + permutation filtered     | 33            | 0.6383 | TBD       | SC-004 degradation PASS; 1/98 prediction diff |
| experiment_v11 (combined) | [run-017](run-017-combined-multiscale-mbes.md) | BTM-33 + MBES-8 (CatBoost)                       | 41            | 0.7829 | TBD       | +14.5 pp vs BTM-only; SGAM recall 0.142       |
| experiment_v11 (selected) | [run-017](run-017-combined-multiscale-mbes.md) | As above, permutation filtered                   | 38            | 0.7829 | TBD       | SC-004 degradation PASS; 0/98 prediction diff |

### Why experiment_v10 (0.6377) is far below R04/R06 (0.8024) — two separate causes

**Cause 1 — Missing MBES core 8 features (primary cause, ~+0.15 pp effect)**

R04/R06 ran through the `benthic_model` pipeline, which _always_ includes the 8 MBES point-sample features derived directly from the bathymetry and backscatter rasters at each sample location:
depth, backscatter, slope, VRM, complexity, max*curvature, northness, eastness.
These are the strongest single predictors on this dataset (run-014 Phase 1: RF on MBES-8 alone = CV 0.797,
Kaggle 0.731). The BTM terrain derivatives are additive on top of them, not a replacement.
`experiment_v10` is a \_BTM-feature-only* standalone script — it never injects these MBES-8 columns.
The 0.8024 score is therefore _not achievable_ from BTM features alone regardless of model choice.

**Cause 2 — Model choice (secondary cause, ~+0.006 pp if RF used instead)**

run-014 Phase 3 benchmarked all model types on the same BTM-winner feature set inside `benthic_model`:

| Model                                                                              | CV F1 (same features) |
| ---------------------------------------------------------------------------------- | --------------------- |
| **RF** (`n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample'`) | **0.8024**            |
| CatBoost                                                                           | 0.7965                |
| LightGBM                                                                           | 0.7954                |
| RF+LGBM ensemble                                                                   | 0.7967                |

RF outperforms CatBoost by ~0.006 points on these features, likely because
`class_weight='balanced_subsample'` handles the 5-class imbalance better than CatBoost's
`auto_class_weights`. So yes — replacing CatBoost with RF in `experiment_v10` will improve
the CV score, but by a modest 0.5–1 pp, not 14 pp.

**Recommended next experiments**

| Priority  | Experiment                                                                                                               | Expected gain         | Rationale                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------- | --------------------------------------------------------------- |
| High      | Add RF to `experiment_v10.py` as third model (`n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample'`) | +0.5–1 pp vs CatBoost | Establishes true BTM-only RF baseline with multi-scale features |
| Very High | New experiment combining multi-scale BTM + MBES-8 features via `benthic_model` pipeline                                  | +12–15 pp estimated   | Addresses Cause 1; the correctly comparable experiment to R04   |

The correct like-for-like baseline for multi-scale BTM _alone_ is BTM-base CatBoost (~0.52 CV F1),
against which `experiment_v10` shows a genuine +11 pp gain.
