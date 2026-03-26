# BTM Classification Methods

This document explains the two classification approaches available in this
repository, when to use each one, and how to combine them.

---

## 1. Rule-based classification (BTM CON cascade)

### What it does

BTM's rule-based classifier assigns a terrain class to every pixel by
evaluating an ordered list of expert-defined rules against four derived
rasters:

| Input                        | Computed by                       |
| ---------------------------- | --------------------------------- |
| Broad-scale standardised BPI | `btm-bpi` → `btm-standardize-bpi` |
| Fine-scale standardised BPI  | `btm-bpi` → `btm-standardize-bpi` |
| Slope (degrees)              | `btm-slope`                       |
| Depth (raw bathymetry)       | input raster                      |

Each rule specifies an optional numeric range for each of the four inputs. A
pixel matches a class when **all supplied ranges are satisfied**. Rules are
evaluated in order; the **first match wins**.

```
# Example from fagatelebay.csv
Class 1  (Reef Crest):    BroadBPI < -100  AND  FineBPI < -100
Class 2  (Mid-Slope Ridges):  BroadBPI < -100  AND  FineBPI in [-100, 100]
…
```

### Strengths

- **No training data required** — works on any bathymetric raster.
- **Fully deterministic** — every pixel decision is explainable by rule.
- **Ecologically grounded** — rules are authored by domain scientists.
- **Portable** — same rules apply across survey areas with similar
  geomorphological regimes.

### Limitations

- Rules are hard thresholds; real class boundaries are gradients.
- Transition zones (the edges between classes) are misclassified by
  whichever adjacent class has the wider range.
- No use of backscatter / acoustic substrate information.
- Requires a classification dictionary written by a domain expert for the
  target environment.

### When to use

- Rapid habitat mapping of a new area with no field samples.
- When interpretability and reproducibility are mandatory.
- As a first-pass product before deploying a trained model.
- Small dataset (<100 labelled points): insufficient for ML training.

### How to run

```bash
# Full pipeline in one command
btm-run-model \
  --bathy     data/bathy.tif \
  --classdict data/classification.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner   1 --fine-outer   5 \
  --outdir    outputs/rule_based

# See TESTING.md for a complete Fagatele Bay walkthrough
```

---

## 2. ML classification (benthic_model — XGBoost / LightGBM)

### What it does

`benthic_model` (see
[niwacolours/benthic-terrain-model-kaggle](https://github.com/niwacolours/benthic-terrain-model-kaggle))
is a supervised machine-learning classifier trained on labelled field samples
(e.g. drop-camera or grab-sample points). It uses gradient-boosted tree
ensembles (XGBoost, LightGBM) to learn a mapping from per-pixel feature
vectors to habitat classes.

Default feature set from MBES surveys:

| Feature group      | Examples                                            |
| ------------------ | --------------------------------------------------- |
| Raw acoustic       | `bathymetry`, `backscatter`                         |
| Terrain morphology | `slope`, `TPI` (square window), `rugosity`          |
| Focal statistics   | `bathymetry_std_3`, `bathymetry_std_9`, focal means |
| Spatial context    | z-scores, relative position, rank features          |
| Interaction terms  | `bathymetry × backscatter`, `slope × backscatter`   |

### Strengths

- Learns non-linear, multi-feature decision boundaries from data.
- Exploits backscatter (substrate hardness) — unavailable to BTM.
- Class-weighted training and optional SMOTE handle sample imbalance.
- Spatial blocked cross-validation prevents over-optimistic metrics.

### Limitations

- Requires labelled training samples — quality determines everything.
- A model trained in one area may not transfer to another without retraining.
- Predictions are harder to explain without SHAP / feature importance tools.
- No spatial smoothness guarantee — can produce salt-and-pepper outputs.

### When to use

- Ground-truth data (labelled points) are available.
- Backscatter is available as a co-registered layer.
- You want to distinguish substrate types (soft sediment vs hard bottom)
  that BTM cannot separate by bathymetry alone.

---

## 3. Hybrid classification (BTM features + ML model)

### Why combining them is stronger

The two systems are **complementary, not competing**:

```
BTM knowledge:                          ML learning:
  annular BPI (ecologically motivated)     learns soft boundaries
  VRM (tri-axial roughness)                uses backscatter
  surface-to-planar ratio                  handles class imbalance
  expert CON cascade as a prior            cross-validates spatially
         ↓                                          ↓
         └──────── combined feature vector ─────────┘
                            ↓
                   XGBoost / LightGBM
                            ↓
                   predicted habitat class
```

**Specific improvements expected:**

1. **Standardised BPI outperforms TPI for seafloor landscapes.**
   BPI's annular footprint excludes the cell's immediate neighbours, making
   it sensitive to position relative to the _surrounding landscape_ rather
   than only the local window. For example, a reef crest sits atop a rise
   whose base is outside typical TPI windows (10–30 cell radii).

2. **VRM is theoretically stronger than std-based roughness.**
   VRM captures the tri-axial variance of surface normals (x, y, z
   components), which is scale-invariant and sensitive to small-scale
   structural complexity (biogenic reefs, rock outcrop). Rugosity based on
   bathymetric std is a rougher proxy.

3. **Cross-scale BPI reveals habitat nesting.**
   Broad BPI characterises landscape position (shelf ridge vs depression);
   fine BPI characterises local structure (individual pinnacle vs flat).
   Their _product_ and _magnitude_ give the ML model information about
   nested terrain features that neither scale provides alone.

4. **Rule-class as a spatial prior.**
   Passing the BTM CON class code as an integer feature gives the model a
   domain-expert prior that is essentially free. The model can learn which
   BTM classes correspond to its target classes and adjust from there.

### Architecture

```
bathymetry_tif
     │
     ├─► btm/core/         ── BPI, slope, VRM, surface_ratio
     │        │
     │   btm/features/extract_btm_features()
     │        │
     │        ▼
     │   btm_* columns (sampled at ground-truth points)
     │
     └─► benthic_model/data/raster_extract.extract_mbes_features()
              │
              ▼
         mbes columns (bathymetry, backscatter, TPI, rugosity, …)

              merge on ID
                   │
                   ▼
         combined feature DataFrame
                   │
                   ▼
         benthic_model/features/engineering.engineer_features()
                   │
                   ▼
         XGBoost / LightGBM training + spatial CV
                   │
                   ▼
         trained model (artifacts/runs/<run_id>/model.joblib)
```

### New features added to the ML feature vector

| Column                 | Description                       | Why it helps                          |
| ---------------------- | --------------------------------- | ------------------------------------- |
| `btm_broad_bpi`        | Raw broad BPI                     | Landscape-scale position              |
| `btm_fine_bpi`         | Raw fine BPI                      | Local-scale structure                 |
| `btm_broad_std`        | Z-normalised broad BPI ×100       | Scale-independent position signal     |
| `btm_fine_std`         | Z-normalised fine BPI ×100        | Scale-independent local signal        |
| `btm_slope`            | Horn (1981) slope in degrees      | Cross-check against TPI-derived slope |
| `btm_vrm`              | Sappington VRM [0, 1]             | Tri-axial roughness                   |
| `btm_surface_ratio`    | Jenness surface/planar ratio      | Fine-scale complexity                 |
| `btm_rule_class`       | CON cascade class code (optional) | Domain expert prior                   |
| `btm_bpi_magnitude`    | √(broad²+fine²)                   | Strength of terrain relief signal     |
| `btm_broad_x_fine_std` | broad_std × fine_std              | Sign-coherent multi-scale             |
| `btm_rough_total`      | vrm + (surface_ratio − 1)         | Unified roughness index               |

### When to use

- You have labelled ground-truth **and** bathymetry only (no backscatter):
  BTM derivatives are the only texture features available.
- You have **both** bathymetry and backscatter: the hybrid should outperform
  either method alone because the two information sources are independent.
- Any time the rule-based classification shows obvious misclassification in
  transition zones — the ML model will soften those boundaries.

---

## 4. Choosing a method — decision guide

```
Do you have labelled training samples?
  │
  ├── NO  ──►  Use BTM rule-based (btm-run-model)
  │
  └── YES
        │
        ├── Do you have backscatter?
        │        │
        │        ├── YES  ──►  Hybrid (best option)
        │        │               benthic_model + extract_btm_features()
        │        │
        │        └── NO   ──►  Hybrid (BTM features are the main signal)
        │                       benthic_model + extract_btm_features()
        │
        └── How many labelled samples?
                 │
                 ├── < 50     ──►  Rule-based (ML will overfit)
                 ├── 50–200   ──►  RF baseline with BTM features
                 └── > 200    ──►  LightGBM or XGBoost hybrid
```

---

## 5. Installing both repos

```bash
# BTM core (this repo)
pip install -e ".[dev,ml]"

# benthic_model (from your Kaggle repo)
git clone https://github.com/niwacolours/benthic-terrain-model-kaggle
cd benthic-terrain-model-kaggle
git checkout 002-boost-weighted-f1
pip install -e .
```

Both run under Python 3.11+. BTM has no conflicting dependencies with
`benthic_model`.

---

## 6. Exporting BTM features as CSV (standalone)

A `btm-export-features` CLI command is available to export BTM derivatives
for any set of (x, y) sample points without running the full ML pipeline:

```bash
btm-export-features \
  --bathy      data/bathy.tif \
  --points     data/train.csv \
  --broad-inner 10 --broad-outer 30 \
  --fine-inner   1 --fine-outer   5 \
  --output     data/train_btm_features.csv

# With optional rule-class column:
btm-export-features \
  --bathy      data/bathy.tif \
  --points     data/train.csv \
  --classdict  data/classification.csv \
  --include-rule-class \
  --output     data/train_btm_features.csv
```

The output CSV can then be merged into any ML pipeline, not just
`benthic_model`.

See [docs/runsheet-hybrid-kaggle.md](runsheet-hybrid-kaggle.md) for a
complete end-to-end example.
