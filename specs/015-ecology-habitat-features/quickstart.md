# Quickstart: Ecology-Informed Habitat Features

**Branch**: `015-ecology-habitat-features` | **Date**: 2026-03-31

---

## Prerequisites

- Branch `015-ecology-habitat-features` checked out
- Existing `data/train_btm.csv` present (from the run-014 BTM extraction)
- `data/MBES/bathymetry.tif` available
- `.venv` activated: `.venv\Scripts\activate`

---

## Step 1 — Submit the GPU CatBoost holdover (Priority 1)

The artifact already exists at `artifacts/runs/candidate-20260330203952/`. No training needed.

```powershell
# Generate prediction CSV
benthic-model predict `
  --run-id candidate-20260330203952 `
  --test-csv data/test.csv `
  --output data/submission_catboost_gpu.csv

# Submit to Kaggle
kaggle competitions submit `
  -c <competition-name> `
  -f data/submission_catboost_gpu.csv `
  -m "candidate-20260330203952 CatBoost GPU task_type=GPU symmetric trees"
```

Expected CV F1: **0.8139**. Record the returned Kaggle F1 in `artifacts/experiments/kaggle_scores.csv`.

---

## Step 2 — Extract eco raster features

Add northness, eastness, max curvature, and complexity to the training and test CSVs:

```powershell
# Training CSV
btm-export-features `
  --bathy data/MBES/bathymetry.tif `
  --points data/train_btm.csv `
  --output data/train_btm_eco.csv `
  --include-eco-features

# Test CSV (use same command on test CSV if it has x,y columns)
btm-export-features `
  --bathy data/MBES/bathymetry.tif `
  --points data/test_btm.csv `
  --output data/test_btm_eco.csv `
  --include-eco-features
```

Verify eco-feature columns are present and NaN rate < 10%:

```powershell
python -c "
import pandas as pd
df = pd.read_csv('data/train_btm_eco.csv')
eco = ['btm_northness','btm_eastness','btm_max_curvature','btm_complexity']
print(df[eco].isna().mean().round(3))
print(df[eco].describe().round(3))
"
```

---

## Step 3 — Run the four eco-feature training experiments

Run each config in priority order (highest Kaggle potential first):

```powershell
# R15: RF + BTM + depth zones only (eco depth signal, minimal new features)
benthic-model train `
  --config configs/rf-btm-eco-depth.yaml `
  --train-csv data/train_btm_eco.csv

# R16: RF + BTM + all eco derivatives (full paper feature set)
benthic-model train `
  --config configs/rf-btm-eco-full.yaml `
  --train-csv data/train_btm_eco.csv

# R17: RF + BTM + pairwise interactions (combines best feature sets)
benthic-model train `
  --config configs/rf-btm-interactions.yaml `
  --train-csv data/train_btm.csv

# R18: RF + BTM + tuned hyperparameters (n_estimators=500, max_features=sqrt)
benthic-model train `
  --config configs/rf-btm-tuned.yaml `
  --train-csv data/train_btm.csv
```

After each run, note the CV weighted F1 printed to the terminal. Compare to the current best: **RF+BTM CV=0.8024**.

---

## Step 4 — Generate and submit today's remaining Kaggle batch (4 submissions)

```powershell
# After each training run, get its run_id from: artifacts/experiments/run_registry.jsonl
# Then generate predictions and submit:

foreach ($run_id in @("rf-btm-eco-depth-XXXXXXXX", "rf-btm-eco-full-XXXXXXXX", "rf-btm-interactions-XXXXXXXX", "rf-btm-tuned-XXXXXXXX")) {
    benthic-model predict `
      --run-id $run_id `
      --test-csv data/test_btm_eco.csv `
      --output "data/submission_$run_id.csv"

    kaggle competitions submit `
      -c <competition-name> `
      -f "data/submission_$run_id.csv" `
      -m $run_id
}
```

Record all scores in `artifacts/experiments/kaggle_scores.csv`.

---

## Step 5 — Check SGAM per-class F1

After eco-feature runs, compare SGAM F1 across runs:

```powershell
python -c "
import json, glob
for path in sorted(glob.glob('artifacts/runs/*/metrics.json')):
    m = json.load(open(path))
    run = path.split('/')[-2]
    sgam_f1 = m.get('class_f1', {}).get('SGAM', m.get('per_class_f1', {}).get('SGAM', 'n/a'))
    overall = m.get('metric_weighted_f1', 'n/a')
    print(f'{run:50s}  overall={overall:.4f}  SGAM={sgam_f1}')
"
```

Target: at least one eco-feature run with SGAM F1 > 0.043 (current best across all 014 runs).

---

## Eco-Feature Flag Reference

| Config | `include_btm_features` | `include_eco_features` | `include_interactions` | `model_params` |
|---|---|---|---|---|
| `rf-btm-eco-depth.yaml` | true | true | false | — |
| `rf-btm-eco-full.yaml` | true | true | false | — |
| `rf-btm-interactions.yaml` | true | false | true | — |
| `rf-btm-tuned.yaml` | true | false | false | n_estimators=500, max_features=sqrt |

---

## Using `BTM_USE_GPU` with eco-feature experiments

The eco-feature configs use RF (not gradient boosters), so `BTM_USE_GPU` has no effect on them. For CatBoost or LGBM runs with eco features, the GPU flag works as before:

```powershell
$env:BTM_USE_GPU=1  # force GPU
benthic-model train --config configs/catboost-candidate.yaml --train-csv data/train_btm_eco.csv
```
