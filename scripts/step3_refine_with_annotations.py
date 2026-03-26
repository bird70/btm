"""
Step 3: Refine predictions using expert-annotated GeoPackage.

Reads the analyst's expert labels from the annotated GeoPackage and uses them
as additional features / stratification for the spatial ensemble. Produces:

  - data/submission_v4.csv              — Kaggle-format submission
  - outputs/gis_layers/reports/v4_reproducibility.md  — Method documentation
  - outputs/gis_layers/reports/v4_metrics.json        — Machine-readable metrics

Usage:
  python scripts/step3_refine_with_annotations.py [--gpkg PATH] [--output CSV]
  python scripts/step3_refine_with_annotations.py --no-annotations  # automated fallback

If --no-annotations is passed, the script runs the full ensemble using only
automated zone features (equivalent to v3 with extended features).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform as rtransform
from numpy.linalg import LinAlgError
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from btm.core.slope import compute_slope
from btm.core.vrm import compute_vrm
from btm.features.extract import sample_raster_at_points

_DATA = _REPO / "data"
_OUT = _REPO / "outputs" / "gis_layers"
CRS_EPSG = 28355


def _load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform


def _sample_zone(zone_raster, transform, xs, ys):
    rows, cols = rtransform.rowcol(transform, xs, ys)
    rows = np.clip(np.asarray(rows, dtype=int), 0, zone_raster.shape[0] - 1)
    cols = np.clip(np.asarray(cols, dtype=int), 0, zone_raster.shape[1] - 1)
    return zone_raster[rows, cols]


def _indicator_kriging(xs_tr, ys_tr, y_tr, xs_te, ys_te, n_classes, max_tr=3000):
    rng = np.random.RandomState(42)
    proba = np.zeros((len(xs_te), n_classes))
    sub = rng.choice(len(xs_tr), min(max_tr, len(xs_tr)), replace=False)
    for c in range(n_classes):
        ind = (y_tr[sub] == c).astype(float)
        if ind.sum() < 5 or ind.sum() > len(ind) - 5:
            proba[:, c] = ind.mean()
            continue
        try:
            ok = OrdinaryKriging(xs_tr[sub], ys_tr[sub], ind,
                                 variogram_model="exponential",
                                 verbose=False, enable_plotting=False, nlags=20)
            z_vals, _ = ok.execute("points", xs_te, ys_te)
            proba[:, c] = np.clip(z_vals, 0, 1)
        except (ValueError, LinAlgError):
            proba[:, c] = ind.mean()
    return proba


def _stratified_kriging(xs_tr, ys_tr, y_tr, zones_tr,
                         xs_te, ys_te, zones_te,
                         n_classes, n_zones, max_per_zone=1500):
    rng = np.random.RandomState(42)
    proba = np.zeros((len(xs_te), n_classes))
    for z in range(n_zones):
        te_mask = zones_te == z
        if not te_mask.any():
            continue
        tr_mask = np.abs(zones_tr.astype(int) - z) <= 1
        if tr_mask.sum() < 20:
            tr_mask = np.ones(len(zones_tr), dtype=bool)
        tr_idx = np.where(tr_mask)[0]
        if len(tr_idx) > max_per_zone:
            tr_idx = rng.choice(tr_idx, max_per_zone, replace=False)
        te_idx = np.where(te_mask)[0]
        for c in range(n_classes):
            ind = (y_tr[tr_idx] == c).astype(float)
            if ind.sum() < 3 or ind.sum() > len(ind) - 3:
                proba[te_idx, c] = ind.mean()
                continue
            try:
                ok = OrdinaryKriging(xs_tr[tr_idx], ys_tr[tr_idx], ind,
                                     variogram_model="exponential",
                                     verbose=False, enable_plotting=False, nlags=15)
                z_vals, _ = ok.execute("points", xs_te[te_idx], ys_te[te_idx])
                proba[te_idx, c] = np.clip(z_vals, 0, 1)
            except (ValueError, LinAlgError):
                proba[te_idx, c] = ind.mean()
    return proba


def extract_features(xs, ys, bathy, back, bathy_f, back_f, back_smooth,
                     bt, bkt, cell_size,
                     bz_raster, af_raster, dz_raster,
                     isobath_depths, depths_all,
                     expert_zone_map=None):
    """Build feature matrix. If expert_zone_map is provided, adds expert zone features."""
    feats = {}
    feats["x"] = xs.copy()
    feats["y"] = ys.copy()
    feats["bathy"] = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs, ys, nodata=-10000)
    feats["back"] = sample_raster_at_points(
        np.nan_to_num(back, nan=-10000), bkt, xs, ys, nodata=-10000)

    # Zone features
    feats["back_zone"] = _sample_zone(bz_raster, bt, xs, ys).astype(float)
    feats["acoustic_facies"] = _sample_zone(af_raster, bt, xs, ys).astype(float)
    feats["depth_zone"] = _sample_zone(dz_raster, bt, xs, ys).astype(float)

    # Expert zone feature (if available)
    if expert_zone_map is not None:
        feats["expert_zone"] = expert_zone_map.astype(float)

    # BTM derivatives
    slope_arr = compute_slope(bathy_f, cell_size, nodata=None)
    feats["slope"] = sample_raster_at_points(slope_arr, bt, xs, ys)
    vrm_arr = compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size)
    feats["vrm"] = sample_raster_at_points(vrm_arr, bt, xs, ys)

    # Focal stats
    for r in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        for arr, p in [(bathy_f, "b_"), (back_f, "k_")]:
            fm = uniform_filter(arr, size=size)
            fsq = uniform_filter(arr ** 2, size=size)
            feats[f"{p}fm_{size}"] = sample_raster_at_points(fm, bt, xs, ys)
            feats[f"{p}std_{size}"] = sample_raster_at_points(
                np.sqrt(np.maximum(fsq - fm ** 2, 0)), bt, xs, ys)
            feats[f"{p}tpi_{size}"] = sample_raster_at_points(arr - fm, bt, xs, ys)

    # Curvature
    for sigma in [1, 2, 4, 8, 16]:
        smooth = gaussian_filter(bathy_f, sigma=sigma)
        dx = np.gradient(smooth, cell_size, axis=1)
        dy = np.gradient(smooth, cell_size, axis=0)
        d2x = np.gradient(dx, cell_size, axis=1)
        d2y = np.gradient(dy, cell_size, axis=0)
        dxy = np.gradient(dx, cell_size, axis=0)
        feats[f"mcurv_s{sigma}"] = sample_raster_at_points((d2x + d2y) / 2, bt, xs, ys)
        feats[f"gcurv_s{sigma}"] = sample_raster_at_points(d2x * d2y - dxy ** 2, bt, xs, ys)
        feats[f"mslope_s{sigma}"] = sample_raster_at_points(np.sqrt(dx ** 2 + dy ** 2), bt, xs, ys)

    # Aspect
    dx = np.gradient(bathy_f, cell_size, axis=1)
    dy = np.gradient(bathy_f, cell_size, axis=0)
    aspect = np.arctan2(-dy, dx)
    feats["asp_sin"] = sample_raster_at_points(np.sin(aspect), bt, xs, ys)
    feats["asp_cos"] = sample_raster_at_points(np.cos(aspect), bt, xs, ys)

    # Relative backscatter
    for r in [2.5, 10.0, 25.0, 50.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        feats[f"rel_back_{size}"] = sample_raster_at_points(
            back_f - uniform_filter(back_f, size=size), bkt, xs, ys)

    # Backscatter gradient
    bk_dx = np.gradient(back_f, cell_size, axis=1)
    bk_dy = np.gradient(back_f, cell_size, axis=0)
    feats["back_grad"] = sample_raster_at_points(np.sqrt(bk_dx ** 2 + bk_dy ** 2), bkt, xs, ys)

    # Interactions
    feats["depth_x_back"] = np.abs(feats["bathy"]) * feats["back"]
    feats["slope_x_back"] = feats["slope"] * feats["back"]
    feats["acoustic_hard"] = feats["back"] / (np.abs(feats["bathy"]) + 1.0)
    feats["vrm_x_back"] = feats["vrm"] * feats["back"]

    # Isobath distances
    for d_iso in isobath_depths:
        on_contour = np.abs(bathy_f - d_iso) < 0.5
        if on_contour.sum() == 0:
            on_contour = np.abs(bathy_f - d_iso) < 1.0
        dist_px = distance_transform_edt(~on_contour)
        dist_m = (dist_px * cell_size).astype(np.float32)
        feats[f"iso_dist_{d_iso:.0f}"] = sample_raster_at_points(dist_m, bt, xs, ys)

    return pd.DataFrame(feats)


def _load_expert_annotations(gpkg_path, xs, ys, crs_epsg):
    """Load expert labels from GeoPackage, spatial-join to points."""
    gdf_zones = gpd.read_file(gpkg_path, layer="acoustic_facies_zones")
    if "expert_label" not in gdf_zones.columns:
        return None, None

    annotated = gdf_zones[gdf_zones["expert_label"].notna()].copy()
    if len(annotated) == 0:
        return None, None

    # Encode expert labels as integers
    unique_labels = sorted(annotated["expert_label"].unique())
    label_map = {lab: i for i, lab in enumerate(unique_labels)}

    # Spatial join: assign each point to the expert zone it falls within
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xs, ys), crs=f"EPSG:{crs_epsg}"
    )
    joined = gpd.sjoin(pts, annotated[["geometry", "expert_label"]], how="left", predicate="within")

    expert_zones = np.full(len(xs), -1, dtype=int)
    for idx, row in joined.iterrows():
        if pd.notna(row.get("expert_label")):
            expert_zones[idx] = label_map[row["expert_label"]]

    return expert_zones, unique_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpkg", default=str(_OUT / "vectors" / "annotation_layers.gpkg"))
    parser.add_argument("--output", default="data/submission_v4.csv")
    parser.add_argument("--no-annotations", action="store_true",
                        help="Skip expert annotations (automated fallback)")
    parser.add_argument("--blend", default="0.60,0.15,0.10,0.15",
                        help="kriging_depth,kriging_global,catboost,lgbm")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n-cv-blocks", default=10, type=int)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args(argv)

    w = [float(x) for x in args.blend.split(",")]
    if len(w) != 4 or abs(sum(w) - 1.0) > 0.01:
        print("ERROR: --blend must be 4 weights summing to 1.0", file=sys.stderr)
        return 1
    w_kd, w_kg, w_cb, w_lg = w

    t_start = time.time()

    # Load data
    print("Loading data...")
    train = pd.read_csv(_DATA / "train.csv")
    test = pd.read_csv(_DATA / "test.csv")
    if "ID" not in train.columns:
        train.insert(0, "ID", range(1, len(train) + 1))

    bathy, bt = _load_raster(_DATA / "MBES" / "bathymetry.tif")
    back, bkt = _load_raster(_DATA / "MBES" / "backscatter.tif")
    cell_size = abs(bt.a)
    bathy_f = np.nan_to_num(bathy, nan=0.0)
    back_f = np.nan_to_num(back, nan=0.0)
    valid_mask = np.isfinite(bathy) & np.isfinite(back)

    enc = LabelEncoder()
    y = enc.fit_transform(train["class"])
    n_classes = len(enc.classes_)
    xs_tr = train["x"].to_numpy(float)
    ys_tr = train["y"].to_numpy(float)
    xs_te = test["x"].to_numpy(float)
    ys_te = test["y"].to_numpy(float)
    coords = train[["x", "y"]].values

    depths_tr = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs_tr, ys_tr, nodata=-10000)

    # Depth zones
    depth_edges = np.quantile(depths_tr[np.isfinite(depths_tr)], np.linspace(0, 1, 6))
    depth_edges[0] -= 1; depth_edges[-1] += 1
    n_dz = 5
    dz_raster = np.clip(np.digitize(bathy_f, depth_edges) - 1, 0, n_dz - 1).astype(np.int32)
    dz_raster[~valid_mask] = -1
    dz_tr = _sample_zone(dz_raster, bt, xs_tr, ys_tr)
    dz_te = _sample_zone(dz_raster, bt, xs_te, ys_te)

    # Backscatter zones
    back_smooth = gaussian_filter(back_f, sigma=10)
    vr, vc = np.where(valid_mask)
    rng = np.random.RandomState(42)
    si = rng.choice(len(vr), min(50000, len(vr)), replace=False)
    n_bz = 5
    km_bz = KMeans(n_clusters=n_bz, random_state=42, n_init=10)
    km_bz.fit(back_smooth[vr[si], vc[si]].reshape(-1, 1))
    bz_raster = np.full(back_smooth.shape, -1, dtype=np.int32)
    bz_raster[valid_mask] = km_bz.predict(back_smooth[valid_mask].reshape(-1, 1))
    zone_means = [back_smooth[bz_raster == z].mean() for z in range(n_bz)]
    remap = np.zeros(n_bz, dtype=int)
    for new_l, old_l in enumerate(np.argsort(zone_means)):
        remap[old_l] = new_l
    bz_raster = np.where(bz_raster >= 0, remap[bz_raster], -1)

    # Acoustic facies
    n_af = 8
    scaler_2d = StandardScaler()
    feats_2d = scaler_2d.fit_transform(np.column_stack([
        bathy_f[vr[si], vc[si]], back_smooth[vr[si], vc[si]]]))
    km_af = KMeans(n_clusters=n_af, random_state=42, n_init=10)
    km_af.fit(feats_2d)
    af_raster = np.full(bathy_f.shape, -1, dtype=np.int32)
    af_raster[valid_mask] = km_af.predict(scaler_2d.transform(
        np.column_stack([bathy_f[valid_mask], back_smooth[valid_mask]])))

    isobath_depths = np.quantile(depths_tr[np.isfinite(depths_tr)], [0.1, 0.25, 0.5, 0.75, 0.9])

    # Expert annotations
    expert_tr = expert_te = expert_labels = None
    if not args.no_annotations:
        gpkg = Path(args.gpkg)
        if gpkg.exists():
            print("Loading expert annotations...")
            expert_tr, expert_labels = _load_expert_annotations(gpkg, xs_tr, ys_tr, CRS_EPSG)
            if expert_tr is not None:
                expert_te, _ = _load_expert_annotations(gpkg, xs_te, ys_te, CRS_EPSG)
                n_annotated = (expert_tr >= 0).sum()
                print(f"  {n_annotated}/{len(expert_tr)} training points have expert labels")
                print(f"  Expert label classes: {expert_labels}")
            else:
                print("  No expert annotations found in GeoPackage (expert_label all NULL)")
        else:
            print(f"  GeoPackage not found: {gpkg}")

    has_expert = expert_tr is not None and (expert_tr >= 0).sum() > 0

    # Extract features
    print("Extracting features (train)...")
    X_train = extract_features(
        xs_tr, ys_tr, bathy, back, bathy_f, back_f, back_smooth,
        bt, bkt, cell_size, bz_raster, af_raster, dz_raster,
        isobath_depths, depths_tr,
        expert_zone_map=expert_tr,
    )
    n_feats = X_train.shape[1]
    print(f"  {n_feats} features")

    print("Extracting features (test)...")
    X_test = extract_features(
        xs_te, ys_te, bathy, back, bathy_f, back_f, back_smooth,
        bt, bkt, cell_size, bz_raster, af_raster, dz_raster,
        isobath_depths, depths_tr,
        expert_zone_map=expert_te,
    )

    med = X_train.median()
    X_tr_clean = X_train.fillna(med).replace([np.inf, -np.inf], 0)
    X_te_clean = X_test.fillna(med).replace([np.inf, -np.inf], 0)
    feat_cols = list(X_tr_clean.columns)

    # Spatial CV
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    cv_scores = {}
    if not args.skip_cv:
        print(f"\n{'='*60}")
        print(f"Spatial-block CV ({args.n_cv_blocks} blocks)")
        print(f"{'='*60}")

        km_sp = KMeans(n_clusters=args.n_cv_blocks, random_state=args.seed, n_init=10)
        blocks = km_sp.fit_predict(coords)

        model_names = ["kriging_depth", "kriging_global", "catboost", "lgbm"]
        oof = {k: np.zeros((len(y), n_classes)) for k in model_names}

        for b in range(args.n_cv_blocks):
            va = np.where(blocks == b)[0]
            tr = np.where(blocks != b)[0]
            t0 = time.time()

            oof["kriging_depth"][va] = _stratified_kriging(
                xs_tr[tr], ys_tr[tr], y[tr], dz_tr[tr],
                xs_tr[va], ys_tr[va], dz_tr[va], n_classes, n_dz)
            oof["kriging_global"][va] = _indicator_kriging(
                xs_tr[tr], ys_tr[tr], y[tr], xs_tr[va], ys_tr[va], n_classes)

            cb = CatBoostClassifier(
                iterations=1000, depth=6, learning_rate=0.02,
                l2_leaf_reg=10.0, random_seed=args.seed, verbose=0,
                auto_class_weights="Balanced")
            cb.fit(X_tr_clean.iloc[tr], y[tr])
            oof["catboost"][va] = cb.predict_proba(X_tr_clean.iloc[va])

            lgbm = LGBMClassifier(
                n_estimators=1000, max_depth=8, learning_rate=0.02,
                num_leaves=63, subsample=0.7, colsample_bytree=0.5,
                min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
                class_weight="balanced", random_state=args.seed, n_jobs=-1, verbose=-1)
            lgbm.fit(X_tr_clean.iloc[tr], y[tr])
            oof["lgbm"][va] = lgbm.predict_proba(X_tr_clean.iloc[va])

            elapsed = time.time() - t0
            scores = "  ".join(
                f"{k}={f1_score(y[va], np.argmax(oof[k][va], 1), average='weighted'):.4f}"
                for k in model_names)
            print(f"  Block {b:2d} ({len(va):4d} pts, {elapsed:5.0f}s): {scores}")

        for name in model_names:
            f1 = f1_score(y, np.argmax(oof[name], 1), average="weighted")
            cv_scores[name] = f1
            print(f"  OOF {name:20s}: {f1:.4f}")

        blend_oof = (w_kd * oof["kriging_depth"] + w_kg * oof["kriging_global"] +
                     w_cb * oof["catboost"] + w_lg * oof["lgbm"])
        blend_preds = enc.classes_[np.argmax(blend_oof, axis=1)]
        blend_f1 = f1_score(enc.inverse_transform(y), blend_preds, average="weighted")
        cv_scores["blend"] = blend_f1
        print(f"\n  Blend ({w_kd}/{w_kg}/{w_cb}/{w_lg}): {blend_f1:.4f}")
        print(classification_report(enc.inverse_transform(y), blend_preds))

    # Train final models on all data
    print("Training final models...")

    print("  Depth-stratified kriging...")
    p_kd = _stratified_kriging(xs_tr, ys_tr, y, dz_tr, xs_te, ys_te, dz_te,
                                n_classes, n_dz)
    print("  Global kriging...")
    p_kg = _indicator_kriging(xs_tr, ys_tr, y, xs_te, ys_te, n_classes)

    print("  CatBoost...")
    cb_final = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.02,
        l2_leaf_reg=10.0, random_seed=args.seed, verbose=0,
        auto_class_weights="Balanced")
    cb_final.fit(X_tr_clean, y)
    p_cb = cb_final.predict_proba(X_te_clean)

    print("  LightGBM...")
    lgbm_final = LGBMClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.02,
        num_leaves=63, subsample=0.7, colsample_bytree=0.5,
        min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
        class_weight="balanced", random_state=args.seed, n_jobs=-1, verbose=-1)
    lgbm_final.fit(X_tr_clean, y)
    p_lg = lgbm_final.predict_proba(X_te_clean)

    test_blend = w_kd * p_kd + w_kg * p_kg + w_cb * p_cb + w_lg * p_lg
    predictions = enc.inverse_transform(np.argmax(test_blend, axis=1))

    # Write submission
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"ID": test["ID"].values, "class": predictions})
    submission.to_csv(output, index=False)
    print(f"\nSubmission: {output.resolve()}")
    print(f"Rows: {len(submission)}")
    print(f"Class distribution:\n{submission['class'].value_counts().to_string()}")

    # Reproducibility report
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO), text=True).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(_REPO), text=True).strip()
    except Exception:
        git_rev = git_branch = "unknown"

    version = "v4" if has_expert else "v4-auto"
    elapsed_total = time.time() - t_start

    report = f"""# Reproducibility Report: submission_{version}

**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Git revision**: {git_rev}
**Branch**: {git_branch}
**Runtime**: {elapsed_total:.0f}s

## Method

**Ensemble**: Depth-stratified indicator kriging + global indicator kriging + CatBoost + LightGBM
**Blend weights**: kriging_depth={w_kd}, kriging_global={w_kg}, catboost={w_cb}, lgbm={w_lg}
**Features**: {n_feats} ({n_feats - 1 if has_expert else n_feats} terrain + {'1 expert zone' if has_expert else 'no expert zones'})
**Expert annotations**: {'Yes — ' + str(expert_labels) if has_expert else 'No (automated fallback)'}

### Kriging Configuration
- Variogram model: exponential
- Max training points per kriging call: 3000 (global), 1500 (per zone)
- Depth zones: 5 (quantile-based)
- Adjacent zone overlap: ±1

### GBDT Configuration
- CatBoost: 1000 iter, depth=6, lr=0.02, l2=10, balanced weights
- LightGBM: 1000 iter, max_depth=8, lr=0.02, num_leaves=63, balanced weights

### Cross-Validation
- Spatial-block CV: {args.n_cv_blocks} folds (KMeans on coordinates)
- Metric: weighted F1 score
{chr(10).join(f'- {k}: {v:.4f}' for k, v in cv_scores.items()) if cv_scores else '- Skipped (--skip-cv)'}

### Feature Set
- Coordinates (x, y)
- Raw bathymetry + backscatter
- Multi-scale focal statistics (8 radii × 2 rasters × 3 stats)
- Multi-scale curvature (5 Gaussian sigmas × 3)
- BTM terrain derivatives (slope, VRM)
- Aspect (sin, cos)
- Relative backscatter (4 scales)
- Backscatter gradient magnitude
- Interaction features (depth×back, slope×back, acoustic hardness, VRM×back)
- Zone features (depth zone, backscatter zone, acoustic facies)
- Isobath distances (5 quantile contours)
{'- Expert zone labels (spatial join from annotated GeoPackage)' if has_expert else ''}

## Output
- Submission: {output.name}
- Rows: {len(submission)}
- Class distribution: {dict(submission['class'].value_counts())}

## Reproduction Command
```
python scripts/step3_refine_with_annotations.py {'--gpkg ' + args.gpkg if has_expert else '--no-annotations'} --output {args.output} --blend {args.blend} --seed {args.seed}
```
"""

    rpt_dir = _OUT / "reports"
    rpt_dir.mkdir(parents=True, exist_ok=True)
    (rpt_dir / f"{version}_reproducibility.md").write_text(report, encoding="utf-8")

    metrics = {
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev,
        "blend_weights": {"kriging_depth": w_kd, "kriging_global": w_kg,
                          "catboost": w_cb, "lgbm": w_lg},
        "n_features": n_feats,
        "has_expert_annotations": has_expert,
        "cv_scores": cv_scores,
        "submission_file": str(output),
        "class_distribution": dict(submission["class"].value_counts()),
    }
    (rpt_dir / f"{version}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print(f"\nReport: {rpt_dir / f'{version}_reproducibility.md'}")
    print(f"Metrics: {rpt_dir / f'{version}_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
