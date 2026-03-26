"""
Spatial Kriging Ensemble submission v3 for GeoHab 2026 MLWG Kaggle competition.

Approach:  60% Depth-Stratified Kriging + 15% Global Kriging + 10% CatBoost + 15% LightGBM
           probability-weighted blend

Key insight: Depth-stratified indicator kriging captures local spatial
structure within bathymetric zones. Blending with global kriging adds
complementary large-scale spatial trends, while CatBoost/LightGBM contribute
terrain-morphology generalisation from 87 features (including isobath
distance, backscatter zones, and interaction features).

Spatial-block CV (10-fold KMeans on coordinates): weighted-F1 ≈ 0.77

Output: data/submission_v3.csv — ready for Kaggle upload in ID,class format
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numpy.linalg import LinAlgError
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from btm.core.slope import compute_slope
from btm.core.vrm import compute_vrm
from btm.features.extract import sample_raster_at_points

_DATA = _REPO / "data"


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------

def _load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform


def _sample_zone_at_points(zone_raster, transform, xs, ys):
    """Sample integer zone raster at (x, y) points."""
    import rasterio.transform as rtransform
    rows, cols = rtransform.rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    h, w = zone_raster.shape
    rows = np.clip(rows, 0, h - 1)
    cols = np.clip(cols, 0, w - 1)
    return zone_raster[rows, cols]


# ---------------------------------------------------------------------------
# Feature extraction (87 features)
# ---------------------------------------------------------------------------

def extract_features(
    xs, ys,
    bathy, back, bathy_f, back_f, back_smooth,
    bt, bkt, cell_size,
    back_zone_raster, combined_zone_raster,
    isobath_depths, depths_all,
):
    feats = {}

    feats["x"] = xs.copy()
    feats["y"] = ys.copy()
    feats["bathy"] = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs, ys, nodata=-10000,
    )
    feats["back"] = sample_raster_at_points(
        np.nan_to_num(back, nan=-10000), bkt, xs, ys, nodata=-10000,
    )

    # Zone features
    feats["back_zone"] = _sample_zone_at_points(back_zone_raster, bt, xs, ys).astype(float)
    feats["combined_zone"] = _sample_zone_at_points(combined_zone_raster, bt, xs, ys).astype(float)

    # BTM derivatives
    slope_arr = compute_slope(bathy_f, cell_size, nodata=None)
    feats["slope"] = sample_raster_at_points(slope_arr, bt, xs, ys)
    vrm_arr = compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size)
    feats["vrm"] = sample_raster_at_points(vrm_arr, bt, xs, ys)

    # Focal stats (8 radii × 2 rasters × 3 stats = 48 features)
    for r in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        for arr, prefix in [(bathy_f, "b_"), (back_f, "k_")]:
            fm = uniform_filter(arr, size=size)
            fsq = uniform_filter(arr ** 2, size=size)
            feats[f"{prefix}fm_{size}"] = sample_raster_at_points(fm, bt, xs, ys)
            feats[f"{prefix}std_{size}"] = sample_raster_at_points(
                np.sqrt(np.maximum(fsq - fm ** 2, 0)), bt, xs, ys
            )
            feats[f"{prefix}tpi_{size}"] = sample_raster_at_points(arr - fm, bt, xs, ys)

    # Curvature (5 sigmas × 3 = 15 features)
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
            back_f - uniform_filter(back_f, size=size), bkt, xs, ys
        )

    # Backscatter gradient
    bk_dx = np.gradient(back_f, cell_size, axis=1)
    bk_dy = np.gradient(back_f, cell_size, axis=0)
    feats["back_grad"] = sample_raster_at_points(np.sqrt(bk_dx ** 2 + bk_dy ** 2), bkt, xs, ys)

    # Interactions
    feats["depth_x_back"] = np.abs(feats["bathy"]) * feats["back"]
    feats["slope_x_back"] = feats["slope"] * feats["back"]
    feats["acoustic_hard"] = feats["back"] / (np.abs(feats["bathy"]) + 1.0)
    feats["vrm_x_back"] = feats["vrm"] * feats["back"]

    # Isobath distances (5 features)
    for d_iso in isobath_depths:
        on_contour = np.abs(bathy_f - d_iso) < 0.5
        if on_contour.sum() == 0:
            on_contour = np.abs(bathy_f - d_iso) < 1.0
        dist_px = distance_transform_edt(~on_contour)
        dist_m = (dist_px * cell_size).astype(np.float32)
        feats[f"iso_dist_{d_iso:.0f}"] = sample_raster_at_points(dist_m, bt, xs, ys)

    return pd.DataFrame(feats)


# ---------------------------------------------------------------------------
# Kriging
# ---------------------------------------------------------------------------

def _indicator_kriging_proba(xs_tr, ys_tr, y_tr, xs_te, ys_te,
                              n_classes, max_tr=3000):
    """Global indicator kriging."""
    rng = np.random.RandomState(42)
    n_te = len(xs_te)
    proba = np.zeros((n_te, n_classes))
    sub = rng.choice(len(xs_tr), min(max_tr, len(xs_tr)), replace=False)
    for c in range(n_classes):
        indicator = (y_tr[sub] == c).astype(float)
        if indicator.sum() < 5 or indicator.sum() > len(indicator) - 5:
            proba[:, c] = indicator.mean()
            continue
        try:
            ok = OrdinaryKriging(
                xs_tr[sub], ys_tr[sub], indicator,
                variogram_model="exponential",
                verbose=False, enable_plotting=False, nlags=20,
            )
            z_vals, _ = ok.execute("points", xs_te, ys_te)
            proba[:, c] = np.clip(z_vals, 0, 1)
        except (ValueError, LinAlgError):
            proba[:, c] = indicator.mean()
    return proba


def _stratified_kriging_proba(xs_tr, ys_tr, y_tr, zones_tr,
                               xs_te, ys_te, zones_te,
                               n_classes, n_zones, max_per_zone=1500):
    """Indicator kriging stratified by depth zone."""
    rng = np.random.RandomState(42)
    n_te = len(xs_te)
    proba = np.zeros((n_te, n_classes))
    for z in range(n_zones):
        te_mask = zones_te == z
        if not te_mask.any():
            continue
        # Same zone + adjacent zones
        tr_mask = np.abs(zones_tr.astype(int) - z) <= 1
        if tr_mask.sum() < 20:
            tr_mask = np.ones(len(zones_tr), dtype=bool)
        tr_idx = np.where(tr_mask)[0]
        if len(tr_idx) > max_per_zone:
            tr_idx = rng.choice(tr_idx, max_per_zone, replace=False)
        te_idx = np.where(te_mask)[0]
        for c in range(n_classes):
            indicator = (y_tr[tr_idx] == c).astype(float)
            if indicator.sum() < 3 or indicator.sum() > len(indicator) - 3:
                proba[te_idx, c] = indicator.mean()
                continue
            try:
                ok = OrdinaryKriging(
                    xs_tr[tr_idx], ys_tr[tr_idx], indicator,
                    variogram_model="exponential",
                    verbose=False, enable_plotting=False, nlags=15,
                )
                z_vals, _ = ok.execute("points", xs_te[te_idx], ys_te[te_idx])
                proba[te_idx, c] = np.clip(z_vals, 0, 1)
            except (ValueError, LinAlgError):
                proba[te_idx, c] = indicator.mean()
    return proba


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/submission_v3.csv")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n-cv-blocks", default=10, type=int)
    parser.add_argument(
        "--blend", default="0.60,0.15,0.10,0.15",
        help="kriging_depth,kriging_global,catboost,lgbm weights (default: 0.60,0.15,0.10,0.15)",
    )
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip spatial CV (faster, just produce submission)")
    args = parser.parse_args(argv)

    w = [float(x) for x in args.blend.split(",")]
    if len(w) != 4 or abs(sum(w) - 1.0) > 0.01:
        print("ERROR: --blend must be 4 weights summing to 1.0", file=sys.stderr)
        return 1
    w_kd, w_kg, w_cb, w_lg = w

    # ── Load data ─────────────────────────────────────────────────────
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

    enc = LabelEncoder()
    y = enc.fit_transform(train["class"])
    n_classes = len(enc.classes_)
    xs_tr = train["x"].to_numpy(float)
    ys_tr = train["y"].to_numpy(float)
    xs_te = test["x"].to_numpy(float)
    ys_te = test["y"].to_numpy(float)
    coords = train[["x", "y"]].values

    print(f"Train: {len(train)} rows, Test: {len(test)} rows, Classes: {list(enc.classes_)}")

    # ── Depth zones ───────────────────────────────────────────────────
    print("Computing depth zones...")
    depths_tr = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs_tr, ys_tr, nodata=-10000
    )
    depths_te = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs_te, ys_te, nodata=-10000
    )
    depth_edges = np.quantile(depths_tr[np.isfinite(depths_tr)], np.linspace(0, 1, 6))
    depth_edges[0] -= 1
    depth_edges[-1] += 1
    n_depth_zones = 5
    dz_tr = np.clip(np.digitize(depths_tr, depth_edges) - 1, 0, n_depth_zones - 1)
    dz_te = np.clip(np.digitize(depths_te, depth_edges) - 1, 0, n_depth_zones - 1)

    # ── Backscatter zones (for features) ──────────────────────────────
    print("Computing backscatter and combined zones...")
    back_smooth = gaussian_filter(back_f, sigma=10)
    valid_mask = np.isfinite(bathy) & np.isfinite(back)
    valid_rows, valid_cols = np.where(valid_mask)
    rng = np.random.RandomState(42)
    n_sample = 50000
    si = rng.choice(len(valid_rows), min(n_sample, len(valid_rows)), replace=False)

    # Backscatter zones
    n_bz = 5
    km_bz = KMeans(n_clusters=n_bz, random_state=42, n_init=10)
    km_bz.fit(back_smooth[valid_rows[si], valid_cols[si]].reshape(-1, 1))
    bz_raster = np.full(back_smooth.shape, -1, dtype=np.int32)
    bz_raster[valid_mask] = km_bz.predict(back_smooth[valid_mask].reshape(-1, 1))
    zone_means = [back_smooth[bz_raster == z].mean() for z in range(n_bz)]
    remap = np.zeros(n_bz, dtype=int)
    for new_l, old_l in enumerate(np.argsort(zone_means)):
        remap[old_l] = new_l
    bz_raster = np.where(bz_raster >= 0, remap[bz_raster], -1)

    # Combined depth×backscatter zones
    from sklearn.preprocessing import StandardScaler
    n_cz = 8
    scaler_2d = StandardScaler()
    feats_2d = scaler_2d.fit_transform(np.column_stack([
        bathy_f[valid_rows[si], valid_cols[si]],
        back_smooth[valid_rows[si], valid_cols[si]],
    ]))
    km_cz = KMeans(n_clusters=n_cz, random_state=42, n_init=10)
    km_cz.fit(feats_2d)
    cz_raster = np.full(bathy_f.shape, -1, dtype=np.int32)
    cz_raster[valid_mask] = km_cz.predict(scaler_2d.transform(
        np.column_stack([bathy_f[valid_mask], back_smooth[valid_mask]])
    ))

    # Isobath depths
    isobath_depths = np.quantile(
        depths_tr[np.isfinite(depths_tr)], [0.1, 0.25, 0.5, 0.75, 0.9]
    )

    # ── Extract features ──────────────────────────────────────────────
    print("Extracting features (train)...")
    X_train = extract_features(
        xs_tr, ys_tr, bathy, back, bathy_f, back_f, back_smooth,
        bt, bkt, cell_size, bz_raster, cz_raster, isobath_depths, depths_tr,
    )
    print(f"  {X_train.shape[1]} features")
    print("Extracting features (test)...")
    X_test = extract_features(
        xs_te, ys_te, bathy, back, bathy_f, back_f, back_smooth,
        bt, bkt, cell_size, bz_raster, cz_raster, isobath_depths, depths_tr,
    )

    med = X_train.median()
    X_tr_clean = X_train.fillna(med).replace([np.inf, -np.inf], 0)
    X_te_clean = X_test.fillna(med).replace([np.inf, -np.inf], 0)
    feat_cols = list(X_tr_clean.columns)

    # ── Spatial block CV ─────────────────────────────────────────────
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    if not args.skip_cv:
        print(f"\n{'='*60}")
        print(f"Spatial-block CV ({args.n_cv_blocks} blocks)")
        print(f"{'='*60}")

        km_sp = KMeans(n_clusters=args.n_cv_blocks, random_state=args.seed, n_init=10)
        blocks = km_sp.fit_predict(coords)

        model_names = ["kriging_depth", "kriging_global", "knn5", "catboost", "lgbm"]
        oof = {k: np.zeros((len(y), n_classes)) for k in model_names}

        t_total = time.time()
        for b in range(args.n_cv_blocks):
            va = np.where(blocks == b)[0]
            tr = np.where(blocks != b)[0]
            t0 = time.time()

            oof["kriging_depth"][va] = _stratified_kriging_proba(
                xs_tr[tr], ys_tr[tr], y[tr], dz_tr[tr],
                xs_tr[va], ys_tr[va], dz_tr[va],
                n_classes, n_depth_zones,
            )
            oof["kriging_global"][va] = _indicator_kriging_proba(
                xs_tr[tr], ys_tr[tr], y[tr], xs_tr[va], ys_tr[va], n_classes,
            )
            knn = KNeighborsClassifier(5, weights="distance")
            knn.fit(coords[tr], y[tr])
            oof["knn5"][va] = knn.predict_proba(coords[va])

            cb = CatBoostClassifier(
                iterations=1000, depth=6, learning_rate=0.02,
                l2_leaf_reg=10.0, random_seed=args.seed, verbose=0,
                auto_class_weights="Balanced",
            )
            cb.fit(X_tr_clean.iloc[tr], y[tr])
            oof["catboost"][va] = cb.predict_proba(X_tr_clean.iloc[va])

            lgbm = LGBMClassifier(
                n_estimators=1000, max_depth=8, learning_rate=0.02,
                num_leaves=63, subsample=0.7, colsample_bytree=0.5,
                min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
                class_weight="balanced", random_state=args.seed, n_jobs=-1, verbose=-1,
            )
            lgbm.fit(X_tr_clean.iloc[tr], y[tr])
            oof["lgbm"][va] = lgbm.predict_proba(X_tr_clean.iloc[va])

            elapsed = time.time() - t0
            scores = "  ".join(
                f"{k}={f1_score(y[va], np.argmax(oof[k][va], 1), average='weighted'):.4f}"
                for k in model_names
            )
            print(f"  Block {b:2d} ({len(va):4d} pts, {elapsed:5.0f}s): {scores}")

        print(f"\nTotal CV time: {time.time()-t_total:.0f}s\n")

        # OOF scores
        print("Individual model OOF scores:")
        for name in model_names:
            f1 = f1_score(y, np.argmax(oof[name], 1), average="weighted")
            print(f"  {name:25s}: {f1:.4f}")

        # Blend
        blend_oof = (w_kd * oof["kriging_depth"] + w_kg * oof["kriging_global"] +
                     w_cb * oof["catboost"] + w_lg * oof["lgbm"])
        blend_preds = enc.classes_[np.argmax(blend_oof, axis=1)]
        blend_f1 = f1_score(enc.inverse_transform(y), blend_preds, average="weighted")
        print(f"\n  Blend ({w_kd}/{w_kg}/{w_cb}/{w_lg}): {blend_f1:.4f}")
        print(classification_report(enc.inverse_transform(y), blend_preds))

        # Also try with KNN replacing kriging_global
        for knn_w in [0.05, 0.10, 0.15]:
            rest = 1.0 - w_kd - knn_w
            for cb_w in np.arange(0.05, rest + 0.01, 0.05):
                lg_w = rest - cb_w
                if lg_w < 0:
                    continue
                b_oof = (w_kd * oof["kriging_depth"] + knn_w * oof["knn5"] +
                         cb_w * oof["catboost"] + lg_w * oof["lgbm"])
                f1 = f1_score(y, np.argmax(b_oof, 1), average="weighted")
                if f1 > blend_f1:
                    blend_f1 = f1
                    print(f"  ** Better: kd({w_kd})+knn({knn_w})+cb({cb_w:.2f})+lg({lg_w:.2f}) = {f1:.4f}")

    # ── Train final models on ALL data ────────────────────────────────
    print("\nTraining final models on all training data...")
    t0 = time.time()

    # Kriging (depth-stratified)
    print("  Depth-stratified kriging...")
    p_kd = _stratified_kriging_proba(
        xs_tr, ys_tr, y, dz_tr,
        xs_te, ys_te, dz_te,
        n_classes, n_depth_zones,
    )

    # Kriging (global)
    print("  Global kriging...")
    p_kg = _indicator_kriging_proba(xs_tr, ys_tr, y, xs_te, ys_te, n_classes)

    # CatBoost
    print("  CatBoost...")
    cb_final = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.02,
        l2_leaf_reg=10.0, random_seed=args.seed, verbose=0,
        auto_class_weights="Balanced",
    )
    cb_final.fit(X_tr_clean, y)
    p_cb = cb_final.predict_proba(X_te_clean)

    # LightGBM
    print("  LightGBM...")
    lgbm_final = LGBMClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.02,
        num_leaves=63, subsample=0.7, colsample_bytree=0.5,
        min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
        class_weight="balanced", random_state=args.seed, n_jobs=-1, verbose=-1,
    )
    lgbm_final.fit(X_tr_clean, y)
    p_lg = lgbm_final.predict_proba(X_te_clean)

    print(f"  Training time: {time.time()-t0:.0f}s")

    # ── Blend + submission ────────────────────────────────────────────
    test_blend = w_kd * p_kd + w_kg * p_kg + w_cb * p_cb + w_lg * p_lg
    predictions = enc.inverse_transform(np.argmax(test_blend, axis=1))

    submission = pd.DataFrame({"ID": test["ID"].values, "class": predictions})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)

    print(f"\nSubmission saved: {output.resolve()}")
    print(f"Rows: {len(submission)}")
    print(f"Class distribution:\n{submission['class'].value_counts().to_string()}")

    # Sanity check
    allowed = set(train["class"].unique())
    bad = set(predictions) - allowed
    if bad:
        print(f"WARNING: unknown class labels: {bad}", file=sys.stderr)
        return 1

    # Show kriging probabilities for first few test points
    print("\nTest point kriging probabilities (first 5):")
    for i in range(min(5, len(xs_te))):
        kd = p_kd[i]
        kg = p_kg[i]
        bl = test_blend[i]
        pred = predictions[i]
        print(f"  ID={test['ID'].iloc[i]:4d}: kriging_depth={kd}, "
              f"global={kg}, blend={np.round(bl, 3)}, pred={pred}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
