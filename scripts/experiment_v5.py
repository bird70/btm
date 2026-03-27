"""Experiment v5: GIS-derived feature engineering for GeoHab 2026.

Adds slope, backscatter zones and acoustic facies (80/20/20 weighting)
on top of the v2 comprehensive feature set, with optional zone-stratified
kriging smoothing for test-point inference (Run B).

Three phases
------------
baseline  -- v2 feature set only (no GIS)
run_a     -- v2 + raw point-sampled GIS features
run_b     -- v2 + zone-stratified kriging-smoothed GIS features (test only)

Constants
---------
WEIGHT_SLOPE = 80, WEIGHT_BZ = 20, WEIGHT_AF = 20
N_BZ_CLUSTERS = 5, N_AF_CLUSTERS = 8, SEED = 42
N_SPATIAL_BLOCKS = 10, KRIGING_MIN_ZONE_PTS = 10
KRIGING_VARIOGRAM = "spherical", KRIGING_NLAGS = 6

Spec:  specs/008-gis-feature-engineering/spec.md
Plan:  specs/008-gis-feature-engineering/plan.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from btm.core.slope import compute_slope  # noqa: E402
from btm.core.vrm import compute_vrm  # noqa: E402
from btm.features.extract import sample_raster_at_points  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_SLOPE: float = 80.0
WEIGHT_BZ: float = 20.0
WEIGHT_AF: float = 20.0

N_BZ_CLUSTERS: int = 5
N_AF_CLUSTERS: int = 8

SEED: int = 42
N_SPATIAL_BLOCKS: int = 10
KRIGING_MIN_ZONE_PTS: int = 10
KRIGING_VARIOGRAM: str = "spherical"
KRIGING_NLAGS: int = 6

_DERIVED_DIR = _REPO / "outputs" / "gis_layers" / "rasters"


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------


def _load_rasters():
    """Load MBES bathymetry and backscatter rasters.

    Returns (bathy, back, bathy_f, back_f, bt, bkt, cell_size)
    """
    data = _REPO / "data"
    with rasterio.open(data / "MBES" / "bathymetry.tif") as src:
        bathy = src.read(1).astype(np.float32)
        bt = src.transform
        bathy[bathy == src.nodata] = np.nan
    with rasterio.open(data / "MBES" / "backscatter.tif") as src:
        back = src.read(1).astype(np.float32)
        bkt = src.transform
        back[back == src.nodata] = np.nan
    cell_size = abs(bt.a)
    bathy_f = np.nan_to_num(bathy, nan=0.0)
    back_f = np.nan_to_num(back, nan=0.0)
    return bathy, back, bathy_f, back_f, bt, bkt, cell_size


def _ensure_derived_rasters():
    """Check all three derived rasters exist; raise FileNotFoundError if any missing."""
    missing = []
    for name in ("slope.tif", "backscatter_zones.tif", "acoustic_facies.tif"):
        path = _DERIVED_DIR / name
        if not path.exists():
            missing.append(str(path))
    if missing:
        msg = (
            "Missing derived rasters:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun scripts/step1_prepare_gis_layers.py to generate them."
        )
        raise FileNotFoundError(msg)
    print(f"Derived rasters OK: {_DERIVED_DIR}")


def _load_derived_rasters():
    """Load slope, backscatter_zones, acoustic_facies from disk.

    Returns (slope_arr, bz_arr, af_arr, derived_transform)
    where derived_transform is the affine transform from slope.tif.
    """
    with rasterio.open(_DERIVED_DIR / "slope.tif") as src:
        slope_arr = src.read(1).astype(np.float32)
        derived_transform = src.transform
    with rasterio.open(_DERIVED_DIR / "backscatter_zones.tif") as src:
        bz_arr = src.read(1).astype(np.int32)
    with rasterio.open(_DERIVED_DIR / "acoustic_facies.tif") as src:
        af_arr = src.read(1).astype(np.int32)
    return slope_arr, bz_arr, af_arr, derived_transform


# ---------------------------------------------------------------------------
# Pixel coordinate helper (for patch extraction)
# ---------------------------------------------------------------------------


def _pixel_coords(xs, ys, transform):
    inv = ~transform
    cols, rows = inv * (xs, ys)
    return np.round(rows).astype(int), np.round(cols).astype(int)


# ---------------------------------------------------------------------------
# Patch extraction helper
# ---------------------------------------------------------------------------


def _extract_patches(arr, rows, cols, half_size):
    h, w = arr.shape
    ps = 2 * half_size + 1
    patches = np.full((len(rows), ps, ps), np.nan, dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        r0, r1 = max(0, r - half_size), min(h, r + half_size + 1)
        c0, c1 = max(0, c - half_size), min(w, c + half_size + 1)
        pr = half_size - (r - r0)
        pc = half_size - (c - c0)
        patches[i, pr : pr + (r1 - r0), pc : pc + (c1 - c0)] = arr[r0:r1, c0:c1]
    return patches


# ---------------------------------------------------------------------------
# GLCM texture features
# ---------------------------------------------------------------------------


def _compute_glcm_features(patches, n_levels=32):
    """Compute GLCM texture features from 2D patches."""
    from skimage.feature import graycomatrix, graycoprops

    n = len(patches)
    prop_names = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    props = {p: np.zeros(n) for p in prop_names}

    for i in range(n):
        patch = patches[i]
        valid_mask = np.isfinite(patch)
        if valid_mask.sum() < 9:
            continue
        p = patch.copy()
        p[~valid_mask] = np.nanmean(p)
        pmin, pmax = np.nanmin(p), np.nanmax(p)
        if pmax - pmin < 1e-8:
            continue
        p_int = ((p - pmin) / (pmax - pmin) * (n_levels - 1)).astype(np.uint8)
        glcm = graycomatrix(
            p_int,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=n_levels,
            symmetric=True,
            normed=True,
        )
        for prop_name in prop_names:
            props[prop_name][i] = graycoprops(glcm, prop_name).mean()

    return props


# ---------------------------------------------------------------------------
# Full v2 feature extraction (verbatim from experiment_v2.py)
# ---------------------------------------------------------------------------


def extract_all_features(xs, ys, bathy, back, bathy_f, back_f, bt, bkt, cell_size):
    """Extract the full v2 feature set (~114 columns).

    Verbatim copy from scripts/experiment_v2.py.
    """
    feats = {}
    rows, cols = _pixel_coords(xs, ys, bt)

    feats["x"] = xs.copy()
    feats["y"] = ys.copy()

    feats["bathy"] = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs, ys, nodata=-10000
    )
    feats["back"] = sample_raster_at_points(
        np.nan_to_num(back, nan=-10000), bkt, xs, ys, nodata=-10000
    )

    print("  BTM slope + VRM...")
    slope_arr = compute_slope(bathy_f, cell_size, nodata=None)
    feats["slope"] = sample_raster_at_points(slope_arr, bt, xs, ys)
    vrm_arr = compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size)
    feats["vrm"] = sample_raster_at_points(vrm_arr, bt, xs, ys)

    print("  Multi-scale focal stats...")
    radii = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
    for r in radii:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        for arr, prefix in [(bathy_f, "b_"), (back_f, "k_")]:
            fm = uniform_filter(arr, size=size)
            fsq = uniform_filter(arr**2, size=size)
            fstd = np.sqrt(np.maximum(fsq - fm**2, 0))
            tpi = arr - fm
            feats[f"{prefix}fm_{size}"] = sample_raster_at_points(fm, bt, xs, ys)
            feats[f"{prefix}std_{size}"] = sample_raster_at_points(fstd, bt, xs, ys)
            feats[f"{prefix}tpi_{size}"] = sample_raster_at_points(tpi, bt, xs, ys)

    print("  Multi-scale curvature...")
    for sigma in [1, 2, 4, 8, 16]:
        smooth = gaussian_filter(bathy_f, sigma=sigma)
        dx = np.gradient(smooth, cell_size, axis=1)
        dy = np.gradient(smooth, cell_size, axis=0)
        d2x = np.gradient(dx, cell_size, axis=1)
        d2y = np.gradient(dy, cell_size, axis=0)
        dxy = np.gradient(dx, cell_size, axis=0)
        feats[f"mcurv_s{sigma}"] = sample_raster_at_points((d2x + d2y) / 2, bt, xs, ys)
        feats[f"gcurv_s{sigma}"] = sample_raster_at_points(d2x * d2y - dxy**2, bt, xs, ys)
        feats[f"mslope_s{sigma}"] = sample_raster_at_points(
            np.sqrt(dx**2 + dy**2), bt, xs, ys
        )

    dx = np.gradient(bathy_f, cell_size, axis=1)
    dy = np.gradient(bathy_f, cell_size, axis=0)
    aspect = np.arctan2(-dy, dx)
    feats["asp_sin"] = sample_raster_at_points(np.sin(aspect), bt, xs, ys)
    feats["asp_cos"] = sample_raster_at_points(np.cos(aspect), bt, xs, ys)

    print("  Relative backscatter...")
    for r in [2.5, 10.0, 25.0, 50.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        local_mean = uniform_filter(back_f, size=size)
        feats[f"rel_back_{size}"] = sample_raster_at_points(back_f - local_mean, bkt, xs, ys)

    bk_dx = np.gradient(back_f, cell_size, axis=1)
    bk_dy = np.gradient(back_f, cell_size, axis=0)
    feats["back_grad"] = sample_raster_at_points(
        np.sqrt(bk_dx**2 + bk_dy**2), bkt, xs, ys
    )

    print("  Patch statistics...")
    for half in [2, 5, 10, 20]:
        bp = _extract_patches(bathy_f, rows, cols, half)
        kp = _extract_patches(back_f, rows, cols, half)
        sp = _extract_patches(slope_arr, rows, cols, half)
        bp_flat = bp.reshape(len(xs), -1)
        kp_flat = kp.reshape(len(xs), -1)
        sp_flat = sp.reshape(len(xs), -1)
        feats[f"b_range_{half}"] = np.nanmax(bp_flat, 1) - np.nanmin(bp_flat, 1)
        feats[f"b_iqr_{half}"] = (
            np.nanpercentile(bp_flat, 75, 1) - np.nanpercentile(bp_flat, 25, 1)
        )
        feats[f"k_range_{half}"] = np.nanmax(kp_flat, 1) - np.nanmin(kp_flat, 1)
        feats[f"k_iqr_{half}"] = (
            np.nanpercentile(kp_flat, 75, 1) - np.nanpercentile(kp_flat, 25, 1)
        )
        feats[f"k_skew_{half}"] = np.nan_to_num(
            (np.nanmean(kp_flat, 1) - np.nanmedian(kp_flat, 1))
            / (np.nanstd(kp_flat, 1) + 1e-8)
        )
        feats[f"s_max_{half}"] = np.nanmax(sp_flat, 1)
        feats[f"s_std_{half}"] = np.nanstd(sp_flat, 1)

    print("  GLCM texture features...")
    glcm_patches = _extract_patches(back_f, rows, cols, 10)
    glcm = _compute_glcm_features(glcm_patches)
    for key, vals in glcm.items():
        feats[f"glcm_{key}"] = vals

    feats["depth_x_back"] = np.abs(feats["bathy"]) * feats["back"]
    feats["slope_x_back"] = feats["slope"] * feats["back"]
    feats["acoustic_hard"] = feats["back"] / (np.abs(feats["bathy"]) + 1.0)
    feats["vrm_x_back"] = feats["vrm"] * feats["back"]
    feats["slope_x_vrm"] = feats["slope"] * feats["vrm"]

    df = pd.DataFrame(feats)
    print(f"  Total features: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# GIS feature helpers
# ---------------------------------------------------------------------------


def _add_gis_features_raw(base_df, slope_arr, bz_arr, af_arr, derived_transform, xs, ys):
    """Sample GIS rasters at (xs, ys), one-hot encode, weight, append 14 columns.

    Columns added
    -------------
    gis_slope              : slope value × WEIGHT_SLOPE
    gis_bz_0 … gis_bz_4   : one-hot encoded backscatter zone × WEIGHT_BZ
    gis_af_0 … gis_af_7   : one-hot encoded acoustic facies × WEIGHT_AF

    Global-median fallback is applied for any NaN sampled values.
    """
    slope_vals = sample_raster_at_points(
        slope_arr.astype(np.float32), derived_transform, xs, ys
    )
    bz_vals = sample_raster_at_points(
        bz_arr.astype(np.float32), derived_transform, xs, ys
    )
    af_vals = sample_raster_at_points(
        af_arr.astype(np.float32), derived_transform, xs, ys
    )

    # Global-median NaN fallback
    slope_vals = np.where(np.isnan(slope_vals), float(np.nanmedian(slope_vals)), slope_vals)
    bz_vals = np.where(np.isnan(bz_vals), float(np.nanmedian(bz_vals)), bz_vals)
    af_vals = np.where(np.isnan(af_vals), float(np.nanmedian(af_vals)), af_vals)

    bz_int = np.clip(np.round(bz_vals).astype(int), 0, N_BZ_CLUSTERS - 1)
    af_int = np.clip(np.round(af_vals).astype(int), 0, N_AF_CLUSTERS - 1)

    idx = base_df.index
    df = base_df.copy()
    df["gis_slope"] = slope_vals * WEIGHT_SLOPE

    # OHE with forced categories to ensure all zone columns always exist
    bz_series = pd.Series(
        pd.Categorical(bz_int, categories=list(range(N_BZ_CLUSTERS))), index=idx
    )
    bz_ohe = pd.get_dummies(bz_series, prefix="gis_bz").astype(float) * WEIGHT_BZ

    af_series = pd.Series(
        pd.Categorical(af_int, categories=list(range(N_AF_CLUSTERS))), index=idx
    )
    af_ohe = pd.get_dummies(af_series, prefix="gis_af").astype(float) * WEIGHT_AF

    df = pd.concat([df, bz_ohe, af_ohe], axis=1)
    return df


def _krige_gis_features(
    train_df, test_df, slope_tr, bz_tr, af_tr, slope_te_raw, bz_te_raw, af_te_raw
):
    """Zone-stratified ordinary kriging of GIS features for test points.

    Stratifies training data by backscatter zone (bz_tr) and applies
    OrdinaryKriging per zone. Falls back to KNeighborsRegressor for zones
    with fewer than KRIGING_MIN_ZONE_PTS training points.

    Note: Run A and Run B CV F1 are equal by design. Kriging applies only to
    test-point inference, not to cross-validation (see spec US2/AC2).

    Returns
    -------
    pd.DataFrame
        Columns: krige_slope, krige_bz, krige_af. No NaN values.
    """
    xs_tr = train_df["x"].to_numpy(float)
    ys_tr = train_df["y"].to_numpy(float)
    xs_te = test_df["x"].to_numpy(float)
    ys_te = test_df["y"].to_numpy(float)

    krige_slope = np.full(len(test_df), np.nan, dtype=np.float64)
    krige_bz = np.full(len(test_df), np.nan, dtype=np.float64)
    krige_af = np.full(len(test_df), np.nan, dtype=np.float64)

    bz_tr_int = np.clip(np.round(np.asarray(bz_tr)).astype(int), 0, N_BZ_CLUSTERS - 1)
    bz_te_int = np.clip(np.round(np.asarray(bz_te_raw)).astype(int), 0, N_BZ_CLUSTERS - 1)

    feature_triples = [
        (np.asarray(slope_tr, dtype=float), krige_slope, "slope"),
        (np.asarray(bz_tr, dtype=float), krige_bz, "bz"),
        (np.asarray(af_tr, dtype=float), krige_af, "af"),
    ]

    for z in range(N_BZ_CLUSTERS):
        tr_mask = bz_tr_int == z
        te_mask = bz_te_int == z
        te_idx = np.where(te_mask)[0]
        if te_idx.size == 0:
            continue

        n_pts = int(tr_mask.sum())
        xs_z = xs_tr[tr_mask]
        ys_z = ys_tr[tr_mask]

        for values_tr_arr, out_arr, feat_name in feature_triples:
            vals_z = values_tr_arr[tr_mask]
            xs_te_z = xs_te[te_idx]
            ys_te_z = ys_te[te_idx]

            if n_pts >= KRIGING_MIN_ZONE_PTS:
                try:
                    ok = OrdinaryKriging(
                        xs_z,
                        ys_z,
                        vals_z,
                        variogram_model=KRIGING_VARIOGRAM,
                        nlags=KRIGING_NLAGS,
                        weight=True,
                        verbose=False,
                        enable_plotting=False,
                    )
                    z_vals, _ = ok.execute("points", xs_te_z, ys_te_z)
                    out_arr[te_idx] = np.asarray(z_vals, dtype=float)
                except Exception as exc:
                    warnings.warn(
                        f"Zone {z} {feat_name}: kriging failed ({exc}), using KNN"
                    )
                    knn = KNeighborsRegressor(n_neighbors=min(5, n_pts))
                    knn.fit(np.column_stack([xs_z, ys_z]), vals_z)
                    out_arr[te_idx] = knn.predict(np.column_stack([xs_te_z, ys_te_z]))
            elif n_pts > 0:
                k = min(5, n_pts)
                warnings.warn(f"Zone {z}: KNN fallback ({n_pts} pts)")
                knn = KNeighborsRegressor(n_neighbors=k)
                knn.fit(np.column_stack([xs_z, ys_z]), vals_z)
                out_arr[te_idx] = knn.predict(np.column_stack([xs_te_z, ys_te_z]))
            else:
                out_arr[te_idx] = float(np.median(values_tr_arr))

    # Global-median fallback for any remaining NaN (unassigned test points)
    for global_vals, out_arr, _feat_name in feature_triples:
        nan_mask = np.isnan(out_arr)
        if nan_mask.any():
            out_arr[nan_mask] = float(np.median(global_vals))

    return pd.DataFrame(
        {"krige_slope": krige_slope, "krige_bz": krige_bz, "krige_af": krige_af},
        index=test_df.index,
    )


def _add_gis_features_kriged(base_df, kriged_df):
    """Build the 14-column kriged GIS feature block and append to base_df."""
    idx = base_df.index
    df = base_df.copy()
    df["gis_slope"] = kriged_df["krige_slope"].values * WEIGHT_SLOPE

    bz_int = np.clip(
        np.round(kriged_df["krige_bz"].values).astype(int), 0, N_BZ_CLUSTERS - 1
    )
    af_int = np.clip(
        np.round(kriged_df["krige_af"].values).astype(int), 0, N_AF_CLUSTERS - 1
    )

    bz_series = pd.Series(
        pd.Categorical(bz_int, categories=list(range(N_BZ_CLUSTERS))), index=idx
    )
    bz_ohe = pd.get_dummies(bz_series, prefix="gis_bz").astype(float) * WEIGHT_BZ

    af_series = pd.Series(
        pd.Categorical(af_int, categories=list(range(N_AF_CLUSTERS))), index=idx
    )
    af_ohe = pd.get_dummies(af_series, prefix="gis_af").astype(float) * WEIGHT_AF

    df = pd.concat([df, bz_ohe, af_ohe], axis=1)
    return df


# ---------------------------------------------------------------------------
# Model definitions (verbatim from experiment_v2.py)
# ---------------------------------------------------------------------------


def get_models(seed=42):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "lgbm": lambda: LGBMClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=63,
            subsample=0.7,
            colsample_bytree=0.5,
            min_child_samples=15,
            reg_alpha=2.0,
            reg_lambda=10.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": lambda: XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.5,
            min_child_weight=10,
            reg_alpha=2.0,
            reg_lambda=10.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        ),
        "catboost": lambda: CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.02,
            l2_leaf_reg=10.0,
            random_seed=seed,
            verbose=0,
            auto_class_weights="Balanced",
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Spatial block CV
# ---------------------------------------------------------------------------


def _run_spatial_block_cv(X, y_labels, enc, train_coords, seed, n_blocks):
    """Run spatial block cross-validation with an ensemble of 4 models.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix — must be all-numeric and NaN/inf-free.
    y_labels : np.ndarray
        Integer-encoded class labels (from a fitted LabelEncoder).
    enc : LabelEncoder
        Fitted encoder for inverse_transform.
    train_coords : np.ndarray, shape (n, 2)
        [x, y] coordinates used for KMeans block assignment.
    seed : int
    n_blocks : int

    Returns
    -------
    tuple
        (weighted_f1, per_class_f1_dict, mean_importances_series, elapsed_seconds)
    """
    from sklearn.utils.class_weight import compute_sample_weight

    t0 = time.time()
    n_classes = len(enc.classes_)
    models = get_models(seed)

    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    blocks = km.fit_predict(train_coords)

    oof_proba = {}
    lgbm_fold_imps: list[np.ndarray] = []

    for name, make_model in models.items():
        print(f"    CV {name}...")
        oof_p = np.zeros((len(y_labels), n_classes))
        fold_imps: list[np.ndarray] = []

        for b in range(n_blocks):
            va_idx = np.where(blocks == b)[0]
            tr_idx = np.where(blocks != b)[0]
            m = make_model()
            if name == "xgb":
                sw = compute_sample_weight("balanced", y_labels[tr_idx])
                m.fit(X.iloc[tr_idx], y_labels[tr_idx], sample_weight=sw)
            else:
                m.fit(X.iloc[tr_idx], y_labels[tr_idx])
            oof_p[va_idx] = m.predict_proba(X.iloc[va_idx])
            if name == "lgbm":
                fold_imps.append(m.feature_importances_)

        oof_proba[name] = oof_p
        if fold_imps:
            lgbm_fold_imps.append(np.mean(fold_imps, axis=0))

        preds_name = enc.classes_[np.argmax(oof_p, axis=1)]
        f1_name = f1_score(enc.inverse_transform(y_labels), preds_name, average="weighted")
        print(f"    {name} spatial-CV weighted-F1 = {f1_name:.4f}")

    avg_proba = sum(oof_proba.values()) / len(oof_proba)
    ens_preds = enc.classes_[np.argmax(avg_proba, axis=1)]
    weighted_f1 = f1_score(enc.inverse_transform(y_labels), ens_preds, average="weighted")
    per_class_arr = f1_score(
        enc.inverse_transform(y_labels), ens_preds, average=None, labels=enc.classes_
    )
    per_class_f1_dict = dict(zip(enc.classes_.tolist(), per_class_arr.tolist()))

    mean_imps: pd.Series
    if lgbm_fold_imps:
        mean_imps = pd.Series(
            np.mean(lgbm_fold_imps, axis=0), index=X.columns
        ).sort_values(ascending=False)
    else:
        mean_imps = pd.Series(dtype=float)

    elapsed = time.time() - t0
    return weighted_f1, per_class_f1_dict, mean_imps, elapsed


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_submission(out_path, pred_labels, test_ids, run_name):
    """Write a submission CSV with columns ID, class."""
    assert len(pred_labels) == len(test_ids), (
        f"_write_submission: length mismatch pred={len(pred_labels)} ids={len(test_ids)}"
    )
    sub = pd.DataFrame({"ID": test_ids, "class": pred_labels})
    sub.to_csv(out_path, index=False)
    print(f"[{run_name}] Submission → {out_path}  (rows={len(sub)})")


def _write_run_report(out_dir, results):
    """Write a Markdown run report to out_dir/run-008-gis-feature-engineering.md.

    Parameters
    ----------
    out_dir : path-like
        Output directory.  In main() this is ``_REPO / "docs"``.
    results : dict
        Keys are run names (baseline, run_a, run_b).  Values are either a
        plain float (weighted-F1) or a dict with keys f1, per_class,
        importances, elapsed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "run-008-gis-feature-engineering.md"

    def _f1(v) -> float:
        return float(v) if isinstance(v, (int, float)) else float(v["f1"])

    def _elapsed(v) -> str:
        if isinstance(v, dict) and "elapsed" in v:
            return f"{v['elapsed']:.0f}s"
        return "N/A"

    def _top3(v) -> str:
        if isinstance(v, dict) and "importances" in v:
            imps = v["importances"]
            if isinstance(imps, pd.Series) and len(imps) > 0:
                return ", ".join(imps.head(3).index.tolist())
        return "N/A"

    def _per_class_row(v) -> str:
        if isinstance(v, dict) and "per_class" in v:
            pc = v["per_class"]
            return " | ".join(f"{cls}: {f:0.3f}" for cls, f in sorted(pc.items()))
        return "N/A"

    lines = [
        "# Run 008 – GIS Feature Engineering\n",
        "**Branch**: `008-gis-feature-engineering`\n",
        "**Spec**: `specs/008-gis-feature-engineering/spec.md`\n",
        "",
        "## Parameters\n",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| WEIGHT_SLOPE | {WEIGHT_SLOPE} |",
        f"| WEIGHT_BZ | {WEIGHT_BZ} |",
        f"| WEIGHT_AF | {WEIGHT_AF} |",
        f"| N_BZ_CLUSTERS | {N_BZ_CLUSTERS} |",
        f"| N_AF_CLUSTERS | {N_AF_CLUSTERS} |",
        f"| SEED | {SEED} |",
        f"| N_SPATIAL_BLOCKS | {N_SPATIAL_BLOCKS} |",
        f"| KRIGING_MIN_ZONE_PTS | {KRIGING_MIN_ZONE_PTS} |",
        f"| KRIGING_VARIOGRAM | {KRIGING_VARIOGRAM} |",
        f"| KRIGING_NLAGS | {KRIGING_NLAGS} |",
        "",
        "## Metrics Comparison\n",
        "| Run | Weighted-F1 | Wall-clock | Top-3 features |",
        "|-----|-------------|------------|----------------|",
    ]

    for run_name in ("baseline", "run_a", "run_b"):
        if run_name in results:
            v = results[run_name]
            lines.append(
                f"| {run_name} | {_f1(v):.4f} | {_elapsed(v)} | {_top3(v)} |"
            )

    lines += [
        "",
        "### Per-class F1\n",
    ]
    for run_name in ("baseline", "run_a", "run_b"):
        if run_name in results:
            lines.append(f"**{run_name}**: {_per_class_row(results[run_name])}")

    lines += [
        "",
        "## Note on Run A vs Run B CV F1\n",
        "Run A and Run B report the **same CV weighted-F1** by design.",
        "Kriging applies only to test-point inference in Run B; "
        "it does not affect cross-validation scores (see spec US2/AC2).",
        "Any difference in leaderboard performance can only be assessed "
        "via submission to the competition.",
        "",
        "## Best Run\n",
    ]

    if "run_a" in results and "run_b" in results:
        f1_a = _f1(results["run_a"])
        f1_b = _f1(results["run_b"])
        best = "run_b" if f1_b >= f1_a else "run_a"
        lines.append(
            f"Best run: **{best}** (CV weighted-F1 = {_f1(results[best]):.4f})"
        )
        if "baseline" in results:
            delta = _f1(results["run_a"]) - _f1(results["baseline"])
            lines += [
                "",
                "## Interpretation\n",
                f"GIS features (slope/bz/af) produced a weighted-F1 delta of "
                f"**{delta:+.4f}** vs baseline.",
                "A positive delta indicates the weighted GIS-derived features "
                "improved ensemble discrimination.",
            ]
    else:
        lines.append("(Not all runs were executed — partial report.)")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Run report → {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    from sklearn.utils.class_weight import compute_sample_weight

    parser = argparse.ArgumentParser(description="GIS feature engineering experiment v5")
    parser.add_argument(
        "--phase",
        default="baseline,run_a,run_b",
        help="Comma-separated phases: baseline,run_a,run_b  (default: all)",
    )
    args = parser.parse_args()
    phases = {p.strip() for p in args.phase.split(",")}

    data_dir = _REPO / "data"
    print("Loading CSV data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    if "ID" not in train_df.columns:
        train_df.insert(0, "ID", range(1, len(train_df) + 1))

    print("Loading MBES rasters...")
    bathy, back, bathy_f, back_f, bt, bkt, cell_size = _load_rasters()

    print("Checking derived rasters...")
    _ensure_derived_rasters()
    slope_arr_d, bz_arr_d, af_arr_d, derived_transform = _load_derived_rasters()

    # ── Base feature extraction ──
    print("\n=== Extracting base features (v2) ===")
    print("  Training set...")
    X_train_base = extract_all_features(
        train_df["x"].values,
        train_df["y"].values,
        bathy,
        back,
        bathy_f,
        back_f,
        bt,
        bkt,
        cell_size,
    )
    print("  Test set...")
    X_test_base = extract_all_features(
        test_df["x"].values,
        test_df["y"].values,
        bathy,
        back,
        bathy_f,
        back_f,
        bt,
        bkt,
        cell_size,
    )

    enc = LabelEncoder()
    y = enc.fit_transform(train_df["class"])
    n_classes = len(enc.classes_)

    feature_cols = [
        c for c in X_train_base.columns if pd.api.types.is_numeric_dtype(X_train_base[c])
    ]
    median_vals = X_train_base[feature_cols].median()
    X_tr_base = (
        X_train_base[feature_cols].fillna(median_vals).replace([np.inf, -np.inf], 0)
    )
    X_te_base = (
        X_test_base[feature_cols].fillna(median_vals).replace([np.inf, -np.inf], 0)
    )

    xs_tr = train_df["x"].to_numpy(float)
    ys_tr = train_df["y"].to_numpy(float)
    xs_te = test_df["x"].to_numpy(float)
    ys_te = test_df["y"].to_numpy(float)
    train_coords = train_df[["x", "y"]].values

    results: dict = {}

    # ── Phase: baseline ──
    if "baseline" in phases:
        print("\n=== Phase: baseline (v2 features, no GIS) ===")
        f1, per_class, imps, elapsed = _run_spatial_block_cv(
            X_tr_base, y, enc, train_coords, SEED, N_SPATIAL_BLOCKS
        )
        results["baseline"] = {
            "f1": f1,
            "per_class": per_class,
            "importances": imps,
            "elapsed": elapsed,
        }
        print(f"\n  Baseline ensemble spatial-CV weighted-F1 = {f1:.4f}  ({elapsed:.0f}s)")
        for cls, v in sorted(per_class.items()):
            print(f"    {cls}: {v:.4f}")

    # ── Shared: sample GIS features for run_a / run_b ──
    X_tr_aug: pd.DataFrame  # populated below; referenced in run_a and run_b blocks
    if phases & {"run_a", "run_b"}:
        print("\n=== Sampling GIS features from derived rasters ===")
        slope_tr = sample_raster_at_points(
            slope_arr_d.astype(np.float32), derived_transform, xs_tr, ys_tr
        )
        bz_tr_raw = sample_raster_at_points(
            bz_arr_d.astype(np.float32), derived_transform, xs_tr, ys_tr
        )
        af_tr_raw = sample_raster_at_points(
            af_arr_d.astype(np.float32), derived_transform, xs_tr, ys_tr
        )
        slope_te_raw = sample_raster_at_points(
            slope_arr_d.astype(np.float32), derived_transform, xs_te, ys_te
        )
        bz_te_raw = sample_raster_at_points(
            bz_arr_d.astype(np.float32), derived_transform, xs_te, ys_te
        )
        af_te_raw = sample_raster_at_points(
            af_arr_d.astype(np.float32), derived_transform, xs_te, ys_te
        )
        print(
            f"  slope train: mean={np.nanmean(slope_tr):.2f}, "
            f"NaN={np.isnan(slope_tr).sum()}"
        )
        print(f"  bz    train: unique={np.unique(np.round(bz_tr_raw).astype(int))}")
        print(f"  af    train: unique={np.unique(np.round(af_tr_raw).astype(int))}")

        # Augmented train features (raw-sampled): used for both Run A and Run B CV
        X_tr_aug_df = _add_gis_features_raw(
            X_tr_base, slope_arr_d, bz_arr_d, af_arr_d, derived_transform, xs_tr, ys_tr
        )
        aug_cols = list(X_tr_aug_df.columns)
        X_tr_aug = (
            X_tr_aug_df.fillna(X_tr_aug_df.median()).replace([np.inf, -np.inf], 0)
        )

    # ── Phase: run_a ──
    if "run_a" in phases:
        print("\n=== Phase: run_a (v2 + raw GIS features) ===")
        f1, per_class, imps, elapsed = _run_spatial_block_cv(
            X_tr_aug, y, enc, train_coords, SEED, N_SPATIAL_BLOCKS
        )
        results["run_a"] = {
            "f1": f1,
            "per_class": per_class,
            "importances": imps,
            "elapsed": elapsed,
        }
        print(f"\n  Run A ensemble spatial-CV weighted-F1 = {f1:.4f}  ({elapsed:.0f}s)")

        print("  Training final Run A ensemble on all data...")
        models_a = get_models(SEED)
        test_proba_a = np.zeros((len(test_df), n_classes))

        X_te_aug_a_df = _add_gis_features_raw(
            X_te_base, slope_arr_d, bz_arr_d, af_arr_d, derived_transform, xs_te, ys_te
        )
        X_te_aug_a = (
            X_te_aug_a_df.reindex(columns=aug_cols)
            .fillna(X_tr_aug.median())
            .replace([np.inf, -np.inf], 0)
        )

        for name, make_model in models_a.items():
            m = make_model()
            if name == "xgb":
                sw = compute_sample_weight("balanced", y)
                m.fit(X_tr_aug, y, sample_weight=sw)
            else:
                m.fit(X_tr_aug, y)
            test_proba_a += m.predict_proba(X_te_aug_a)
            print(f"    {name} final trained")

        test_proba_a /= len(models_a)
        final_preds_a = enc.inverse_transform(np.argmax(test_proba_a, axis=1))
        _write_submission(
            data_dir / "submission_v5_gis_features.csv",
            final_preds_a,
            test_df["ID"].values,
            "run_a",
        )

    # ── Phase: run_b ──
    if "run_b" in phases:
        print("\n=== Phase: run_b (v2 + kriged GIS features at test inference) ===")
        print("  CV uses same raw-augmented features as run_a (kriging skipped for CV)")
        f1, per_class, imps, elapsed = _run_spatial_block_cv(
            X_tr_aug, y, enc, train_coords, SEED, N_SPATIAL_BLOCKS
        )
        results["run_b"] = {
            "f1": f1,
            "per_class": per_class,
            "importances": imps,
            "elapsed": elapsed,
        }
        print(f"\n  Run B ensemble spatial-CV weighted-F1 = {f1:.4f}  ({elapsed:.0f}s)")
        print("  [Note: Run B CV F1 == Run A CV F1 by design — see spec US2/AC2]")

        print("  Running zone-stratified kriging for test-point inference...")
        bz_tr_int = np.clip(np.round(bz_tr_raw).astype(int), 0, N_BZ_CLUSTERS - 1)
        af_tr_int = np.clip(np.round(af_tr_raw).astype(int), 0, N_AF_CLUSTERS - 1)
        bz_te_int = np.clip(np.round(bz_te_raw).astype(int), 0, N_BZ_CLUSTERS - 1)
        af_te_int = np.clip(np.round(af_te_raw).astype(int), 0, N_AF_CLUSTERS - 1)

        kriged_df = _krige_gis_features(
            train_df,
            test_df,
            slope_tr,
            bz_tr_int,
            af_tr_int,
            slope_te_raw,
            bz_te_int,
            af_te_int,
        )

        X_te_aug_b_df = _add_gis_features_kriged(X_te_base, kriged_df)
        X_te_aug_b = (
            X_te_aug_b_df.reindex(columns=aug_cols)
            .fillna(X_tr_aug.median())
            .replace([np.inf, -np.inf], 0)
        )

        print("  Training final Run B ensemble on all data...")
        models_b = get_models(SEED)
        test_proba_b = np.zeros((len(test_df), n_classes))

        for name, make_model in models_b.items():
            m = make_model()
            if name == "xgb":
                sw = compute_sample_weight("balanced", y)
                m.fit(X_tr_aug, y, sample_weight=sw)
            else:
                m.fit(X_tr_aug, y)
            test_proba_b += m.predict_proba(X_te_aug_b)
            print(f"    {name} final trained")

        test_proba_b /= len(models_b)
        final_preds_b = enc.inverse_transform(np.argmax(test_proba_b, axis=1))
        _write_submission(
            data_dir / "submission_v5_gis_features_kriging.csv",
            final_preds_b,
            test_df["ID"].values,
            "run_b",
        )

    # ── Best submission selection ──
    if "run_a" in results and "run_b" in results:
        f1_a = float(results["run_a"]["f1"])
        f1_b = float(results["run_b"]["f1"])
        best_run = "run_b" if f1_b >= f1_a else "run_a"
        src_path = (
            data_dir / "submission_v5_gis_features_kriging.csv"
            if best_run == "run_b"
            else data_dir / "submission_v5_gis_features.csv"
        )
        shutil.copy(src_path, data_dir / "submission_v5_best.csv")
        print(
            f"\nBest run: {best_run}  "
            f"(CV weighted-F1 = {results[best_run]['f1']:.4f})"
        )
        print(f"Best submission → {data_dir / 'submission_v5_best.csv'}")

    # ── Run report ──
    if results:
        _write_run_report(_REPO / "docs", results)

    print("\n=== Experiment v5 complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
