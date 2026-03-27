"""
Experiment v2: Comprehensive feature engineering + ensemble for GeoHab 2026.

Key improvements over v1:
1. x,y coordinates as features (spatial interpolation for same-area test)
2. Multi-scale terrain derivatives (6 radii including very broad)
3. Curvature features (mean, Gaussian, profile) at 4 Gaussian sigmas
4. GLCM texture features from backscatter patches
5. Patch-level statistics (range, IQR, skewness, kurtosis)
6. Relative backscatter (normalised by local mean)
7. 4-model ensemble: LightGBM + XGBoost + CatBoost + RandomForest
8. Spatial block CV for realistic performance estimation
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter, gaussian_filter
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from btm.features.extract import sample_raster_at_points  # noqa: E402
from btm.core.slope import compute_slope  # noqa: E402
from btm.core.vrm import compute_vrm  # noqa: E402


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------

def load_rasters():
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


# ---------------------------------------------------------------------------
# Pixel coordinate helper
# ---------------------------------------------------------------------------

def pixel_coords(xs, ys, transform):
    inv = ~transform
    cols, rows = inv * (xs, ys)
    return np.round(rows).astype(int), np.round(cols).astype(int)


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(arr, rows, cols, half_size):
    h, w = arr.shape
    ps = 2 * half_size + 1
    patches = np.full((len(rows), ps, ps), np.nan, dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        r0, r1 = max(0, r - half_size), min(h, r + half_size + 1)
        c0, c1 = max(0, c - half_size), min(w, c + half_size + 1)
        pr = half_size - (r - r0)
        pc = half_size - (c - c0)
        patches[i, pr:pr + (r1 - r0), pc:pc + (c1 - c0)] = arr[r0:r1, c0:c1]
    return patches


# ---------------------------------------------------------------------------
# GLCM texture features from patches
# ---------------------------------------------------------------------------

def compute_glcm_features(patches, n_levels=32):
    """Compute GLCM texture features from 2D patches.
    Returns: dict of arrays (contrast, dissimilarity, homogeneity, energy, correlation)
    """
    from skimage.feature import graycomatrix, graycoprops

    n = len(patches)
    props = {p: np.zeros(n) for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']}

    for i in range(n):
        patch = patches[i]
        # Skip if too many NaNs
        valid_mask = np.isfinite(patch)
        if valid_mask.sum() < 9:
            for p in props:
                props[p][i] = 0.0
            continue

        # Quantize to integer levels
        p = patch.copy()
        p[~valid_mask] = np.nanmean(p)
        pmin, pmax = np.nanmin(p), np.nanmax(p)
        if pmax - pmin < 1e-8:
            for prop_name in props:
                props[prop_name][i] = 0.0
            continue
        p_int = ((p - pmin) / (pmax - pmin) * (n_levels - 1)).astype(np.uint8)

        # GLCM at 4 directions, average
        glcm = graycomatrix(p_int, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=n_levels, symmetric=True, normed=True)
        for prop_name in props:
            props[prop_name][i] = graycoprops(glcm, prop_name).mean()

    return props


# ---------------------------------------------------------------------------
# Full feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(xs, ys, bathy, back, bathy_f, back_f, bt, bkt, cell_size):
    feats = {}
    rows, cols = pixel_coords(xs, ys, bt)

    # ── Coordinates (spatial interpolation) ──
    feats['x'] = xs.copy()
    feats['y'] = ys.copy()

    # ── Raw values ──
    feats['bathy'] = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs, ys, nodata=-10000)
    feats['back'] = sample_raster_at_points(
        np.nan_to_num(back, nan=-10000), bkt, xs, ys, nodata=-10000)

    # ── BTM derivatives ──
    print("  BTM slope + VRM...")
    slope_arr = compute_slope(bathy_f, cell_size, nodata=None)
    feats['slope'] = sample_raster_at_points(slope_arr, bt, xs, ys)
    vrm_arr = compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size)
    feats['vrm'] = sample_raster_at_points(vrm_arr, bt, xs, ys)

    # ── Multi-scale focal statistics ──
    print("  Multi-scale focal stats...")
    radii = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
    for r in radii:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        for arr, prefix in [(bathy_f, 'b_'), (back_f, 'k_')]:
            fm = uniform_filter(arr, size=size)
            fsq = uniform_filter(arr ** 2, size=size)
            fstd = np.sqrt(np.maximum(fsq - fm ** 2, 0))
            tpi = arr - fm

            feats[f'{prefix}fm_{size}'] = sample_raster_at_points(fm, bt, xs, ys)
            feats[f'{prefix}std_{size}'] = sample_raster_at_points(fstd, bt, xs, ys)
            feats[f'{prefix}tpi_{size}'] = sample_raster_at_points(tpi, bt, xs, ys)

    # ── Curvature at multiple Gaussian sigmas ──
    print("  Multi-scale curvature...")
    for sigma in [1, 2, 4, 8, 16]:
        smooth = gaussian_filter(bathy_f, sigma=sigma)
        dx = np.gradient(smooth, cell_size, axis=1)
        dy = np.gradient(smooth, cell_size, axis=0)
        d2x = np.gradient(dx, cell_size, axis=1)
        d2y = np.gradient(dy, cell_size, axis=0)
        dxy = np.gradient(dx, cell_size, axis=0)

        # Mean curvature
        feats[f'mcurv_s{sigma}'] = sample_raster_at_points((d2x + d2y) / 2, bt, xs, ys)
        # Gaussian curvature
        feats[f'gcurv_s{sigma}'] = sample_raster_at_points(d2x * d2y - dxy ** 2, bt, xs, ys)
        # Multi-scale slope
        feats[f'mslope_s{sigma}'] = sample_raster_at_points(np.sqrt(dx ** 2 + dy ** 2), bt, xs, ys)

    # ── Aspect (sin/cos) ──
    dx = np.gradient(bathy_f, cell_size, axis=1)
    dy = np.gradient(bathy_f, cell_size, axis=0)
    aspect = np.arctan2(-dy, dx)
    feats['asp_sin'] = sample_raster_at_points(np.sin(aspect), bt, xs, ys)
    feats['asp_cos'] = sample_raster_at_points(np.cos(aspect), bt, xs, ys)

    # ── Relative backscatter ──
    print("  Relative backscatter...")
    for r in [2.5, 10.0, 25.0, 50.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        local_mean = uniform_filter(back_f, size=size)
        feats[f'rel_back_{size}'] = sample_raster_at_points(back_f - local_mean, bkt, xs, ys)

    # ── Backscatter gradient ──
    bk_dx = np.gradient(back_f, cell_size, axis=1)
    bk_dy = np.gradient(back_f, cell_size, axis=0)
    feats['back_grad'] = sample_raster_at_points(
        np.sqrt(bk_dx ** 2 + bk_dy ** 2), bkt, xs, ys)

    # ── Patch statistics ──
    print("  Patch statistics...")
    for half in [2, 5, 10, 20]:
        bp = extract_patches(bathy_f, rows, cols, half)
        kp = extract_patches(back_f, rows, cols, half)
        sp = extract_patches(slope_arr, rows, cols, half)

        bp_flat = bp.reshape(len(xs), -1)
        kp_flat = kp.reshape(len(xs), -1)
        sp_flat = sp.reshape(len(xs), -1)

        feats[f'b_range_{half}'] = np.nanmax(bp_flat, 1) - np.nanmin(bp_flat, 1)
        feats[f'b_iqr_{half}'] = np.nanpercentile(bp_flat, 75, 1) - np.nanpercentile(bp_flat, 25, 1)
        feats[f'k_range_{half}'] = np.nanmax(kp_flat, 1) - np.nanmin(kp_flat, 1)
        feats[f'k_iqr_{half}'] = np.nanpercentile(kp_flat, 75, 1) - np.nanpercentile(kp_flat, 25, 1)
        feats[f'k_skew_{half}'] = np.nan_to_num(
            (np.nanmean(kp_flat, 1) - np.nanmedian(kp_flat, 1)) / (np.nanstd(kp_flat, 1) + 1e-8)
        )
        feats[f's_max_{half}'] = np.nanmax(sp_flat, 1)
        feats[f's_std_{half}'] = np.nanstd(sp_flat, 1)

    # ── GLCM texture (backscatter patches) ──
    print("  GLCM texture features...")
    glcm_patches = extract_patches(back_f, rows, cols, 10)  # 21x21
    glcm = compute_glcm_features(glcm_patches)
    for key, vals in glcm.items():
        feats[f'glcm_{key}'] = vals

    # ── Interactions ──
    feats['depth_x_back'] = np.abs(feats['bathy']) * feats['back']
    feats['slope_x_back'] = feats['slope'] * feats['back']
    feats['acoustic_hard'] = feats['back'] / (np.abs(feats['bathy']) + 1.0)
    feats['vrm_x_back'] = feats['vrm'] * feats['back']
    feats['slope_x_vrm'] = feats['slope'] * feats['vrm']

    df = pd.DataFrame(feats)
    print(f"  Total features: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(seed=42):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier

    return {
        'lgbm': lambda: LGBMClassifier(
            n_estimators=1000, max_depth=8, learning_rate=0.02,
            num_leaves=63, subsample=0.7, colsample_bytree=0.5,
            min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
            class_weight='balanced', random_state=seed, n_jobs=-1, verbose=-1,
        ),
        'xgb': lambda: XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.5, min_child_weight=10,
            reg_alpha=2.0, reg_lambda=10.0, random_state=seed, n_jobs=-1,
            eval_metric='mlogloss', verbosity=0,
        ),
        'catboost': lambda: CatBoostClassifier(
            iterations=1000, depth=6, learning_rate=0.02,
            l2_leaf_reg=10.0, random_seed=seed, verbose=0,
            auto_class_weights='Balanced',
        ),
        'rf': lambda: RandomForestClassifier(
            n_estimators=1000, max_depth=20, min_samples_leaf=5,
            class_weight='balanced', random_state=seed, n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seed = 42
    data = _REPO / "data"

    print("Loading data...")
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    if "ID" not in train.columns:
        train.insert(0, "ID", range(1, len(train) + 1))

    bathy, back, bathy_f, back_f, bt, bkt, cell_size = load_rasters()

    # ── Extract features ──
    print("\n=== Extracting training features ===")
    X_train = extract_all_features(
        train['x'].values, train['y'].values,
        bathy, back, bathy_f, back_f, bt, bkt, cell_size
    )
    print("\n=== Extracting test features ===")
    X_test = extract_all_features(
        test['x'].values, test['y'].values,
        bathy, back, bathy_f, back_f, bt, bkt, cell_size
    )

    enc = LabelEncoder()
    y = enc.fit_transform(train['class'])
    feature_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]

    # Clean up
    median_vals = X_train[feature_cols].median()
    X_tr = X_train[feature_cols].fillna(median_vals).replace([np.inf, -np.inf], 0)
    X_te = X_test[feature_cols].fillna(median_vals).replace([np.inf, -np.inf], 0)

    # ── Spatial CV ──
    print("\n=== Spatial Block CV (10 blocks) ===")
    km = KMeans(n_clusters=10, random_state=seed, n_init=10)
    blocks = km.fit_predict(train[['x', 'y']].values)

    models = get_models(seed)
    oof_proba = {}
    n_classes = len(enc.classes_)

    for name, make_model in models.items():
        print(f"\n  {name}...")
        oof_p = np.zeros((len(y), n_classes))
        for b in range(10):
            va = np.where(blocks == b)[0]
            tr = np.where(blocks != b)[0]
            m = make_model()
            if name == 'xgb':
                from sklearn.utils.class_weight import compute_sample_weight
                sw = compute_sample_weight('balanced', y[tr])
                m.fit(X_tr.iloc[tr], y[tr], sample_weight=sw)
            else:
                m.fit(X_tr.iloc[tr], y[tr])
            oof_p[va] = m.predict_proba(X_tr.iloc[va])

        oof_proba[name] = oof_p
        preds_name = enc.classes_[np.argmax(oof_p, axis=1)]
        f1 = f1_score(enc.inverse_transform(y), preds_name, average='weighted')
        print(f"  {name} spatial CV weighted-F1 = {f1:.4f}")

    # Ensemble
    avg_proba = sum(oof_proba.values()) / len(oof_proba)
    ens_preds = enc.classes_[np.argmax(avg_proba, axis=1)]
    f1_ens = f1_score(enc.inverse_transform(y), ens_preds, average='weighted')
    print(f"\n  ENSEMBLE spatial CV weighted-F1 = {f1_ens:.4f}")
    print("\nSpatial CV classification report (ensemble):")
    print(classification_report(enc.inverse_transform(y), ens_preds))

    # ── Standard StratifiedKFold for comparison ──
    print("=== Standard StratifiedKFold CV ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_std = np.zeros((len(y), n_classes))
    for name, make_model in [('lgbm', models['lgbm'])]:
        for tr, va in skf.split(X_tr, y):
            m = make_model()
            m.fit(X_tr.iloc[tr], y[tr])
            oof_std[va] = m.predict_proba(X_tr.iloc[va])
    std_preds = enc.classes_[np.argmax(oof_std, axis=1)]
    print(f"  lgbm StratKFold weighted-F1 = {f1_score(enc.inverse_transform(y), std_preds, average='weighted'):.4f}")

    # ── Train final ensemble on ALL data + predict test ──
    print("\n=== Training final models on all data ===")
    test_proba = np.zeros((len(test), n_classes))
    for name, make_model in models.items():
        m = make_model()
        if name == 'xgb':
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight('balanced', y)
            m.fit(X_tr, y, sample_weight=sw)
        else:
            m.fit(X_tr, y)
        test_proba += m.predict_proba(X_te)
        print(f"  {name} trained on {len(X_tr)} samples")

    test_proba /= len(models)
    final_preds = enc.inverse_transform(np.argmax(test_proba, axis=1))

    # ── Write submission ──
    submission = pd.DataFrame({"ID": test["ID"], "class": final_preds})
    out_path = data / "submission_v2.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission written to {out_path}")
    print(f"Rows: {len(submission)}")
    print(f"Class distribution:\n{submission['class'].value_counts().to_string()}")

    # Feature importance from final LightGBM
    m_lgbm = models['lgbm']()
    m_lgbm.fit(X_tr, y)
    imp = pd.Series(m_lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop 20 features:")
    print(imp.head(20).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
