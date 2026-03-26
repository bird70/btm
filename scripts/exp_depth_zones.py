"""
Experiment: Depth-zone stratification + stratified kriging.

Hypothesis: benthic habitats follow depth contours (isobaths).
By segmenting the bathymetry into depth zones we can:
1. Add depth-zone membership as a feature
2. Add distance-to-isobath features
3. Run stratified indicator kriging (separate variogram per zone)
4. Build zone-specific classifiers
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numpy.linalg import LinAlgError
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import gaussian_filter, uniform_filter
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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train = pd.read_csv(_REPO / "data/train.csv")
test = pd.read_csv(_REPO / "data/test.csv")
if "ID" not in train.columns:
    train.insert(0, "ID", range(1, len(train) + 1))

with rasterio.open(_REPO / "data/MBES/bathymetry.tif") as src:
    bathy = src.read(1).astype(np.float32)
    bt = src.transform
    bathy[bathy == src.nodata] = np.nan
with rasterio.open(_REPO / "data/MBES/backscatter.tif") as src:
    back = src.read(1).astype(np.float32)
    bkt = src.transform
    back[back == src.nodata] = np.nan

cell_size = abs(bt.a)
bathy_f = np.nan_to_num(bathy, nan=0.0)
back_f = np.nan_to_num(back, nan=0.0)

enc = LabelEncoder()
y = enc.fit_transform(train["class"])
n_classes = len(enc.classes_)
coords = train[["x", "y"]].values

km = KMeans(n_clusters=10, random_state=42, n_init=10)
blocks = km.fit_predict(coords)


# ---------------------------------------------------------------------------
# Part 1: Explore depth-class relationship
# ---------------------------------------------------------------------------
print("\n=== Part 1: Depth-class analysis ===")
xs_tr = train["x"].to_numpy(float)
ys_tr = train["y"].to_numpy(float)
depths = sample_raster_at_points(
    np.nan_to_num(bathy, nan=-10000), bt, xs_tr, ys_tr, nodata=-10000
)

# Depth statistics per class
for cls in sorted(train["class"].unique()):
    mask = train["class"] == cls
    d = depths[mask]
    print(f"  {cls}: depth range [{d.min():.1f}, {d.max():.1f}], "
          f"mean={d.mean():.1f}, std={d.std():.1f}, n={mask.sum()}")

# Quantile-based depth zones
print("\nDepth zone analysis:")
n_zones = 5
zone_edges = np.quantile(depths[np.isfinite(depths)], np.linspace(0, 1, n_zones + 1))
zone_edges[0] -= 1  # ensure all points covered
zone_edges[-1] += 1
zone_labels = np.digitize(depths, zone_edges) - 1
zone_labels = np.clip(zone_labels, 0, n_zones - 1)

for z in range(n_zones):
    mask = zone_labels == z
    cls_dist = train.loc[mask, "class"].value_counts()
    purity = cls_dist.iloc[0] / mask.sum() if mask.sum() > 0 else 0
    print(f"  Zone {z} (depth {zone_edges[z]:.1f} to {zone_edges[z+1]:.1f}): "
          f"n={mask.sum()}, dominant={cls_dist.index[0]} ({purity:.0%})")


# ---------------------------------------------------------------------------
# Part 2: Distance-to-isobath features
# ---------------------------------------------------------------------------
print("\n=== Part 2: Isobath distance features ===")
from scipy.ndimage import distance_transform_edt

# Key isobaths at ecologically meaningful depths
# Use the zone edges + some fixed depths
bathy_valid = np.where(np.isfinite(bathy), bathy, np.nan)
bathy_for_contour = np.nan_to_num(bathy, nan=0.0)

# Choose isobath depths based on class depth boundaries
all_depths_sorted = np.sort(depths[np.isfinite(depths)])
isobath_depths = np.quantile(all_depths_sorted, [0.1, 0.25, 0.5, 0.75, 0.9])
print(f"  Isobath depths: {[f'{d:.1f}' for d in isobath_depths]}")


def compute_isobath_distances(arr, transform, isobath_depths, xs, ys):
    """Compute distance from each point to closest pixel on each isobath."""
    feats = {}
    for d in isobath_depths:
        # Binary mask: pixels within 0.5m of this isobath
        threshold = 0.5  # metres
        on_contour = np.abs(arr - d) < threshold
        if on_contour.sum() == 0:
            on_contour = np.abs(arr - d) < 1.0  # widen
        # EDT from contour (in pixel units)
        dist_pixels = distance_transform_edt(~on_contour)
        dist_m = dist_pixels * abs(transform.a)
        feats[f"iso_dist_{d:.0f}"] = sample_raster_at_points(
            dist_m.astype(np.float32), transform, xs, ys
        )
    return feats


iso_feats_tr = compute_isobath_distances(bathy_f, bt, isobath_depths, xs_tr, ys_tr)
print(f"  Generated {len(iso_feats_tr)} isobath distance features")


# ---------------------------------------------------------------------------
# Part 3: Full feature extraction with depth zones
# ---------------------------------------------------------------------------
print("\n=== Part 3: Extended feature extraction ===")


def extract_extended_features(xs, ys, include_xy=True):
    feats = {}
    if include_xy:
        feats["x"] = xs.copy()
        feats["y"] = ys.copy()

    # Raw values
    feats["bathy"] = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs, ys, nodata=-10000
    )
    feats["back"] = sample_raster_at_points(
        np.nan_to_num(back, nan=-10000), bkt, xs, ys, nodata=-10000
    )

    # Depth zone (categorical encoded as int)
    d = feats["bathy"]
    feats["depth_zone"] = np.clip(np.digitize(d, zone_edges) - 1, 0, n_zones - 1).astype(float)
    # Relative position within zone (0 = zone boundary shallow, 1 = zone boundary deep)
    for z in range(n_zones):
        mask_z = feats["depth_zone"] == z
        if mask_z.any():
            zmin, zmax = zone_edges[z], zone_edges[z + 1]
            feats["depth_zone_frac"] = np.where(
                mask_z, (d - zmin) / (zmax - zmin + 1e-8), feats.get("depth_zone_frac", 0.0)
            )

    # BTM derivatives
    slope_arr = compute_slope(bathy_f, cell_size, nodata=None)
    feats["slope"] = sample_raster_at_points(slope_arr, bt, xs, ys)
    vrm_arr = compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size)
    feats["vrm"] = sample_raster_at_points(vrm_arr, bt, xs, ys)

    # Multi-scale focal stats
    for r in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        for arr, p in [(bathy_f, "b_"), (back_f, "k_")]:
            fm = uniform_filter(arr, size=size)
            fsq = uniform_filter(arr ** 2, size=size)
            fstd = np.sqrt(np.maximum(fsq - fm ** 2, 0))
            feats[f"{p}fm_{size}"] = sample_raster_at_points(fm, bt, xs, ys)
            feats[f"{p}std_{size}"] = sample_raster_at_points(fstd, bt, xs, ys)
            feats[f"{p}tpi_{size}"] = sample_raster_at_points(arr - fm, bt, xs, ys)

    # Multi-scale curvature
    for sigma in [1, 2, 4, 8, 16]:
        smooth = gaussian_filter(bathy_f, sigma=sigma)
        dx = np.gradient(smooth, cell_size, axis=1)
        dy = np.gradient(smooth, cell_size, axis=0)
        d2x = np.gradient(dx, cell_size, axis=1)
        d2y = np.gradient(dy, cell_size, axis=0)
        dxy = np.gradient(dx, cell_size, axis=0)
        feats[f"mcurv_s{sigma}"] = sample_raster_at_points((d2x + d2y) / 2, bt, xs, ys)
        feats[f"gcurv_s{sigma}"] = sample_raster_at_points(d2x * d2y - dxy ** 2, bt, xs, ys)
        feats[f"mslope_s{sigma}"] = sample_raster_at_points(
            np.sqrt(dx ** 2 + dy ** 2), bt, xs, ys
        )

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
    feats["back_grad"] = sample_raster_at_points(
        np.sqrt(bk_dx ** 2 + bk_dy ** 2), bkt, xs, ys
    )

    # Interactions
    feats["depth_x_back"] = np.abs(feats["bathy"]) * feats["back"]
    feats["slope_x_back"] = feats["slope"] * feats["back"]
    feats["acoustic_hard"] = feats["back"] / (np.abs(feats["bathy"]) + 1.0)
    feats["vrm_x_back"] = feats["vrm"] * feats["back"]

    # Isobath distance features
    iso = compute_isobath_distances(bathy_f, bt, isobath_depths, xs, ys)
    feats.update(iso)

    return pd.DataFrame(feats)


print("Extracting training features...")
X_train = extract_extended_features(xs_tr, ys_tr)
print(f"  {X_train.shape[1]} features")

xs_te = test["x"].to_numpy(float)
ys_te = test["y"].to_numpy(float)
print("Extracting test features...")
X_test = extract_extended_features(xs_te, ys_te)

med = X_train.median()
X_tr = X_train.fillna(med).replace([np.inf, -np.inf], 0)
X_te = X_test.fillna(med).replace([np.inf, -np.inf], 0)
feat_cols = list(X_tr.columns)


# ---------------------------------------------------------------------------
# Part 4: Spatial CV — everything
# ---------------------------------------------------------------------------
print("\n=== Part 4: Spatial CV experiments ===")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def indicator_kriging_proba(xs_tr, ys_tr, y_tr, xs_va, ys_va, n_classes,
                            max_tr=3000, vmodel="exponential"):
    """Run indicator kriging, return (n_va, n_classes) probability matrix."""
    n_va = len(xs_va)
    proba = np.zeros((n_va, n_classes))
    rng = np.random.RandomState(42)
    if len(xs_tr) > max_tr:
        sub = rng.choice(len(xs_tr), max_tr, replace=False)
    else:
        sub = np.arange(len(xs_tr))
    for c in range(n_classes):
        indicator = (y_tr[sub] == c).astype(float)
        if indicator.sum() < 5 or indicator.sum() > len(indicator) - 5:
            proba[:, c] = indicator.mean()
            continue
        try:
            ok = OrdinaryKriging(
                xs_tr[sub], ys_tr[sub], indicator,
                variogram_model=vmodel,
                verbose=False, enable_plotting=False, nlags=20,
            )
            z, _ = ok.execute("points", xs_va, ys_va)
            proba[:, c] = np.clip(z, 0, 1)
        except (ValueError, LinAlgError):
            proba[:, c] = indicator.mean()
    return proba


def stratified_kriging_proba(xs_tr, ys_tr, y_tr, depths_tr,
                              xs_va, ys_va, depths_va,
                              n_classes, zone_edges, max_tr_per_zone=1500):
    """Indicator kriging stratified by depth zone.
    
    Within each depth zone, fit a separate variogram. Points are assigned
    to the zone of their depth, and kriged only against training points
    in the same zone (+adjacent zones for robustness).
    """
    n_va = len(xs_va)
    n_zones = len(zone_edges) - 1
    proba = np.zeros((n_va, n_classes))
    
    zones_tr = np.clip(np.digitize(depths_tr, zone_edges) - 1, 0, n_zones - 1)
    zones_va = np.clip(np.digitize(depths_va, zone_edges) - 1, 0, n_zones - 1)
    
    rng = np.random.RandomState(42)
    
    for z in range(n_zones):
        va_mask = zones_va == z
        if not va_mask.any():
            continue
        # Use training points from this zone + adjacent zones
        tr_mask = np.abs(zones_tr - z) <= 1
        if tr_mask.sum() < 30:
            tr_mask = np.ones(len(zones_tr), dtype=bool)
        
        tr_idx = np.where(tr_mask)[0]
        if len(tr_idx) > max_tr_per_zone:
            tr_idx = rng.choice(tr_idx, max_tr_per_zone, replace=False)
        
        va_idx = np.where(va_mask)[0]
        
        for c in range(n_classes):
            indicator = (y_tr[tr_idx] == c).astype(float)
            if indicator.sum() < 3 or indicator.sum() > len(indicator) - 3:
                proba[va_idx, c] = indicator.mean()
                continue
            try:
                ok = OrdinaryKriging(
                    xs_tr[tr_idx], ys_tr[tr_idx], indicator,
                    variogram_model="exponential",
                    verbose=False, enable_plotting=False, nlags=15,
                )
                z_vals, _ = ok.execute("points", xs_va[va_idx], ys_va[va_idx])
                proba[va_idx, c] = np.clip(z_vals, 0, 1)
            except (ValueError, LinAlgError):
                proba[va_idx, c] = indicator.mean()
    return proba


# Run spatial CV
oof = {
    "knn": np.zeros((len(y), n_classes)),
    "kriging": np.zeros((len(y), n_classes)),
    "strat_kriging": np.zeros((len(y), n_classes)),
    "catboost": np.zeros((len(y), n_classes)),
    "lgbm": np.zeros((len(y), n_classes)),
}

depths_tr_all = depths.copy()
depths_te = sample_raster_at_points(
    np.nan_to_num(bathy, nan=-10000), bt, xs_te, ys_te, nodata=-10000
)

t_total = time.time()
for b in range(10):
    va = np.where(blocks == b)[0]
    tr = np.where(blocks != b)[0]
    t0 = time.time()

    # KNN
    knn = KNeighborsClassifier(5, weights="distance")
    knn.fit(coords[tr], y[tr])
    oof["knn"][va] = knn.predict_proba(coords[va])

    # Indicator kriging (global)
    oof["kriging"][va] = indicator_kriging_proba(
        xs_tr[tr], ys_tr[tr], y[tr], xs_tr[va], ys_tr[va], n_classes
    )

    # Stratified kriging (by depth zone)
    oof["strat_kriging"][va] = stratified_kriging_proba(
        xs_tr[tr], ys_tr[tr], y[tr], depths_tr_all[tr],
        xs_tr[va], ys_tr[va], depths_tr_all[va],
        n_classes, zone_edges,
    )

    # CatBoost (with depth zone + isobath features)
    cb = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.02,
        l2_leaf_reg=10.0, random_seed=42, verbose=0,
        auto_class_weights="Balanced",
    )
    cb.fit(X_tr.iloc[tr], y[tr])
    oof["catboost"][va] = cb.predict_proba(X_tr.iloc[va])

    # LightGBM (with depth zone + isobath features)
    lgbm = LGBMClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.02,
        num_leaves=63, subsample=0.7, colsample_bytree=0.5,
        min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_tr.iloc[tr], y[tr])
    oof["lgbm"][va] = lgbm.predict_proba(X_tr.iloc[va])

    # Report
    scores = {}
    for name, arr in oof.items():
        p = enc.classes_[np.argmax(arr[va], axis=1)]
        scores[name] = f1_score(enc.inverse_transform(y[va]), p, average="weighted")
    elapsed = time.time() - t0
    print(f"Block {b} ({len(va)} pts, {elapsed:.0f}s): " +
          "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

print(f"\nTotal time: {time.time()-t_total:.0f}s\n")

# Overall OOF scores
print("=== Overall Spatial CV Results ===")
for name in oof:
    preds = enc.classes_[np.argmax(oof[name], axis=1)]
    f1 = f1_score(enc.inverse_transform(y), preds, average="weighted")
    print(f"  {name:20s}: {f1:.4f}")

# Blends
print("\n=== Blend search ===")
best_f1, best_label = 0, ""
# Grid search over 3-model blends (knn/kriging variants + catboost + lgbm)
spatial_models = ["knn", "kriging", "strat_kriging"]
for sp_name in spatial_models:
    for w_sp in np.arange(0.2, 0.7, 0.05):
        for w_cb in np.arange(0.1, 0.7, 0.05):
            w_lg = 1.0 - w_sp - w_cb
            if w_lg < 0.05 or w_lg > 0.6:
                continue
            blend = w_sp * oof[sp_name] + w_cb * oof["catboost"] + w_lg * oof["lgbm"]
            bp = enc.classes_[np.argmax(blend, axis=1)]
            f1 = f1_score(enc.inverse_transform(y), bp, average="weighted")
            if f1 > best_f1:
                best_f1 = f1
                best_label = f"{sp_name}({w_sp:.2f})+cb({w_cb:.2f})+lgbm({w_lg:.2f})"

print(f"  Best 3-model: {best_label} = {best_f1:.4f}")

# Also try 4-model blends
for w_knn in np.arange(0.1, 0.5, 0.05):
    for w_kr in np.arange(0.05, 0.4, 0.05):
        for w_cb in np.arange(0.1, 0.5, 0.05):
            w_lg = 1.0 - w_knn - w_kr - w_cb
            if w_lg < 0.05 or w_lg > 0.4:
                continue
            blend = (w_knn * oof["knn"] + w_kr * oof["strat_kriging"] +
                     w_cb * oof["catboost"] + w_lg * oof["lgbm"])
            bp = enc.classes_[np.argmax(blend, axis=1)]
            f1 = f1_score(enc.inverse_transform(y), bp, average="weighted")
            if f1 > best_f1:
                best_f1 = f1
                best_label = (f"knn({w_knn:.2f})+strat_kr({w_kr:.2f})+"
                              f"cb({w_cb:.2f})+lgbm({w_lg:.2f})")

print(f"  Best 4-model: {best_label} = {best_f1:.4f}")

# Print classification report for best
# Parse weights from label
print(f"\n=== Best blend: {best_label} ({best_f1:.4f}) ===")
