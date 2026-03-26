"""
Experiment: Backscatter-based acoustic zonation + stratified kriging.

The backscatter raster directly images substrate type:
  - Bright (high dB) → hard substrate (rock, reef, coarse sediment)
  - Dark (low dB) → soft substrate (sand, mud, fine sediment)

The visible grey-value zonation in the backscatter image corresponds to
acoustic facies — homogeneous areas with similar substrate character.
These zones should align with habitat classes better than depth alone.

We test three segmentation approaches:
  A. Depth-only quantile zones (baseline from exp_depth_zones.py)
  B. Backscatter intensity zones (quantile + k-means)
  C. Combined depth × backscatter 2D segmentation (k-means on raster)
  D. Smoothed backscatter segmentation (Gaussian blur first, then k-means)

For each, we run stratified indicator kriging and blend with GBDT.
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
xs_tr = train["x"].to_numpy(float)
ys_tr = train["y"].to_numpy(float)
xs_te = test["x"].to_numpy(float)
ys_te = test["y"].to_numpy(float)

km_spatial = KMeans(n_clusters=10, random_state=42, n_init=10)
blocks = km_spatial.fit_predict(coords)

# Sample raster values at train/test points
depths_tr = sample_raster_at_points(
    np.nan_to_num(bathy, nan=-10000), bt, xs_tr, ys_tr, nodata=-10000
)
backs_tr = sample_raster_at_points(
    np.nan_to_num(back, nan=-10000), bkt, xs_tr, ys_tr, nodata=-10000
)
depths_te = sample_raster_at_points(
    np.nan_to_num(bathy, nan=-10000), bt, xs_te, ys_te, nodata=-10000
)
backs_te = sample_raster_at_points(
    np.nan_to_num(back, nan=-10000), bkt, xs_te, ys_te, nodata=-10000
)


# ---------------------------------------------------------------------------
# Part 1: Explore backscatter-class relationship
# ---------------------------------------------------------------------------
print("\n=== Part 1: Backscatter-class analysis ===")
for cls in sorted(train["class"].unique()):
    mask = train["class"] == cls
    b = backs_tr[mask]
    d = depths_tr[mask]
    print(f"  {cls}: back [{b.min():.1f}, {b.max():.1f}] "
          f"mean={b.mean():.1f} std={b.std():.1f} | "
          f"depth [{d.min():.1f}, {d.max():.1f}] mean={d.mean():.1f}")


# ---------------------------------------------------------------------------
# Part 2: Segmentation approaches
# ---------------------------------------------------------------------------
print("\n=== Part 2: Raster-based segmentation ===")

# Create a smoothed backscatter for segmentation (reduce speckle noise)
back_smooth = gaussian_filter(back_f, sigma=10)  # ~2.5m smoothing

# Approach B: Backscatter k-means zones on the raster
# Subsample raster pixels for k-means fitting (full raster is 19M pixels)
valid_mask = np.isfinite(bathy) & np.isfinite(back)
valid_rows, valid_cols = np.where(valid_mask)
rng = np.random.RandomState(42)
n_sample_pixels = 50000
sample_idx = rng.choice(len(valid_rows), min(n_sample_pixels, len(valid_rows)), replace=False)
sample_back = back_smooth[valid_rows[sample_idx], valid_cols[sample_idx]].reshape(-1, 1)

n_back_zones = 5
km_back = KMeans(n_clusters=n_back_zones, random_state=42, n_init=10)
km_back.fit(sample_back)

# Assign every raster pixel to a backscatter zone
back_zone_raster = np.full(back_smooth.shape, -1, dtype=np.int32)
back_zone_raster[valid_mask] = km_back.predict(
    back_smooth[valid_mask].reshape(-1, 1)
)
# Sort zone labels by backscatter intensity (0 = softest, 4 = hardest)
zone_means = [back_smooth[back_zone_raster == z].mean() for z in range(n_back_zones)]
sort_order = np.argsort(zone_means)
remap = np.zeros(n_back_zones, dtype=int)
for new_label, old_label in enumerate(sort_order):
    remap[old_label] = new_label
back_zone_raster_sorted = np.where(back_zone_raster >= 0,
                                    remap[back_zone_raster], -1)

print("Backscatter zones (sorted by intensity):")
for z in range(n_back_zones):
    mask_z = back_zone_raster_sorted == z
    bval = back_smooth[mask_z]
    print(f"  Zone {z}: pixels={mask_z.sum()}, back=[{bval.min():.1f}, {bval.max():.1f}], "
          f"mean={bval.mean():.1f}")

# Approach C: Combined depth × backscatter k-means
sample_depth = bathy_f[valid_rows[sample_idx], valid_cols[sample_idx]]
# Standardise before k-means
from sklearn.preprocessing import StandardScaler
scaler_2d = StandardScaler()
features_2d = scaler_2d.fit_transform(
    np.column_stack([sample_depth, sample_back.ravel()])
)
n_combined_zones = 8
km_2d = KMeans(n_clusters=n_combined_zones, random_state=42, n_init=10)
km_2d.fit(features_2d)

combined_zone_raster = np.full(bathy_f.shape, -1, dtype=np.int32)
all_valid_features = scaler_2d.transform(
    np.column_stack([bathy_f[valid_mask], back_smooth[valid_mask]])
)
combined_zone_raster[valid_mask] = km_2d.predict(all_valid_features)

print("\nCombined depth×backscatter zones:")
for z in range(n_combined_zones):
    mask_z = combined_zone_raster == z
    dval = bathy_f[mask_z]
    bval = back_smooth[mask_z]
    print(f"  Zone {z}: px={mask_z.sum():>8d}, "
          f"depth=[{dval.min():.1f},{dval.max():.1f}] mean={dval.mean():.1f}, "
          f"back mean={bval.mean():.1f}")

# Sample zone assignments at training/test points
def sample_zone_at_points(zone_raster, transform, xs, ys):
    inv = ~transform
    cols, rows = inv * (xs, ys)
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)
    h, w = zone_raster.shape
    rows = np.clip(rows, 0, h - 1)
    cols = np.clip(cols, 0, w - 1)
    return zone_raster[rows, cols]

back_zones_tr = sample_zone_at_points(back_zone_raster_sorted, bt, xs_tr, ys_tr)
back_zones_te = sample_zone_at_points(back_zone_raster_sorted, bt, xs_te, ys_te)
combined_zones_tr = sample_zone_at_points(combined_zone_raster, bt, xs_tr, ys_tr)
combined_zones_te = sample_zone_at_points(combined_zone_raster, bt, xs_te, ys_te)

# How well do zones separate classes?
print("\nBackscatter zone ↔ class confusion:")
for z in range(n_back_zones):
    mask = back_zones_tr == z
    if mask.sum() == 0:
        continue
    dist = train.loc[mask, "class"].value_counts()
    purity = dist.iloc[0] / mask.sum()
    print(f"  Zone {z}: n={mask.sum()}, dominant={dist.index[0]} ({purity:.0%}), "
          f"classes={dict(dist)}")

print("\nCombined zone ↔ class confusion:")
for z in range(n_combined_zones):
    mask = combined_zones_tr == z
    if mask.sum() == 0:
        continue
    dist = train.loc[mask, "class"].value_counts()
    purity = dist.iloc[0] / mask.sum()
    print(f"  Zone {z}: n={mask.sum()}, dominant={dist.index[0]} ({purity:.0%}), "
          f"classes={dict(dist)}")


# ---------------------------------------------------------------------------
# Part 3: Stratified kriging with different zone definitions
# ---------------------------------------------------------------------------
print("\n=== Part 3: Stratified kriging comparison ===")


def stratified_kriging(xs_tr, ys_tr, y_tr, zones_tr,
                       xs_va, ys_va, zones_va,
                       n_classes, n_zones, max_per_zone=1500):
    """Indicator kriging stratified by pre-computed zones."""
    n_va = len(xs_va)
    proba = np.zeros((n_va, n_classes))
    rng = np.random.RandomState(42)

    for z in range(n_zones):
        va_mask = zones_va == z
        if not va_mask.any():
            continue
        # Training: same zone + adjacent (±1) zones
        tr_mask = np.abs(zones_tr.astype(int) - z) <= 1
        if tr_mask.sum() < 20:
            tr_mask = np.ones(len(zones_tr), dtype=bool)

        tr_idx = np.where(tr_mask)[0]
        if len(tr_idx) > max_per_zone:
            tr_idx = rng.choice(tr_idx, max_per_zone, replace=False)
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


def global_kriging(xs_tr, ys_tr, y_tr, xs_va, ys_va, n_classes, max_tr=3000):
    """Global indicator kriging (no stratification)."""
    n_va = len(xs_va)
    proba = np.zeros((n_va, n_classes))
    rng = np.random.RandomState(42)
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
            z_vals, _ = ok.execute("points", xs_va, ys_va)
            proba[:, c] = np.clip(z_vals, 0, 1)
        except (ValueError, LinAlgError):
            proba[:, c] = indicator.mean()
    return proba


# Depth zone edges (from previous experiment)
depth_zone_edges = np.quantile(
    depths_tr[np.isfinite(depths_tr)], np.linspace(0, 1, 6)
)
depth_zone_edges[0] -= 1
depth_zone_edges[-1] += 1
depth_zones_tr = np.clip(np.digitize(depths_tr, depth_zone_edges) - 1, 0, 4)
depth_zones_te = np.clip(np.digitize(depths_te, depth_zone_edges) - 1, 0, 4)

# Run spatial CV for all stratification approaches
strategies = {
    "knn5": None,
    "kriging_global": None,
    "kriging_depth": ("depth", depth_zones_tr, 5),
    "kriging_back": ("back", back_zones_tr, n_back_zones),
    "kriging_combined": ("combined", combined_zones_tr, n_combined_zones),
}

oof = {k: np.zeros((len(y), n_classes)) for k in strategies}

t_total = time.time()
for b in range(10):
    va = np.where(blocks == b)[0]
    tr = np.where(blocks != b)[0]
    t0 = time.time()

    # KNN
    knn = KNeighborsClassifier(5, weights="distance")
    knn.fit(coords[tr], y[tr])
    oof["knn5"][va] = knn.predict_proba(coords[va])

    # Global kriging
    oof["kriging_global"][va] = global_kriging(
        xs_tr[tr], ys_tr[tr], y[tr], xs_tr[va], ys_tr[va], n_classes
    )

    # Stratified kriging variants
    for name, (_, zones_all, nz) in [(k, v) for k, v in strategies.items() if v]:
        oof[name][va] = stratified_kriging(
            xs_tr[tr], ys_tr[tr], y[tr], zones_all[tr],
            xs_tr[va], ys_tr[va], zones_all[va],
            n_classes, nz,
        )

    elapsed = time.time() - t0
    scores_str = "  ".join(
        f"{k}={f1_score(y[va], np.argmax(oof[k][va], 1), average='weighted'):.4f}"
        for k in oof
    )
    print(f"Block {b} ({len(va)} pts, {elapsed:.0f}s): {scores_str}")

print(f"\nTotal time: {time.time()-t_total:.0f}s\n")

print("=== Overall Spatial CV ===")
for name in oof:
    preds = enc.classes_[np.argmax(oof[name], axis=1)]
    f1 = f1_score(enc.inverse_transform(y), preds, average="weighted")
    print(f"  {name:25s}: {f1:.4f}")


# ---------------------------------------------------------------------------
# Part 4: Blend spatial models with GBDT (using extended features)
# ---------------------------------------------------------------------------
print("\n=== Part 4: Add GBDT and blend ===")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.ndimage import distance_transform_edt


def extract_features_with_zones(xs, ys, back_zones, combined_zones):
    feats = {}
    feats["x"] = xs.copy()
    feats["y"] = ys.copy()
    feats["bathy"] = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs, ys, nodata=-10000
    )
    feats["back"] = sample_raster_at_points(
        np.nan_to_num(back, nan=-10000), bkt, xs, ys, nodata=-10000
    )
    feats["back_zone"] = back_zones.astype(float)
    feats["combined_zone"] = combined_zones.astype(float)

    slope_arr = compute_slope(bathy_f, cell_size, nodata=None)
    feats["slope"] = sample_raster_at_points(slope_arr, bt, xs, ys)
    vrm_arr = compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size)
    feats["vrm"] = sample_raster_at_points(vrm_arr, bt, xs, ys)

    for r in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        for arr, p in [(bathy_f, "b_"), (back_f, "k_")]:
            fm = uniform_filter(arr, size=size)
            fsq = uniform_filter(arr ** 2, size=size)
            feats[f"{p}fm_{size}"] = sample_raster_at_points(fm, bt, xs, ys)
            feats[f"{p}std_{size}"] = sample_raster_at_points(
                np.sqrt(np.maximum(fsq - fm ** 2, 0)), bt, xs, ys
            )
            feats[f"{p}tpi_{size}"] = sample_raster_at_points(arr - fm, bt, xs, ys)

    for sigma in [1, 2, 4, 8, 16]:
        smooth = gaussian_filter(bathy_f, sigma=sigma)
        dx = np.gradient(smooth, cell_size, axis=1)
        dy = np.gradient(smooth, cell_size, axis=0)
        d2x = np.gradient(dx, cell_size, axis=1)
        d2y = np.gradient(dy, cell_size, axis=0)
        dxy = np.gradient(dx, cell_size, axis=0)
        feats[f"mcurv_s{sigma}"] = sample_raster_at_points((d2x + d2y) / 2, bt, xs, ys)
        feats[f"gcurv_s{sigma}"] = sample_raster_at_points(
            d2x * d2y - dxy ** 2, bt, xs, ys
        )
        feats[f"mslope_s{sigma}"] = sample_raster_at_points(
            np.sqrt(dx ** 2 + dy ** 2), bt, xs, ys
        )

    dx = np.gradient(bathy_f, cell_size, axis=1)
    dy = np.gradient(bathy_f, cell_size, axis=0)
    aspect = np.arctan2(-dy, dx)
    feats["asp_sin"] = sample_raster_at_points(np.sin(aspect), bt, xs, ys)
    feats["asp_cos"] = sample_raster_at_points(np.cos(aspect), bt, xs, ys)

    for r in [2.5, 10.0, 25.0, 50.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        feats[f"rel_back_{size}"] = sample_raster_at_points(
            back_f - uniform_filter(back_f, size=size), bkt, xs, ys
        )

    bk_dx = np.gradient(back_f, cell_size, axis=1)
    bk_dy = np.gradient(back_f, cell_size, axis=0)
    feats["back_grad"] = sample_raster_at_points(
        np.sqrt(bk_dx ** 2 + bk_dy ** 2), bkt, xs, ys
    )

    feats["depth_x_back"] = np.abs(feats["bathy"]) * feats["back"]
    feats["slope_x_back"] = feats["slope"] * feats["back"]
    feats["acoustic_hard"] = feats["back"] / (np.abs(feats["bathy"]) + 1.0)
    feats["vrm_x_back"] = feats["vrm"] * feats["back"]

    # Isobath distance features
    isobath_depths = np.quantile(
        depths_tr[np.isfinite(depths_tr)], [0.1, 0.25, 0.5, 0.75, 0.9]
    )
    for d_iso in isobath_depths:
        on_contour = np.abs(bathy_f - d_iso) < 0.5
        if on_contour.sum() == 0:
            on_contour = np.abs(bathy_f - d_iso) < 1.0
        dist_px = distance_transform_edt(~on_contour)
        dist_m = (dist_px * cell_size).astype(np.float32)
        feats[f"iso_dist_{d_iso:.0f}"] = sample_raster_at_points(dist_m, bt, xs, ys)

    return pd.DataFrame(feats)


print("Extracting features (train)...")
X_train = extract_features_with_zones(xs_tr, ys_tr, back_zones_tr, combined_zones_tr)
print(f"  {X_train.shape[1]} features")
print("Extracting features (test)...")
X_test = extract_features_with_zones(xs_te, ys_te, back_zones_te, combined_zones_te)

med = X_train.median()
X_tr_clean = X_train.fillna(med).replace([np.inf, -np.inf], 0)
X_te_clean = X_test.fillna(med).replace([np.inf, -np.inf], 0)
feat_cols = list(X_tr_clean.columns)

# Add GBDT OOF
oof["catboost"] = np.zeros((len(y), n_classes))
oof["lgbm"] = np.zeros((len(y), n_classes))

for b in range(10):
    va = np.where(blocks == b)[0]
    tr = np.where(blocks != b)[0]

    cb = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.02,
        l2_leaf_reg=10.0, random_seed=42, verbose=0,
        auto_class_weights="Balanced",
    )
    cb.fit(X_tr_clean.iloc[tr], y[tr])
    oof["catboost"][va] = cb.predict_proba(X_tr_clean.iloc[va])

    lgbm = LGBMClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.02,
        num_leaves=63, subsample=0.7, colsample_bytree=0.5,
        min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_tr_clean.iloc[tr], y[tr])
    oof["lgbm"][va] = lgbm.predict_proba(X_tr_clean.iloc[va])

    cb_f1 = f1_score(y[va], np.argmax(oof["catboost"][va], 1), average="weighted")
    lg_f1 = f1_score(y[va], np.argmax(oof["lgbm"][va], 1), average="weighted")
    print(f"  Block {b}: catboost={cb_f1:.4f}  lgbm={lg_f1:.4f}")

print("\nAll models OOF:")
for name in oof:
    preds = enc.classes_[np.argmax(oof[name], axis=1)]
    f1 = f1_score(enc.inverse_transform(y), preds, average="weighted")
    print(f"  {name:25s}: {f1:.4f}")

# Exhaustive blend search
print("\n=== Blend search (all spatial × GBDT combos) ===")
spatial_names = ["knn5", "kriging_global", "kriging_depth", "kriging_back", "kriging_combined"]
gbdt_names = ["catboost", "lgbm"]

best_f1, best_label, best_weights = 0, "", {}

# 2-spatial + 2-gbdt blend
for sp1 in spatial_names:
    for sp2 in spatial_names:
        if sp2 <= sp1:
            continue  # avoid duplicate pairs
        for w1 in np.arange(0.05, 0.65, 0.05):
            for w2 in np.arange(0.05, 0.65, 0.05):
                for wcb in np.arange(0.05, 0.55, 0.05):
                    wlg = 1.0 - w1 - w2 - wcb
                    if wlg < 0.0 or wlg > 0.5:
                        continue
                    blend = (w1 * oof[sp1] + w2 * oof[sp2] +
                             wcb * oof["catboost"] + wlg * oof["lgbm"])
                    bp = enc.classes_[np.argmax(blend, axis=1)]
                    f1 = f1_score(enc.inverse_transform(y), bp, average="weighted")
                    if f1 > best_f1:
                        best_f1 = f1
                        best_label = f"{sp1}({w1:.2f})+{sp2}({w2:.2f})+cb({wcb:.2f})+lgbm({wlg:.2f})"
                        best_weights = {sp1: w1, sp2: w2, "catboost": wcb, "lgbm": wlg}

print(f"  Best 4-model: {best_label} = {best_f1:.4f}")

# Also try single-spatial + gbdt blends
for sp in spatial_names:
    for wsp in np.arange(0.1, 0.8, 0.05):
        for wcb in np.arange(0.05, 0.7, 0.05):
            wlg = 1.0 - wsp - wcb
            if wlg < 0.0 or wlg > 0.5:
                continue
            blend = wsp * oof[sp] + wcb * oof["catboost"] + wlg * oof["lgbm"]
            bp = enc.classes_[np.argmax(blend, axis=1)]
            f1 = f1_score(enc.inverse_transform(y), bp, average="weighted")
            if f1 > best_f1:
                best_f1 = f1
                best_label = f"{sp}({wsp:.2f})+cb({wcb:.2f})+lgbm({wlg:.2f})"
                best_weights = {sp: wsp, "catboost": wcb, "lgbm": wlg}

print(f"  Overall best: {best_label} = {best_f1:.4f}")

# Print classification report for best
# Reconstruct the blend
blend_final = sum(w * oof[n] for n, w in best_weights.items())
best_preds = enc.classes_[np.argmax(blend_final, axis=1)]
print(f"\n=== Best blend classification report ({best_f1:.4f}) ===")
print(classification_report(enc.inverse_transform(y), best_preds))
