"""
Step 1: Prepare GIS-ready layers for expert visual analysis.

Produces:
  outputs/gis_layers/
  ├── rasters/
  │   ├── depth_zones.tif          — 5-class categorical (quantile depth zones)
  │   ├── backscatter_zones.tif    — 5-class categorical (k-means on smoothed backscatter)
  │   ├── acoustic_facies.tif      — 8-class combined depth×backscatter k-means
  │   ├── slope.tif                — Horn (1981) slope in degrees
  │   ├── mean_curvature.tif       — Gaussian-smoothed mean curvature (sigma=4)
  │   ├── vrm.tif                  — Vector Ruggedness Measure (3×3)
  │   ├── backscatter_smooth.tif   — Gaussian-smoothed backscatter (sigma=10)
  │   └── kriging_dominant.tif     — Dominant kriging-predicted class (full-data)
  ├── vectors/
  │   └── annotation_layers.gpkg   — GeoPackage with:
  │       ├── acoustic_facies_zones — Polygonal zones with editable expert_label
  │       ├── depth_zones          — Depth zone polygons
  │       ├── backscatter_zones    — Backscatter zone polygons
  │       ├── training_points      — Training samples (class, predicted, confidence)
  │       └── test_points          — Test samples (predicted class, confidence)
  └── reports/
      └── layer_summary.md         — Description of all layers

All outputs in EPSG:28355 (GDA94 MGA Zone 55).
"""
from __future__ import annotations

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
from rasterio.features import shapes as raster_shapes
from rasterio.transform import from_bounds
from numpy.linalg import LinAlgError
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import gaussian_filter, uniform_filter
from shapely.geometry import shape as shapely_shape
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from btm.core.slope import compute_slope
from btm.core.vrm import compute_vrm
from btm.features.extract import sample_raster_at_points

CRS_EPSG = 28355
_DATA = _REPO / "data"
_OUT = _REPO / "outputs" / "gis_layers"
_OUT_R = _OUT / "rasters"
_OUT_V = _OUT / "vectors"
_OUT_RPT = _OUT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform, crs


def _write_raster(path: Path, data: np.ndarray, transform, crs,
                  dtype="float32", nodata=np.nan):
    """Write a single-band GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1], count=1,
        dtype=dtype, crs=crs, transform=transform,
        nodata=nodata, compress="lzw",
    ) as dst:
        dst.write(data.astype(dtype), 1)
    print(f"  Wrote {path.name} ({data.shape})")


def _indicator_kriging_full(xs_tr, ys_tr, y_tr, target_rows, target_cols,
                             transform, n_classes, max_tr=3000):
    """Indicator kriging on a grid of target points (row, col indices)."""
    # Convert grid points to coordinates
    xs_te = transform.c + (target_cols + 0.5) * transform.a
    ys_te = transform.f + (target_rows + 0.5) * transform.e

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


def _raster_to_polygons(zone_raster, transform, crs, valid_mask=None):
    """Convert a categorical raster to polygons via rasterio.features.shapes."""
    mask = (zone_raster >= 0)
    if valid_mask is not None:
        mask = mask & valid_mask
    data = zone_raster.astype(np.int32)
    geoms = []
    vals = []
    for geom, val in raster_shapes(data, mask=mask, transform=transform):
        geoms.append(shapely_shape(geom))
        vals.append(int(val))
    return geoms, vals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    for d in (_OUT_R, _OUT_V, _OUT_RPT):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading data...")
    train = pd.read_csv(_DATA / "train.csv")
    test = pd.read_csv(_DATA / "test.csv")
    if "ID" not in train.columns:
        train.insert(0, "ID", range(1, len(train) + 1))

    bathy, bt, crs = _load_raster(_DATA / "MBES" / "bathymetry.tif")
    back, bkt, _ = _load_raster(_DATA / "MBES" / "backscatter.tif")
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

    depths_tr = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs_tr, ys_tr, nodata=-10000
    )

    # ══════════════════════════════════════════════════════════════════
    # RASTER PRODUCTS
    # ══════════════════════════════════════════════════════════════════
    print("\n═══ Producing raster layers ═══")

    # --- Terrain derivatives ---
    print("Computing terrain derivatives...")
    slope_arr = compute_slope(bathy_f, cell_size, nodata=None)
    slope_arr[~valid_mask] = np.nan
    _write_raster(_OUT_R / "slope.tif", slope_arr, bt, crs)

    vrm_arr = compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size)
    vrm_arr[~valid_mask] = np.nan
    _write_raster(_OUT_R / "vrm.tif", vrm_arr, bt, crs)

    smooth4 = gaussian_filter(bathy_f, sigma=4)
    dx = np.gradient(smooth4, cell_size, axis=1)
    dy = np.gradient(smooth4, cell_size, axis=0)
    d2x = np.gradient(dx, cell_size, axis=1)
    d2y = np.gradient(dy, cell_size, axis=0)
    mcurv = (d2x + d2y) / 2
    mcurv[~valid_mask] = np.nan
    _write_raster(_OUT_R / "mean_curvature.tif", mcurv, bt, crs)

    # --- Smoothed backscatter ---
    back_smooth = gaussian_filter(back_f, sigma=10)
    back_smooth_out = back_smooth.copy()
    back_smooth_out[~valid_mask] = np.nan
    _write_raster(_OUT_R / "backscatter_smooth.tif", back_smooth_out, bt, crs)

    # --- Depth zones (5 quantile classes) ---
    print("Computing depth zones...")
    depth_edges = np.quantile(
        depths_tr[np.isfinite(depths_tr)], np.linspace(0, 1, 6)
    )
    depth_edges[0] = np.nanmin(bathy) - 1
    depth_edges[-1] = np.nanmax(bathy) + 1
    n_depth_zones = 5
    dz_raster = np.clip(
        np.digitize(bathy_f, depth_edges) - 1, 0, n_depth_zones - 1
    ).astype(np.int32)
    dz_raster[~valid_mask] = -1
    _write_raster(_OUT_R / "depth_zones.tif", dz_raster.astype(np.float32),
                  bt, crs, nodata=-1)

    # --- Backscatter zones (5 k-means classes) ---
    print("Computing backscatter zones...")
    vr, vc = np.where(valid_mask)
    rng = np.random.RandomState(42)
    n_sample = 50000
    si = rng.choice(len(vr), min(n_sample, len(vr)), replace=False)

    n_bz = 5
    km_bz = KMeans(n_clusters=n_bz, random_state=42, n_init=10)
    km_bz.fit(back_smooth[vr[si], vc[si]].reshape(-1, 1))

    bz_raster = np.full(back_smooth.shape, -1, dtype=np.int32)
    bz_raster[valid_mask] = km_bz.predict(back_smooth[valid_mask].reshape(-1, 1))

    # Sort zone labels by intensity (0=softest/darkest, 4=hardest/brightest)
    zone_means = [back_smooth[bz_raster == z].mean() for z in range(n_bz)]
    remap = np.zeros(n_bz, dtype=int)
    for new_l, old_l in enumerate(np.argsort(zone_means)):
        remap[old_l] = new_l
    bz_raster = np.where(bz_raster >= 0, remap[bz_raster], -1)
    _write_raster(_OUT_R / "backscatter_zones.tif", bz_raster.astype(np.float32),
                  bt, crs, nodata=-1)

    # --- Combined acoustic facies (8 k-means on depth×backscatter) ---
    print("Computing acoustic facies zones...")
    n_af = 8
    scaler_2d = StandardScaler()
    feats_2d = scaler_2d.fit_transform(np.column_stack([
        bathy_f[vr[si], vc[si]],
        back_smooth[vr[si], vc[si]],
    ]))
    km_af = KMeans(n_clusters=n_af, random_state=42, n_init=10)
    km_af.fit(feats_2d)

    af_raster = np.full(bathy_f.shape, -1, dtype=np.int32)
    af_raster[valid_mask] = km_af.predict(scaler_2d.transform(
        np.column_stack([bathy_f[valid_mask], back_smooth[valid_mask]])
    ))
    _write_raster(_OUT_R / "acoustic_facies.tif", af_raster.astype(np.float32),
                  bt, crs, nodata=-1)

    # --- Kriging dominant class raster (subsampled grid) ---
    print("Computing kriging predictions (subsampled grid)...")
    # Subsample grid for feasible kriging: every 20th pixel = ~5m resolution
    step = 20
    grid_rows = np.arange(0, bathy.shape[0], step)
    grid_cols = np.arange(0, bathy.shape[1], step)
    gr, gc = np.meshgrid(grid_rows, grid_cols, indexing="ij")
    gr_flat = gr.ravel()
    gc_flat = gc.ravel()
    # Keep only valid pixels
    krig_valid = valid_mask[gr_flat, gc_flat]
    gr_v = gr_flat[krig_valid]
    gc_v = gc_flat[krig_valid]

    print(f"  Kriging grid: {len(gr_v)} points ({step}×{step} subsample)...")
    proba_grid = _indicator_kriging_full(
        xs_tr, ys_tr, y, gr_v, gc_v, bt, n_classes, max_tr=3000
    )
    krig_class = np.argmax(proba_grid, axis=1)

    # Build kriging raster at subsampled resolution
    krig_h = len(grid_rows)
    krig_w = len(grid_cols)
    krig_raster_sub = np.full((krig_h, krig_w), -1, dtype=np.int32)
    # Map back to sub-grid indices
    ri_map = {r: i for i, r in enumerate(grid_rows)}
    ci_map = {c: i for i, c in enumerate(grid_cols)}
    for idx, (r, c) in enumerate(zip(gr_v, gc_v)):
        krig_raster_sub[ri_map[r], ci_map[c]] = krig_class[idx]

    # Write at subsampled resolution with adjusted transform
    krig_transform = rasterio.transform.from_bounds(
        bt.c, bt.f + bt.e * bathy.shape[0],
        bt.c + bt.a * bathy.shape[1], bt.f,
        krig_w, krig_h,
    )
    _write_raster(_OUT_R / "kriging_dominant.tif",
                  krig_raster_sub.astype(np.float32),
                  krig_transform, crs, nodata=-1)

    # ══════════════════════════════════════════════════════════════════
    # VECTOR PRODUCTS (GeoPackage)
    # ══════════════════════════════════════════════════════════════════
    print("\n═══ Producing vector layers (GeoPackage) ═══")
    gpkg_path = _OUT_V / "annotation_layers.gpkg"

    # --- Acoustic facies zone polygons ---
    print("Vectorising acoustic facies zones...")
    af_geoms, af_vals = _raster_to_polygons(af_raster, bt, crs, valid_mask)

    # Compute zone statistics
    af_records = []
    for geom, zone_id in zip(af_geoms, af_vals):
        # Get per-zone stats from raster
        mask_z = af_raster == zone_id
        if mask_z.sum() == 0:
            continue
        af_records.append({
            "geometry": geom,
            "zone_id": zone_id,
            "zone_type": "acoustic_facies",
            "mean_depth": float(np.nanmean(bathy[mask_z & valid_mask])) if (mask_z & valid_mask).any() else None,
            "mean_backscatter": float(np.nanmean(back[mask_z & valid_mask])) if (mask_z & valid_mask).any() else None,
            "pixel_count": int(mask_z.sum()),
            "dominant_class": None,  # Will fill after spatial join with training
            "confidence": None,
            "expert_label": None,    # ← For the analyst to fill in
            "notes": None,           # ← Free-text notes field
        })

    gdf_af = gpd.GeoDataFrame(af_records, crs=f"EPSG:{CRS_EPSG}")
    # Fill dominant_class from training points spatial join
    train_gdf = gpd.GeoDataFrame(
        train, geometry=gpd.points_from_xy(train["x"], train["y"]),
        crs=f"EPSG:{CRS_EPSG}",
    )
    joined = gpd.sjoin(train_gdf, gdf_af[["geometry", "zone_id"]].drop_duplicates("zone_id"),
                       how="left", predicate="within")

    for zid in gdf_af["zone_id"].unique():
        pts_in_zone = joined[joined["zone_id"] == zid]
        if len(pts_in_zone) > 0:
            dominant = pts_in_zone["class"].mode()
            if len(dominant) > 0:
                mask = gdf_af["zone_id"] == zid
                gdf_af.loc[mask, "dominant_class"] = dominant.iloc[0]
                gdf_af.loc[mask, "confidence"] = float(
                    (pts_in_zone["class"] == dominant.iloc[0]).mean()
                )

    # Dissolve small slivers — keep only polygons > 100 m²
    gdf_af["area_m2"] = gdf_af.geometry.area
    gdf_af = gdf_af[gdf_af["area_m2"] > 100].copy()
    gdf_af = gdf_af.drop(columns=["area_m2"])

    print(f"  {len(gdf_af)} acoustic facies polygons")
    gdf_af.to_file(gpkg_path, layer="acoustic_facies_zones", driver="GPKG")

    # --- Depth zone polygons ---
    print("Vectorising depth zones...")
    dz_geoms, dz_vals = _raster_to_polygons(dz_raster, bt, crs, valid_mask)
    dz_records = []
    for geom, zone_id in zip(dz_geoms, dz_vals):
        if zone_id < 0:
            continue
        mask_z = dz_raster == zone_id
        dz_records.append({
            "geometry": geom,
            "zone_id": zone_id,
            "zone_type": "depth_zone",
            "depth_min": float(depth_edges[zone_id]),
            "depth_max": float(depth_edges[zone_id + 1]),
            "mean_depth": float(np.nanmean(bathy[mask_z & valid_mask])) if (mask_z & valid_mask).any() else None,
            "mean_backscatter": float(np.nanmean(back[mask_z & valid_mask])) if (mask_z & valid_mask).any() else None,
            "expert_label": None,
            "notes": None,
        })
    gdf_dz = gpd.GeoDataFrame(dz_records, crs=f"EPSG:{CRS_EPSG}")
    gdf_dz["area_m2"] = gdf_dz.geometry.area
    gdf_dz = gdf_dz[gdf_dz["area_m2"] > 100].drop(columns=["area_m2"])
    print(f"  {len(gdf_dz)} depth zone polygons")
    gdf_dz.to_file(gpkg_path, layer="depth_zones", driver="GPKG")

    # --- Backscatter zone polygons ---
    print("Vectorising backscatter zones...")
    bz_geoms, bz_vals = _raster_to_polygons(bz_raster, bt, crs, valid_mask)
    bz_records = []
    for geom, zone_id in zip(bz_geoms, bz_vals):
        if zone_id < 0:
            continue
        mask_z = bz_raster == zone_id
        bz_records.append({
            "geometry": geom,
            "zone_id": zone_id,
            "zone_type": "backscatter_zone",
            "mean_depth": float(np.nanmean(bathy[mask_z & valid_mask])) if (mask_z & valid_mask).any() else None,
            "mean_backscatter": float(np.nanmean(back[mask_z & valid_mask])) if (mask_z & valid_mask).any() else None,
            "expert_label": None,
            "notes": None,
        })
    gdf_bz = gpd.GeoDataFrame(bz_records, crs=f"EPSG:{CRS_EPSG}")
    gdf_bz["area_m2"] = gdf_bz.geometry.area
    gdf_bz = gdf_bz[gdf_bz["area_m2"] > 100].drop(columns=["area_m2"])
    print(f"  {len(gdf_bz)} backscatter zone polygons")
    gdf_bz.to_file(gpkg_path, layer="backscatter_zones", driver="GPKG")

    # --- Training points ---
    print("Creating training point layer...")
    # KNN predictions for confidence
    knn = KNeighborsClassifier(5, weights="distance")
    coords_tr = train[["x", "y"]].values
    knn.fit(coords_tr, y)
    knn_proba_tr = knn.predict_proba(coords_tr)
    knn_pred_tr = enc.classes_[np.argmax(knn_proba_tr, axis=1)]
    knn_conf_tr = np.max(knn_proba_tr, axis=1)

    # Sample zone IDs at training points
    import rasterio.transform as rtransform
    tr_rows, tr_cols = rtransform.rowcol(bt, xs_tr, ys_tr)
    tr_rows = np.clip(np.asarray(tr_rows, dtype=int), 0, bathy.shape[0] - 1)
    tr_cols = np.clip(np.asarray(tr_cols, dtype=int), 0, bathy.shape[1] - 1)

    train_pt = gpd.GeoDataFrame({
        "ID": train["ID"].values if "ID" in train.columns else range(1, len(train) + 1),
        "class": train["class"].values,
        "predicted_class": knn_pred_tr,
        "knn_confidence": np.round(knn_conf_tr, 3),
        "depth": np.round(depths_tr, 2),
        "backscatter": np.round(
            sample_raster_at_points(np.nan_to_num(back, nan=-10000), bkt, xs_tr, ys_tr, nodata=-10000), 2
        ),
        "depth_zone": dz_raster[tr_rows, tr_cols],
        "backscatter_zone": bz_raster[tr_rows, tr_cols],
        "acoustic_facies": af_raster[tr_rows, tr_cols],
        "correct": (train["class"].values == knn_pred_tr).astype(int),
    }, geometry=gpd.points_from_xy(xs_tr, ys_tr), crs=f"EPSG:{CRS_EPSG}")
    train_pt.to_file(gpkg_path, layer="training_points", driver="GPKG")
    print(f"  {len(train_pt)} training points")

    # --- Test points ---
    print("Creating test point layer...")
    knn_proba_te = knn.predict_proba(test[["x", "y"]].values)
    knn_pred_te = enc.classes_[np.argmax(knn_proba_te, axis=1)]
    knn_conf_te = np.max(knn_proba_te, axis=1)

    te_rows, te_cols = rtransform.rowcol(bt, xs_te, ys_te)
    te_rows = np.clip(np.asarray(te_rows, dtype=int), 0, bathy.shape[0] - 1)
    te_cols = np.clip(np.asarray(te_cols, dtype=int), 0, bathy.shape[1] - 1)

    test_pt = gpd.GeoDataFrame({
        "ID": test["ID"].values,
        "predicted_class": knn_pred_te,
        "knn_confidence": np.round(knn_conf_te, 3),
        "depth": np.round(
            sample_raster_at_points(np.nan_to_num(bathy, nan=-10000), bt, xs_te, ys_te, nodata=-10000), 2
        ),
        "backscatter": np.round(
            sample_raster_at_points(np.nan_to_num(back, nan=-10000), bkt, xs_te, ys_te, nodata=-10000), 2
        ),
        "depth_zone": dz_raster[te_rows, te_cols],
        "backscatter_zone": bz_raster[te_rows, te_cols],
        "acoustic_facies": af_raster[te_rows, te_cols],
    }, geometry=gpd.points_from_xy(xs_te, ys_te), crs=f"EPSG:{CRS_EPSG}")
    test_pt.to_file(gpkg_path, layer="test_points", driver="GPKG")
    print(f"  {len(test_pt)} test points")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ══════════════════════════════════════════════════════════════════
    print("\n═══ Writing summary report ═══")

    # Get git revision
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO), text=True
        ).strip()
    except Exception:
        git_rev = "unknown"

    depth_zone_desc = "\n".join(
        f"  - Zone {i}: depth [{depth_edges[i]:.1f}, {depth_edges[i+1]:.1f}] m"
        for i in range(n_depth_zones)
    )

    bz_desc_lines = []
    for z in range(n_bz):
        mask_z = bz_raster == z
        if mask_z.sum() > 0:
            bmean = back_smooth[mask_z].mean()
            bz_desc_lines.append(f"  - Zone {z}: mean backscatter = {bmean:.1f} dB")
    bz_desc = "\n".join(bz_desc_lines)

    af_desc_lines = []
    for z in range(n_af):
        mask_z = af_raster == z
        if mask_z.sum() > 0:
            dmean = np.nanmean(bathy[mask_z & valid_mask])
            bmean = np.nanmean(back[mask_z & valid_mask])
            af_desc_lines.append(
                f"  - Facies {z}: mean depth = {dmean:.1f} m, "
                f"mean backscatter = {bmean:.1f} dB, "
                f"pixels = {mask_z.sum()}"
            )
    af_desc = "\n".join(af_desc_lines)

    report = f"""# GIS Layer Summary — Human-in-the-Loop Habitat Mapping

**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Git revision**: {git_rev}
**Branch**: 007-gis-expert-annotation
**CRS**: EPSG:{CRS_EPSG} (GDA94 MGA Zone 55)
**Raster resolution**: {cell_size} m
**Raster shape**: {bathy.shape[0]} rows × {bathy.shape[1]} cols
**Study area bounds**: E [{bt.c:.1f}, {bt.c + bt.a * bathy.shape[1]:.1f}], N [{bt.f + bt.e * bathy.shape[0]:.1f}, {bt.f:.1f}]

## Raster Layers

| File | Description | Type |
|------|-------------|------|
| slope.tif | Horn (1981) slope in degrees | Continuous |
| vrm.tif | Vector Ruggedness Measure (3×3 window) | Continuous |
| mean_curvature.tif | Gaussian-smoothed mean curvature (σ=4) | Continuous |
| backscatter_smooth.tif | Gaussian-smoothed backscatter (σ=10, ~2.5m) | Continuous |
| depth_zones.tif | 5 quantile-based depth zones | Categorical (0–4) |
| backscatter_zones.tif | 5 k-means backscatter zones (sorted by intensity) | Categorical (0–4) |
| acoustic_facies.tif | 8 combined depth×backscatter k-means zones | Categorical (0–7) |
| kriging_dominant.tif | Dominant kriging class ({step}×{step} subsample) | Categorical (0–{n_classes-1}) |

### Depth Zones
{depth_zone_desc}

### Backscatter Zones (sorted: 0=softest/darkest → 4=hardest/brightest)
{bz_desc}

### Acoustic Facies (depth × backscatter k-means)
{af_desc}

### Kriging Class Legend
{chr(10).join(f'  - Class {i}: {c}' for i, c in enumerate(enc.classes_))}

## Vector Layers (annotation_layers.gpkg)

| Layer | Features | Description |
|-------|----------|-------------|
| acoustic_facies_zones | {len(gdf_af)} | Zone polygons with editable `expert_label` field |
| depth_zones | {len(gdf_dz)} | Depth zone polygons |
| backscatter_zones | {len(gdf_bz)} | Backscatter zone polygons |
| training_points | {len(train_pt)} | Training samples with class, prediction, confidence |
| test_points | {len(test_pt)} | Test samples with predicted class |

## Annotation Workflow

### Step 2: Expert Visual Analysis (your turn)

1. Open `outputs/gis_layers/vectors/annotation_layers.gpkg` in QGIS
2. Add the raster layers from `outputs/gis_layers/rasters/` as base maps
3. Inspect the `acoustic_facies_zones` layer against the backscatter/bathymetry
4. For each zone polygon, edit the `expert_label` field:
   - Use habitat descriptors (e.g., "hard_reef", "sand_plain", "mixed_rubble", "algal_cover", "seagrass")
   - Or confirm the automated `dominant_class` by copying it into `expert_label`
   - Use the `notes` field for any uncertainty or observations
5. You may split or merge polygons as needed — the refinement script uses spatial joins
6. Save the GeoPackage when done

### Step 3: Return the annotated GeoPackage

Save the file and run the refinement script (Step 3, to be provided) which will:
- Read your expert labels and use them as additional features/stratification
- Produce a refined Kaggle submission with reproducibility report

## Training Data Summary

| Class | Count | Proportion |
|-------|-------|------------|
{chr(10).join(f'| {c} | {(train["class"] == c).sum()} | {(train["class"] == c).mean():.1%} |' for c in sorted(train["class"].unique()))}

## Previous Results (Baselines)

| Version | Method | Spatial CV | Kaggle F1 |
|---------|--------|-----------|-----------|
| v1 | LightGBM only (25 features) | ~0.48 | ~0.70 |
| v2 | 40% KNN + 40% CatBoost + 20% LightGBM (80 features) | 0.7521 | 0.764 |
| v3 | 60% Depth-kriging + 15% Global-kriging + 10% CB + 15% LGBM (87 feat) | 0.7668 | 0.764 |
"""

    report_path = _OUT_RPT / "layer_summary.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report: {report_path}")

    # Also write machine-readable metadata
    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev,
        "branch": "007-gis-expert-annotation",
        "crs_epsg": CRS_EPSG,
        "cell_size_m": cell_size,
        "raster_shape": list(bathy.shape),
        "n_train": len(train),
        "n_test": len(test),
        "classes": list(enc.classes_),
        "depth_zone_edges": [float(e) for e in depth_edges],
        "n_acoustic_facies": n_af,
        "n_backscatter_zones": n_bz,
        "rasters": [f.name for f in sorted(_OUT_R.glob("*.tif"))],
        "gpkg_layers": ["acoustic_facies_zones", "depth_zones", "backscatter_zones",
                        "training_points", "test_points"],
    }
    meta_path = _OUT_RPT / "layer_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Metadata: {meta_path}")

    elapsed = time.time() - t_start
    print(f"\n✓ All GIS layers ready in {elapsed:.0f}s")
    print(f"  Rasters: {_OUT_R}")
    print(f"  Vectors: {gpkg_path}")
    print(f"  Report:  {_OUT_RPT}")


if __name__ == "__main__":
    main()
