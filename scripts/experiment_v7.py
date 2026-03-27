"""
Experiment v7: Enhanced Feature Set + SMOTE + LightGBM/CatBoost
================================================================

Builds on experiment_v6 (OBIA + PB hybrid, Ierodiaconou et al. 2018) and
integrates the best ideas from the companion repo
(niwacolours/benthic-terrain-model-kaggle, branch 002-boost-weighted-f1).

Key improvements over v6
------------------------
1. Multi-scale focal std: bathy_std_{3,5,9}, back_std_{3,5,9} (roughness texture)
2. TPI at 9 and 25 cells (topographic position index at two scales)
3. Interaction terms: bathy×back, slope×back, bathy_roughness_ratio, relief_index
4. Spatial context: relative x/y coordinates, z-score of bathy and back
5. SMOTE — fold-safe oversampling of SGAM minority class (170 vs 3036 NVB)
6. Smaller, targeted ensemble: LightGBM + CatBoost only (drop RF and XGBoost)
7. Tuned LightGBM: lr=0.03, n_estimators=600, num_leaves=31 (from companion repo)
8. CV: n_splits=6, n_clusters=6 (closer to companion repo's spatial_bins=6)
9. Uppercase 'ID' in submission CSV (required by Kaggle)
10. Global Moran's I EDA in run report for the 5 strongest features

OBIA features retained from v6
-------------------------------
- SLIC segmentation: per-segment mean/std/skew of bathy/back/VRM + pixel count
- Same min-max normalisation, same 3-channel (bathy, back, VRM) stack

Target
------
Beat Kaggle public test score of 0.76413 (current best across both repos).

Usage
-----
    python scripts/experiment_v7.py
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy import ndimage
from scipy.stats import skew as scipy_skew
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
import catboost as cb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATHY_PATH = Path("data/MBES/bathymetry.tif")
BACK_PATH = Path("data/MBES/backscatter.tif")
TRAIN_CSV = Path("data/train.csv")
TEST_CSV = Path("data/test.csv")
OUTPUT_SUBMISSION = Path("data/submission_v7_best.csv")
OUTPUT_REPORT = Path("docs/run-010-enhanced-features-lgb.md")

# PB feature column names (paper-derived, v6 base)
PB_FEATURE_COLS = [
    "depth",
    "backscatter",
    "slope",
    "vrm",
    "complexity",
    "max_curvature",
    "northness",
    "eastness",
]

# Multi-scale focal texture features (new in v7)
FOCAL_FEATURE_COLS = [
    "bathy_std_3",
    "bathy_std_5",
    "bathy_std_9",
    "back_std_3",
    "back_std_5",
    "back_std_9",
]

# TPI at 2 scales (new in v7)
TPI_FEATURE_COLS = [
    "tpi_9",
    "tpi_25",
]

# Interaction and derived terms (new in v7)
INTERACTION_FEATURE_COLS = [
    "bathy_x_back",
    "slope_x_back",
    "bathy_roughness_ratio",
    "relief_index",
]

# Spatial context (relative + z-score) from companion repo
SPATIAL_CTX_COLS = [
    "x_rel",
    "y_rel",
    "depth_z",
    "backscatter_z",
]

# OB feature column names (v6 OBIA, unchanged)
OB_FEATURE_COLS = [
    "seg_bathy_mean",
    "seg_bathy_std",
    "seg_bathy_skew",
    "seg_back_mean",
    "seg_back_std",
    "seg_back_skew",
    "seg_vrm_mean",
    "seg_vrm_std",
    "seg_vrm_skew",
    "seg_pixel_count",
]

# Full combined feature set
COMBINED_FEATURE_COLS = (
    PB_FEATURE_COLS
    + FOCAL_FEATURE_COLS
    + TPI_FEATURE_COLS
    + INTERACTION_FEATURE_COLS
    + SPATIAL_CTX_COLS
    + OB_FEATURE_COLS
)

# Segmentation parameters (same as v6)
SLIC_N_SEGMENTS = 4000
SLIC_COMPACTNESS = 0.01

# Cross-validation (6 splits / 6 clusters, matching companion repo)
CV_N_SPLITS = 6
CV_RANDOM_STATE = 42
CV_N_CLUSTERS = 6

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raster loading (same as v6)
# ---------------------------------------------------------------------------


def _load_rasters(
    bathy_path: Path, back_path: Path
) -> tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, float]:
    """Load bathymetry and backscatter GeoTIFFs, masking NoData to NaN."""
    with rasterio.open(bathy_path) as src:
        bathy_arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            bathy_arr[bathy_arr == nodata] = np.nan
        transform = src.transform
        cell_size = float(abs(transform.a))

    with rasterio.open(back_path) as src:
        back_arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            back_arr[back_arr == nodata] = np.nan

    log.info(
        "Loaded rasters: bathy %s (%.0f%% valid), back %s (%.0f%% valid), cell=%.4f m",
        bathy_arr.shape,
        100 * np.isfinite(bathy_arr).mean(),
        back_arr.shape,
        100 * np.isfinite(back_arr).mean(),
        cell_size,
    )
    return bathy_arr, back_arr, transform, cell_size


# ---------------------------------------------------------------------------
# Pixel-based features  (PB + focal + TPI + interaction — v7 extension of v6)
# ---------------------------------------------------------------------------


def _focal_std(arr: np.ndarray, size: int) -> np.ndarray:
    """Compute local standard deviation in a square (size×size) window.

    Uses E[X²] – E[X]² identity for efficiency (all operations on NaN-zeroed arrays).
    """
    a = np.nan_to_num(arr, nan=0.0).astype(np.float64)
    mean = ndimage.uniform_filter(a, size=size)
    mean_sq = ndimage.uniform_filter(a**2, size=size)
    variance = np.maximum(mean_sq - mean**2, 0.0)
    return np.sqrt(variance).astype(np.float32)


def _tpi(bathy: np.ndarray, size: int) -> np.ndarray:
    """Topographic Position Index: depth minus focal mean in size×size window."""
    bathy_f = np.nan_to_num(bathy, nan=0.0).astype(np.float64)
    focal_mean = ndimage.uniform_filter(bathy_f, size=size)
    return (bathy_f - focal_mean).astype(np.float32)


def _compute_all_features(
    bathy: np.ndarray,
    back: np.ndarray,
    cell_size: float,
) -> dict[str, np.ndarray]:
    """Compute all pixel-level features: PB + focal + TPI + interactions.

    Returns a dict with one float32 (H,W) array per feature name.
    NaN at NoData pixels throughout.
    """
    from btm.core import slope as slope_mod
    from btm.core import vrm as vrm_mod

    bathy_f = np.nan_to_num(bathy, nan=0.0)
    nodata_mask = ~np.isfinite(bathy)

    # ── Paper PB features (v6) ──────────────────────────────────────────────
    depth = bathy.copy()
    bs = back.copy()

    slope_deg = slope_mod.compute_slope(bathy_f, cell_size, nodata=None).astype(np.float32)
    slope_deg[nodata_mask] = np.nan

    vrm_arr = vrm_mod.compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size).astype(
        np.float32
    )
    vrm_arr[nodata_mask] = np.nan

    slope_rad = np.deg2rad(slope_deg)
    slope_rad_smooth = ndimage.uniform_filter(np.nan_to_num(slope_rad, nan=0.0), size=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        complexity = (1.0 / np.cos(slope_rad_smooth)).astype(np.float32)
    complexity[nodata_mask] = np.nan

    lap = ndimage.laplace(bathy_f)
    max_curvature = (np.abs(lap) / (cell_size**2)).astype(np.float32)
    max_curvature[nodata_mask] = np.nan

    dz_dy, dz_dx = np.gradient(bathy_f, cell_size)
    aspect_rad = np.arctan2(dz_dy, dz_dx)
    northness = np.cos(aspect_rad).astype(np.float32)
    eastness = np.sin(aspect_rad).astype(np.float32)
    northness[nodata_mask] = np.nan
    eastness[nodata_mask] = np.nan

    # ── Multi-scale focal std (v7 new) ─────────────────────────────────────
    bathy_std_3 = _focal_std(bathy, size=3)
    bathy_std_5 = _focal_std(bathy, size=5)
    bathy_std_9 = _focal_std(bathy, size=9)
    back_std_3 = _focal_std(back, size=3)
    back_std_5 = _focal_std(back, size=5)
    back_std_9 = _focal_std(back, size=9)
    for arr in (bathy_std_3, bathy_std_5, bathy_std_9, back_std_3, back_std_5, back_std_9):
        arr[nodata_mask] = np.nan

    # ── TPI at 2 scales (v7 new) ───────────────────────────────────────────
    tpi_9 = _tpi(bathy, size=9)
    tpi_25 = _tpi(bathy, size=25)
    tpi_9[nodata_mask] = np.nan
    tpi_25[nodata_mask] = np.nan

    # ── Interaction terms (v7 new) ─────────────────────────────────────────
    back_f = np.nan_to_num(back, nan=0.0)
    bathy_x_back = (bathy_f * back_f).astype(np.float32)
    bathy_x_back[nodata_mask] = np.nan

    slope_f = np.nan_to_num(slope_deg, nan=0.0)
    slope_x_back = (slope_f * back_f).astype(np.float32)
    slope_x_back[nodata_mask] = np.nan

    # Multi-scale roughness ratio: coarse/fine bathymetric texture
    bathy_roughness_ratio = (bathy_std_9 / (bathy_std_3 + 1e-6)).astype(np.float32)
    bathy_roughness_ratio[nodata_mask] = np.nan

    # Relief index: fine bathy + fine back texture
    relief_index = (bathy_std_3 + back_std_3).astype(np.float32)
    relief_index[nodata_mask] = np.nan

    # Replace inf with NaN (guard)
    all_arrs = dict(
        depth=depth,
        backscatter=bs,
        slope=slope_deg,
        vrm=vrm_arr,
        complexity=complexity,
        max_curvature=max_curvature,
        northness=northness,
        eastness=eastness,
        bathy_std_3=bathy_std_3,
        bathy_std_5=bathy_std_5,
        bathy_std_9=bathy_std_9,
        back_std_3=back_std_3,
        back_std_5=back_std_5,
        back_std_9=back_std_9,
        tpi_9=tpi_9,
        tpi_25=tpi_25,
        bathy_x_back=bathy_x_back,
        slope_x_back=slope_x_back,
        bathy_roughness_ratio=bathy_roughness_ratio,
        relief_index=relief_index,
    )
    for key in all_arrs:
        a = all_arrs[key]
        a[~np.isfinite(a)] = np.nan
        log.info("Feature '%s': nan_px=%d", key, int(np.isnan(a).sum()))

    return all_arrs


# ---------------------------------------------------------------------------
# Segmentation (same as v6, unchanged)
# ---------------------------------------------------------------------------


def _segment_rasters(
    bathy: np.ndarray,
    back: np.ndarray,
    vrm_arr: np.ndarray,
    n_segments: int = SLIC_N_SEGMENTS,
    compactness: float = SLIC_COMPACTNESS,
) -> np.ndarray:
    """SLIC segmentation on normalised [bathy, back, VRM] stack."""

    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    seg_stack = np.stack(
        [_minmax(bathy), _minmax(back), _minmax(vrm_arr)], axis=-1
    ).astype(np.float64)
    seg_stack = np.nan_to_num(seg_stack, nan=0.0)

    labels = slic(
        seg_stack,
        n_segments=n_segments,
        compactness=compactness,
        enforce_connectivity=True,
        start_label=0,
    ).astype(np.int32)

    n_unique = len(np.unique(labels))
    mean_size = labels.size / n_unique
    log.info(
        "Segmentation: %d labels, mean size %.1f px (%.1f m²)",
        n_unique, mean_size, mean_size * 0.25 * 0.25,
    )
    return labels


def _compute_segment_stats(
    bathy: np.ndarray,
    back: np.ndarray,
    vrm_arr: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Per-segment mean/std/skew of bathy/back/VRM + pixel count."""
    unique_labels = np.unique(labels)
    arrays = [("bathy", bathy), ("back", back), ("vrm", vrm_arr)]

    global_skew = {}
    global_median = {}
    for name, arr in arrays:
        valid_all = arr[np.isfinite(arr)]
        global_skew[name] = float(scipy_skew(valid_all)) if len(valid_all) >= 3 else 0.0
        median_val = float(np.nanmedian(arr)) if len(valid_all) > 0 else 0.0
        global_median[name] = 0.0 if not np.isfinite(median_val) else median_val

    records = []
    for seg_id in unique_labels:
        mask = labels == seg_id
        row: dict[str, float] = {}
        for name, arr in arrays:
            vals = arr[mask]
            valid = vals[np.isfinite(vals)]
            n = len(valid)
            if n == 0:
                row[f"seg_{name}_mean"] = global_median[name]
                row[f"seg_{name}_std"] = 0.0
                row[f"seg_{name}_skew"] = global_skew[name]
            elif n < 3:
                row[f"seg_{name}_mean"] = float(valid.mean())
                row[f"seg_{name}_std"] = float(valid.std())
                row[f"seg_{name}_skew"] = global_skew[name]
            else:
                row[f"seg_{name}_mean"] = float(valid.mean())
                row[f"seg_{name}_std"] = float(valid.std())
                row[f"seg_{name}_skew"] = float(scipy_skew(valid))
        row["seg_pixel_count"] = int(mask.sum())
        records.append((seg_id, row))

    df = pd.DataFrame(
        [r for _, r in records],
        index=[s for s, _ in records],
        columns=OB_FEATURE_COLS,
    )
    df.index.name = "segment_label"
    log.info("Segment stats: %d segs, NaN count: %d", len(df), int(df.isna().sum().sum()))
    return df


# ---------------------------------------------------------------------------
# Point feature extraction
# ---------------------------------------------------------------------------


def _extract_point_features(
    feature_dict: dict[str, np.ndarray],
    xy_coords: np.ndarray,
    transform: rasterio.transform.Affine,
    col_list: list[str],
) -> pd.DataFrame:
    """Sample raster feature arrays at point coordinates."""
    xs, ys = xy_coords[:, 0], xy_coords[:, 1]
    rows, cols = rowcol(transform, xs, ys)
    first_arr = next(iter(feature_dict.values()))
    h, w = first_arr.shape
    rows = np.clip(np.array(rows), 0, h - 1)
    cols = np.clip(np.array(cols), 0, w - 1)
    return pd.DataFrame(
        {key: feature_dict[key][rows, cols] for key in col_list},
        columns=col_list,
    )


def _assign_segment_features(
    point_df: pd.DataFrame,
    labels: np.ndarray,
    seg_stats_df: pd.DataFrame,
    transform: rasterio.transform.Affine,
    xy_coords: np.ndarray,
) -> pd.DataFrame:
    """Assign per-segment OB statistics to each point."""
    xs, ys = xy_coords[:, 0], xy_coords[:, 1]
    rows, cols = rowcol(transform, xs, ys)
    h, w = labels.shape
    rows = np.clip(np.array(rows), 0, h - 1)
    cols = np.clip(np.array(cols), 0, w - 1)
    seg_labels_at_pts = labels[rows, cols]
    col_medians = seg_stats_df.median(axis=0).to_dict()
    ob_rows = []
    for seg_lbl in seg_labels_at_pts:
        if seg_lbl in seg_stats_df.index:
            ob_rows.append(seg_stats_df.loc[seg_lbl, OB_FEATURE_COLS].tolist())
        else:
            ob_rows.append([col_medians[c] for c in OB_FEATURE_COLS])
    ob_df = pd.DataFrame(ob_rows, columns=OB_FEATURE_COLS, index=point_df.index)
    return pd.concat([point_df.reset_index(drop=True), ob_df.reset_index(drop=True)], axis=1)


def _add_spatial_context(
    df: pd.DataFrame, xy_coords: np.ndarray, all_xy_coords: np.ndarray
) -> pd.DataFrame:
    """Add relative x/y coordinates and z-score of depth/backscatter.

    The normalisation is computed over the FULL dataset (train+test combined)
    so that train and test see identical normalisation — no leakage.
    """
    x_min, x_max = float(all_xy_coords[:, 0].min()), float(all_xy_coords[:, 0].max())
    y_min, y_max = float(all_xy_coords[:, 1].min()), float(all_xy_coords[:, 1].max())
    df = df.copy()
    df["x_rel"] = (xy_coords[:, 0] - x_min) / max(x_max - x_min, 1.0)
    df["y_rel"] = (xy_coords[:, 1] - y_min) / max(y_max - y_min, 1.0)
    # z-scores over train+test combined (passed in df must already have depth/backscatter)
    # These are computed column-wise over the full df passed in — but we need global stats.
    # This function is called after concatenating; caller slices back.
    return df


# ---------------------------------------------------------------------------
# SMOTE utils
# ---------------------------------------------------------------------------


def _apply_smote(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """Fold-safe SMOTE with RandomOverSampler fallback.

    Applied only inside CV training folds — never on the test fold.
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler

    min_count = int(y.value_counts().min())
    if min_count < 2:
        sampler = RandomOverSampler(random_state=seed)
    else:
        k = min(5, min_count - 1)
        try:
            sampler = SMOTE(k_neighbors=k, random_state=seed)
        except Exception:
            sampler = RandomOverSampler(random_state=seed)

    X_res, y_res = sampler.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


# ---------------------------------------------------------------------------
# Spatial CV
# ---------------------------------------------------------------------------


def _get_spatial_cv_groups(coords_xy: np.ndarray) -> np.ndarray:
    """KMeans spatial block CV groups (n_clusters=CV_N_CLUSTERS, seed=42)."""
    km = KMeans(n_clusters=CV_N_CLUSTERS, random_state=CV_RANDOM_STATE, n_init=10)
    return km.fit_predict(coords_xy).astype(int)


# ---------------------------------------------------------------------------
# Models (LightGBM + CatBoost only — targeted ensemble)
# ---------------------------------------------------------------------------


def _get_models() -> dict:
    """Return LightGBM and CatBoost factory functions.

    Parameters tuned from the companion repo results
    (lightgbm-20260324225233 CV F1=0.8357, spatial_bins=6, n_splits=6).
    """
    return {
        "lgb": lambda: lgb.LGBMClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=CV_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "cat": lambda: cb.CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=6,
            auto_class_weights="Balanced",
            random_seed=CV_RANDOM_STATE,
            verbose=0,
        ),
    }


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------


def _run_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    label: str,
    use_smote: bool = True,
) -> dict:
    """GroupKFold CV with SMOTE on training folds."""
    col_idx = [COMBINED_FEATURE_COLS.index(c) for c in feature_cols]
    X = X_train[:, col_idx]
    n_classes = len(np.unique(y_train))
    classes = np.sort(np.unique(y_train))

    models = _get_models()
    gkf = GroupKFold(n_splits=CV_N_SPLITS)

    best_name = None
    best_f1 = -1.0
    best_oof_proba = None
    best_fold_scores = None

    for name, make_model in models.items():
        oof_proba = np.zeros((len(y_train), n_classes), dtype=np.float32)

        for tr_idx, va_idx in gkf.split(X, y_train, groups):
            X_tr, y_tr = pd.DataFrame(X[tr_idx], columns=feature_cols), pd.Series(
                classes[y_train[tr_idx]]
            )
            X_va = pd.DataFrame(X[va_idx], columns=feature_cols)

            if use_smote:
                X_tr, y_tr = _apply_smote(X_tr, y_tr, seed=CV_RANDOM_STATE)
                # Re-encode after SMOTE (may have reordered labels)
                y_smote_int = np.searchsorted(classes, y_tr.values)
            else:
                y_smote_int = y_train[tr_idx]

            m = make_model()
            m.fit(X_tr, y_smote_int)
            oof_proba[va_idx] = m.predict_proba(X_va)

        oof_preds_int = np.argmax(oof_proba, axis=1)
        oof_preds_labels = classes[oof_preds_int]
        f1 = f1_score(classes[y_train], oof_preds_labels, average="weighted")

        fold_scores = []
        for tr_idx, va_idx in gkf.split(X, y_train, groups):
            preds_va = classes[np.argmax(oof_proba[va_idx], axis=1)]
            fold_scores.append(f1_score(classes[y_train[va_idx]], preds_va, average="weighted"))

        log.info("  [%s] %s: CV F1=%.4f ± %.4f", label, name, f1, np.std(fold_scores))

        if f1 > best_f1:
            best_f1, best_name, best_oof_proba, best_fold_scores = (
                f1, name, oof_proba, fold_scores
            )

    log.info("[%s] Best: %s (F1=%.4f)", label, best_name, best_f1)
    return {
        "label": label,
        "best_model_name": best_name,
        "cv_f1_mean": float(best_f1),
        "cv_f1_std": float(np.std(best_fold_scores)),
        "oof_proba": best_oof_proba,
        "fold_scores": best_fold_scores,
        "best_model_factory": _get_models()[best_name],
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Moran's I (EDA only, in run report)
# ---------------------------------------------------------------------------


def _global_morans_i(values: np.ndarray, xy: np.ndarray, k: int = 8) -> float:
    """Compute Global Moran's I using k-nearest-neighbour spatial weights.

    Values should be z-score normalised before calling.
    Returns float in [-1, 1]; +1 = perfect clustering, -1 = perfect dispersion.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(values)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    distances, indices = nbrs.kneighbors(xy)
    # Build row-normalised weight matrix (binary, k-NN)
    z = values - values.mean()
    # Spatial lag: mean of z values at k nearest neighbours (row-normalised weights)
    w_z_lag = np.array([z[indices[i, 1:]].mean() for i in range(n)])
    numer = float(np.sum(z * w_z_lag))
    denom = float(np.sum(z**2))
    if denom == 0:
        return 0.0
    # Standard formula: I = N/W * sum_ij(w_ij z_i z_j) / sum_i(z_i^2)
    # With binary k-NN weights W = n*k; sum_j(w_ij z_j) = k * lag_i
    # => I = (1/k) * sum_i(z_i * k*lag_i) / sum(z^2) = sum(z*lag) / sum(z^2)
    return numer / denom


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------


def _write_run_report(
    config_results: list[dict],
    seg_params: dict,
    best_label: str,
    output_path: Path,
    classes: np.ndarray,
    n_total_features: int,
    feature_importances: dict | None = None,
    morans_i: dict | None = None,
    report_elapsed_sec: float | None = None,
) -> None:
    """Write Markdown run report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "---",
        f"date: {now}",
        "branch: 010-enhanced-features-lgb",
        "script: scripts/experiment_v7.py",
        "builds_on: experiment_v6.py (009-obia-pixel-hybrid)",
        "reference_companion: niwacolours/benthic-terrain-model-kaggle@002-boost-weighted-f1",
        "---",
        "",
        "# Run Report: Enhanced Features + SMOTE + LightGBM/CatBoost (Experiment v7)",
        "",
        "## Approach Summary",
        "",
        "Extends the v6 OBIA+PB hybrid with multi-scale focal texture, TPI,",
        "interaction terms, spatial context, and fold-safe SMOTE.",
        "Ensemble reduced to LightGBM + CatBoost (best models from both repos).",
        "",
        "| Feature Group | Count | Source |",
        "|---------------|-------|--------|",
        f"| PB paper features (depth/slope/VRM/etc.) | {len(PB_FEATURE_COLS)} | Ierodiaconou 2018 |",
        f"| Multi-scale focal std (3/5/9 cells) | {len(FOCAL_FEATURE_COLS)} | companion repo |",
        f"| TPI at 9 and 25 cells | {len(TPI_FEATURE_COLS)} | companion repo |",
        f"| Interaction terms | {len(INTERACTION_FEATURE_COLS)} | companion repo |",
        f"| Spatial context (x_rel/y_rel/z-score) | {len(SPATIAL_CTX_COLS)} | companion repo |",
        f"| OBIA segment stats (v6) | {len(OB_FEATURE_COLS)} | Ierodiaconou 2018 |",
        f"| **Total** | **{n_total_features}** | |",
        "",
        "## Segmentation Parameters",
        "",
        f"| n_segments (target) | {seg_params.get('n_segments_target', SLIC_N_SEGMENTS)} |",
        f"| n_segments (actual) | {seg_params.get('n_unique', '?')} |",
        f"| Mean segment size (px) | {seg_params.get('mean_size_px', 0):.1f} |",
        f"| Mean segment size (m²) | {seg_params.get('mean_size_m2', 0):.1f} |",
        "",
        "## CV Results",
        "",
        f"Spatial block GroupKFold (n_splits={CV_N_SPLITS}, KMeans n_clusters={CV_N_CLUSTERS}, seed=42).",
        "SMOTE applied on training folds only (fold-safe). LightGBM + CatBoost ensemble.",
        "",
        "| Configuration | Best Model | CV Weighted-F1 (mean ± std) |",
        "|---------------|------------|----------------------------|",
    ]

    for r in config_results:
        marker = " ← **BEST**" if r["label"] == best_label else ""
        lines.append(
            f"| {r['label']} | {r['best_model_name']} "
            f"| {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}{marker} |"
        )

    lines += [""]

    combined_result = next((r for r in config_results if r["label"] == "combined"), None)
    if combined_result and best_label != "combined":
        combined_f1 = combined_result["cv_f1_mean"]
        best_result = next(r for r in config_results if r["label"] == best_label)
        delta = best_result["cv_f1_mean"] - combined_f1
        lines += [
            "### ⚠ Combined Was Not Best on CV",
            "",
            f"Combined F1={combined_f1:.4f} underperformed {best_label} "
            f"(Δ={delta:.4f}). Submission uses best-CV configuration.",
            "",
        ]

    if feature_importances:
        sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:15]
        lines += [
            "## Top Feature Importances (Combined Config, Best Model)",
            "",
            "| Rank | Feature | Importance |",
            "|------|---------|------------|",
        ]
        for rank, (feat, imp) in enumerate(sorted_fi, 1):
            lines.append(f"| {rank} | {feat} | {imp:.4f} |")
        lines += [""]

    if morans_i:
        sorted_mi = sorted(morans_i.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        lines += [
            "## Spatial Autocorrelation — Global Moran's I (training points, k=8 NN)",
            "",
            "Moran's I > 0 indicates spatial clustering (nearby points similar).",
            "High values suggest spatial autocorrelation leakage risk in non-spatial CV.",
            "",
            "| Feature | Moran's I |",
            "|---------|-----------|",
        ]
        for feat, mi in sorted_mi:
            lines.append(f"| {feat} | {mi:.4f} |")
        lines += [""]

    lines += [
        "## Output Files",
        "",
        f"- Submission: `{OUTPUT_SUBMISSION}`",
        f"- Report: `{OUTPUT_REPORT}`",
        "",
    ]

    if report_elapsed_sec is not None:
        lines += [f"Report generation: {report_elapsed_sec:.1f}s", ""]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Run report written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the enhanced v7 experiment end-to-end."""
    log.info("=== Experiment v7: Enhanced Features + SMOTE + LGB/Cat ===")

    # ── Load rasters ──────────────────────────────────────────────────────
    bathy_arr, back_arr, transform, cell_size = _load_rasters(BATHY_PATH, BACK_PATH)

    # ── Pixel features (PB + focal + TPI + interactions) ──────────────────
    log.info("Computing pixel features (PB + focal + TPI + interactions)...")
    feat_dict = _compute_all_features(bathy_arr, back_arr, cell_size)
    vrm_arr = feat_dict["vrm"]

    # ── OBIA segmentation + segment stats ─────────────────────────────────
    log.info("Running SLIC segmentation (%d target segments)...", SLIC_N_SEGMENTS)
    labels = _segment_rasters(bathy_arr, back_arr, vrm_arr)
    n_unique = len(np.unique(labels))
    mean_size_px = labels.size / n_unique
    seg_params = {
        "n_segments_target": SLIC_N_SEGMENTS,
        "n_unique": n_unique,
        "mean_size_px": mean_size_px,
        "mean_size_m2": mean_size_px * cell_size**2,
    }

    log.info("Computing segment statistics...")
    seg_stats_df = _compute_segment_stats(bathy_arr, back_arr, vrm_arr, labels)

    # ── Load CSV data ──────────────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    x_col = next(c for c in train_df.columns if c.lower() in ("x", "easting", "lon", "longitude"))
    y_col = next(c for c in train_df.columns if c.lower() in ("y", "northing", "lat", "latitude"))
    label_col = next(
        c for c in train_df.columns if c.lower() in ("class", "label", "category", "substrate")
    )

    train_xy = train_df[[x_col, y_col]].values.astype(np.float64)
    test_xy = test_df[[x_col, y_col]].values.astype(np.float64)
    all_xy = np.vstack([train_xy, test_xy])

    classes = np.sort(train_df[label_col].unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_train = train_df[label_col].map(class_to_int).values.astype(int)

    # ── Extract pixel features at sampled points ───────────────────────────
    raster_cols = PB_FEATURE_COLS + FOCAL_FEATURE_COLS + TPI_FEATURE_COLS + INTERACTION_FEATURE_COLS
    log.info("Extracting pixel features at points...")
    train_px = _extract_point_features(feat_dict, train_xy, transform, raster_cols)
    test_px = _extract_point_features(feat_dict, test_xy, transform, raster_cols)

    # ── Assign OB segment features ─────────────────────────────────────────
    log.info("Assigning OB segment features...")
    train_full_raw = _assign_segment_features(train_px, labels, seg_stats_df, transform, train_xy)
    test_full_raw = _assign_segment_features(test_px, labels, seg_stats_df, transform, test_xy)

    # ── Spatial context (relative coordinates + z-scores) ──────────────────
    # Compute global stats over train+test combined for z-score normalisation
    combined_for_stats = pd.concat(
        [train_full_raw[["depth", "backscatter"]], test_full_raw[["depth", "backscatter"]]],
        axis=0,
        ignore_index=True,
    )
    depth_mean = float(combined_for_stats["depth"].mean())
    depth_std = float(combined_for_stats["depth"].std() or 1.0)
    back_mean = float(combined_for_stats["backscatter"].mean())
    back_std_val = float(combined_for_stats["backscatter"].std() or 1.0)

    x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())

    def _add_ctx(df: pd.DataFrame, xy: np.ndarray) -> pd.DataFrame:
        df = df.copy()
        df["x_rel"] = (xy[:, 0] - x_min) / max(x_max - x_min, 1.0)
        df["y_rel"] = (xy[:, 1] - y_min) / max(y_max - y_min, 1.0)
        df["depth_z"] = (df["depth"] - depth_mean) / depth_std
        df["backscatter_z"] = (df["backscatter"] - back_mean) / back_std_val
        return df

    train_full = _add_ctx(train_full_raw, train_xy)
    test_full = _add_ctx(test_full_raw, test_xy)

    # ── Build full feature matrices (N × n_features) ──────────────────────
    # Fill any remaining NaN with column median from training set
    for col in COMBINED_FEATURE_COLS:
        median = float(train_full[col].median()) if col in train_full.columns else 0.0
        if not np.isfinite(median):
            median = 0.0
        if col in train_full.columns:
            train_full[col] = train_full[col].fillna(median)
        if col in test_full.columns:
            test_full[col] = test_full[col].fillna(median)

    X_train_all = train_full[COMBINED_FEATURE_COLS].values.astype(np.float32)
    X_test_all = test_full[COMBINED_FEATURE_COLS].values.astype(np.float32)
    n_total_features = X_train_all.shape[1]
    log.info("Feature matrix: train %s, test %s", X_train_all.shape, X_test_all.shape)

    # ── Spatial CV groups ──────────────────────────────────────────────────
    groups = _get_spatial_cv_groups(train_xy)
    log.info(
        "CV groups: n_clusters=%d, distribution: %s",
        CV_N_CLUSTERS,
        np.bincount(groups).tolist(),
    )

    # ── CV configurations ──────────────────────────────────────────────────
    # Run three configurations; Combined is primary, but we report all three
    log.info("Running PB-only CV...")
    pb_result = _run_cv(X_train_all, y_train, groups, PB_FEATURE_COLS, "pb_only")
    log.info("Running OB-only CV...")
    ob_result = _run_cv(X_train_all, y_train, groups, OB_FEATURE_COLS, "ob_only")
    log.info("Running Combined CV (all %d features)...", n_total_features)
    combined_result = _run_cv(X_train_all, y_train, groups, COMBINED_FEATURE_COLS, "combined")

    config_results = [pb_result, ob_result, combined_result]
    best_result = max(config_results, key=lambda r: r["cv_f1_mean"])
    best_label = best_result["label"]
    log.info("Best configuration: %s (F1=%.4f)", best_label, best_result["cv_f1_mean"])

    # ── Train final model on full training set ────────────────────────────
    best_feat_cols = best_result["feature_cols"]
    best_col_idx = [COMBINED_FEATURE_COLS.index(c) for c in best_feat_cols]
    X_train_best = X_train_all[:, best_col_idx]
    X_test_best = X_test_all[:, best_col_idx]

    # Apply SMOTE to the full training set before final fit
    X_train_df_best = pd.DataFrame(X_train_best, columns=best_feat_cols)
    y_train_labels = pd.Series(classes[y_train])
    X_train_smoted, y_train_smoted = _apply_smote(X_train_df_best, y_train_labels)
    y_train_smoted_int = np.array([class_to_int[c] for c in y_train_smoted])

    final_model = best_result["best_model_factory"]()
    final_model.fit(X_train_smoted.values, y_train_smoted_int)
    test_pred_int = np.asarray(final_model.predict(X_test_best)).ravel().astype(int)
    test_pred_labels = classes[test_pred_int]

    # ── Feature importances from combined model ────────────────────────────
    feature_importances = None
    X_all_df = pd.DataFrame(X_train_all, columns=COMBINED_FEATURE_COLS)
    X_all_smoted, y_all_smoted_labels = _apply_smote(X_all_df, y_train_labels)
    y_all_smoted_int = np.array([class_to_int[c] for c in y_all_smoted_labels])
    cm = _get_models()[combined_result["best_model_name"]]()
    cm.fit(X_all_smoted.values, y_all_smoted_int)
    if hasattr(cm, "feature_importances_"):
        fi = cm.feature_importances_
        feature_importances = dict(zip(COMBINED_FEATURE_COLS, fi.tolist()))

    # ── Global Moran's I for top features ─────────────────────────────────
    log.info("Computing Global Moran's I (EDA)...")
    morans_i = {}
    for col in COMBINED_FEATURE_COLS[:12]:  # first 12 features for speed
        vals = train_full[col].fillna(0.0).values.astype(float)
        std = vals.std()
        if std > 0:
            z = (vals - vals.mean()) / std
            morans_i[col] = _global_morans_i(z, train_xy)
    log.info("Moran's I computed for %d features", len(morans_i))

    # ── Write submission CSV ───────────────────────────────────────────────
    id_col = next(
        (c for c in test_df.columns if c.upper() == "ID"),
        None,
    )
    if id_col is not None:
        submission = pd.DataFrame({"ID": test_df[id_col].values, "class": test_pred_labels})
    else:
        submission = pd.DataFrame(
            {"ID": np.arange(1, len(test_pred_labels) + 1), "class": test_pred_labels}
        )

    OUTPUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_SUBMISSION, index=False)
    log.info("Submission written: %s (%d rows)", OUTPUT_SUBMISSION, len(submission))
    log.info("Predicted class distribution:\n%s", submission["class"].value_counts().to_string())

    # ── Write run report ───────────────────────────────────────────────────
    t0 = datetime.now()
    _write_run_report(
        config_results=config_results,
        seg_params=seg_params,
        best_label=best_label,
        output_path=OUTPUT_REPORT,
        classes=classes,
        n_total_features=n_total_features,
        feature_importances=feature_importances,
        morans_i=morans_i,
        report_elapsed_sec=(datetime.now() - t0).total_seconds(),
    )

    log.info("=== Experiment v7 complete ===")
    for r in config_results:
        log.info("  %s: F1=%.4f ± %.4f", r["label"], r["cv_f1_mean"], r["cv_f1_std"])
    log.info("  Submission: %s", OUTPUT_SUBMISSION)
    log.info("  Report:     %s", OUTPUT_REPORT)


if __name__ == "__main__":
    main()
