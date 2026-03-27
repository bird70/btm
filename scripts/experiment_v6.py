"""
Experiment v6: OBIA + Pixel-Based Hybrid Classification
========================================================

Implements the combined Object-Based Image Analysis (OBIA) + pixel-based (PB)
hybrid habitat classification approach from:

    Ierodiaconou et al. (2018). "Comparison of automated classification
    techniques for predicting benthic biological communities using hydroacoustics
    and video observations." Marine Geodesy, 41(5), 442-461.
    DOI: 10.1007/s11001-017-9338-z

The paper used the same Refuge Cove MBES dataset (0.25 m resolution, 5 benthic
classes) and reported combined PB+OB accuracy of 83.6% vs 72.5% PB-only.
This experiment attempts to replicate and improve on that result using
modern Python tooling.

Approach:
    - 8 pixel-based (PB) features: depth, backscatter, slope, vrm, complexity,
      max_curvature, northness, eastness
    - 10 object-based (OB) features: per-segment mean/std/skewness of bathy,
      backscatter, vrm + segment pixel count
    - 3 CV configurations: PB-only, OB-only, Combined (18 features)
    - Ensemble: LightGBM, XGBoost, CatBoost, RandomForest
    - Spatial block cross-validation (KMeans, n_clusters=10, seed=42)

Usage:
    python scripts/experiment_v6.py
"""

import logging
from datetime import datetime
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.transform import rowcol
from scipy import ndimage
from scipy.stats import skew as scipy_skew
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATHY_PATH = Path("data/MBES/bathymetry.tif")
BACK_PATH = Path("data/MBES/backscatter.tif")
TRAIN_CSV = Path("data/train.csv")
TEST_CSV = Path("data/test.csv")
OUTPUT_SUBMISSION = Path("data/submission_v6_best.csv")
OUTPUT_REPORT = Path("docs/run-009-obia-pixel-hybrid.md")

# PB feature column names (must match _compute_pb_features dict keys)
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

# OB feature column names (must match _compute_segment_stats output columns)
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

COMBINED_FEATURE_COLS = PB_FEATURE_COLS + OB_FEATURE_COLS

# Segmentation parameters
SLIC_N_SEGMENTS = 4000  # targets ~300 m² mean object on 0.25 m grid
SLIC_COMPACTNESS = 0.01  # low = irregular, natural boundaries

# Cross-validation
CV_N_SPLITS = 10
CV_RANDOM_STATE = 42

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------


def _load_rasters(
    bathy_path: Path, back_path: Path
) -> tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, float]:
    """Load bathymetry and backscatter GeoTIFFs, masking NoData to NaN.

    Returns:
        bathy_arr: float32 array (H, W), NoData → NaN
        back_arr:  float32 array (H, W), NoData → NaN
        transform: rasterio Affine transform for raster→world coordinate mapping
        cell_size: pixel size in metres (abs of transform.a)
    """
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
# Pixel-based features
# ---------------------------------------------------------------------------


def _compute_pb_features(
    bathy: np.ndarray, back: np.ndarray, cell_size: float
) -> dict[str, np.ndarray]:
    """Compute 8 pixel-based spatial derivatives from MBES rasters.

    Algorithm citations:
        slope: Horn (1981)
        vrm: Sappington et al. (2007) — reused from btm.core.vrm
        complexity: Jenness (2004) surface-area ratio = 1/cos(slope_rad)
        max_curvature: Zevenbergen & Thorne (1987) via Laplace approximation
        northness/eastness: Roberts (1986) circular aspect decomposition

    Args:
        bathy: bathymetry array (H, W), NaN at NoData
        back: backscatter array (H, W), NaN at NoData
        cell_size: pixel size in metres

    Returns:
        dict with keys matching PB_FEATURE_COLS; each value is (H, W) float32
    """
    from btm.core import slope as slope_mod
    from btm.core import vrm as vrm_mod

    # Build a NaN-free copy for derivative computations (NaN fills with 0.0).
    # This matches the pattern in experiment_v2 and prevents NaN propagation
    # through np.gradient and scipy.ndimage operations across the NoData border.
    bathy_f = np.nan_to_num(bathy, nan=0.0)
    nodata_mask = ~np.isfinite(bathy)  # original NoData locations

    # depth — bathymetry itself
    depth = bathy.copy()

    # backscatter — as-is
    bs = back.copy()

    # slope (degrees) — use NaN-free array; nodata=None → no extra masking
    slope_deg = slope_mod.compute_slope(bathy_f, cell_size, nodata=None).astype(np.float32)
    slope_deg[nodata_mask] = np.nan

    # vrm — Vector Ruggedness Measure (Sappington et al. 2007); 3×3 neighbourhood
    vrm_arr = vrm_mod.compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size).astype(
        np.float32
    )
    vrm_arr[nodata_mask] = np.nan

    # complexity — surface area ratio = 1/cos(slope_rad), smoothed 3×3
    slope_rad = np.deg2rad(slope_deg)
    slope_rad_smooth = ndimage.uniform_filter(np.nan_to_num(slope_rad, nan=0.0), size=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        complexity = (1.0 / np.cos(slope_rad_smooth)).astype(np.float32)
    complexity[nodata_mask] = np.nan

    # max_curvature — Laplace approximation: |∇²z| / cell_size²
    lap = ndimage.laplace(bathy_f)
    max_curvature = (np.abs(lap) / (cell_size**2)).astype(np.float32)
    max_curvature[nodata_mask] = np.nan

    # northness and eastness — circular-safe aspect decomposition
    dz_dy, dz_dx = np.gradient(bathy_f, cell_size)
    aspect_rad = np.arctan2(dz_dy, dz_dx)
    northness = np.cos(aspect_rad).astype(np.float32)
    eastness = np.sin(aspect_rad).astype(np.float32)
    northness[nodata_mask] = np.nan
    eastness[nodata_mask] = np.nan

    features = {
        "depth": depth,
        "backscatter": bs,
        "slope": slope_deg,
        "vrm": vrm_arr,
        "complexity": complexity,
        "max_curvature": max_curvature,
        "northness": northness,
        "eastness": eastness,
    }

    # Replace inf with NaN (guard against edge cases in cos/curvature)
    for key in features:
        arr = features[key]
        arr[~np.isfinite(arr)] = np.nan
        features[key] = arr
        log.info(
            "PB feature '%s': shape=%s, nan_px=%d",
            key,
            arr.shape,
            int(np.isnan(arr).sum()),
        )

    return features


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _segment_rasters(
    bathy: np.ndarray,
    back: np.ndarray,
    vrm_arr: np.ndarray,
    n_segments: int = SLIC_N_SEGMENTS,
    compactness: float = SLIC_COMPACTNESS,
) -> np.ndarray:
    """Segment MBES rasters into spatially contiguous objects using SLIC.

    Algorithm: Achanta et al. (2012) SLIC Superpixels.
    Channels: normalised [bathymetry, backscatter, vrm] (min-max per channel).
    NaN pixels are zeroed before segmentation and ignored in statistics.

    Args:
        bathy, back, vrm_arr: (H, W) float32 input arrays
        n_segments: approximate number of segments (default 4000 → ~300 m² mean)
        compactness: SLIC compactness (0.01 = irregular boundaries)

    Returns:
        labels: (H, W) int32 array; label ≥ 0 for every pixel
    """

    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if hi > lo:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr)

    seg_stack = np.stack(
        [_minmax(bathy), _minmax(back), _minmax(vrm_arr)], axis=-1
    ).astype(np.float64)

    # NaN → 0 before passing to SLIC
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
        "Segmentation: %d unique labels, mean size %.1f px (%.1f m²)",
        n_unique,
        mean_size,
        mean_size * 0.25 * 0.25,
    )
    return labels


# ---------------------------------------------------------------------------
# Segment statistics
# ---------------------------------------------------------------------------


def _compute_segment_stats(
    bathy: np.ndarray,
    back: np.ndarray,
    vrm_arr: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute per-segment statistics for bathymetry, backscatter, and VRM.

    For each segment: mean, std, skewness × 3 channels + pixel count = 10 columns.

    Fallback rules (per spec edge case):
        - Segments with < 3 valid pixels: use global mean for skewness
        - Segments with 0 valid pixels: use global median for all stats

    Returns:
        DataFrame indexed by segment label, columns = OB_FEATURE_COLS
    """
    unique_labels = np.unique(labels)
    arrays = [("bathy", bathy), ("back", back), ("vrm", vrm_arr)]

    # Pre-compute global stats for fallback (only for segments with >=3 pixels)
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
    log.info(
        "Segment stats: %d segments, %d columns, NaN count: %d",
        len(df),
        len(df.columns),
        int(df.isna().sum().sum()),
    )
    return df


# ---------------------------------------------------------------------------
# Point feature extraction
# ---------------------------------------------------------------------------


def _extract_point_features(
    feature_dict: dict[str, np.ndarray],
    xy_coords: np.ndarray,
    transform: rasterio.transform.Affine,
) -> pd.DataFrame:
    """Sample raster feature arrays at point coordinates.

    Args:
        feature_dict: dict of (H, W) arrays, keys = PB_FEATURE_COLS
        xy_coords: (N, 2) array of (x, y) world coordinates
        transform: rasterio Affine transform

    Returns:
        DataFrame with len(xy_coords) rows and PB_FEATURE_COLS columns
    """
    xs, ys = xy_coords[:, 0], xy_coords[:, 1]
    rows, cols = rowcol(transform, xs, ys)

    # Get array shape from first feature
    first_arr = next(iter(feature_dict.values()))
    h, w = first_arr.shape

    # Clip indices to valid bounds
    rows = np.clip(np.array(rows), 0, h - 1)
    cols = np.clip(np.array(cols), 0, w - 1)

    data = {}
    for key in PB_FEATURE_COLS:
        arr = feature_dict[key]
        data[key] = arr[rows, cols]

    return pd.DataFrame(data, columns=PB_FEATURE_COLS)


def _assign_segment_features(
    point_df: pd.DataFrame,
    labels: np.ndarray,
    seg_stats_df: pd.DataFrame,
    transform: rasterio.transform.Affine,
    xy_coords: np.ndarray,
) -> pd.DataFrame:
    """Assign per-segment OB statistics to each point via segment membership.

    Args:
        point_df: DataFrame with PB features (N rows)
        labels: (H, W) int32 segment label array
        seg_stats_df: DataFrame indexed by segment_label, OB_FEATURE_COLS columns
        transform: rasterio Affine transform
        xy_coords: (N, 2) world coordinates for the points

    Returns:
        point_df with 10 OB columns appended (N × 18 total)
    """
    xs, ys = xy_coords[:, 0], xy_coords[:, 1]
    rows, cols = rowcol(transform, xs, ys)

    h, w = labels.shape
    rows = np.clip(np.array(rows), 0, h - 1)
    cols = np.clip(np.array(cols), 0, w - 1)

    seg_labels_at_pts = labels[rows, cols]

    # Build per-point OB feature rows via lookup; fall back to column medians
    col_medians = seg_stats_df.median(axis=0).to_dict()

    ob_rows = []
    for seg_lbl in seg_labels_at_pts:
        if seg_lbl in seg_stats_df.index:
            ob_rows.append(seg_stats_df.loc[seg_lbl, OB_FEATURE_COLS].tolist())
        else:
            ob_rows.append([col_medians[c] for c in OB_FEATURE_COLS])

    ob_df = pd.DataFrame(ob_rows, columns=OB_FEATURE_COLS, index=point_df.index)
    return pd.concat([point_df.reset_index(drop=True), ob_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Models and CV
# ---------------------------------------------------------------------------


def get_models() -> dict:
    """Return a dict of model factory functions (name → callable returning estimator).

    Reused from experiment_v2 approach. Each call returns a fresh untrained model.
    """
    return {
        "lgb": lambda: lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
        "xgb": lambda: xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        ),
        "cat": lambda: cb.CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def _get_spatial_cv_groups(coords_xy: np.ndarray) -> np.ndarray:
    """Assign spatial block IDs to training points using KMeans clustering.

    Uses identical parameters to experiment_v2 for CV comparability.

    Args:
        coords_xy: (N, 2) array of (x, y) world coordinates

    Returns:
        groups: (N,) integer array with values in [0, CV_N_SPLITS-1]
    """
    km = KMeans(n_clusters=CV_N_SPLITS, random_state=CV_RANDOM_STATE, n_init=10)
    return km.fit_predict(coords_xy).astype(int)


def _run_cv_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    label: str,
) -> dict:
    """Run grouped K-fold cross-validation for all models on a feature subset.

    Args:
        X_train: full feature matrix (N × 18); sliced by feature_cols internally
        y_train: integer class labels (N,)
        groups: spatial block assignments (N,)
        feature_cols: list of column names to use (subset of COMBINED_FEATURE_COLS)
        label: config label for reporting ("pb_only", "ob_only", "combined")

    Returns:
        dict with keys: label, best_model_name, cv_f1_mean, cv_f1_std,
                        oof_preds (N,), oof_proba (N × n_classes), fold_scores,
                        best_model_factory (callable)
    """
    col_idx = [COMBINED_FEATURE_COLS.index(c) for c in feature_cols]
    X = X_train[:, col_idx]
    n_classes = len(np.unique(y_train))

    models = get_models()
    gkf = GroupKFold(n_splits=CV_N_SPLITS)

    best_name = None
    best_f1 = -1.0
    best_oof_proba = None
    best_fold_scores = None

    for name, make_model in models.items():
        oof_proba = np.zeros((len(y_train), n_classes), dtype=np.float32)

        for tr_idx, va_idx in gkf.split(X, y_train, groups):
            m = make_model()
            X_tr, y_tr = X[tr_idx], y_train[tr_idx]
            X_va = X[va_idx]

            if name == "xgb":
                sw = compute_sample_weight("balanced", y_tr)
                m.fit(X_tr, y_tr, sample_weight=sw)
            else:
                m.fit(X_tr, y_tr)

            oof_proba[va_idx] = m.predict_proba(X_va)

        oof_preds = np.argmax(oof_proba, axis=1)
        # Map back to original class values if needed
        classes = np.sort(np.unique(y_train))
        oof_preds_labels = classes[oof_preds]
        f1 = f1_score(y_train, oof_preds_labels, average="weighted")

        # Per-fold F1
        fold_scores = []
        for tr_idx, va_idx in gkf.split(X, y_train, groups):
            preds_va = classes[np.argmax(oof_proba[va_idx], axis=1)]
            fold_scores.append(f1_score(y_train[va_idx], preds_va, average="weighted"))

        log.info("  [%s] %s: CV weighted-F1 = %.4f ± %.4f", label, name, f1, np.std(fold_scores))

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_oof_proba = oof_proba
            best_fold_scores = fold_scores

    log.info("[%s] Best model: %s (F1=%.4f)", label, best_name, best_f1)

    return {
        "label": label,
        "best_model_name": best_name,
        "cv_f1_mean": float(best_f1),
        "cv_f1_std": float(np.std(best_fold_scores)),
        "oof_proba": best_oof_proba,
        "fold_scores": best_fold_scores,
        "best_model_factory": get_models()[best_name],
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------


def _write_run_report(
    config_results: list[dict],
    seg_params: dict,
    best_label: str,
    output_path: Path,
    classes: np.ndarray,
    feature_importances: dict | None = None,
    report_elapsed_sec: float | None = None,
) -> None:
    """Write Markdown run report documenting the experiment.

    Args:
        config_results: list of result dicts from _run_cv_config
        seg_params: dict with segmentation details (n_segments, n_unique, mean_size_px, etc.)
        best_label: label of the best-CV configuration
        output_path: path to write .md file
        classes: sorted class label array
        feature_importances: dict mapping feature name → mean importance (from best combined model)
        report_elapsed_sec: time taken for report generation
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    combined_result = next((r for r in config_results if r["label"] == "combined"), None)

    lines = [
        "---",
        f"date: {now}",
        "branch: 009-obia-pixel-hybrid",
        "script: scripts/experiment_v6.py",
        "reference: Ierodiaconou et al. (2018) DOI:10.1007/s11001-017-9338-z",
        "---",
        "",
        "# Run Report: OBIA + Pixel-Based Hybrid (Experiment v6)",
        "",
        "## Approach",
        "",
        "Combined pixel-based (PB) + object-based (OB) habitat classification following",
        "Ierodiaconou et al. (2018) on the Refuge Cove MBES dataset (0.25 m, 5 classes).",
        "Paper baseline: PB=72.5%, OB=78.5%, Combined=83.6% overall accuracy.",
        "",
        "## Segmentation Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Algorithm | SLIC (scikit-image 0.26, Achanta et al. 2012) |",
        "| Input channels | bathymetry, backscatter, VRM (min-max normalised) |",
        f"| n_segments (target) | {seg_params.get('n_segments_target', SLIC_N_SEGMENTS)} |",
        f"| n_segments (actual) | {seg_params.get('n_unique', '?')} |",
        f"| compactness | {seg_params.get('compactness', SLIC_COMPACTNESS)} |",
        f"| Mean segment size (px) | {seg_params.get('mean_size_px', '?'):.1f} |",
        f"| Mean segment size (m²) | {seg_params.get('mean_size_m2', '?'):.1f} |",
        "| Target object size (m²) | ~300 (paper scale=41 equivalent) |",
        "",
        "## CV Results",
        "",
        "Spatial block GroupKFold (n_splits=10, KMeans n_clusters=10, seed=42).",
        "Identical CV setup to experiment_v2 for direct comparability.",
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

    # Flag if combined is not best
    if combined_result and best_label != "combined":
        combined_f1 = combined_result["cv_f1_mean"]
        best_result = next(r for r in config_results if r["label"] == best_label)
        delta = best_result["cv_f1_mean"] - combined_f1
        lines += [
            "### ⚠ Combined Configuration Was Not Best on CV",
            "",
            f"The combined PB+OB configuration (F1={combined_f1:.4f}) underperformed the "
            f"best configuration ({best_label}, F1={best_result['cv_f1_mean']:.4f}) "
            f"by Δ={delta:.4f} on CV.",
            "Per SC-002: submission contains best-CV configuration predictions.",
            "The paper's rationale (spatial coherence + spectral diversity) may not hold",
            "equally for this specific fold structure. Consider reviewing segment scale.",
            "",
        ]

    # Feature importances
    if feature_importances:
        sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        lines += [
            "## Top Feature Importances (Combined Config, Best Model)",
            "",
            "| Rank | Feature | Importance |",
            "|------|---------|------------|",
        ]
        for rank, (feat, imp) in enumerate(sorted_fi, 1):
            lines.append(f"| {rank} | {feat} | {imp:.4f} |")
        lines += [""]

    lines += [
        "## Output Files",
        "",
        f"- Submission: `{OUTPUT_SUBMISSION}`",
        f"- Report: `{OUTPUT_REPORT}`",
        "",
    ]

    if report_elapsed_sec is not None:
        lines += [
            "## Timing",
            "",
            "| Phase | Duration |",
            "|-------|----------|",
            f"| Report generation | {report_elapsed_sec:.1f}s |",
            "",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Run report written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the OBIA + pixel-based hybrid experiment end-to-end."""
    log.info("=== Experiment v6: OBIA + Pixel-Based Hybrid ===")

    # ── Load rasters ──────────────────────────────────────────────────────
    bathy_arr, back_arr, transform, cell_size = _load_rasters(BATHY_PATH, BACK_PATH)

    # ── PB features ───────────────────────────────────────────────────────
    log.info("Computing pixel-based features...")
    pb_features = _compute_pb_features(bathy_arr, back_arr, cell_size)
    vrm_arr = pb_features["vrm"]

    # ── Segmentation ──────────────────────────────────────────────────────
    log.info("Running SLIC segmentation (%d target segments)...", SLIC_N_SEGMENTS)
    labels = _segment_rasters(bathy_arr, back_arr, vrm_arr)
    n_unique = len(np.unique(labels))
    mean_size_px = labels.size / n_unique
    seg_params = {
        "n_segments_target": SLIC_N_SEGMENTS,
        "n_unique": n_unique,
        "compactness": SLIC_COMPACTNESS,
        "mean_size_px": mean_size_px,
        "mean_size_m2": mean_size_px * cell_size * cell_size,
    }

    # ── Segment statistics ─────────────────────────────────────────────────
    log.info("Computing segment statistics...")
    seg_stats_df = _compute_segment_stats(bathy_arr, back_arr, vrm_arr, labels)

    # ── Load CSV data ──────────────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # Detect coordinate and label columns
    x_col = next(c for c in train_df.columns if c.lower() in ("x", "easting", "lon", "longitude"))
    y_col = next(c for c in train_df.columns if c.lower() in ("y", "northing", "lat", "latitude"))
    label_col = next(
        c for c in train_df.columns if c.lower() in ("class", "label", "category", "substrate")
    )

    train_xy = train_df[[x_col, y_col]].values.astype(np.float64)
    test_xy = test_df[[x_col, y_col]].values.astype(np.float64)

    # Integer-encode class labels
    classes = np.sort(train_df[label_col].unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_train = train_df[label_col].map(class_to_int).values.astype(int)

    # ── Extract PB point features ──────────────────────────────────────────
    log.info("Extracting PB point features...")
    train_pb = _extract_point_features(pb_features, train_xy, transform)
    test_pb = _extract_point_features(pb_features, test_xy, transform)

    # ── Assign OB segment features ─────────────────────────────────────────
    log.info("Assigning OB segment features...")
    train_full = _assign_segment_features(train_pb, labels, seg_stats_df, transform, train_xy)
    test_full = _assign_segment_features(test_pb, labels, seg_stats_df, transform, test_xy)

    # Build combined feature matrices (N × 18)
    X_train_all = train_full[COMBINED_FEATURE_COLS].values.astype(np.float32)
    X_test_all = test_full[COMBINED_FEATURE_COLS].values.astype(np.float32)

    # ── Spatial CV groups ──────────────────────────────────────────────────
    groups = _get_spatial_cv_groups(train_xy)
    log.info(
        "CV groups: %d clusters, distribution: %s",
        CV_N_SPLITS,
        np.bincount(groups).tolist(),
    )

    # ── Run CV configurations ──────────────────────────────────────────────
    log.info("Running PB-only CV...")
    pb_result = _run_cv_config(X_train_all, y_train, groups, PB_FEATURE_COLS, "pb_only")
    log.info("Running OB-only CV...")
    ob_result = _run_cv_config(X_train_all, y_train, groups, OB_FEATURE_COLS, "ob_only")
    log.info("Running Combined CV...")
    combined_result = _run_cv_config(
        X_train_all, y_train, groups, COMBINED_FEATURE_COLS, "combined"
    )

    config_results = [pb_result, ob_result, combined_result]

    # ── Select best configuration ──────────────────────────────────────────
    best_result = max(config_results, key=lambda r: r["cv_f1_mean"])
    best_label = best_result["label"]
    log.info(
        "Best configuration: %s (F1=%.4f)", best_label, best_result["cv_f1_mean"]
    )

    # ── Train final model on full training set ─────────────────────────────
    best_feat_cols = best_result["feature_cols"]
    best_col_idx = [COMBINED_FEATURE_COLS.index(c) for c in best_feat_cols]
    X_train_best = X_train_all[:, best_col_idx]
    X_test_best = X_test_all[:, best_col_idx]

    final_model = best_result["best_model_factory"]()
    if best_result["best_model_name"] == "xgb":
        sw = compute_sample_weight("balanced", y_train)
        final_model.fit(X_train_best, y_train, sample_weight=sw)
    else:
        final_model.fit(X_train_best, y_train)

    test_pred_int = np.asarray(final_model.predict(X_test_best)).ravel().astype(int)
    test_pred_labels = classes[test_pred_int]

    # ── Feature importances for combined model ─────────────────────────────
    feature_importances = None
    combined_model_factory = combined_result["best_model_factory"]
    cm = combined_model_factory()
    if combined_result["best_model_name"] == "xgb":
        sw = compute_sample_weight("balanced", y_train)
        cm.fit(X_train_all, y_train, sample_weight=sw)
    else:
        cm.fit(X_train_all, y_train)

    if hasattr(cm, "feature_importances_"):
        fi_vals = cm.feature_importances_
        feature_importances = dict(zip(COMBINED_FEATURE_COLS, fi_vals.tolist()))
    elif hasattr(cm, "coef_"):
        fi_vals = np.abs(cm.coef_).mean(axis=0)
        feature_importances = dict(zip(COMBINED_FEATURE_COLS, fi_vals.tolist()))

    # ── Write submission CSV ───────────────────────────────────────────────
    id_col = next(
        (c for c in test_df.columns if c.lower() in ("id", "index", "rowid")),
        None,
    )
    # Flatten prediction output (some estimators return 2-D arrays)
    test_pred_labels = np.asarray(test_pred_labels).ravel()
    if id_col is not None:
        submission = pd.DataFrame({"id": test_df[id_col].values, "class": test_pred_labels})
    else:
        submission = pd.DataFrame(
            {"id": np.arange(len(test_pred_labels)), "class": test_pred_labels}
        )

    OUTPUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_SUBMISSION, index=False)
    log.info("Submission written: %s (%d rows)", OUTPUT_SUBMISSION, len(submission))

    # ── Write run report ───────────────────────────────────────────────────
    t_report_start = datetime.now()
    _write_run_report(
        config_results=config_results,
        seg_params=seg_params,
        best_label=best_label,
        output_path=OUTPUT_REPORT,
        classes=classes,
        feature_importances=feature_importances,
    )
    report_elapsed = (datetime.now() - t_report_start).total_seconds()
    # Re-write with timing
    _write_run_report(
        config_results=config_results,
        seg_params=seg_params,
        best_label=best_label,
        output_path=OUTPUT_REPORT,
        classes=classes,
        feature_importances=feature_importances,
        report_elapsed_sec=report_elapsed,
    )

    log.info("=== Experiment v6 complete ===")
    log.info("  Submission: %s", OUTPUT_SUBMISSION)
    log.info("  Report:     %s", OUTPUT_REPORT)
    for r in config_results:
        log.info("  %s: F1=%.4f ± %.4f", r["label"], r["cv_f1_mean"], r["cv_f1_std"])


if __name__ == "__main__":
    main()
