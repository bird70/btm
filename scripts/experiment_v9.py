"""
Experiment v9: v6 Architecture + Texture Features + Stronger Boosting
======================================================================

ROOT CAUSE OF v8 FAILURE (Kaggle 0.47)
---------------------------------------
v8 selected Random Forest (PB-only) as the best-CV model.
RF does not generalise as well as CatBoost for this acoustic/terrain data.
v6 used CatBoost on COMBINED PB+OB features and scored 0.7639 on Kaggle.
v8's high CV std (0.2975) was the warning sign: RF was unstable across folds.

v9 DESIGN PRINCIPLES
---------------------
1. Return to v6's proven architecture: CatBoost + LGB on COMBINED PB+OB features.
2. Extend PB features with 3 non-leaky texture/TPI features (proven no leakage):
       bathy_std_9  — bathymetric roughness at 2.25 m scale (9-cell window)
       back_std_9   — acoustic texture at 2.25 m scale
       tpi_9        — topographic position index (am I on a ridge or hollow?)
3. NO spatial coordinates (removed from v7 where they caused spatial memorisation).
4. Stronger CatBoost (iterations=800, depth=7 vs v6's 400, depth=6).
5. Stronger LGB (companion repo's tuned params: n_estimators=600, lr=0.03,
   num_leaves=31 vs v6's n_estimators=400, lr=0.05, num_leaves=63).
6. Add soft-vote CatBoost+LGB ensemble as a 4th model candidate.
7. Keep 10-fold spatial GroupKFold (same as v6, directly comparable).
8. Best-CV config selected for submission; combined expected to win.

FEATURE SET (21 total)
----------------------
PB (11):  depth, backscatter, slope, vrm, complexity, max_curvature,
          northness, eastness,
          bathy_std_9, back_std_9, tpi_9   ← NEW: texture+TPI, no leakage
OB (10):  seg_{bathy,back,vrm}_{mean,std,skew} + seg_pixel_count  (v6 unchanged)
Total:    21 features, zero spatial coordinate leakage

TARGET
------
Kaggle public test F1 > 0.764 (v6 = 0.76394).

USAGE
-----
    python scripts/experiment_v9.py
"""

import logging
from datetime import datetime
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
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
BACK_PATH  = Path("data/MBES/backscatter.tif")
TRAIN_CSV  = Path("data/train.csv")
TEST_CSV   = Path("data/test.csv")
OUTPUT_SUBMISSION = Path("data/submission_v9_best.csv")
OUTPUT_REPORT     = Path("docs/run-012-v9-catboost-lgb-texture.md")

# PB features: v6's 8 paper features + 3 new texture/TPI (no spatial coords)
PB_FEATURE_COLS = [
    "depth",
    "backscatter",
    "slope",
    "vrm",
    "complexity",
    "max_curvature",
    "northness",
    "eastness",
    # ↓ new in v9: non-leaky texture and topographic position
    "bathy_std_9",
    "back_std_9",
    "tpi_9",
]

# OB features: same 10 SLIC segment statistics as v6 (unchanged)
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

COMBINED_FEATURE_COLS = PB_FEATURE_COLS + OB_FEATURE_COLS  # 21 features

# Segmentation (same as v6)
SLIC_N_SEGMENTS  = 4000
SLIC_COMPACTNESS = 0.01

# CV (same as v6 for direct F1 comparability)
CV_N_SPLITS     = 10
CV_RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raster loading (identical to v6)
# ---------------------------------------------------------------------------

def _load_rasters(
    bathy_path: Path, back_path: Path
) -> tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, float]:
    with rasterio.open(bathy_path) as src:
        bathy = src.read(1).astype(np.float32)
        if src.nodata is not None:
            bathy[bathy == src.nodata] = np.nan
        transform = src.transform
        cell_size = float(abs(transform.a))
    with rasterio.open(back_path) as src:
        back = src.read(1).astype(np.float32)
        if src.nodata is not None:
            back[back == src.nodata] = np.nan
    log.info(
        "Rasters: bathy %s (%.0f%% valid)  back %s  cell=%.4f m",
        bathy.shape, 100 * np.isfinite(bathy).mean(), back.shape, cell_size,
    )
    return bathy, back, transform, cell_size


# ---------------------------------------------------------------------------
# Pixel-based features (v6 PB features + 3 texture/TPI)
# ---------------------------------------------------------------------------

def _focal_std(arr: np.ndarray, size: int) -> np.ndarray:
    """Local standard deviation via E[X²] – E[X]² identity."""
    a = np.nan_to_num(arr, nan=0.0).astype(np.float64)
    return np.sqrt(
        np.maximum(
            ndimage.uniform_filter(a ** 2, size=size)
            - ndimage.uniform_filter(a, size=size) ** 2,
            0.0,
        )
    ).astype(np.float32)


def _compute_pb_features(
    bathy: np.ndarray,
    back: np.ndarray,
    cell_size: float,
) -> dict[str, np.ndarray]:
    """Compute all 11 pixel-based features (8 paper + 3 texture/TPI).

    NaN at NoData pixels throughout. No spatial coordinate features.
    """
    from btm.core import slope as slope_mod
    from btm.core import vrm as vrm_mod

    nodata_mask = ~np.isfinite(bathy)
    bathy_f = np.nan_to_num(bathy, nan=0.0)

    # ── Paper PB features (same as v6) ────────────────────────────────────
    depth = bathy.copy()
    bs    = back.copy()

    slope_deg = slope_mod.compute_slope(bathy_f, cell_size, nodata=None).astype(np.float32)
    slope_deg[nodata_mask] = np.nan

    vrm_arr = vrm_mod.compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size).astype(np.float32)
    vrm_arr[nodata_mask] = np.nan

    slope_rad = np.deg2rad(slope_deg)
    with np.errstate(invalid="ignore", divide="ignore"):
        complexity = (1.0 / np.cos(
            ndimage.uniform_filter(np.nan_to_num(slope_rad, nan=0.0), size=3)
        )).astype(np.float32)
    complexity[nodata_mask] = np.nan

    max_curvature = (np.abs(ndimage.laplace(bathy_f)) / cell_size ** 2).astype(np.float32)
    max_curvature[nodata_mask] = np.nan

    dy, dx   = np.gradient(bathy_f, cell_size)
    aspect   = np.arctan2(dy, dx)
    northness = np.cos(aspect).astype(np.float32)
    eastness  = np.sin(aspect).astype(np.float32)
    northness[nodata_mask] = np.nan
    eastness[nodata_mask]  = np.nan

    # ── Texture / TPI (new in v9, no spatial leakage) ─────────────────────
    bathy_std_9 = _focal_std(bathy, size=9)
    bathy_std_9[nodata_mask] = np.nan

    back_std_9 = _focal_std(back, size=9)
    back_std_9[nodata_mask] = np.nan

    tpi_9      = (bathy_f - ndimage.uniform_filter(bathy_f, size=9)).astype(np.float32)
    tpi_9[nodata_mask] = np.nan

    features = {
        "depth":         depth,
        "backscatter":   bs,
        "slope":         slope_deg,
        "vrm":           vrm_arr,
        "complexity":    complexity,
        "max_curvature": max_curvature,
        "northness":     northness,
        "eastness":      eastness,
        "bathy_std_9":   bathy_std_9,
        "back_std_9":    back_std_9,
        "tpi_9":         tpi_9,
    }
    for k, v in features.items():
        v[~np.isfinite(v)] = np.nan
        log.info("Feature '%s': nan_px=%d", k, int(np.isnan(v).sum()))
    return features


# ---------------------------------------------------------------------------
# Segmentation + OB statistics (identical to v6)
# ---------------------------------------------------------------------------

def _segment_rasters(
    bathy: np.ndarray,
    back: np.ndarray,
    vrm: np.ndarray,
) -> np.ndarray:
    def _mm(a):
        lo, hi = np.nanmin(a), np.nanmax(a)
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    stack = np.stack([_mm(bathy), _mm(back), _mm(vrm)], axis=-1).astype(np.float64)
    stack = np.nan_to_num(stack, nan=0.0)
    labels = slic(
        stack,
        n_segments=SLIC_N_SEGMENTS,
        compactness=SLIC_COMPACTNESS,
        enforce_connectivity=True,
        start_label=0,
    ).astype(np.int32)
    n_unique  = len(np.unique(labels))
    mean_size = labels.size / n_unique
    log.info(
        "Segmentation: %d labels, mean %.1f px (%.1f m²), target ~300 m²",
        n_unique, mean_size, mean_size * 0.25 ** 2,
    )
    return labels


def _compute_segment_stats(
    bathy: np.ndarray,
    back: np.ndarray,
    vrm: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Per-segment mean/std/skewness of bathy/back/VRM + pixel count (v6 identical)."""
    unique_labels = np.unique(labels)
    arrays = [("bathy", bathy), ("back", back), ("vrm", vrm)]

    global_skew = {n: float(scipy_skew(a[np.isfinite(a)])) if np.isfinite(a).any() else 0.0
                   for n, a in arrays}
    global_med  = {n: float(np.nanmedian(a)) for n, a in arrays}

    records = []
    for seg_id in unique_labels:
        mask = labels == seg_id
        row: dict = {}
        for name, arr in arrays:
            vals  = arr[mask]
            valid = vals[np.isfinite(vals)]
            n     = len(valid)
            gsk   = global_skew[name]
            gmd   = global_med[name]
            if n == 0:
                row[f"seg_{name}_mean"] = gmd;  row[f"seg_{name}_std"] = 0.0;  row[f"seg_{name}_skew"] = gsk
            elif n < 3:
                row[f"seg_{name}_mean"] = float(valid.mean());  row[f"seg_{name}_std"] = float(valid.std());  row[f"seg_{name}_skew"] = gsk
            else:
                row[f"seg_{name}_mean"] = float(valid.mean());  row[f"seg_{name}_std"] = float(valid.std());  row[f"seg_{name}_skew"] = float(scipy_skew(valid))
        row["seg_pixel_count"] = int(mask.sum())
        records.append((seg_id, row))

    df = pd.DataFrame([r for _, r in records], index=[s for s, _ in records], columns=OB_FEATURE_COLS)
    df.index.name = "segment_label"
    log.info("Segment stats: %d segs, NaN=%d", len(df), int(df.isna().sum().sum()))
    return df


# ---------------------------------------------------------------------------
# Point feature extraction (identical to v6)
# ---------------------------------------------------------------------------

def _extract_pb_features_at_points(
    feature_dict: dict[str, np.ndarray],
    xy_coords: np.ndarray,
    transform: rasterio.transform.Affine,
) -> pd.DataFrame:
    xs, ys = xy_coords[:, 0], xy_coords[:, 1]
    first  = next(iter(feature_dict.values()))
    h, w   = first.shape
    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(np.array(rows), 0, h - 1)
    cols = np.clip(np.array(cols), 0, w - 1)
    return pd.DataFrame(
        {key: feature_dict[key][rows, cols] for key in PB_FEATURE_COLS},
        columns=PB_FEATURE_COLS,
    )


def _assign_segment_features(
    point_df: pd.DataFrame,
    labels: np.ndarray,
    seg_stats_df: pd.DataFrame,
    transform: rasterio.transform.Affine,
    xy_coords: np.ndarray,
) -> pd.DataFrame:
    xs, ys = xy_coords[:, 0], xy_coords[:, 1]
    rows, cols = rowcol(transform, xs, ys)
    h, w = labels.shape
    rows = np.clip(np.array(rows), 0, h - 1)
    cols = np.clip(np.array(cols), 0, w - 1)
    seg_labels_at_pts = labels[rows, cols]
    col_medians = seg_stats_df.median(axis=0).to_dict()
    ob_rows = []
    for sl in seg_labels_at_pts:
        if sl in seg_stats_df.index:
            ob_rows.append(seg_stats_df.loc[sl, OB_FEATURE_COLS].tolist())
        else:
            ob_rows.append([col_medians[c] for c in OB_FEATURE_COLS])
    ob_df = pd.DataFrame(ob_rows, columns=OB_FEATURE_COLS, index=point_df.index)
    return pd.concat([point_df.reset_index(drop=True), ob_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Models (v6 architecture but with stronger CatBoost + LGB, plus ensemble)
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """CatBoost, LightGBM, RF and soft-vote CatBoost+LGB ensemble.

    CatBoost and LGB use v9-tuned params (stronger than v6).
    RF kept as a benchmark.
    Ensemble averages CatBoost + LGB probabilities.
    """
    return {
        "lgb": lambda: lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            class_weight="balanced",
            random_state=CV_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "cat": lambda: cb.CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=7,
            auto_class_weights="Balanced",
            random_seed=CV_RANDOM_STATE,
            verbose=0,
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=CV_RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# CV (same structure as v6's _run_cv_config)
# ---------------------------------------------------------------------------

def _run_cv_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    label: str,
) -> dict:
    """10-fold spatial GroupKFold CV; returns best single-model + soft-vote ensemble."""
    col_idx = [COMBINED_FEATURE_COLS.index(c) for c in feature_cols]
    X = X_train[:, col_idx]
    classes    = np.sort(np.unique(y_train))
    n_classes  = len(classes)
    models     = get_models()
    gkf        = GroupKFold(n_splits=CV_N_SPLITS)

    best_name, best_f1, best_oof, best_folds = None, -1.0, None, None

    # ── Individual models ──────────────────────────────────────────────────
    all_oofs = {}
    for name, make_model in models.items():
        oof = np.zeros((len(y_train), n_classes), dtype=np.float32)

        for tr_idx, va_idx in gkf.split(X, y_train, groups):
            m = make_model()
            m.fit(X[tr_idx], y_train[tr_idx])
            oof[va_idx] = m.predict_proba(X[va_idx])

        all_oofs[name] = oof
        preds = classes[np.argmax(oof, axis=1)]
        f1    = f1_score(y_train, preds, average="weighted")

        fold_scores = []
        for tr_idx, va_idx in gkf.split(X, y_train, groups):
            va_preds = classes[np.argmax(oof[va_idx], axis=1)]
            fold_scores.append(f1_score(y_train[va_idx], va_preds, average="weighted"))

        log.info(
            "  [%s] %s: CV F1=%.4f ± %.4f", label, name, f1, np.std(fold_scores)
        )

        if f1 > best_f1:
            best_f1, best_name, best_oof, best_folds = f1, name, oof, fold_scores

    # ── Soft-vote CatBoost + LGB ensemble ─────────────────────────────────
    if "cat" in all_oofs and "lgb" in all_oofs:
        oof_ens = (all_oofs["cat"] + all_oofs["lgb"]) / 2.0
        preds_ens = classes[np.argmax(oof_ens, axis=1)]
        f1_ens    = f1_score(y_train, preds_ens, average="weighted")
        fold_ens  = []
        for tr_idx, va_idx in gkf.split(X, y_train, groups):
            va_preds = classes[np.argmax(oof_ens[va_idx], axis=1)]
            fold_ens.append(f1_score(y_train[va_idx], va_preds, average="weighted"))
        log.info("  [%s] cat+lgb ensemble: CV F1=%.4f ± %.4f", label, f1_ens, np.std(fold_ens))
        if f1_ens > best_f1:
            best_f1, best_name, best_oof, best_folds = f1_ens, "cat+lgb_ensemble", oof_ens, fold_ens

    log.info("[%s] Best: %s (F1=%.4f)", label, best_name, best_f1)
    return {
        "label":          label,
        "best_model_name": best_name,
        "cv_f1_mean":     float(best_f1),
        "cv_f1_std":      float(np.std(best_folds)),
        "oof_proba":      best_oof,
        "fold_scores":    best_folds,
        "feature_cols":   feature_cols,
    }


# ---------------------------------------------------------------------------
# Spatial CV grouping (same as v6)
# ---------------------------------------------------------------------------

def _get_spatial_cv_groups(coords_xy: np.ndarray) -> np.ndarray:
    km = KMeans(n_clusters=CV_N_SPLITS, random_state=CV_RANDOM_STATE, n_init=10)
    return km.fit_predict(coords_xy).astype(int)


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------

def _write_run_report(
    config_results: list[dict],
    best_label: str,
    seg_params: dict,
    output_path: Path,
    n_features: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "---",
        f"date: {now}",
        "branch: 012-v9-catboost-lgb-texture",
        "script: scripts/experiment_v9.py",
        "v6_kaggle: 0.76394  (CatBoost combined 18 features, CV=0.5758)",
        "v8_kaggle: ~0.47    (RF pb-only, spatial memorisation failure)",
        "fix: returned to CatBoost+LGB on combined features; added 3 texture features",
        "---",
        "",
        "# Run Report: CatBoost+LGB + Texture Features (Experiment v9)",
        "",
        "## Changes vs v6",
        "",
        "| Added | Reason |",
        "|-------|--------|",
        "| bathy_std_9 | Bathymetric roughness — distinguishes rocky vs sediment substrate |",
        "| back_std_9  | Acoustic texture — local backscatter variance |",
        "| tpi_9       | Topographic position — ridge vs hollow vs slope |",
        "| CatBoost depth=7, iterations=800 | More capacity (was depth=6, iterations=400) |",
        "| LGB: lr=0.03, n_est=600 | Companion repo tuned params (was lr=0.05, n_est=400) |",
        "| Soft-vote CatBoost+LGB ensemble | New 4th candidate |",
        "",
        "| Removed | Reason |",
        "|---------|--------|",
        "| XGBoost | Slowest; CatBoost consistently outperformed it on this data |",
        "| x_rel, y_rel, z-scores | Caused spatial memorisation in v7 (0.47 Kaggle) |",
        "| SMOTE | Non-physical samples for spatially autocorrelated acoustic data |",
        "",
        f"Feature set: {n_features} total (11 PB + 10 OB), no spatial coordinates.",
        "",
        "## Segmentation",
        "",
        f"| n_segments target | {SLIC_N_SEGMENTS} |",
        f"| n_segments actual | {seg_params.get('n_unique','?')} |",
        f"| Mean segment size | {seg_params.get('mean_px',0):.1f} px = {seg_params.get('mean_m2',0):.1f} m² |",
        "",
        "## CV Results",
        "",
        f"10-fold spatial GroupKFold (KMeans, k={CV_N_SPLITS}, seed=42). Same as v6.",
        "",
        "| Configuration | Best Model | CV F1 mean ± std |",
        "|---------------|------------|-----------------|",
    ]
    for r in config_results:
        marker = " ← **BEST**" if r["label"] == best_label else ""
        lines.append(
            f"| {r['label']} | {r['best_model_name']} "
            f"| {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}{marker} |"
        )
    lines += ["", f"- Submission: `{OUTPUT_SUBMISSION}`", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Experiment v9: v6 + Texture + Stronger Boosting ===")

    # ── Rasters ──────────────────────────────────────────────────────────
    bathy, back, transform, cell_size = _load_rasters(BATHY_PATH, BACK_PATH)

    # ── PB features ──────────────────────────────────────────────────────
    log.info("Computing PB features (11)...")
    pb_dict = _compute_pb_features(bathy, back, cell_size)
    vrm_arr = pb_dict["vrm"]

    # ── Segmentation + OB stats ─────────────────────────────────────────
    log.info("SLIC segmentation (%d target segments)...", SLIC_N_SEGMENTS)
    labels = _segment_rasters(bathy, back, vrm_arr)
    n_uniq = len(np.unique(labels))
    mpx    = labels.size / n_uniq
    seg_params = {"n_unique": n_uniq, "mean_px": mpx, "mean_m2": mpx * cell_size ** 2}

    log.info("Computing segment statistics...")
    seg_stats = _compute_segment_stats(bathy, back, vrm_arr, labels)

    # ── Load CSVs ─────────────────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    log.info("Train: %d  Test: %d", len(train_df), len(test_df))

    x_col   = next(c for c in train_df.columns if c.lower() in ("x", "easting"))
    y_col   = next(c for c in train_df.columns if c.lower() in ("y", "northing"))
    lbl_col = next(c for c in train_df.columns if c.lower() in ("class", "label", "substrate"))

    train_xy = train_df[[x_col, y_col]].values.astype(np.float64)
    test_xy  = test_df[[x_col, y_col]].values.astype(np.float64)

    classes      = np.sort(train_df[lbl_col].unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_train      = train_df[lbl_col].map(class_to_int).values.astype(int)

    # ── Extract features ─────────────────────────────────────────────────
    log.info("Extracting features at training points...")
    train_feat = _assign_segment_features(
        _extract_pb_features_at_points(pb_dict, train_xy, transform),
        labels, seg_stats, transform, train_xy,
    )
    log.info("Extracting features at test points...")
    test_feat = _assign_segment_features(
        _extract_pb_features_at_points(pb_dict, test_xy, transform),
        labels, seg_stats, transform, test_xy,
    )

    # Median-impute any NaN (raster edge effects — same as v6)
    for col in COMBINED_FEATURE_COLS:
        med = float(train_feat[col].median()) if col in train_feat.columns else 0.0
        if not np.isfinite(med):
            med = 0.0
        train_feat[col] = train_feat[col].fillna(med)
        test_feat[col]  = test_feat[col].fillna(med)

    X_train = train_feat[COMBINED_FEATURE_COLS].values.astype(np.float32)
    X_test  = test_feat[COMBINED_FEATURE_COLS].values.astype(np.float32)
    log.info("Feature matrix: train %s  test %s", X_train.shape, X_test.shape)

    # Anti-leakage assertion
    for forbidden in ("x_rel", "y_rel"):
        assert forbidden not in COMBINED_FEATURE_COLS, f"Leaky spatial feature '{forbidden}' found!"

    # ── Spatial CV groups ─────────────────────────────────────────────────
    groups = _get_spatial_cv_groups(train_xy)
    log.info("Spatial CV groups: %s", np.bincount(groups).tolist())

    # ── CV: three configurations ──────────────────────────────────────────
    log.info("Running PB-only CV (11 features)...")
    pb_result = _run_cv_config(X_train, y_train, groups, PB_FEATURE_COLS, "pb_only")

    log.info("Running OB-only CV (10 features)...")
    ob_result = _run_cv_config(X_train, y_train, groups, OB_FEATURE_COLS, "ob_only")

    log.info("Running Combined CV (21 features)...")
    comb_result = _run_cv_config(X_train, y_train, groups, COMBINED_FEATURE_COLS, "combined")

    config_results = [pb_result, ob_result, comb_result]
    best_result    = max(config_results, key=lambda r: r["cv_f1_mean"])
    best_label     = best_result["label"]
    log.info("Best config: %s  F1=%.4f", best_label, best_result["cv_f1_mean"])

    # ── Final model: train on ALL training data, predict test ─────────────
    best_feat_cols = best_result["feature_cols"]
    best_feat_idx  = [COMBINED_FEATURE_COLS.index(c) for c in best_feat_cols]
    X_tr_best = X_train[:, best_feat_idx]
    X_te_best = X_test[:, best_feat_idx]
    best_model_name = best_result["best_model_name"]

    if best_model_name == "cat+lgb_ensemble":
        # Retrain both CatBoost and LGB on full training set, average probabilities
        models     = get_models()
        test_proba = np.zeros((len(X_te_best), len(classes)), dtype=np.float32)
        for m_name in ("cat", "lgb"):
            m = models[m_name]()
            m.fit(X_tr_best, y_train)
            test_proba += m.predict_proba(X_te_best)
        test_proba /= 2.0
    else:
        m = get_models()[best_model_name]()
        m.fit(X_tr_best, y_train)
        test_proba = m.predict_proba(X_te_best)

    test_pred_labels = classes[np.argmax(test_proba, axis=1).astype(int)]

    # ── Submission CSV ────────────────────────────────────────────────────
    id_col = next((c for c in test_df.columns if c.upper() == "ID"), None)
    submission = pd.DataFrame({
        "ID":    test_df[id_col].values if id_col else np.arange(1, len(test_pred_labels) + 1),
        "class": test_pred_labels,
    })
    OUTPUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_SUBMISSION, index=False)
    log.info("Submission: %s (%d rows)", OUTPUT_SUBMISSION, len(submission))
    log.info("Class distribution:\n%s", submission["class"].value_counts().to_string())

    # ── Write report ──────────────────────────────────────────────────────
    _write_run_report(config_results, best_label, seg_params, OUTPUT_REPORT, len(COMBINED_FEATURE_COLS))

    log.info("=== v9 complete ===")
    for r in config_results:
        log.info("  %-15s  F1=%.4f ± %.4f  model=%s", r["label"], r["cv_f1_mean"], r["cv_f1_std"], r["best_model_name"])


if __name__ == "__main__":
    main()
