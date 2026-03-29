"""
Experiment v8: Clean Acoustic+Terrain Features — KNN / RF / LightGBM
======================================================================

ROOT CAUSE OF v7 FAILURE
-------------------------
x_rel and y_rel (relative geographic coordinates) had 36.6 % and 11.8 %
feature importance in v7 — the model learned WHERE things are, not WHAT
they sound like acoustically. On the Kaggle test set the spatial pattern
does not transfer, so v7 scored the lowest of all 13 submissions.

v8 DESIGN PRINCIPLES
---------------------
1. NO spatial coordinates (x_rel, y_rel, z-scores of depth/backscatter).
2. NO interaction terms or multi-scale ratio features.
3. NO SMOTE — spatially autocorrelated data; SMOTE creates non-physical
   samples that violate the spatial structure.
4. Pure acoustic + terrain features that generalise across the survey area.
5. Three model candidates covering the spectrum:
     KNN  — acoustically similar neighbourhood lookup (distance-weighted,
             StandardScaler preprocessing)
     RF   — what the paper used; handles class imbalance via balanced weights
     LGB  — state-of-art boosted trees for tabular data
6. Soft-voting ensemble of all three as a fourth candidate.
7. 10-fold spatial GroupKFold CV — identical to v6, directly comparable.

FEATURE SET (21 total — no spatial leakage)
--------------------------------------------
PB paper (8):    depth, backscatter, slope, vrm, complexity,
                 max_curvature, northness, eastness
PB texture (3):  bathy_std_9 (local roughness at 2.25 m scale),
                 back_std_9 (acoustic texture at 2.25 m scale),
                 tpi_9 (topographic position: am I on a ridge/valley?).
OB (10):         seg_{bathy,back,vrm}_{mean,std,skew} + seg_pixel_count
                 (SLIC segments on [bathy, backscatter, VRM] — same as v6).

KNN RATIONALE
-------------
KNN in normalised acoustic/terrain feature space is a natural fit:
- Two points with identical depth, backscatter and rugosity should be the
  same habitat — KNN exploits that directly.
- No risk of spatial overfitting; the model knows nothing about location.
- Interpretable: predictions are driven by the k most acoustically similar
  training points.

TARGET
------
Kaggle public test F1 > 0.764 (current best hybrid = 0.76394, v6).

USAGE
-----
    python scripts/experiment_v8.py
"""

import logging
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy import ndimage
from scipy.stats import skew as scipy_skew
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATHY_PATH = Path("data/MBES/bathymetry.tif")
BACK_PATH = Path("data/MBES/backscatter.tif")
TRAIN_CSV = Path("data/train.csv")
TEST_CSV = Path("data/test.csv")

OUTPUT_SUBMISSION = Path("data/submission_v8_best.csv")
OUTPUT_REPORT = Path("docs/run-011-clean-knn-rf-lgb.md")

# Paper PB features (8 from Ierodiaconou 2018) + texture/TPI (3 new, no leakage)
PB_FEATURE_COLS = [
    "depth",
    "backscatter",
    "slope",
    "vrm",
    "complexity",
    "max_curvature",
    "northness",
    "eastness",
    # -- texture / TPI (new in v8, not in v6; no spatial coords) --
    "bathy_std_9",
    "back_std_9",
    "tpi_9",
]

# Object-Based features from SLIC segmentation (same as v6, 10 features)
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

ALL_FEATURE_COLS = PB_FEATURE_COLS + OB_FEATURE_COLS  # 21, no spatial coords

# SLIC segmentation parameters (same as v6)
SLIC_N_SEGMENTS = 4000
SLIC_COMPACTNESS = 0.01

# Spatial CV (10-fold, matching v6 for direct comparability)
CV_N_SPLITS = 10
CV_N_CLUSTERS = 10
CV_RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raster loading
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
        "Rasters: bathy %s (%.0f%% valid) back %s (%.0f%% valid) cell=%.4f m",
        bathy.shape,
        100 * np.isfinite(bathy).mean(),
        back.shape,
        100 * np.isfinite(back).mean(),
        cell_size,
    )
    return bathy, back, transform, cell_size


# ---------------------------------------------------------------------------
# Pixel-based feature computation
# ---------------------------------------------------------------------------


def _focal_std(arr: np.ndarray, size: int) -> np.ndarray:
    """Local std in a square window via E[X²] – E[X]² trick."""
    a = np.nan_to_num(arr, nan=0.0).astype(np.float64)
    m1 = ndimage.uniform_filter(a, size=size)
    m2 = ndimage.uniform_filter(a**2, size=size)
    return np.sqrt(np.maximum(m2 - m1**2, 0.0)).astype(np.float32)


def _compute_pb_features(
    bathy: np.ndarray,
    back: np.ndarray,
    cell_size: float,
) -> dict[str, np.ndarray]:
    """Compute all 11 pixel-based features; NaN at NoData pixels."""
    from btm.core import slope as slope_mod
    from btm.core import vrm as vrm_mod

    nodata_mask = ~np.isfinite(bathy)
    bathy_f = np.nan_to_num(bathy, nan=0.0)

    # ── Paper PB features ────────────────────────────────────────────────
    depth = bathy.copy()
    bs = back.copy()

    slope_deg = slope_mod.compute_slope(bathy_f, cell_size, nodata=None).astype(
        np.float32
    )
    slope_deg[nodata_mask] = np.nan

    vrm_arr = vrm_mod.compute_vrm(
        bathy_f, neighborhood_size=3, cell_size=cell_size
    ).astype(np.float32)
    vrm_arr[nodata_mask] = np.nan

    slope_rad = np.deg2rad(slope_deg)
    with np.errstate(invalid="ignore", divide="ignore"):
        complexity = (
            1.0
            / np.cos(ndimage.uniform_filter(np.nan_to_num(slope_rad, nan=0.0), size=3))
        ).astype(np.float32)
    complexity[nodata_mask] = np.nan

    max_curvature = (np.abs(ndimage.laplace(bathy_f)) / cell_size**2).astype(np.float32)
    max_curvature[nodata_mask] = np.nan

    dy, dx = np.gradient(bathy_f, cell_size)
    aspect = np.arctan2(dy, dx)
    northness = np.cos(aspect).astype(np.float32)
    eastness = np.sin(aspect).astype(np.float32)
    northness[nodata_mask] = np.nan
    eastness[nodata_mask] = np.nan

    # ── Texture / TPI (new, no spatial coords) ───────────────────────────
    bathy_std_9 = _focal_std(bathy, size=9)
    bathy_std_9[nodata_mask] = np.nan

    back_std_9 = _focal_std(back, size=9)
    back_std_9[nodata_mask] = np.nan

    bathy_mean_9 = ndimage.uniform_filter(bathy_f, size=9).astype(np.float32)
    tpi_9 = (bathy_f - bathy_mean_9).astype(np.float32)
    tpi_9[nodata_mask] = np.nan

    features = {
        "depth": depth,
        "backscatter": bs,
        "slope": slope_deg,
        "vrm": vrm_arr,
        "complexity": complexity,
        "max_curvature": max_curvature,
        "northness": northness,
        "eastness": eastness,
        "bathy_std_9": bathy_std_9,
        "back_std_9": back_std_9,
        "tpi_9": tpi_9,
    }
    for k, v in features.items():
        v[~np.isfinite(v)] = np.nan  # guard infs
    return features


# ---------------------------------------------------------------------------
# Segmentation + OB statistics (same as v6)
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
    n_unique = len(np.unique(labels))
    log.info("Segmentation: %d labels, mean %.1f px", n_unique, labels.size / n_unique)
    return labels


def _compute_segment_stats(
    bathy: np.ndarray,
    back: np.ndarray,
    vrm: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    unique_labels = np.unique(labels)
    arrays = [("bathy", bathy), ("back", back), ("vrm", vrm)]

    global_skew = {
        n: float(scipy_skew(a[np.isfinite(a)]))
        for n, a in arrays
        if np.isfinite(a).any()
    }
    global_med = {n: float(np.nanmedian(a)) for n, a in arrays}

    records = []
    for seg_id in unique_labels:
        mask = labels == seg_id
        row: dict = {}
        for name, arr in arrays:
            vals = arr[mask]
            valid = vals[np.isfinite(vals)]
            n = len(valid)
            gsk = global_skew.get(name, 0.0)
            gmd = global_med.get(name, 0.0)
            if n == 0:
                row[f"seg_{name}_mean"] = gmd
                row[f"seg_{name}_std"] = 0.0
                row[f"seg_{name}_skew"] = gsk
            elif n < 3:
                row[f"seg_{name}_mean"] = float(valid.mean())
                row[f"seg_{name}_std"] = float(valid.std())
                row[f"seg_{name}_skew"] = gsk
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
    log.info("Segment stats: %d segs, NaN=%d", len(df), int(df.isna().sum().sum()))
    return df


# ---------------------------------------------------------------------------
# Point-level feature extraction
# ---------------------------------------------------------------------------


def _extract_features_at_points(
    pb_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    seg_stats: pd.DataFrame,
    xy_coords: np.ndarray,
    transform: rasterio.transform.Affine,
) -> pd.DataFrame:
    """Sample all 21 features at (x, y) geographic coordinates."""
    xs, ys = xy_coords[:, 0], xy_coords[:, 1]
    first = next(iter(pb_dict.values()))
    h, w = first.shape
    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(np.array(rows), 0, h - 1)
    cols = np.clip(np.array(cols), 0, w - 1)

    # PB features
    pb_df = pd.DataFrame(
        {key: pb_dict[key][rows, cols] for key in PB_FEATURE_COLS},
        columns=PB_FEATURE_COLS,
    )

    # OB features via segment lookup
    seg_labels_at_pts = labels[rows, cols]
    col_medians = seg_stats.median(axis=0).to_dict()
    ob_rows = []
    for sl in seg_labels_at_pts:
        if sl in seg_stats.index:
            ob_rows.append(seg_stats.loc[sl, OB_FEATURE_COLS].tolist())
        else:
            ob_rows.append([col_medians[c] for c in OB_FEATURE_COLS])
    ob_df = pd.DataFrame(ob_rows, columns=OB_FEATURE_COLS)

    return pd.concat([pb_df, ob_df], axis=1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _get_models(classes: np.ndarray) -> dict:
    """Return individual model factories and a soft-voting ensemble."""
    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1)),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=CV_RANDOM_STATE,
        n_jobs=-1,
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        min_child_samples=10,
        random_state=CV_RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    return {"knn": knn, "rf": rf, "lgb": lgb_clf}


# ---------------------------------------------------------------------------
# Spatial CV
# ---------------------------------------------------------------------------


def _spatial_groups(xy: np.ndarray) -> np.ndarray:
    km = KMeans(n_clusters=CV_N_CLUSTERS, random_state=CV_RANDOM_STATE, n_init=10)
    return km.fit_predict(xy).astype(int)


def _run_spatial_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    feature_cols: list[str],
    config_label: str,
) -> dict:
    """10-fold spatial GroupKFold; returns best model name + CV F1."""
    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    models = _get_models(classes)

    best_name, best_f1, best_oof, best_folds = None, -1.0, None, None

    for name, m in models.items():
        import copy

        n_cls = len(classes)
        oof = np.zeros((len(y), n_cls), dtype=np.float32)
        fold_f1 = []

        for tr_idx, va_idx in gkf.split(X, y, groups):
            clf = copy.deepcopy(m)
            clf.fit(X[tr_idx], y[tr_idx])
            proba = clf.predict_proba(X[va_idx])
            # Ensure column order matches classes
            if hasattr(clf, "classes_"):
                col_order = [np.where(clf.classes_ == c)[0][0] for c in range(n_cls)]
                proba = proba[:, col_order]
            elif hasattr(clf, "named_steps"):
                inner = clf.named_steps["clf"]
                if hasattr(inner, "classes_"):
                    col_order = [
                        np.where(inner.classes_ == c)[0][0] for c in range(n_cls)
                    ]
                    proba = proba[:, col_order]
            oof[va_idx] = proba

        oof_preds = np.argmax(oof, axis=1)
        y_str = classes[y]
        oof_str = classes[oof_preds]
        f1 = f1_score(y_str, oof_str, labels=classes, average="weighted")

        for tr_idx, va_idx in gkf.split(X, y, groups):
            va_preds = classes[np.argmax(oof[va_idx], axis=1)]
            fold_f1.append(
                f1_score(
                    classes[y[va_idx]], va_preds, labels=classes, average="weighted"
                )
            )

        log.info(
            "  [%s] %s: F1=%.4f ± %.4f",
            config_label,
            name,
            f1,
            np.std(fold_f1),
        )

        if f1 > best_f1:
            best_f1, best_name, best_oof, best_folds = f1, name, oof, fold_f1

    log.info("[%s] Best: %s F1=%.4f", config_label, best_name, best_f1)
    return {
        "label": config_label,
        "best_model": best_name,
        "cv_f1_mean": float(best_f1),
        "cv_f1_std": float(np.std(best_folds)),
        "oof_proba": best_oof,
        "best_factory": _get_models,  # re-construct when needed
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Soft-voting ensemble
# ---------------------------------------------------------------------------


def _run_ensemble_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    feature_cols: list[str],
) -> dict:
    """Average OOF probabilities from all three models."""
    import copy

    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    models = _get_models(classes)
    n_cls = len(classes)

    oof_sum = np.zeros((len(y), n_cls), dtype=np.float32)
    fold_f1 = []

    for tr_idx, va_idx in gkf.split(X, y, groups):
        fold_proba = np.zeros((len(va_idx), n_cls), dtype=np.float32)
        for name, m in models.items():
            clf = copy.deepcopy(m)
            clf.fit(X[tr_idx], y[tr_idx])
            proba = clf.predict_proba(X[va_idx])
            fold_proba += proba
        oof_sum[va_idx] = fold_proba / len(models)

    oof_preds = np.argmax(oof_sum, axis=1)
    f1 = f1_score(classes[y], classes[oof_preds], labels=classes, average="weighted")

    for tr_idx, va_idx in gkf.split(X, y, groups):
        va_preds = classes[np.argmax(oof_sum[va_idx], axis=1)]
        fold_f1.append(
            f1_score(classes[y[va_idx]], va_preds, labels=classes, average="weighted")
        )

    log.info("  [combined] ensemble (soft vote): F1=%.4f ± %.4f", f1, np.std(fold_f1))
    return {
        "label": "combined_ensemble",
        "best_model": "ensemble",
        "cv_f1_mean": float(f1),
        "cv_f1_std": float(np.std(fold_f1)),
        "oof_proba": oof_sum,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------


def _write_report(
    results: list[dict],
    best_label: str,
    seg_params: dict,
    n_features: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "---",
        f"date: {now}",
        "branch: 011-clean-knn-rf-lgb",
        "script: scripts/experiment_v8.py",
        "builds_on: experiment_v6.py (OBIA+PB hybrid, Kaggle 0.76394)",
        "root_cause_of_v7_failure: x_rel/y_rel geographic coords (36.6%+11.8% importance) caused spatial memorisation",
        "---",
        "",
        "# Run Report: Clean Features — KNN / RF / LightGBM (Experiment v8)",
        "",
        "## What Changed vs v7",
        "",
        "| Removed from v7 | Reason |",
        "|-----------------|--------|",
        "| x_rel, y_rel (spatial coords) | Caused spatial memorisation — LOWEST Kaggle score |",
        "| depth_z, backscatter_z (z-scores) | Redundant duplicates of depth + backscatter |",
        "| SMOTE | Creates non-physical samples for spatially autocorrelated data |",
        "| 4 interaction terms | Collinear with inputs; added noise |",
        "| 6 multi-scale focal stds | Replaced by single best-scale bathy_std_9, back_std_9 |",
        "| XGBoost, CatBoost | Replaced by KNN (new) to explore acoustic similarity |",
        "",
        "## Feature Set",
        "",
        f"| Group | Count | Features |",
        f"|-------|-------|----------|",
        f"| PB paper (Ierodiaconou 2018) | 8 | depth, backscatter, slope, vrm, complexity, max_curvature, northness, eastness |",
        f"| PB texture/TPI (new) | 3 | bathy_std_9, back_std_9, tpi_9 |",
        f"| OB SLIC segments (v6) | 10 | seg_{{bathy,back,vrm}}_{{mean,std,skew}}, seg_pixel_count |",
        f"| **Total** | **{n_features}** | no spatial coordinates |",
        "",
        "## Segmentation",
        "",
        f"| n_segments (target) | {SLIC_N_SEGMENTS} |",
        f"| n_segments (actual) | {seg_params.get('n_unique', '?')} |",
        f"| Mean size (px) | {seg_params.get('mean_px', 0):.1f} |",
        f"| Mean size (m²) | {seg_params.get('mean_m2', 0):.1f} |",
        "",
        "## CV Results",
        "",
        f"Spatial GroupKFold (n_splits={CV_N_SPLITS}, KMeans n_clusters={CV_N_CLUSTERS}, seed=42).",
        "No SMOTE. StandardScaler inside KNN pipeline; raw features for RF/LGB.",
        "",
        "| Configuration | Model | CV F1 mean ± std |",
        "|---------------|-------|-----------------|",
    ]
    for r in results:
        marker = " ← **BEST**" if r["label"] == best_label else ""
        lines.append(
            f"| {r['label']} | {r['best_model']} "
            f"| {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}{marker} |"
        )
    lines += [
        "",
        "## Output",
        "",
        f"- Submission: `{OUTPUT_SUBMISSION}`",
        f"- Report: `{OUTPUT_REPORT}`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("=== Experiment v8: Clean Features + KNN/RF/LGB ===")

    # ── Rasters ──────────────────────────────────────────────────────────
    bathy, back, transform, cell_size = _load_rasters(BATHY_PATH, BACK_PATH)

    # ── Pixel features ───────────────────────────────────────────────────
    log.info("Computing PB features (11)...")
    pb_dict = _compute_pb_features(bathy, back, cell_size)
    vrm_arr = pb_dict["vrm"]

    # ── Segmentation + OB stats ─────────────────────────────────────────
    log.info("Segmenting (%d target segs)...", SLIC_N_SEGMENTS)
    labels = _segment_rasters(bathy, back, vrm_arr)
    n_unique = len(np.unique(labels))
    mean_px = labels.size / n_unique
    seg_params = {
        "n_unique": n_unique,
        "mean_px": mean_px,
        "mean_m2": mean_px * cell_size**2,
    }

    log.info("Computing segment statistics...")
    seg_stats = _compute_segment_stats(bathy, back, vrm_arr, labels)

    # ── Load CSVs ─────────────────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    log.info("Train: %d rows  Test: %d rows", len(train_df), len(test_df))

    x_col = next(c for c in train_df.columns if c.lower() in ("x", "easting"))
    y_col = next(c for c in train_df.columns if c.lower() in ("y", "northing"))
    lbl_col = next(
        c for c in train_df.columns if c.lower() in ("class", "label", "substrate")
    )

    train_xy = train_df[[x_col, y_col]].values.astype(np.float64)
    test_xy = test_df[[x_col, y_col]].values.astype(np.float64)

    classes = np.sort(train_df[lbl_col].unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_train = train_df[lbl_col].map(class_to_int).values.astype(int)

    # ── Feature extraction ───────────────────────────────────────────────
    log.info("Extracting features at training points...")
    train_feat = _extract_features_at_points(
        pb_dict, labels, seg_stats, train_xy, transform
    )
    log.info("Extracting features at test points...")
    test_feat = _extract_features_at_points(
        pb_dict, labels, seg_stats, test_xy, transform
    )

    # Median-impute any remaining NaN (raster edge effects)
    for col in ALL_FEATURE_COLS:
        med = float(train_feat[col].median()) if col in train_feat.columns else 0.0
        if not np.isfinite(med):
            med = 0.0
        train_feat[col] = train_feat[col].fillna(med)
        test_feat[col] = test_feat[col].fillna(med)

    X_train = train_feat[ALL_FEATURE_COLS].values.astype(np.float32)
    X_test = test_feat[ALL_FEATURE_COLS].values.astype(np.float32)
    log.info("Feature matrix: train %s  test %s", X_train.shape, X_test.shape)

    # Anti-leakage assertion — no spatial coords in features
    for forbidden in ("x_rel", "y_rel", "depth_z", "backscatter_z"):
        assert (
            forbidden not in ALL_FEATURE_COLS
        ), f"LEAKY FEATURE {forbidden} in ALL_FEATURE_COLS!"

    # ── Spatial CV groups ─────────────────────────────────────────────────
    groups = _spatial_groups(train_xy)
    log.info("CV groups: %s", np.bincount(groups).tolist())

    # ── CV: individual models + ensemble ─────────────────────────────────
    log.info("Running PB-only CV (11 features)...")
    pb_idx = [ALL_FEATURE_COLS.index(c) for c in PB_FEATURE_COLS]
    pb_result = _run_spatial_cv(
        X_train[:, pb_idx], y_train, groups, classes, PB_FEATURE_COLS, "pb_only"
    )

    log.info("Running OB-only CV (10 features)...")
    ob_idx = [ALL_FEATURE_COLS.index(c) for c in OB_FEATURE_COLS]
    ob_result = _run_spatial_cv(
        X_train[:, ob_idx], y_train, groups, classes, OB_FEATURE_COLS, "ob_only"
    )

    log.info("Running combined CV (21 features) — individual models...")
    comb_result = _run_spatial_cv(
        X_train, y_train, groups, classes, ALL_FEATURE_COLS, "combined_best_single"
    )

    log.info("Running combined CV (21 features) — soft-vote ensemble...")
    ens_result = _run_ensemble_cv(X_train, y_train, groups, classes, ALL_FEATURE_COLS)

    all_results = [pb_result, ob_result, comb_result, ens_result]
    best_result = max(all_results, key=lambda r: r["cv_f1_mean"])
    best_label = best_result["label"]
    log.info("Best CV config: %s  F1=%.4f", best_label, best_result["cv_f1_mean"])

    # ── Final model: train on full dataset, predict test ─────────────────
    import copy

    best_feat_cols = best_result["feature_cols"]
    best_feat_idx = [ALL_FEATURE_COLS.index(c) for c in best_feat_cols]
    X_tr_best = X_train[:, best_feat_idx]
    X_te_best = X_test[:, best_feat_idx]

    best_model_name = best_result["best_model"]
    if best_model_name == "ensemble":
        # Retrain all three and average probabilities
        models = _get_models(classes)
        test_proba = np.zeros((len(X_te_best), len(classes)), dtype=np.float32)
        for name, m in models.items():
            clf = copy.deepcopy(m)
            clf.fit(X_tr_best, y_train)
            test_proba += clf.predict_proba(X_te_best)
        test_proba /= len(models)
    else:
        models = _get_models(classes)
        final_clf = copy.deepcopy(models[best_model_name])
        final_clf.fit(X_tr_best, y_train)
        test_proba = final_clf.predict_proba(X_te_best)

    test_pred_int = np.argmax(test_proba, axis=1).astype(int)
    test_pred_labels = classes[test_pred_int]

    # ── Submission CSV ────────────────────────────────────────────────────
    id_col = next((c for c in test_df.columns if c.upper() == "ID"), None)
    submission = pd.DataFrame(
        {
            "ID": (
                test_df[id_col].values
                if id_col
                else np.arange(1, len(test_pred_labels) + 1)
            ),
            "class": test_pred_labels,
        }
    )
    OUTPUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_SUBMISSION, index=False)
    log.info("Submission: %s (%d rows)", OUTPUT_SUBMISSION, len(submission))
    log.info(
        "Predicted class dist:\n%s", submission["class"].value_counts().to_string()
    )

    # ── Report ────────────────────────────────────────────────────────────
    _write_report(
        all_results, best_label, seg_params, len(ALL_FEATURE_COLS), OUTPUT_REPORT
    )

    log.info("=== v8 complete ===")
    for r in all_results:
        log.info("  %-30s F1=%.4f ± %.4f", r["label"], r["cv_f1_mean"], r["cv_f1_std"])


if __name__ == "__main__":
    main()
