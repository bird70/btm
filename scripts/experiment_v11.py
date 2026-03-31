"""
Experiment v11: Combined Multi-Scale BTM + MBES-8 Features
===========================================================

HYPOTHESIS
----------
Run-016 multi-scale BTM features (CV F1=0.6377) fell short of R04/R06
baseline (0.8024) due to two identified causes:
  Cause 1 (~14 pp): missing MBES-8 point-sample features
  Cause 2 (~0.6 pp): CatBoost instead of Random Forest
Combining the 33 selected BTM features with 8 MBES point-sample features
and adding RF as a third model should close both gaps.

v11 DESIGN PRINCIPLES
---------------------
1. MBES-8 extracted directly from bathymetry/backscatter rasters (v9 pattern).
2. BTM-33 loaded from spec-016 CSV cache (no re-extraction).
3. Three models: RF (R04 params), CatBoost (v10 params), LightGBM (v10 params).
4. 5-fold spatial-blocked CV matching R04/R06 via iter_spatial_blocked_folds.
5. Feature selection: Spearman corr pre-filter + permutation importance.
6. SGAM minority class monitoring (SC-003).

FEATURE SET
-----------
MBES-8:  depth, backscatter, slope, vrm, complexity, max_curvature,
         northness, eastness
BTM-33:  33 selected multi-scale terrain + GLCM features from spec-016
Total:   ~41 combined features before selection

BASELINES
---------
Kaggle: 0.79518  (best: R04/R06 RF)
CV F1:  0.8024   (R04/R06 RF, 5-fold spatial-blocked)

SUCCESS CRITERIA (spec-017)
---------------------------
SC-001: RF CV weighted-F1 >= 0.8024 (same scheme as R04/R06)
SC-002: Best model Kaggle F1 >= 0.79518 OR CV improvement >= 0.005 over RF
SC-003: SGAM recall maintained or improved vs R04/R06 (~0.045)
SC-004: Feature selection to <= 25 features, CV degradation <= 0.01
SC-005: Extraction time < 15 minutes for train + test

USAGE
-----
    python scripts/experiment_v11.py                   # full run
    python scripts/experiment_v11.py --extract-only    # extraction timing only
    python scripts/experiment_v11.py --dry-run         # 5 points, rapid test
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATHY_PATH = Path("data/MBES/bathymetry.tif")
BACK_PATH = Path("data/MBES/backscatter.tif")
TRAIN_CSV = Path("data/train.csv")
TEST_CSV = Path("data/test.csv")
SAMPLE_SUBMISSION = Path("data/sample_submission.csv")

# Spec-016 caches
CACHE_TRAIN_V10 = Path("reports/metrics/cache_train_feats_v10.csv")
CACHE_TEST_V10 = Path("reports/metrics/cache_test_feats_v10.csv")
FEATURE_SELECTION_V10 = Path("reports/metrics/feature_selection_v10.csv")

# v11 outputs
OUTPUT_SUBMISSION = Path("data/submission_v11.csv")
OUTPUT_SUBMISSION_SELECTED = Path("data/submission_v11_selected.csv")
OUTPUT_IMPORTANCE = Path("reports/metrics/feature_importance_v11.csv")
OUTPUT_SELECTION = Path("reports/metrics/feature_selection_v11.csv")
OUTPUT_PER_CLASS = Path("reports/metrics/cv_per_class_v11.csv")
CACHE_COMBINED_TRAIN = Path("reports/metrics/cache_combined_v11.csv")
CACHE_COMBINED_TEST = Path("reports/metrics/cache_combined_test_v11.csv")

CV_N_SPLITS = 5
CV_SPATIAL_BINS = 4
CV_RANDOM_STATE = 42
BASELINE_CV_F1 = 0.8024
BASELINE_KAGGLE = 0.79518
BASELINE_SGAM_RECALL = 0.045
KNOWN_CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]

MBES8_COLUMNS = [
    "depth",
    "backscatter",
    "slope",
    "vrm",
    "complexity",
    "max_curvature",
    "northness",
    "eastness",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers (tested in tests/unit/test_experiment_v11.py)
# ---------------------------------------------------------------------------


def _write_feature_importance(
    importances_mean: list[float],
    importances_std: list[float],
    feature_names: list[str],
    path: str,
) -> None:
    """Write feature importance CSV (partial FR-009).

    Columns: feature_name, importance_mean, importance_std, rank.
    The ``selected`` column is added in Phase 4 after feature selection.
    """
    df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance_mean": importances_mean,
            "importance_std": importances_std,
        }
    )
    df["rank"] = (
        df["importance_mean"].rank(ascending=False, na_option="bottom").astype(int)
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Feature importance written: %s", out)


def _write_submission(predictions: pd.DataFrame, path: str) -> None:
    """Write a Kaggle submission CSV (FR-010).

    ``predictions`` must have columns matching ``data/sample_submission.csv``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    log.info("Submission written: %s (%d rows)", out, len(predictions))


# ---------------------------------------------------------------------------
# Raster loading (from experiment_v9.py pattern)
# ---------------------------------------------------------------------------


def _load_rasters(
    bathy_path: Path,
    back_path: Path,
):
    """Open bathymetry and backscatter GeoTIFFs; return arrays + metadata."""
    import rasterio

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
        bathy.shape,
        100 * np.isfinite(bathy).mean(),
        back.shape,
        cell_size,
    )
    return bathy, back, transform, cell_size


# ---------------------------------------------------------------------------
# MBES-8 feature extraction (FR-001, EC-1)
# ---------------------------------------------------------------------------


def _extract_mbes8_features(
    bathy: np.ndarray,
    back: np.ndarray,
    cell_size: float,
    xy_coords: np.ndarray,
    transform=None,
) -> pd.DataFrame:
    """Compute 8 MBES point-sample features from raster arrays.

    Returns DataFrame with columns: depth, backscatter, slope, vrm,
    complexity, max_curvature, northness, eastness.

    NaN values (from out-of-bounds or nodata pixels) are filled with
    column median before returning (EC-1).
    """
    from btm.core import slope as slope_mod
    from btm.core import vrm as vrm_mod

    nodata_mask = ~np.isfinite(bathy)
    bathy_f = np.nan_to_num(bathy, nan=0.0)

    # Slope (Horn 1981)
    slope_deg = slope_mod.compute_slope(bathy_f, cell_size, nodata=None).astype(
        np.float32
    )
    slope_deg[nodata_mask] = np.nan

    # VRM (Sappington 2007)
    vrm_arr = vrm_mod.compute_vrm(
        bathy_f, neighborhood_size=3, cell_size=cell_size
    ).astype(np.float32)
    vrm_arr[nodata_mask] = np.nan

    # Complexity (Wilson 2007)
    slope_rad = np.deg2rad(slope_deg)
    with np.errstate(invalid="ignore", divide="ignore"):
        complexity = (
            1.0
            / np.cos(ndimage.uniform_filter(np.nan_to_num(slope_rad, nan=0.0), size=3))
        ).astype(np.float32)
    complexity[nodata_mask] = np.nan

    # Max curvature (Evans 1980)
    max_curvature = (np.abs(ndimage.laplace(bathy_f)) / cell_size**2).astype(np.float32)
    max_curvature[nodata_mask] = np.nan

    # Aspect-derived: northness, eastness (Horn 1981)
    dy, dx = np.gradient(bathy_f, cell_size)
    aspect = np.arctan2(dy, dx)
    northness = np.cos(aspect).astype(np.float32)
    eastness = np.sin(aspect).astype(np.float32)
    northness[nodata_mask] = np.nan
    eastness[nodata_mask] = np.nan

    # Build feature raster dict
    feature_rasters = {
        "depth": bathy,
        "backscatter": back,
        "slope": slope_deg,
        "vrm": vrm_arr,
        "complexity": complexity,
        "max_curvature": max_curvature,
        "northness": northness,
        "eastness": eastness,
    }

    # Sample at point locations
    if transform is not None:
        import rasterio

        rows, cols = rasterio.transform.rowcol(
            transform, xy_coords[:, 0], xy_coords[:, 1]
        )
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
    else:
        # For unit tests: treat xy_coords as (row, col) directly
        rows = xy_coords[:, 1].astype(int)
        cols = xy_coords[:, 0].astype(int)

    h, w = bathy.shape
    records = []
    for i in range(len(rows)):
        r, c = rows[i], cols[i]
        row_dict = {}
        for fname, raster in feature_rasters.items():
            if 0 <= r < h and 0 <= c < w:
                row_dict[fname] = float(raster[r, c])
            else:
                row_dict[fname] = np.nan
        records.append(row_dict)

    df = pd.DataFrame(records, columns=MBES8_COLUMNS)

    # EC-1: fill NaN with column median
    for col in df.columns:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

    return df


# ---------------------------------------------------------------------------
# BTM-33 loading (FR-002)
# ---------------------------------------------------------------------------


def _load_btm33_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load spec-016 cached features filtered to the 33 selected columns."""
    train_cache = pd.read_csv(CACHE_TRAIN_V10)
    test_cache = pd.read_csv(CACHE_TEST_V10)
    sel_df = pd.read_csv(FEATURE_SELECTION_V10)

    selected_names = sel_df.loc[
        sel_df["selected"] == True, "feature_name"
    ].tolist()  # noqa: E712
    log.info("BTM-33 selected features: %d columns", len(selected_names))

    # Keep ID column if present for merging
    keep_cols = [c for c in selected_names if c in train_cache.columns]
    train_btm = train_cache[keep_cols].copy()
    test_btm = test_cache[keep_cols].copy()

    return train_btm, test_btm, keep_cols


# ---------------------------------------------------------------------------
# Feature merge (FR-003)
# ---------------------------------------------------------------------------


def _merge_features(
    mbes_df: pd.DataFrame,
    btm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge MBES-8 and BTM-33 DataFrames by positional alignment.

    Both DataFrames must be aligned to the same point ordering.
    """
    assert len(mbes_df) == len(
        btm_df
    ), f"Length mismatch: MBES={len(mbes_df)}, BTM={len(btm_df)}"

    # Reset indices to ensure positional alignment
    mbes_reset = mbes_df.reset_index(drop=True)
    btm_reset = btm_df.reset_index(drop=True)

    combined = pd.concat([mbes_reset, btm_reset], axis=1)

    # Check for duplicate column names
    dupes = combined.columns[combined.columns.duplicated()].tolist()
    if dupes:
        log.warning("Duplicate columns detected: %s — dropping duplicates", dupes)
        combined = combined.loc[:, ~combined.columns.duplicated()]

    log.info("Combined feature matrix: %d columns", len(combined.columns))
    return combined


# ---------------------------------------------------------------------------
# Extract and cache (FR-011, FR-012)
# ---------------------------------------------------------------------------


def _extract_and_cache(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Coordinate MBES-8 extraction + BTM-33 loading with CSV caching."""
    # Check cache first
    if CACHE_COMBINED_TRAIN.exists() and CACHE_COMBINED_TEST.exists():
        log.info("Loading combined feature cache from %s", CACHE_COMBINED_TRAIN)
        train_combined = pd.read_csv(CACHE_COMBINED_TRAIN)
        test_combined = pd.read_csv(CACHE_COMBINED_TEST)
        feature_cols = [
            c
            for c in train_combined.columns
            if c not in ("ID", "class", "x", "y", "label")
        ]
        return train_combined, test_combined, feature_cols

    t0 = time.time()

    # Load rasters
    bathy, back, transform, cell_size = _load_rasters(BATHY_PATH, BACK_PATH)

    # MBES-8 for train
    xy_train = train_raw[["x", "y"]].values
    mbes_train = _extract_mbes8_features(bathy, back, cell_size, xy_train, transform)
    log.info("MBES-8 train: %d rows × %d cols", *mbes_train.shape)

    # MBES-8 for test
    xy_test = test_raw[["x", "y"]].values
    mbes_test = _extract_mbes8_features(bathy, back, cell_size, xy_test, transform)
    log.info("MBES-8 test: %d rows × %d cols", *mbes_test.shape)

    # BTM-33
    btm_train, btm_test, btm_cols = _load_btm33_features()

    # Handle dry-run truncation
    if dry_run:
        btm_train = btm_train.head(len(train_raw))
        btm_test = btm_test.head(len(test_raw))

    # Merge
    train_combined = _merge_features(mbes_train, btm_train)
    test_combined = _merge_features(mbes_test, btm_test)

    elapsed = time.time() - t0
    log.info("Feature extraction completed in %.1fs", elapsed)
    if elapsed >= 900:
        log.warning("SC-005 FAIL: extraction took %.1fs (> 900s limit)", elapsed)
    else:
        log.info("SC-005 PASS: extraction under 15 minutes (%.1fs)", elapsed)

    # Cache
    if not dry_run:
        CACHE_COMBINED_TRAIN.parent.mkdir(parents=True, exist_ok=True)
        train_combined.to_csv(CACHE_COMBINED_TRAIN, index=False)
        test_combined.to_csv(CACHE_COMBINED_TEST, index=False)
        log.info("Combined feature cache written")

    feature_cols = list(train_combined.columns)
    return train_combined, test_combined, feature_cols


# ---------------------------------------------------------------------------
# CV (FR-004, FR-005, FR-006, FR-007, FR-012, EC-3)
# ---------------------------------------------------------------------------


def _run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    label: str,
) -> dict:
    """5-fold spatial-blocked CV with RF, CatBoost, LightGBM + ensemble."""
    import catboost as cb
    import lightgbm as lgb

    from benthic_model.evaluation.cv import iter_spatial_blocked_folds

    feature_cols = list(X.columns)
    classes = np.array(sorted(y.dropna().unique().astype(str)))
    n_classes = len(classes)

    models = {
        "rf": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=CV_RANDOM_STATE,
            n_jobs=-1,
        ),
        "cat": lambda: cb.CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=7,
            auto_class_weights="Balanced",
            random_seed=CV_RANDOM_STATE,
            verbose=0,
        ),
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
    }

    best_name, best_f1, best_oof = None, -1.0, None
    all_oofs: dict[str, np.ndarray] = {}
    all_fold_scores: dict[str, list[float]] = {}
    per_class_rows: list[dict] = []
    model_times: dict[str, float] = {}

    y_np = y.to_numpy(dtype=str, na_value="NVB")

    # Build fold iterator (materialise once, reuse per model)
    fold_splits = list(
        iter_spatial_blocked_folds(
            x=x_coords,
            y=y_coords,
            n_splits=CV_N_SPLITS,
            spatial_bins=CV_SPATIAL_BINS,
            random_state=CV_RANDOM_STATE,
        )
    )

    for name, make_model in models.items():
        t_model_start = time.time()
        oof = np.zeros((len(y), n_classes), dtype=np.float32)
        fold_scores: list[float] = []

        for fold_i, (tr_idx, va_idx) in enumerate(fold_splits):
            m = make_model()
            m.fit(X.values[tr_idx], y_np[tr_idx])
            proba = m.predict_proba(X.values[va_idx])
            # Map model's class subset to full class array
            if hasattr(m, "classes_"):
                model_classes = list(m.classes_)
            else:
                model_classes = list(classes)
            for ci, cls_name in enumerate(model_classes):
                full_idx = np.where(classes == cls_name)[0]
                if len(full_idx) > 0:
                    oof[va_idx, full_idx[0]] = proba[:, ci]
            va_preds = classes[np.argmax(oof[va_idx], axis=1)]
            fold_f1 = f1_score(y_np[va_idx], va_preds, average="weighted")
            fold_scores.append(fold_f1)

            # Per-class recall for all models
            report = classification_report(
                y_np[va_idx], va_preds, output_dict=True, zero_division=0
            )
            row = {"model": name, "fold": fold_i, "weighted_f1": fold_f1}
            for cls in KNOWN_CLASSES:
                recall_val = report.get(cls, {}).get("recall", np.nan)
                # EC-3: if class not present in fold, report NaN
                if cls not in y_np[va_idx]:
                    recall_val = np.nan
                row[f"recall_{cls}"] = recall_val
            per_class_rows.append(row)

        model_times[name] = time.time() - t_model_start
        preds_full = classes[np.argmax(oof, axis=1)]
        cv_f1 = f1_score(y_np, preds_full, average="weighted")
        all_oofs[name] = oof
        all_fold_scores[name] = fold_scores
        log.info(
            "  [%s] %s: CV F1=%.4f ± %.4f  (%.1fs)",
            label,
            name,
            cv_f1,
            float(np.std(fold_scores)),
            model_times[name],
        )
        if cv_f1 > best_f1:
            best_f1, best_name, best_oof = cv_f1, name, oof

    # Soft-vote ensemble (RF + CatBoost + LGB)
    t_ens_start = time.time()
    oof_ens = (all_oofs["rf"] + all_oofs["cat"] + all_oofs["lgb"]) / 3.0
    preds_ens = classes[np.argmax(oof_ens, axis=1)]
    f1_ens = f1_score(y_np, preds_ens, average="weighted")
    fold_ens: list[float] = []
    for tr_idx, va_idx in fold_splits:
        va_preds = classes[np.argmax(oof_ens[va_idx], axis=1)]
        fold_ens.append(f1_score(y_np[va_idx], va_preds, average="weighted"))

    model_times["ensemble"] = time.time() - t_ens_start
    log.info(
        "  [%s] rf+cat+lgb ensemble: CV F1=%.4f ± %.4f",
        label,
        f1_ens,
        float(np.std(fold_ens)),
    )
    all_oofs["ensemble"] = oof_ens
    all_fold_scores["ensemble"] = fold_ens

    if f1_ens > best_f1:
        best_f1, best_name, best_oof = f1_ens, "rf+cat+lgb_ensemble", oof_ens

    total_cv_time = sum(model_times.values())
    log.info("[%s] Best: %s (CV F1=%.4f)", label, best_name, best_f1)
    log.info(
        "[%s] Training times: RF=%.1fs Cat=%.1fs LGB=%.1fs Ens=%.1fs Total=%.1fs",
        label,
        model_times["rf"],
        model_times["cat"],
        model_times["lgb"],
        model_times["ensemble"],
        total_cv_time,
    )

    # Per-class metrics log + SC-003 check
    if per_class_rows:
        pc_df = pd.DataFrame(per_class_rows)
        for cls in KNOWN_CLASSES:
            col = f"recall_{cls}"
            if col in pc_df.columns:
                # Exclude NaN folds from mean (EC-3)
                valid = pc_df[col].dropna()
                if len(valid) > 0:
                    log.info(
                        "  %s recall: %.4f ± %.4f (%d folds)",
                        cls,
                        valid.mean(),
                        valid.std(),
                        len(valid),
                    )

    return {
        "label": label,
        "best_model_name": best_name,
        "cv_f1_mean": float(best_f1),
        "cv_f1_std": float(np.std(all_fold_scores.get(best_name, fold_ens))),
        "oof_proba": best_oof,
        "classes": classes,
        "feature_cols": feature_cols,
        "per_class_rows": per_class_rows,
        "all_oofs": all_oofs,
        "model_times": model_times,
        "rf_cv_f1": f1_score(
            y_np,
            classes[np.argmax(all_oofs["rf"], axis=1)],
            average="weighted",
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dry_run: bool = False, extract_only: bool = False) -> None:
    log.info("=== Experiment v11: Combined BTM-33 + MBES-8 ===")
    log.info(
        "Baseline: CV=%.4f  Kaggle=%.4f  SGAM_recall=%.3f",
        BASELINE_CV_F1,
        BASELINE_KAGGLE,
        BASELINE_SGAM_RECALL,
    )

    # ── Load CSVs ──────────────────────────────────────────────────────────
    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw = pd.read_csv(TEST_CSV)
    log.info("Train: %d rows   Test: %d rows", len(train_raw), len(test_raw))

    if dry_run:
        train_raw = train_raw.head(5)
        test_raw = test_raw.head(5)
        log.info("DRY RUN: 5 train/test points")

    sample_cols = pd.read_csv(SAMPLE_SUBMISSION, nrows=0).columns.tolist()
    id_col = sample_cols[0]
    class_col = sample_cols[1]

    # ── Feature extraction + caching ──────────────────────────────────────
    t_extract_start = time.time()
    train_feats, test_feats, feature_cols = _extract_and_cache(
        train_raw, test_raw, dry_run=dry_run
    )
    extraction_time = time.time() - t_extract_start

    log.info(
        "Feature matrix: %d train × %d features  (%d test)  extraction=%.1fs",
        len(train_feats),
        len(feature_cols),
        len(test_feats),
        extraction_time,
    )

    if extract_only:
        log.info("--extract-only flag set; stopping after feature extraction.")
        return

    # ── Prepare X, y ──────────────────────────────────────────────────────
    label_col = "class" if "class" in train_raw.columns else class_col
    y_train = train_raw[label_col].head(len(train_feats))

    X_train = train_feats[feature_cols].fillna(0.0)
    X_test = test_feats[feature_cols].fillna(0.0)

    x_coords = train_raw["x"].head(len(train_feats)).values
    y_coords = train_raw["y"].head(len(train_feats)).values

    # ── Phase 1: Full feature set CV ──────────────────────────────────────
    t_cv_start = time.time()
    res_full = _run_cv(X_train, y_train, x_coords, y_coords, label="combined")
    cv_time = time.time() - t_cv_start
    log.info("Total CV training time: %.1fs", cv_time)

    # SC-001: RF CV F1 check
    rf_f1 = res_full["rf_cv_f1"]
    if rf_f1 >= BASELINE_CV_F1:
        log.info("SC-001 PASS: RF CV F1=%.4f >= %.4f", rf_f1, BASELINE_CV_F1)
    else:
        log.warning("SC-001 FAIL: RF CV F1=%.4f < %.4f", rf_f1, BASELINE_CV_F1)

    # SC-002: Best model check
    cv_f1 = res_full["cv_f1_mean"]
    if cv_f1 - rf_f1 >= 0.005:
        log.info(
            "SC-002 PASS (CV): best=%s improvement=%.4f >= 0.005",
            res_full["best_model_name"],
            cv_f1 - rf_f1,
        )

    # SC-003: SGAM recall
    if res_full["per_class_rows"]:
        pc_df = pd.DataFrame(res_full["per_class_rows"])
        sgam_recall = pc_df["recall_SGAM"].dropna().mean()
        if sgam_recall >= BASELINE_SGAM_RECALL:
            log.info(
                "SC-003 PASS: SGAM recall=%.4f >= %.4f",
                sgam_recall,
                BASELINE_SGAM_RECALL,
            )
        else:
            log.warning(
                "SC-003 WARNING: SGAM recall=%.4f < %.4f (delta=%.4f)",
                sgam_recall,
                BASELINE_SGAM_RECALL,
                sgam_recall - BASELINE_SGAM_RECALL,
            )

    # ── Per-class metrics output ──────────────────────────────────────────
    OUTPUT_PER_CLASS.parent.mkdir(parents=True, exist_ok=True)
    if res_full["per_class_rows"]:
        pc_df = pd.DataFrame(res_full["per_class_rows"])
        pc_df.to_csv(OUTPUT_PER_CLASS, index=False)
        log.info("Per-class metrics written: %s", OUTPUT_PER_CLASS)

    # ── Feature importance (RF model) ─────────────────────────────────────
    rf_final = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=CV_RANDOM_STATE,
        n_jobs=-1,
    )
    rf_final.fit(X_train.values, y_train.to_numpy(dtype=str, na_value="NVB"))
    imp_mean = rf_final.feature_importances_.tolist()
    imp_std = [0.0] * len(feature_cols)
    _write_feature_importance(imp_mean, imp_std, feature_cols, str(OUTPUT_IMPORTANCE))

    # ── Submission: full feature set (FR-010, T021) ───────────────────────
    best_model_name = res_full["best_model_name"]
    log.info("Training final %s model on full training set...", best_model_name)
    final_model = _build_final_model(best_model_name)
    final_model.fit(X_train.values, y_train.to_numpy(dtype=str, na_value="NVB"))
    test_preds_full = _predict(final_model, X_test.values, res_full["classes"])
    sub_full = pd.DataFrame(
        {id_col: test_raw[id_col].head(len(test_feats)), class_col: test_preds_full}
    )
    _write_submission(sub_full, str(OUTPUT_SUBMISSION))

    # ── Phase 2: Feature selection (SC-004, FR-008, T018) ─────────────────
    from benthic_model.features.selection import select_by_permutation_importance

    log.info("Running permutation importance feature selection...")
    sel_result = select_by_permutation_importance(
        X_train, y_train, rf_final, n_repeats=5, importance_threshold=0.0
    )
    n_selected = len(sel_result.selected_features)
    log.info("Feature selection: %d -> %d features", len(feature_cols), n_selected)

    if n_selected <= 25:
        log.info("SC-004 PASS (count): %d <= 25 features", n_selected)
    else:
        log.warning("SC-004 FAIL (count): %d > 25 features", n_selected)

    sel_result.importance_df.to_csv(OUTPUT_SELECTION, index=False)
    log.info("Feature selection written: %s", OUTPUT_SELECTION)

    # Log MBES-8 vs BTM retention
    mbes_retained = [f for f in sel_result.selected_features if f in MBES8_COLUMNS]
    btm_retained = [f for f in sel_result.selected_features if f not in MBES8_COLUMNS]
    log.info(
        "Retained: %d MBES-8 (%s), %d BTM (%s)",
        len(mbes_retained),
        mbes_retained,
        len(btm_retained),
        btm_retained,
    )

    # ── Selected-feature CV (T019) ────────────────────────────────────────
    if sel_result.selected_features:
        X_train_sel = X_train[sel_result.selected_features]
        X_test_sel = X_test[sel_result.selected_features]
        res_sel = _run_cv(X_train_sel, y_train, x_coords, y_coords, label="selected")
        cv_f1_sel = res_sel["cv_f1_mean"]
        degradation = cv_f1 - cv_f1_sel
        log.info(
            "SC-004 CHECK: CV F1 before=%.4f after=%.4f degradation=%.4f",
            cv_f1,
            cv_f1_sel,
            degradation,
        )
        if degradation <= 0.01:
            log.info("SC-004 PASS (degradation): <= 0.01")
        else:
            log.warning("SC-004 FAIL (degradation): %.4f > 0.01", degradation)

        # Submission: selected features (T022)
        final_model_sel = _build_final_model(res_sel["best_model_name"])
        final_model_sel.fit(
            X_train_sel.values, y_train.to_numpy(dtype=str, na_value="NVB")
        )
        test_preds_sel = _predict(
            final_model_sel, X_test_sel.values, res_sel["classes"]
        )
        sub_sel = pd.DataFrame(
            {
                id_col: test_raw[id_col].head(len(test_feats)),
                class_col: test_preds_sel,
            }
        )
        _write_submission(sub_sel, str(OUTPUT_SUBMISSION_SELECTED))

        # Compare predictions
        changed = (sub_full[class_col].values != sub_sel[class_col].values).sum()
        log.info(
            "Predictions changed after feature selection: %d / %d (%.1f%%)",
            changed,
            len(sub_full),
            100 * changed / max(len(sub_full), 1),
        )

    log.info("=== Experiment v11 complete ===")


# ---------------------------------------------------------------------------
# Model factory helper
# ---------------------------------------------------------------------------


def _build_final_model(model_name: str):
    """Build a fresh untrained model instance by name."""
    import catboost as cb
    import lightgbm as lgb

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=CV_RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "cat":
        return cb.CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=7,
            auto_class_weights="Balanced",
            random_seed=CV_RANDOM_STATE,
            verbose=0,
        )
    elif model_name == "lgb":
        return lgb.LGBMClassifier(
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
        )
    else:
        # Ensemble or unknown — default to RF
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=CV_RANDOM_STATE,
            n_jobs=-1,
        )


def _predict(model, X_values, classes):
    """Predict class labels from a fitted model."""
    proba = model.predict_proba(X_values)
    return classes[np.argmax(proba, axis=1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment v11: Combined BTM-33 + MBES-8"
    )
    parser.add_argument("--dry-run", action="store_true", help="5 points, rapid test")
    parser.add_argument(
        "--extract-only", action="store_true", help="Extract features then stop"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, extract_only=args.extract_only)
