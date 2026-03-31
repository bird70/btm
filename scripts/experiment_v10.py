"""
Experiment v10: Multi-Scale Terrain + GLCM Texture Features
============================================================

HYPOTHESIS
----------
Single-scale (3×3) BTM derivatives miss broader geomorphic context.
Nemani et al. (2022) showed no 3×3 features were selected in their best
model; multi-scale features (3×3 – 21×21) improved accuracy 25%+.
Backscatter GLCM contrast at broad scale was their 4th most important feature.

v10 DESIGN PRINCIPLES
---------------------
1. Use ``btm.features.extract.extract_btm_features`` with new ``scales`` +
   ``include_glcm`` params (spec-016 additions).
2. Multi-scale terrain: 7 derivatives × 5 scales + 5 RDMV = 40 new columns.
3. GLCM texture: contrast + homogeneity at 5 scales = 10 new columns.
4. Feature selection: Spearman corr pre-filter + permutation importance
   (``benthic_model.features.selection``) to reduce to ≤ 25 features.
5. Train CatBoost + LGB on full-feature and selected-feature sets.
6. 10-fold spatial GroupKFold CV (same as v9 for direct comparability).
7. Evaluate SGAM (minority class) recall explicitly per SC-003.

FEATURE SET
-----------
Base BTM (11):  broad_bpi, fine_bpi, broad_std, fine_std, slope, vrm,
                surface_ratio, bpi_magnitude, broad_x_fine_std, rough_total
Eco (4):        northness, eastness, max_curvature, complexity
Multi-scale (40): btm_{deriv}_{scale} for 7 derivatives × 5 scales
RDMV (5):       btm_rdmv_{scale} for 5 scales
GLCM (10):      btm_glcm_contrast_{scale} + btm_glcm_homogeneity_{scale}
Total:          ~70 features before selection

BASELINES
---------
Kaggle: 0.79518  (best: v6/v9 CatBoost combined)
CV F1:  0.8024   (best: v9 CatBoost + LGB on combined PB+OB)

SUCCESS CRITERIA (spec-016)
---------------------------
SC-001: CV weighted-F1 ≥ 0.8024
SC-002: Kaggle ≥ 0.79518 OR CV improvement ≥ 0.005
SC-003: SGAM recall maintained or improved (baseline from run registry)
SC-004: Feature selection to ≤ 25 features, CV degradation ≤ 0.01
SC-005: Extraction time < 10 minutes for 6256 training points

USAGE
-----
    python scripts/experiment_v10.py                   # full run
    python scripts/experiment_v10.py --extract-only    # extraction timing only
    python scripts/experiment_v10.py --dry-run         # 5 points, 2 scales
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATHY_PATH = Path("data/MBES/bathymetry.tif")
BACK_PATH = Path("data/MBES/backscatter.tif")
TRAIN_CSV = Path("data/train.csv")
TEST_CSV = Path("data/test.csv")
SAMPLE_SUBMISSION = Path("data/sample_submission.csv")
OUTPUT_SUBMISSION = Path("data/submission_v10.csv")
OUTPUT_SUBMISSION_SELECTED = Path("data/submission_v10_selected.csv")
OUTPUT_IMPORTANCE = Path("reports/metrics/feature_importance_v10.csv")
OUTPUT_PER_CLASS = Path("reports/metrics/cv_per_class_v10.csv")

SCALES = [3, 7, 11, 15, 21]
CV_N_SPLITS = 10
CV_RANDOM_STATE = 42
BASELINE_CV_F1 = 0.8024
BASELINE_KAGGLE = 0.79518
KNOWN_CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers (tested in tests/unit/test_experiment_v10.py)
# ---------------------------------------------------------------------------


def _write_feature_importance(
    importances_mean: list[float],
    importances_std: list[float],
    feature_names: list[str],
    path: str,
) -> None:
    """Write feature importance CSV with required schema (FR-006).

    Columns: feature_name, importance_mean, importance_std, rank, selected.
    ``selected`` is True for features with importance_mean > 0.0.

    Parameters
    ----------
    importances_mean, importances_std:
        Parallel lists of per-feature importance mean and std values.
    feature_names:
        Column name for each feature, same order as importances.
    path:
        Destination file path (parent directory created if needed).
    """
    df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance_mean": importances_mean,
            "importance_std": importances_std,
        }
    )
    df["selected"] = df["importance_mean"] > 0.0
    df["rank"] = (
        df["importance_mean"].rank(ascending=False, na_option="bottom").astype(int)
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Feature importance written: %s", out)


def _write_submission(predictions: pd.DataFrame, path: str) -> None:
    """Write a Kaggle submission CSV (FR-009).

    ``predictions`` must have columns matching ``data/sample_submission.csv``
    (ID column + class column).

    Parameters
    ----------
    predictions:
        DataFrame with exactly the columns in ``data/sample_submission.csv``.
    path:
        Destination file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    log.info("Submission written: %s (%d rows)", out, len(predictions))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scales: list[int],
    include_glcm: bool = True,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract multi-scale BTM features for train and test point sets."""
    from btm.features.extract import extract_btm_features

    common_kwargs = dict(
        bathymetry_tif=BATHY_PATH,
        include_interactions=True,
        include_eco_features=True,
        scales=scales,
        include_glcm=include_glcm,
        backscatter_tif=BACK_PATH,
    )

    t0 = time.time()
    train_feats = extract_btm_features(train_df, **common_kwargs)
    elapsed_train = time.time() - t0
    log.info(
        "Train feature extraction: %d points, %d columns in %.1fs",
        len(train_df),
        len(train_feats.columns),
        elapsed_train,
    )
    if elapsed_train >= 600:
        log.warning("SC-005 FAIL: extraction took %.1fs (> 600s limit)", elapsed_train)
    else:
        log.info("SC-005 PASS: extraction under 10 minutes (%.1fs)", elapsed_train)

    t1 = time.time()
    test_feats = extract_btm_features(test_df, **common_kwargs)
    log.info(
        "Test feature extraction: %d points, %d columns in %.1fs",
        len(test_df),
        len(test_feats.columns),
        time.time() - t1,
    )

    return train_feats, test_feats


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------


def _get_spatial_groups(xy: np.ndarray, n: int = CV_N_SPLITS) -> np.ndarray:
    km = KMeans(n_clusters=n, random_state=CV_RANDOM_STATE, n_init=10)
    return km.fit_predict(xy).astype(int)


def _run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    label: str,
) -> dict:
    """10-fold spatial GroupKFold CV; returns best-model name + metrics."""
    import catboost as cb
    import lightgbm as lgb

    feature_cols = list(X.columns)
    classes = np.sort(y.unique())
    n_classes = len(classes)
    gkf = GroupKFold(n_splits=CV_N_SPLITS)

    models = {
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
    # Accumulate per-fold classification reports for per-class metrics
    per_class_rows: list[dict] = []

    for name, make_model in models.items():
        oof = np.zeros((len(y), n_classes), dtype=np.float32)
        fold_scores: list[float] = []

        for fold_i, (tr_idx, va_idx) in enumerate(
            gkf.split(X.values, y.values, groups)
        ):
            m = make_model()
            m.fit(X.values[tr_idx], y.values[tr_idx])
            oof[va_idx] = m.predict_proba(X.values[va_idx])
            va_preds = classes[np.argmax(oof[va_idx], axis=1)]
            fold_f1 = f1_score(y.values[va_idx], va_preds, average="weighted")
            fold_scores.append(fold_f1)

            if name == "cat":
                report = classification_report(
                    y.values[va_idx], va_preds, output_dict=True, zero_division=0
                )
                row = {"model": name, "fold": fold_i, "weighted_f1": fold_f1}
                for cls in KNOWN_CLASSES:
                    row[f"recall_{cls}"] = report.get(cls, {}).get("recall", 0.0)
                per_class_rows.append(row)

        preds_full = classes[np.argmax(oof, axis=1)]
        cv_f1 = f1_score(y.values, preds_full, average="weighted")
        all_oofs[name] = oof
        all_fold_scores[name] = fold_scores
        log.info(
            "  [%s] %s: CV F1=%.4f ± %.4f",
            label,
            name,
            cv_f1,
            float(np.std(fold_scores)),
        )
        if cv_f1 > best_f1:
            best_f1, best_name, best_oof = cv_f1, name, oof

    # Soft-vote ensemble
    oof_ens = (all_oofs["cat"] + all_oofs["lgb"]) / 2.0
    preds_ens = classes[np.argmax(oof_ens, axis=1)]
    f1_ens = f1_score(y.values, preds_ens, average="weighted")
    fold_ens: list[float] = []
    for tr_idx, va_idx in gkf.split(X.values, y.values, groups):
        va_preds = classes[np.argmax(oof_ens[va_idx], axis=1)]
        fold_ens.append(f1_score(y.values[va_idx], va_preds, average="weighted"))
    log.info(
        "  [%s] cat+lgb ensemble: CV F1=%.4f ± %.4f",
        label,
        f1_ens,
        float(np.std(fold_ens)),
    )
    if f1_ens > best_f1:
        best_f1, best_name, best_oof = f1_ens, "cat+lgb_ensemble", oof_ens

    log.info("[%s] Best: %s (CV F1=%.4f)", label, best_name, best_f1)

    # Per-class metrics log + SC-003 check
    if per_class_rows:
        pc_df = pd.DataFrame(per_class_rows)
        mean_sgam_recall = pc_df["recall_SGAM"].mean()
        log.info(
            "[%s] SGAM mean recall across folds (SC-003): %.4f", label, mean_sgam_recall
        )
        for cls in KNOWN_CLASSES:
            col = f"recall_{cls}"
            if col in pc_df.columns:
                log.info(
                    "  %s recall: %.4f ± %.4f", cls, pc_df[col].mean(), pc_df[col].std()
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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dry_run: bool = False, extract_only: bool = False) -> None:
    log.info("=== Experiment v10: Multi-Scale Terrain + GLCM ===")
    log.info("Baseline: CV=%.4f  Kaggle=%.4f", BASELINE_CV_F1, BASELINE_KAGGLE)

    # ── Load CSVs ──────────────────────────────────────────────────────────
    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw = pd.read_csv(TEST_CSV)
    log.info("Train: %d rows   Test: %d rows", len(train_raw), len(test_raw))

    if dry_run:
        train_raw = train_raw.head(5)
        test_raw = test_raw.head(5)
        run_scales = [3, 7]
        log.info("DRY RUN: 5 train points, scales=%s", run_scales)
    else:
        run_scales = SCALES

    # Identify ID and coordinate columns
    sample_cols = pd.read_csv(SAMPLE_SUBMISSION, nrows=0).columns.tolist()
    id_col = sample_cols[0]
    class_col = sample_cols[1]

    # ── Feature extraction ────────────────────────────────────────────────
    train_feats, test_feats = _extract_features(
        train_raw, test_raw, scales=run_scales, include_glcm=True, dry_run=dry_run
    )

    if extract_only:
        log.info("--extract-only flag set; stopping after feature extraction.")
        return

    # ── Identify feature columns ──────────────────────────────────────────
    btm_cols = [c for c in train_feats.columns if c.startswith("btm_")]
    label_col = "class" if "class" in train_feats.columns else class_col
    if label_col not in train_feats.columns and label_col in train_raw.columns:
        train_feats[label_col] = train_raw[label_col].values

    X_train = train_feats[btm_cols].fillna(0.0)
    y_train = train_feats[label_col]
    X_test = test_feats[btm_cols].fillna(0.0)

    log.info(
        "Feature matrix: %d train × %d features  (%d test)",
        len(X_train),
        len(btm_cols),
        len(X_test),
    )

    # ── Spatial CV groups ─────────────────────────────────────────────────
    xy = train_feats[["x", "y"]].values
    groups = _get_spatial_groups(xy)

    # ── Phase 1: Full feature set ─────────────────────────────────────────
    res_full = _run_cv(X_train, y_train, groups, label="full_multiscale")

    # SC-001 / SC-002 check
    cv_f1 = res_full["cv_f1_mean"]
    if cv_f1 >= BASELINE_CV_F1:
        log.info("SC-001 PASS: CV F1=%.4f ≥ %.4f", cv_f1, BASELINE_CV_F1)
    else:
        log.warning("SC-001 FAIL: CV F1=%.4f < %.4f", cv_f1, BASELINE_CV_F1)

    if cv_f1 - BASELINE_CV_F1 >= 0.005:
        log.info("SC-002 PASS (CV): improvement=%.4f ≥ 0.005", cv_f1 - BASELINE_CV_F1)

    # ── Per-class metrics (SC-003) ────────────────────────────────────────
    OUTPUT_PER_CLASS.parent.mkdir(parents=True, exist_ok=True)
    if res_full["per_class_rows"]:
        pc_df = pd.DataFrame(res_full["per_class_rows"])
        pc_df.to_csv(OUTPUT_PER_CLASS, index=False)
        log.info("Per-class metrics written: %s", OUTPUT_PER_CLASS)

        sgam_recall = pc_df["recall_SGAM"].mean()
        if sgam_recall > 0:
            log.info(
                "SC-003 CHECK: SGAM recall=%.4f (compare to baseline run)", sgam_recall
            )
        else:
            log.warning("SC-003 WARNING: SGAM mean recall=%.4f", sgam_recall)

    # ── Feature importance ────────────────────────────────────────────────
    import catboost as cb

    final_model = cb.CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=7,
        auto_class_weights="Balanced",
        random_seed=CV_RANDOM_STATE,
        verbose=0,
    )
    final_model.fit(X_train.values, y_train.values)
    # Use model's built-in feature importances for ranking (fast)
    imp_mean = final_model.get_feature_importance().tolist()
    imp_std = [0.0] * len(btm_cols)

    _write_feature_importance(imp_mean, imp_std, btm_cols, str(OUTPUT_IMPORTANCE))

    # ── Submission: full feature set ──────────────────────────────────────
    test_preds_full = final_model.predict(X_test.values).ravel()
    sub_full = pd.DataFrame({id_col: test_feats[id_col], class_col: test_preds_full})
    _write_submission(sub_full, str(OUTPUT_SUBMISSION))

    # ── Phase 2: Feature selection (SC-004) ──────────────────────────────
    from benthic_model.features.selection import select_by_permutation_importance

    log.info("Running permutation importance feature selection...")
    sel_result = select_by_permutation_importance(
        X_train, y_train, final_model, n_repeats=5, importance_threshold=0.0
    )
    n_selected = len(sel_result.selected_features)
    log.info(
        "Feature selection: %d → %d features",
        len(btm_cols),
        n_selected,
    )

    if n_selected <= 25:
        log.info("SC-004 PASS (count): %d ≤ 25 features", n_selected)
    else:
        log.warning("SC-004 FAIL (count): %d > 25 features", n_selected)

    sel_result.importance_df.to_csv(
        OUTPUT_IMPORTANCE.parent / "feature_selection_v10.csv", index=False
    )

    if sel_result.selected_features:
        X_train_sel = X_train[sel_result.selected_features]
        X_test_sel = X_test[sel_result.selected_features]
        res_sel = _run_cv(X_train_sel, y_train, groups, label="selected_features")
        cv_f1_sel = res_sel["cv_f1_mean"]
        degradation = cv_f1 - cv_f1_sel
        log.info(
            "SC-004 CHECK: CV F1 before=%.4f after=%.4f degradation=%.4f",
            cv_f1,
            cv_f1_sel,
            degradation,
        )
        if degradation <= 0.01:
            log.info("SC-004 PASS (degradation): ≤ 0.01")
        else:
            log.warning("SC-004 FAIL (degradation): %.4f > 0.01", degradation)

        # Submission: selected features
        final_model_sel = cb.CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=7,
            auto_class_weights="Balanced",
            random_seed=CV_RANDOM_STATE,
            verbose=0,
        )
        final_model_sel.fit(X_train_sel.values, y_train.values)
        test_preds_sel = final_model_sel.predict(X_test_sel.values).ravel()
        sub_sel = pd.DataFrame({id_col: test_feats[id_col], class_col: test_preds_sel})
        _write_submission(sub_sel, str(OUTPUT_SUBMISSION_SELECTED))

        # Compare predictions
        changed = (sub_full[class_col].values != sub_sel[class_col].values).sum()
        log.info(
            "Predictions changed after feature selection: %d / %d (%.1f%%)",
            changed,
            len(sub_full),
            100 * changed / max(len(sub_full), 1),
        )

    log.info("=== Experiment v10 complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment v10: multi-scale terrain + GLCM"
    )
    parser.add_argument("--dry-run", action="store_true", help="5 points, 2 scales")
    parser.add_argument(
        "--extract-only", action="store_true", help="Extract features then stop"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, extract_only=args.extract_only)
