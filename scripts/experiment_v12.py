"""
Experiment v12: MBES-8 + Top-4 BTM Features (RF-first)
=======================================================

DIAGNOSIS FROM v11
------------------
v11 (38 features, CatBoost): CV=0.7829, Kaggle=0.7287, gap=0.054
R04/R06 (~10 features, RF): CV=0.8024, Kaggle=0.7952, gap=0.007

The large CV-Kaggle gap in v11 (0.054 vs R04's 0.007) confirms that the
33 BTM multi-scale features cause spatial overfitting that does not transfer
to the test partition. The same pattern was seen with CatBoost GPU in R05
(CV=0.8139, Kaggle=0.7615, gap=0.052).

HYPOTHESIS
----------
A lean feature set of MBES-8 + only the 4 highest-importance BTM features
from v11's RF importance ranking should:
  1. Eliminate the noise from 29 lower-value BTM columns
  2. Keep the large-scale terrain context that was genuinely informative
  3. Restore RF as the best generalising model (as in R04)

FEATURE SET (12 features)
--------------------------
MBES-8:   depth, backscatter, slope, vrm, complexity, max_curvature,
          northness, eastness
BTM-4:    btm_complexity_21   (v11 RF imp rank 2, 0.126)
          btm_northness_21    (v11 RF imp rank 3, 0.093)
          btm_max_curvature_11 (v11 RF imp rank 4, 0.053)
          btm_northness_11    (v11 RF imp rank 5, 0.045)

TARGET: CV F1 >= 0.8024  /  Kaggle F1 >= 0.79518

USAGE
-----
    python scripts/experiment_v12.py              # full run
    python scripts/experiment_v12.py --dry-run    # 5 points, rapid test
    python scripts/experiment_v12.py --submit     # submit best to Kaggle
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

# v11 combined cache (MBES-8 already extracted there — reuse it)
CACHE_COMBINED_TRAIN = Path("reports/metrics/cache_combined_v11.csv")
CACHE_COMBINED_TEST = Path("reports/metrics/cache_combined_test_v11.csv")

# v12 outputs
OUTPUT_SUBMISSION = Path("data/submission_v12.csv")
OUTPUT_PER_CLASS = Path("reports/metrics/cv_per_class_v12.csv")

CV_N_SPLITS = 5
CV_SPATIAL_BINS = 4
CV_RANDOM_STATE = 42
BASELINE_CV_F1 = 0.8024
BASELINE_KAGGLE = 0.79518
KNOWN_CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
COMPETITION = "geohab-mlwg-competition-2026"

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

# Top-4 BTM by RF permutation importance in v11
BTM4_COLUMNS = [
    "btm_complexity_21",  # imp=0.126, rank 2
    "btm_northness_21",  # imp=0.093, rank 3
    "btm_max_curvature_11",  # imp=0.053, rank 4
    "btm_northness_11",  # imp=0.045, rank 5
]

FEATURE_COLS = MBES8_COLUMNS + BTM4_COLUMNS  # 12 features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_features(dry_run: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 12-feature matrix from the v11 combined cache."""
    if not CACHE_COMBINED_TRAIN.exists():
        raise FileNotFoundError(
            f"v11 combined cache not found: {CACHE_COMBINED_TRAIN}\n"
            "Run experiment_v11.py first to build the cache."
        )
    log.info("Loading v11 combined cache from %s", CACHE_COMBINED_TRAIN)
    train = pd.read_csv(CACHE_COMBINED_TRAIN)
    test = pd.read_csv(CACHE_COMBINED_TEST)

    missing = [c for c in FEATURE_COLS if c not in train.columns]
    if missing:
        raise ValueError(f"Feature columns missing from cache: {missing}")

    if dry_run:
        train = train.head(5)
        test = test.head(5)

    return train[FEATURE_COLS].fillna(0.0), test[FEATURE_COLS].fillna(0.0)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> dict:
    """5-fold spatial-blocked CV: RF (primary) + CatBoost + LGB."""
    import catboost as cb
    import lightgbm as lgb

    from benthic_model.evaluation.cv import iter_spatial_blocked_folds

    classes = np.array(sorted(y.dropna().unique().astype(str)))
    n_classes = len(classes)
    y_np = y.to_numpy(dtype=str, na_value="NVB")

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

    fold_splits = list(
        iter_spatial_blocked_folds(
            x=x_coords,
            y=y_coords,
            n_splits=CV_N_SPLITS,
            spatial_bins=CV_SPATIAL_BINS,
            random_state=CV_RANDOM_STATE,
        )
    )

    all_oofs: dict[str, np.ndarray] = {}
    all_fold_scores: dict[str, list[float]] = {}
    per_class_rows: list[dict] = []
    model_times: dict[str, float] = {}

    for name, make_model in models.items():
        t0 = time.time()
        oof = np.zeros((len(y), n_classes), dtype=np.float32)
        fold_scores: list[float] = []

        for fold_i, (tr_idx, va_idx) in enumerate(fold_splits):
            m = make_model()
            m.fit(X.values[tr_idx], y_np[tr_idx])
            proba = m.predict_proba(X.values[va_idx])
            model_classes = (
                list(m.classes_) if hasattr(m, "classes_") else list(classes)
            )
            for ci, cls_name in enumerate(model_classes):
                full_idx = np.where(classes == cls_name)[0]
                if len(full_idx) > 0:
                    oof[va_idx, full_idx[0]] = proba[:, ci]
            va_preds = classes[np.argmax(oof[va_idx], axis=1)]
            fold_f1 = f1_score(y_np[va_idx], va_preds, average="weighted")
            fold_scores.append(fold_f1)

            report = classification_report(
                y_np[va_idx], va_preds, output_dict=True, zero_division=0
            )
            row = {"model": name, "fold": fold_i, "weighted_f1": fold_f1}
            for cls in KNOWN_CLASSES:
                if cls not in y_np[va_idx]:
                    row[f"recall_{cls}"] = np.nan
                else:
                    row[f"recall_{cls}"] = report.get(cls, {}).get("recall", np.nan)
            per_class_rows.append(row)

        model_times[name] = time.time() - t0
        preds_full = classes[np.argmax(oof, axis=1)]
        cv_f1 = f1_score(y_np, preds_full, average="weighted")
        all_oofs[name] = oof
        all_fold_scores[name] = fold_scores
        log.info(
            "  %s: CV F1=%.4f ± %.4f  (%.1fs)",
            name,
            cv_f1,
            float(np.std(fold_scores)),
            model_times[name],
        )

    # Soft-vote ensemble
    t0 = time.time()
    oof_ens = (all_oofs["rf"] + all_oofs["cat"] + all_oofs["lgb"]) / 3.0
    preds_ens = classes[np.argmax(oof_ens, axis=1)]
    f1_ens = f1_score(y_np, preds_ens, average="weighted")
    fold_ens = []
    for _, va_idx in fold_splits:
        va_preds = classes[np.argmax(oof_ens[va_idx], axis=1)]
        fold_ens.append(f1_score(y_np[va_idx], va_preds, average="weighted"))
    model_times["ensemble"] = time.time() - t0
    all_oofs["ensemble"] = oof_ens
    log.info(
        "  rf+cat+lgb ensemble: CV F1=%.4f ± %.4f",
        f1_ens,
        float(np.std(fold_ens)),
    )

    # Find best
    scores = {
        "rf": f1_score(
            y_np, classes[np.argmax(all_oofs["rf"], axis=1)], average="weighted"
        ),
        "cat": f1_score(
            y_np, classes[np.argmax(all_oofs["cat"], axis=1)], average="weighted"
        ),
        "lgb": f1_score(
            y_np, classes[np.argmax(all_oofs["lgb"], axis=1)], average="weighted"
        ),
        "ensemble": f1_ens,
    }
    best_name = max(scores, key=scores.__getitem__)
    best_f1 = scores[best_name]
    log.info("Best: %s (CV F1=%.4f)", best_name, best_f1)
    log.info(
        "Training times: RF=%.1fs Cat=%.1fs LGB=%.1fs Ens=%.1fs",
        model_times["rf"],
        model_times["cat"],
        model_times["lgb"],
        model_times["ensemble"],
    )

    # Per-class recall summary
    if per_class_rows:
        pc_df = pd.DataFrame(per_class_rows)
        for cls in KNOWN_CLASSES:
            col = f"recall_{cls}"
            if col in pc_df.columns:
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
        "best_model_name": best_name,
        "cv_f1_mean": best_f1,
        "rf_cv_f1": scores["rf"],
        "classes": classes,
        "all_oofs": all_oofs,
        "per_class_rows": per_class_rows,
        "model_times": model_times,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def _write_submission(ids, predictions, classes, path: Path) -> None:
    sample = pd.read_csv(SAMPLE_SUBMISSION, nrows=0)
    id_col, class_col = sample.columns[0], sample.columns[1]
    df = pd.DataFrame({id_col: ids, class_col: predictions})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Submission written: %s (%d rows)", path, len(df))


def _submit_to_kaggle(csv_path: Path, message: str) -> None:
    """Submit a CSV to Kaggle and print the resulting public score."""
    try:
        import kaggle

        api = kaggle.KaggleApi()
        api.authenticate()
        log.info(
            "Submitting %s to Kaggle competition %s ...", csv_path.name, COMPETITION
        )
        api.competition_submit(
            file_name=str(csv_path),
            message=message,
            competition=COMPETITION,
        )
        log.info("Submission accepted — score will appear on the leaderboard shortly.")
        # Poll once for the new score
        import time as _time

        _time.sleep(8)
        subs = api.competition_submissions(COMPETITION)
        if subs:
            latest = subs[0]
            log.info(
                "Latest submission: %s  public_score=%s  status=%s",
                latest.file_name,
                latest.public_score,
                latest.status,
            )
    except Exception as exc:
        log.warning("Kaggle submission failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dry_run: bool = False, submit: bool = False) -> None:
    log.info("=== Experiment v12: MBES-8 + Top-4 BTM (12 features) ===")
    log.info("Baseline: CV=%.4f  Kaggle=%.4f", BASELINE_CV_F1, BASELINE_KAGGLE)
    log.info("Features (%d): %s", len(FEATURE_COLS), ", ".join(FEATURE_COLS))

    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw = pd.read_csv(TEST_CSV)
    log.info("Train: %d rows   Test: %d rows", len(train_raw), len(test_raw))

    if dry_run:
        train_raw = train_raw.head(5)
        test_raw = test_raw.head(5)
        log.info("DRY RUN: 5 points")

    X_train, X_test = _load_features(dry_run)
    y_train = train_raw["class"].head(len(X_train))
    x_coords = train_raw["x"].head(len(X_train)).values
    y_coords = train_raw["y"].head(len(X_train)).values

    log.info(
        "Feature matrix: %d × %d train,  %d × %d test",
        len(X_train),
        len(X_train.columns),
        len(X_test),
        len(X_test.columns),
    )

    # ── Cross-validation ──────────────────────────────────────────────────
    res = _run_cv(X_train, y_train, x_coords, y_coords)

    # Gate checks
    rf_f1 = res["rf_cv_f1"]
    if rf_f1 >= BASELINE_CV_F1:
        log.info("SC-001 PASS: RF CV F1=%.4f >= %.4f", rf_f1, BASELINE_CV_F1)
    else:
        log.warning(
            "SC-001 FAIL: RF CV F1=%.4f < %.4f (gap=%.4f)",
            rf_f1,
            BASELINE_CV_F1,
            BASELINE_CV_F1 - rf_f1,
        )

    if res["per_class_rows"]:
        pc_df = pd.DataFrame(res["per_class_rows"])
        sgam_recall = pc_df["recall_SGAM"].dropna().mean()
        log.info("SGAM recall: %.4f (baseline=0.045)", sgam_recall)

    # ── Save per-class metrics ───────────────────────────────────────────
    OUTPUT_PER_CLASS.parent.mkdir(parents=True, exist_ok=True)
    if res["per_class_rows"]:
        pd.DataFrame(res["per_class_rows"]).to_csv(OUTPUT_PER_CLASS, index=False)
        log.info("Per-class metrics written: %s", OUTPUT_PER_CLASS)

    # ── Train final models and write submissions ──────────────────────────
    classes = res["classes"]
    y_np = y_train.to_numpy(dtype=str, na_value="NVB")

    best_name = res["best_model_name"]
    log.info("Training final %s model on full training set...", best_name)

    def _make_final(name: str):
        import catboost as cb
        import lightgbm as lgb

        if name == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=CV_RANDOM_STATE,
                n_jobs=-1,
            )
        elif name == "cat":
            return cb.CatBoostClassifier(
                iterations=800,
                learning_rate=0.03,
                depth=7,
                auto_class_weights="Balanced",
                random_seed=CV_RANDOM_STATE,
                verbose=0,
            )
        elif name == "lgb":
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
        else:  # ensemble: use RF for final predictions
            return RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=CV_RANDOM_STATE,
                n_jobs=-1,
            )

    final_model = _make_final(best_name)
    final_model.fit(X_train.values, y_np)
    proba = final_model.predict_proba(X_test.values)
    model_classes = list(final_model.classes_)
    pred_proba = np.zeros((len(X_test), len(classes)))
    for ci, cls_name in enumerate(model_classes):
        full_idx = np.where(classes == cls_name)[0]
        if len(full_idx) > 0:
            pred_proba[:, full_idx[0]] = proba[:, ci]
    test_preds = classes[np.argmax(pred_proba, axis=1)]

    _write_submission(
        test_raw["ID"].head(len(X_test)), test_preds, classes, OUTPUT_SUBMISSION
    )

    # Also always write an RF submission (in case best model is CatBoost)
    if best_name != "rf":
        rf_model = _make_final("rf")
        rf_model.fit(X_train.values, y_np)
        rf_proba = rf_model.predict_proba(X_test.values)
        rf_classes = list(rf_model.classes_)
        rf_pred_proba = np.zeros((len(X_test), len(classes)))
        for ci, cls_name in enumerate(rf_classes):
            full_idx = np.where(classes == cls_name)[0]
            if len(full_idx) > 0:
                rf_pred_proba[:, full_idx[0]] = rf_proba[:, ci]
        rf_preds = classes[np.argmax(rf_pred_proba, axis=1)]
        rf_sub_path = Path("data/submission_v12_rf.csv")
        _write_submission(
            test_raw["ID"].head(len(X_test)), rf_preds, classes, rf_sub_path
        )

    log.info("=== Experiment v12 complete ===")
    log.info("Summary:")
    for name, score in res["scores"].items():
        log.info("  %s: %.4f%s", name, score, " <- BEST" if name == best_name else "")
    log.info(
        "  RF vs baseline: %.4f vs %.4f  (delta=+%.4f)",
        rf_f1,
        BASELINE_CV_F1,
        rf_f1 - BASELINE_CV_F1,
    )

    if submit and not dry_run:
        _submit_to_kaggle(
            OUTPUT_SUBMISSION,
            f"v12: MBES-8 + top-4 BTM (12 feat), {best_name}, CV={res['cv_f1_mean']:.4f}",
        )
        if best_name != "rf":
            _submit_to_kaggle(
                Path("data/submission_v12_rf.csv"),
                f"v12-rf: MBES-8 + top-4 BTM (12 feat), RF, CV={rf_f1:.4f}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment v12: MBES-8 + Top-4 BTM (12 features, RF-first)"
    )
    parser.add_argument("--dry-run", action="store_true", help="5 points, rapid test")
    parser.add_argument("--submit", action="store_true", help="Auto-submit to Kaggle")
    args = parser.parse_args()
    main(dry_run=args.dry_run, submit=args.submit)
