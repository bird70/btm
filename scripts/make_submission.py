"""
Hybrid BTM + LightGBM submission for the GeoHab 2026 MLWG Kaggle Competition.

Dataset: Refuge Cove, Victoria, Australia (Ierodiaconou et al. 2018)
  - 0.25 m MBES bathymetry and backscatter
  - 5 benthic habitat classes: ALG, FMAT, NVB, SGAM, SGZ
  - 98 test points to predict

Pipeline
--------
1. Load and validate all competition data
2. Extract features at every labelled/test point:
   a. Raw bathymetry + backscatter (rasterio point sampling)
   b. Focal statistics at 3 scales: 1m, 2.5m, 6.25m radius
      (using scipy.ndimage.uniform_filter — O(n), seconds even on 19 M cells)
   c. BTM terrain derivatives: BTM slope (Horn 1981) and VRM (Sappington 2007)
      both computed by 3×3 kernels — fast on any raster
   d. BTM BPI approximated via TPI (depth - focal mean) at 2 scales:
      fine (~1m) and broad (~5m) — using uniform_filter box windows
      Note: this is a box-window TPI, not BTM's annular BPI; the distinction
      matters ecologically but the box version is still informative and fast
   e. Interaction features: depth×backscatter, slope×backscatter,
      bathy_roughness_ratio, acoustic_hardness_proxy
3. Train LightGBM with class_weight='balanced', using 5-fold stratified CV
   to estimate per-class F1 before fitting the final model
4. Predict class labels for all 98 test points
5. Write submission CSV in ID,class format

Output
------
  data/submission_hybrid_btm.csv   — ready for Kaggle upload

Exit codes: 0 success, 1 input error, 2 feature-extraction error, 3 model error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from btm._logging import configure_cli_logging, get_logger  # noqa: E402
from btm.core.slope import compute_slope  # noqa: E402
from btm.core.vrm import compute_vrm  # noqa: E402
from btm.features.extract import sample_raster_at_points  # noqa: E402

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DATA = _REPO / "data"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _load_raster(path: Path) -> tuple[np.ndarray, object, float | None]:
    """Return (array, transform, nodata)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform, nodata


def _focal_stats(
    arr: np.ndarray,
    transform,
    xs: np.ndarray,
    ys: np.ndarray,
    radii_m: list[float],
    prefix: str,
) -> dict[str, np.ndarray]:
    """
    Compute focal mean and std at several radii (in metres) using a fast
    box-window uniform_filter.  Both mean and std are sampled at (xs, ys).

    The box side length is round(radius_m / cell_size) * 2 + 1 (odd integer).
    """
    cell_size = abs(transform.a)  # assumed square pixels
    results: dict[str, np.ndarray] = {}

    for r in radii_m:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1  # odd window side in cells

        # Replace NaN with local mean for convolution stability
        filled = np.where(np.isfinite(arr), arr, 0.0)
        count_valid = uniform_filter((~np.isnan(arr)).astype(float), size=size)
        focal_sum = uniform_filter(filled, size=size) * (size * size)
        focal_mean = np.where(count_valid > 0, focal_sum / (size * size), np.nan)

        # Focal variance: E[x²] - (E[x])²
        sq_sum = uniform_filter(np.where(np.isfinite(arr), arr**2, 0.0), size=size)
        focal_sq_mean = sq_sum  # already normalised by uniform_filter
        focal_var = focal_sq_mean - focal_mean**2
        focal_std = np.sqrt(np.maximum(focal_var, 0.0))

        key_mean = f"{prefix}focal_mean_{size}"
        key_std = f"{prefix}focal_std_{size}"
        results[key_mean] = sample_raster_at_points(focal_mean, transform, xs, ys)
        results[key_std] = sample_raster_at_points(focal_std, transform, xs, ys)

        # TPI / BPI approximation: depth - focal_mean  (signed: + = high, - = low)
        tpi = arr - focal_mean
        tpi_key = f"{prefix}tpi_{size}"
        results[tpi_key] = sample_raster_at_points(tpi, transform, xs, ys)

    return results


def extract_features(
    points: pd.DataFrame,
    bathy_path: Path,
    backscatter_path: Path,
    focal_radii_m: list[float] | None = None,
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Build a feature matrix for *points* from MBES rasters + BTM derivatives.

    Parameters
    ----------
    points:
        DataFrame with columns ``id_col``, ``x``, ``y``.
    bathy_path, backscatter_path:
        Paths to MBES GeoTIFFs.
    focal_radii_m:
        Focal window radii in metres (box approximation).
        Default: [1.0, 2.5, 6.25].
    id_col:
        Name of the identifier column.
    """
    if focal_radii_m is None:
        focal_radii_m = [1.0, 2.5, 6.25]

    xs = points["x"].to_numpy(dtype=float)
    ys = points["y"].to_numpy(dtype=float)

    _log.info("Loading bathymetry raster …")
    bathy, bathy_transform, bathy_nodata = _load_raster(bathy_path)
    _log.info("Loading backscatter raster …")
    back, back_transform, back_nodata = _load_raster(backscatter_path)

    feats: dict[str, np.ndarray] = {}

    # ── 1. Raw values ────────────────────────────────────────────────────────
    _log.info("Sampling raw bathymetry and backscatter …")
    feats["bathymetry"] = sample_raster_at_points(
        np.where(np.isfinite(bathy), bathy, bathy_nodata or -10000),
        bathy_transform, xs, ys, nodata=bathy_nodata,
    )
    feats["backscatter"] = sample_raster_at_points(
        np.where(np.isfinite(back), back, back_nodata or -10000),
        back_transform, xs, ys, nodata=back_nodata,
    )

    # ── 2. Focal statistics (uniform_filter, O(n)) ───────────────────────────
    _log.info("Computing focal statistics at %s m radii …", focal_radii_m)
    bathy_focal = _focal_stats(bathy, bathy_transform, xs, ys, focal_radii_m, "bathy_")
    back_focal  = _focal_stats(back,  back_transform,  xs, ys, focal_radii_m, "back_")
    feats.update(bathy_focal)
    feats.update(back_focal)

    # ── 3. BTM slope (Horn 1981, 3×3) ───────────────────────────────────────
    cell_size = abs(bathy_transform.a)
    bathy_filled = np.where(np.isfinite(bathy), bathy, 0.0)

    _log.info("Computing BTM slope (Horn 1981) …")
    slope_arr = compute_slope(bathy_filled, cell_size, nodata=None)
    feats["btm_slope"] = sample_raster_at_points(slope_arr, bathy_transform, xs, ys)

    # ── 4. BTM VRM (Sappington 2007, 3×3) ───────────────────────────────────
    _log.info("Computing BTM VRM (Sappington 2007) …")
    vrm_arr = compute_vrm(bathy_filled, neighborhood_size=3, cell_size=cell_size)
    feats["btm_vrm"] = sample_raster_at_points(vrm_arr, bathy_transform, xs, ys)

    # ── 5. Interaction features ──────────────────────────────────────────────
    b = feats["bathymetry"]
    bs = feats["backscatter"]
    slope = feats["btm_slope"]

    # depth × backscatter: deeper + higher backscatter → rocky substrate
    feats["depth_x_backscatter"] = np.abs(b) * bs

    # slope × backscatter: hard rocky slopes are steep and highly reflective
    feats["slope_x_backscatter"] = slope * bs

    # acoustic hardness proxy
    feats["acoustic_hardness"] = bs / (np.abs(b) + 1.0)

    # coarse/fine roughness ratio (TPI at two scales)
    bathy_fine_key  = [k for k in feats if k.startswith("bathy_tpi_") and "3" in k]
    bathy_coarse_key = [k for k in feats if k.startswith("bathy_tpi_") and "51" in k]
    if bathy_fine_key and bathy_coarse_key:
        feats["tpi_ratio"] = feats[bathy_coarse_key[0]] / (
            np.abs(feats[bathy_fine_key[0]]) + 1.0
        )

    result = points[[id_col, "x", "y"]].copy().reset_index(drop=True)
    for name, values in feats.items():
        result[name] = values

    _log.info(
        "Feature extraction complete: %d points × %d features",
        len(result), len(feats),
    )
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_predict(
    train_feats: pd.DataFrame,
    train_labels: pd.Series,
    test_feats: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
    n_splits: int = 5,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Train a LightGBM classifier with stratified k-fold CV, then fit a final
    model on all training data.

    Returns
    -------
    preds:
        Class label array for test rows.
    cv_scores:
        Per-class F1 scores from cross-validation.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    X_train = train_feats[feature_cols].fillna(train_feats[feature_cols].median())
    X_test  = test_feats[feature_cols].fillna(train_feats[feature_cols].median())
    y_raw   = train_labels.astype(str)

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)

    _log.info("Classes: %s", list(enc.classes_))
    _log.info("Training set: %d rows × %d features", len(X_train), len(feature_cols))

    # ── Cross-validation ─────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_true: list[int] = []
    oof_pred: list[int] = []

    _log.info("Running %d-fold stratified CV …", n_splits)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        model = LGBMClassifier(
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
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train.iloc[tr_idx], y[tr_idx])
        oof_pred.extend(model.predict(X_train.iloc[va_idx]).tolist())
        oof_true.extend(y[va_idx].tolist())
        wf1 = f1_score(y[va_idx], model.predict(X_train.iloc[va_idx]), average="weighted")
        _log.info("  fold %d: weighted-F1 = %.4f", fold, wf1)

    cv_weighted = float(f1_score(oof_true, oof_pred, average="weighted"))
    _log.info("OOF weighted-F1 = %.4f", cv_weighted)
    report = classification_report(
        enc.inverse_transform(oof_true),
        enc.inverse_transform(oof_pred),
        output_dict=True,
    )
    cv_scores = {k: v["f1-score"] for k, v in report.items() if k in enc.classes_}
    cv_scores["weighted_avg"] = cv_weighted

    print(f"\n{'='*50}")
    print("Cross-validation results (OOF):")
    print(classification_report(
        enc.inverse_transform(oof_true),
        enc.inverse_transform(oof_pred),
    ))
    print(f"{'='*50}\n")

    # ── Final model on all training data ────────────────────────────────────
    _log.info("Fitting final model on all %d training samples …", len(X_train))
    final_model = LGBMClassifier(
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
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(X_train, y)

    # Feature importance (top 15)
    importances = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("Top 15 features by importance:")
    print(importances.head(15).to_string())
    print()

    preds_encoded = final_model.predict(X_test)
    preds = enc.inverse_transform(preds_encoded)

    return preds, cv_scores


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------


def write_submission(
    test: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
    id_col: str = "ID",
) -> None:
    """Write submission CSV in ID,class format. Validates against competition rules."""
    submission = pd.DataFrame({
        "ID":    test[id_col].values,
        "class": predictions,
    })
    # Competition rule: ID coverage must be complete
    assert len(submission) == len(test), "Row count mismatch"
    assert submission["ID"].nunique() == len(submission), "Duplicate IDs"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    _log.info("Wrote %d predictions to %s", len(submission), output_path)
    print(f"\nSubmission saved: {output_path.resolve()}")
    print(f"Rows: {len(submission)}")
    print(f"Class distribution:\n{submission['class'].value_counts().to_string()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_submission",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir",    default=str(_DATA), metavar="PATH",
                   help="Competition data directory (default: data/)")
    p.add_argument("--output",      default="data/submission_hybrid_btm.csv",
                   metavar="CSV",   help="Output submission CSV path")
    p.add_argument("--seed",        default=42, type=int)
    p.add_argument("--n-splits",    default=5,  type=int,
                   help="Number of CV folds")
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--quiet",       action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(verbose=args.verbose, quiet=args.quiet)

    data_dir = Path(args.data_dir)
    train_csv  = data_dir / "train.csv"
    test_csv   = data_dir / "test.csv"
    bathy_tif  = data_dir / "MBES" / "bathymetry.tif"
    bscat_tif  = data_dir / "MBES" / "backscatter.tif"
    sample_sub = data_dir / "sample_submission.csv"

    for f in (train_csv, test_csv, bathy_tif, bscat_tif):
        if not f.exists():
            print(f"ERROR: required file not found: {f}", file=sys.stderr)
            return 1

    # ── Load CSV data ─────────────────────────────────────────────────────────
    _log.info("Loading competition data …")
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)

    # train.csv has no ID column — add sequential IDs for feature extraction
    if "ID" not in train.columns:
        train.insert(0, "ID", range(1, len(train) + 1))

    # Validate class vocabulary against sample submission
    if sample_sub.exists():
        allowed = set(pd.read_csv(sample_sub)["class"].unique())
        train_classes = set(train["class"].unique())
        unknown = train_classes - allowed
        if unknown:
            _log.warning("Training labels not in sample_submission: %s", unknown)

    _log.info("Train: %d rows  |  Test: %d rows", len(train), len(test))
    _log.info("Classes: %s", sorted(train["class"].unique()))

    # ── Extract features ───────────────────────────────────────────────────────
    _log.info("Extracting features …")
    try:
        train_feats = extract_features(train, bathy_tif, bscat_tif)
        test_feats  = extract_features(test,  bathy_tif, bscat_tif)
    except Exception as exc:
        _log.exception("Feature extraction failed: %s", exc)
        return 2

    # Feature columns = all numeric except ID, x, y
    exclude = {"ID", "x", "y"}
    feature_cols = [
        c for c in train_feats.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(train_feats[c])
    ]
    _log.info("Using %d features: %s", len(feature_cols), feature_cols)

    # Check NaN coverage
    na_counts = train_feats[feature_cols].isna().sum()
    if na_counts.any():
        _log.warning(
            "NaN counts per feature (will be median-filled):\n%s",
            na_counts[na_counts > 0].to_string(),
        )

    # ── Train + predict ────────────────────────────────────────────────────────
    _log.info("Training hybrid LightGBM model …")
    try:
        predictions, cv_scores = train_and_predict(
            train_feats=train_feats,
            train_labels=train["class"],
            test_feats=test_feats,
            feature_cols=feature_cols,
            seed=args.seed,
            n_splits=args.n_splits,
        )
    except Exception as exc:
        _log.exception("Model training/prediction failed: %s", exc)
        return 3

    # ── Write submission ───────────────────────────────────────────────────────
    write_submission(test, predictions, Path(args.output))

    # Validate class labels against competition vocabulary
    allowed_classes = set(train["class"].unique())
    bad = set(predictions) - allowed_classes
    if bad:
        _log.error("Submission contains unknown class labels: %s", bad)
        return 3

    print(f"\nCV scores: {cv_scores}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
