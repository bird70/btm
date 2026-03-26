"""
Spatial Ensemble submission for the GeoHab 2026 MLWG Kaggle Competition.

Approach   40 % spatial KNN  +  40 % CatBoost  +  20 % LightGBM
           probability-weighted blend (softmax voting)

Key insight
-----------
Benthic habitat classes exhibit very strong spatial autocorrelation
(Tobler's First Law of Geography).  A distance-weighted K-Nearest-Neighbours
classifier on raw (x, y) coordinates alone achieves ~0.74 weighted-F1 in
spatial-block cross-validation — competitive with gradient-boosting on 80
terrain features.  Blending the two approaches combines spatial interpolation
power with terrain-morphology generalisation.

Feature set  (80 features)
--------------------------
- (x, y) coordinates
- Raw bathymetry + backscatter at point
- Multi-scale focal statistics (8 radii, 0.5 m – 100 m):
    focal mean, focal std, TPI for both bathymetry and backscatter
- Multi-scale curvature (5 Gaussian sigmas, 1 – 16 cells):
    mean curvature, Gaussian curvature, smoothed slope
- BTM terrain derivatives: slope (Horn 1981), VRM (Sappington 2007)
- Aspect (sin, cos)
- Relative backscatter at 4 scales
- Backscatter gradient magnitude
- Interaction features: depth×backscatter, slope×backscatter,
  acoustic hardness, VRM×backscatter

Cross-validation
----------------
10-fold spatial-block CV (KMeans on coordinates) gives weighted-F1 ≈ 0.75,
a realistic estimate for spatially-structured test data.

Standard stratified K-Fold massively over-estimates (~0.99) due to spatial
leakage and must NOT be used for model selection.

Output
------
  data/submission_v2.csv   — ready for Kaggle upload in ID,class format

Exit codes: 0 success, 1 input error, 2 feature-extraction error, 3 model error
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter, uniform_filter

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from btm._logging import configure_cli_logging, get_logger  # noqa: E402
from btm.core.slope import compute_slope  # noqa: E402
from btm.core.vrm import compute_vrm  # noqa: E402
from btm.features.extract import sample_raster_at_points  # noqa: E402

_log = get_logger(__name__)

_DATA = _REPO / "data"


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------


def _load_raster(path: Path) -> tuple[np.ndarray, object, float | None]:
    """Return (array, transform, nodata) with NaN in place of nodata."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform, nodata


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    xs: np.ndarray,
    ys: np.ndarray,
    bathy: np.ndarray,
    back: np.ndarray,
    bathy_f: np.ndarray,
    back_f: np.ndarray,
    bt,
    bkt,
    cell_size: float,
) -> pd.DataFrame:
    """Build the full 80-feature matrix for a set of (x, y) points."""
    feats: dict[str, np.ndarray] = {}

    # ── Coordinates ──────────────────────────────────────────────────────────
    feats["x"] = xs.copy()
    feats["y"] = ys.copy()

    # ── Raw raster values ────────────────────────────────────────────────────
    _log.info("Sampling raw bathymetry and backscatter …")
    feats["bathy"] = sample_raster_at_points(
        np.nan_to_num(bathy, nan=-10000), bt, xs, ys, nodata=-10000,
    )
    feats["back"] = sample_raster_at_points(
        np.nan_to_num(back, nan=-10000), bkt, xs, ys, nodata=-10000,
    )

    # ── BTM derivatives ──────────────────────────────────────────────────────
    _log.info("Computing BTM slope (Horn 1981) …")
    slope_arr = compute_slope(bathy_f, cell_size, nodata=None)
    feats["slope"] = sample_raster_at_points(slope_arr, bt, xs, ys)

    _log.info("Computing BTM VRM (Sappington 2007) …")
    vrm_arr = compute_vrm(bathy_f, neighborhood_size=3, cell_size=cell_size)
    feats["vrm"] = sample_raster_at_points(vrm_arr, bt, xs, ys)

    # ── Multi-scale focal statistics ─────────────────────────────────────────
    radii_m = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
    _log.info("Computing focal stats at %d radii …", len(radii_m))
    for r in radii_m:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        for arr, prefix in [(bathy_f, "b_"), (back_f, "k_")]:
            fm = uniform_filter(arr, size=size)
            fsq = uniform_filter(arr ** 2, size=size)
            fstd = np.sqrt(np.maximum(fsq - fm ** 2, 0))
            tpi = arr - fm
            feats[f"{prefix}fm_{size}"] = sample_raster_at_points(fm, bt, xs, ys)
            feats[f"{prefix}std_{size}"] = sample_raster_at_points(fstd, bt, xs, ys)
            feats[f"{prefix}tpi_{size}"] = sample_raster_at_points(tpi, bt, xs, ys)

    # ── Multi-scale curvature ────────────────────────────────────────────────
    _log.info("Computing curvature at 5 Gaussian scales …")
    for sigma in [1, 2, 4, 8, 16]:
        smooth = gaussian_filter(bathy_f, sigma=sigma)
        dx = np.gradient(smooth, cell_size, axis=1)
        dy = np.gradient(smooth, cell_size, axis=0)
        d2x = np.gradient(dx, cell_size, axis=1)
        d2y = np.gradient(dy, cell_size, axis=0)
        dxy = np.gradient(dx, cell_size, axis=0)
        feats[f"mcurv_s{sigma}"] = sample_raster_at_points(
            (d2x + d2y) / 2, bt, xs, ys,
        )
        feats[f"gcurv_s{sigma}"] = sample_raster_at_points(
            d2x * d2y - dxy ** 2, bt, xs, ys,
        )
        feats[f"mslope_s{sigma}"] = sample_raster_at_points(
            np.sqrt(dx ** 2 + dy ** 2), bt, xs, ys,
        )

    # ── Aspect ───────────────────────────────────────────────────────────────
    dx = np.gradient(bathy_f, cell_size, axis=1)
    dy = np.gradient(bathy_f, cell_size, axis=0)
    aspect = np.arctan2(-dy, dx)
    feats["asp_sin"] = sample_raster_at_points(np.sin(aspect), bt, xs, ys)
    feats["asp_cos"] = sample_raster_at_points(np.cos(aspect), bt, xs, ys)

    # ── Relative backscatter ─────────────────────────────────────────────────
    _log.info("Computing relative backscatter …")
    for r in [2.5, 10.0, 25.0, 50.0]:
        n = max(1, round(r / cell_size))
        size = 2 * n + 1
        local_mean = uniform_filter(back_f, size=size)
        feats[f"rel_back_{size}"] = sample_raster_at_points(
            back_f - local_mean, bkt, xs, ys,
        )

    # ── Backscatter gradient magnitude ───────────────────────────────────────
    bk_dx = np.gradient(back_f, cell_size, axis=1)
    bk_dy = np.gradient(back_f, cell_size, axis=0)
    feats["back_grad"] = sample_raster_at_points(
        np.sqrt(bk_dx ** 2 + bk_dy ** 2), bkt, xs, ys,
    )

    # ── Interaction features ─────────────────────────────────────────────────
    feats["depth_x_back"] = np.abs(feats["bathy"]) * feats["back"]
    feats["slope_x_back"] = feats["slope"] * feats["back"]
    feats["acoustic_hard"] = feats["back"] / (np.abs(feats["bathy"]) + 1.0)
    feats["vrm_x_back"] = feats["vrm"] * feats["back"]

    df = pd.DataFrame(feats)
    _log.info("Feature extraction complete: %d points × %d features", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Ensemble training + prediction
# ---------------------------------------------------------------------------


def train_and_predict(
    train_feats: pd.DataFrame,
    train_labels: pd.Series,
    train_coords: np.ndarray,
    test_feats: pd.DataFrame,
    test_coords: np.ndarray,
    feature_cols: list[str],
    seed: int = 42,
    n_cv_blocks: int = 10,
    blend_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> tuple[np.ndarray, dict]:
    """
    Spatial ensemble: KNN(5) + CatBoost + LightGBM probability blend.

    Returns (predictions, cv_info).
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.cluster import KMeans
    from sklearn.metrics import classification_report, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder

    enc = LabelEncoder()
    y = enc.fit_transform(train_labels.astype(str))
    n_classes = len(enc.classes_)
    _log.info("Classes: %s", list(enc.classes_))

    X_tr = train_feats[feature_cols].fillna(train_feats[feature_cols].median())
    X_te = test_feats[feature_cols].fillna(train_feats[feature_cols].median())
    X_tr = X_tr.replace([np.inf, -np.inf], 0)
    X_te = X_te.replace([np.inf, -np.inf], 0)

    w_knn, w_cb, w_lgbm = blend_weights

    # ── Spatial block CV ─────────────────────────────────────────────────────
    _log.info("Running %d-block spatial CV …", n_cv_blocks)
    km = KMeans(n_clusters=n_cv_blocks, random_state=seed, n_init=10)
    blocks = km.fit_predict(train_coords)

    oof_knn = np.zeros((len(y), n_classes))
    oof_cb = np.zeros((len(y), n_classes))
    oof_lgbm = np.zeros((len(y), n_classes))

    for b in range(n_cv_blocks):
        va = np.where(blocks == b)[0]
        tr = np.where(blocks != b)[0]

        # KNN on coordinates
        knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
        knn.fit(train_coords[tr], y[tr])
        oof_knn[va] = knn.predict_proba(train_coords[va])

        # CatBoost on features
        cb = CatBoostClassifier(
            iterations=1000, depth=6, learning_rate=0.02,
            l2_leaf_reg=10.0, random_seed=seed, verbose=0,
            auto_class_weights="Balanced",
        )
        cb.fit(X_tr.iloc[tr], y[tr])
        oof_cb[va] = cb.predict_proba(X_tr.iloc[va])

        # LightGBM on features
        lgbm = LGBMClassifier(
            n_estimators=1000, max_depth=8, learning_rate=0.02,
            num_leaves=63, subsample=0.7, colsample_bytree=0.5,
            min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
        )
        lgbm.fit(X_tr.iloc[tr], y[tr])
        oof_lgbm[va] = lgbm.predict_proba(X_tr.iloc[va])

        # Per-block blend score
        blend_va = w_knn * oof_knn[va] + w_cb * oof_cb[va] + w_lgbm * oof_lgbm[va]
        va_preds = enc.classes_[np.argmax(blend_va, axis=1)]
        f1_va = f1_score(enc.inverse_transform(y[va]), va_preds, average="weighted")
        _log.info("  block %d (%d pts): blend weighted-F1 = %.4f", b, len(va), f1_va)

    # OOF scores
    oof_blend = w_knn * oof_knn + w_cb * oof_cb + w_lgbm * oof_lgbm
    oof_preds = enc.classes_[np.argmax(oof_blend, axis=1)]
    oof_f1 = f1_score(enc.inverse_transform(y), oof_preds, average="weighted")

    # Per-model OOF
    f1_knn = f1_score(y, np.argmax(oof_knn, 1), average="weighted")
    f1_cb = f1_score(y, np.argmax(oof_cb, 1), average="weighted")
    f1_lgbm = f1_score(y, np.argmax(oof_lgbm, 1), average="weighted")

    _log.info("OOF KNN(5):    %.4f", f1_knn)
    _log.info("OOF CatBoost:  %.4f", f1_cb)
    _log.info("OOF LightGBM:  %.4f", f1_lgbm)
    _log.info("OOF Blend:     %.4f", oof_f1)

    print(f"\n{'=' * 60}")
    print(f"Spatial-block CV results  (blend {w_knn:.0%} KNN + {w_cb:.0%} CB + {w_lgbm:.0%} LGBM)")
    print(f"{'=' * 60}")
    print(f"  KNN(5) on (x,y):  {f1_knn:.4f}")
    print(f"  CatBoost:         {f1_cb:.4f}")
    print(f"  LightGBM:         {f1_lgbm:.4f}")
    print(f"  Ensemble blend:   {oof_f1:.4f}")
    print()
    print(classification_report(enc.inverse_transform(y), oof_preds))
    print(f"{'=' * 60}\n")

    # ── Train final models on ALL data ───────────────────────────────────────
    _log.info("Training final models on all %d training samples …", len(y))

    knn_final = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn_final.fit(train_coords, y)

    cb_final = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.02,
        l2_leaf_reg=10.0, random_seed=seed, verbose=0,
        auto_class_weights="Balanced",
    )
    cb_final.fit(X_tr, y)

    lgbm_final = LGBMClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.02,
        num_leaves=63, subsample=0.7, colsample_bytree=0.5,
        min_child_samples=15, reg_alpha=2.0, reg_lambda=10.0,
        class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
    )
    lgbm_final.fit(X_tr, y)

    # ── Predict test ─────────────────────────────────────────────────────────
    p_knn = knn_final.predict_proba(test_coords)
    p_cb = cb_final.predict_proba(X_te)
    p_lgbm = lgbm_final.predict_proba(X_te)
    test_blend = w_knn * p_knn + w_cb * p_cb + w_lgbm * p_lgbm
    preds = enc.inverse_transform(np.argmax(test_blend, axis=1))

    # Feature importance from final LightGBM
    imp = pd.Series(
        lgbm_final.feature_importances_, index=feature_cols,
    ).sort_values(ascending=False)
    print("Top 20 features (LightGBM importance):")
    print(imp.head(20).to_string())
    print()

    cv_info = {
        "oof_f1_knn": f1_knn,
        "oof_f1_catboost": f1_cb,
        "oof_f1_lgbm": f1_lgbm,
        "oof_f1_blend": oof_f1,
        "blend_weights": blend_weights,
        "n_cv_blocks": n_cv_blocks,
    }
    return preds, cv_info


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------


def write_submission(
    test: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """Write submission CSV in ID,class format."""
    submission = pd.DataFrame({"ID": test["ID"].values, "class": predictions})
    assert len(submission) == len(test), "Row count mismatch"
    assert submission["ID"].nunique() == len(submission), "Duplicate IDs"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    _log.info("Wrote %d predictions to %s", len(submission), output_path)
    print(f"Submission saved: {output_path.resolve()}")
    print(f"Rows: {len(submission)}")
    print(f"Class distribution:\n{submission['class'].value_counts().to_string()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_submission_v2",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", default=str(_DATA), metavar="PATH",
        help="Competition data directory (default: data/)",
    )
    p.add_argument(
        "--output", default="data/submission_v2.csv", metavar="CSV",
        help="Output submission CSV path",
    )
    p.add_argument("--seed", default=42, type=int)
    p.add_argument(
        "--n-cv-blocks", default=10, type=int,
        help="Number of spatial CV blocks (default: 10)",
    )
    p.add_argument(
        "--blend", default="0.4,0.4,0.2", metavar="W",
        help="KNN,CatBoost,LightGBM blend weights (default: 0.4,0.4,0.2)",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(verbose=args.verbose, quiet=args.quiet)

    blend_weights = tuple(float(w) for w in args.blend.split(","))
    if len(blend_weights) != 3 or abs(sum(blend_weights) - 1.0) > 0.01:
        print("ERROR: --blend must be 3 comma-separated weights summing to 1.0",
              file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    bathy_tif = data_dir / "MBES" / "bathymetry.tif"
    bscat_tif = data_dir / "MBES" / "backscatter.tif"

    for f in (train_csv, test_csv, bathy_tif, bscat_tif):
        if not f.exists():
            print(f"ERROR: required file not found: {f}", file=sys.stderr)
            return 1

    # ── Load data ─────────────────────────────────────────────────────────────
    _log.info("Loading competition data …")
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    if "ID" not in train.columns:
        train.insert(0, "ID", range(1, len(train) + 1))

    _log.info("Train: %d rows  |  Test: %d rows", len(train), len(test))
    _log.info("Classes: %s", sorted(train["class"].unique()))

    # ── Load rasters ──────────────────────────────────────────────────────────
    bathy, bt, _ = _load_raster(bathy_tif)
    back, bkt, _ = _load_raster(bscat_tif)
    cell_size = abs(bt.a)
    bathy_f = np.nan_to_num(bathy, nan=0.0)
    back_f = np.nan_to_num(back, nan=0.0)

    # ── Extract features ──────────────────────────────────────────────────────
    _log.info("Extracting training features …")
    try:
        train_feats = extract_features(
            train["x"].values, train["y"].values,
            bathy, back, bathy_f, back_f, bt, bkt, cell_size,
        )
        _log.info("Extracting test features …")
        test_feats = extract_features(
            test["x"].values, test["y"].values,
            bathy, back, bathy_f, back_f, bt, bkt, cell_size,
        )
    except Exception as exc:
        _log.exception("Feature extraction failed: %s", exc)
        return 2

    feature_cols = [
        c for c in train_feats.columns
        if pd.api.types.is_numeric_dtype(train_feats[c])
    ]
    _log.info("Using %d features", len(feature_cols))

    # ── Train + predict ───────────────────────────────────────────────────────
    try:
        predictions, cv_info = train_and_predict(
            train_feats=train_feats,
            train_labels=train["class"],
            train_coords=train[["x", "y"]].values,
            test_feats=test_feats,
            test_coords=test[["x", "y"]].values,
            feature_cols=feature_cols,
            seed=args.seed,
            n_cv_blocks=args.n_cv_blocks,
            blend_weights=blend_weights,
        )
    except Exception as exc:
        _log.exception("Model training/prediction failed: %s", exc)
        return 3

    # ── Write submission ──────────────────────────────────────────────────────
    write_submission(test, predictions, Path(args.output))

    # Validate class labels
    allowed = set(train["class"].unique())
    bad = set(predictions) - allowed
    if bad:
        _log.error("Submission contains unknown class labels: %s", bad)
        return 3

    print(f"\nCV info: {cv_info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
