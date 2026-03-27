"""Unit tests for experiment_v5 GIS feature engineering helpers.

Tests cover structural/format invariants only — not model accuracy.

T010: test_add_gis_features_raw_shape
T011: test_gis_feature_weights
T012: test_write_submission_format
T018: test_krige_gis_features_no_nan
T024: test_run_report_contains_all_runs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — scripts/ is not a package; add to sys.path for import
# ---------------------------------------------------------------------------
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from experiment_v5 import (  # noqa: E402, I001
    N_AF_CLUSTERS,
    N_BZ_CLUSTERS,
    WEIGHT_AF,
    WEIGHT_BZ,
    WEIGHT_SLOPE,
    _add_gis_features_raw,
    _krige_gis_features,
    _write_run_report,
    _write_submission,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_transform(n: int = 10):
    """Return an affine transform covering a [0, n] × [0, n] geographic extent."""
    from rasterio.transform import from_bounds

    return from_bounds(west=0, south=0, east=n, north=n, width=n, height=n)


def _pixel_xs_ys(n: int = 10):
    """x, y at pixel centres for the first n pixels along the diagonal."""
    # with from_bounds(0,0,n,n,n,n): pixel (row=r, col=c) centre = (c+0.5, n-r-0.5)
    xs = np.array([c + 0.5 for c in range(n)], dtype=float)
    ys = np.array([n - r - 0.5 for r in range(n)], dtype=float)
    return xs, ys


# ---------------------------------------------------------------------------
# T010 — shape and column names
# ---------------------------------------------------------------------------


def test_add_gis_features_raw_shape():
    """Output has n_base + 14 columns and the expected column names."""
    n = 10
    rng = np.random.RandomState(0)
    transform = _simple_transform(n)
    xs, ys = _pixel_xs_ys(n)

    slope_arr = rng.rand(n, n).astype(np.float32)
    bz_arr = rng.randint(0, N_BZ_CLUSTERS, (n, n), dtype=np.int32)
    af_arr = rng.randint(0, N_AF_CLUSTERS, (n, n), dtype=np.int32)

    base_cols = ["f1", "f2"]
    base_df = pd.DataFrame(np.zeros((n, len(base_cols))), columns=base_cols)
    n_base = len(base_cols)

    result = _add_gis_features_raw(base_df, slope_arr, bz_arr, af_arr, transform, xs, ys)

    assert result.shape == (n, n_base + 14), (
        f"Expected {(n, n_base + 14)}, got {result.shape}"
    )
    assert "gis_slope" in result.columns
    assert "gis_bz_0" in result.columns
    assert f"gis_bz_{N_BZ_CLUSTERS - 1}" in result.columns
    assert "gis_af_0" in result.columns
    assert f"gis_af_{N_AF_CLUSTERS - 1}" in result.columns


# ---------------------------------------------------------------------------
# T011 — weight correctness
# ---------------------------------------------------------------------------


def test_gis_feature_weights():
    """Scalar values: slope×WEIGHT_SLOPE, active OHE col×weight, others 0."""
    n = 5
    transform = _simple_transform(n)
    xs, ys = _pixel_xs_ys(n)

    # Uniform rasters: slope=1.0, bz=2~everywhere, af=5~everywhere
    slope_arr = np.ones((n, n), dtype=np.float32)
    bz_arr = np.full((n, n), 2, dtype=np.int32)
    af_arr = np.full((n, n), 5, dtype=np.int32)

    base_df = pd.DataFrame({"dummy": np.zeros(n)})
    result = _add_gis_features_raw(base_df, slope_arr, bz_arr, af_arr, transform, xs, ys)

    # slope: 1.0 * WEIGHT_SLOPE
    np.testing.assert_allclose(
        result["gis_slope"].values, WEIGHT_SLOPE, rtol=1e-5,
        err_msg="gis_slope should equal 1.0 * WEIGHT_SLOPE"
    )

    # bz=2 → gis_bz_2 == WEIGHT_BZ, all others == 0
    np.testing.assert_allclose(result["gis_bz_2"].values, WEIGHT_BZ, rtol=1e-5)
    for c in range(N_BZ_CLUSTERS):
        if c != 2:
            np.testing.assert_allclose(
                result[f"gis_bz_{c}"].values, 0.0, atol=1e-6,
                err_msg=f"gis_bz_{c} should be 0 when bz=2"
            )

    # af=5 → gis_af_5 == WEIGHT_AF, all others == 0
    np.testing.assert_allclose(result["gis_af_5"].values, WEIGHT_AF, rtol=1e-5)
    for c in range(N_AF_CLUSTERS):
        if c != 5:
            np.testing.assert_allclose(
                result[f"gis_af_{c}"].values, 0.0, atol=1e-6,
                err_msg=f"gis_af_{c} should be 0 when af=5"
            )


# ---------------------------------------------------------------------------
# T012 — submission file format
# ---------------------------------------------------------------------------


def test_write_submission_format(tmp_path):
    """Submission CSV has columns [ID, class], correct row count, no duplicate IDs."""
    pred_labels = np.array(["coral", "sand", "rock", "coral", "sand"])
    test_ids = np.array([101, 102, 103, 104, 105])

    out_path = tmp_path / "test_submission.csv"
    _write_submission(out_path, pred_labels, test_ids, "test_run")

    sub = pd.read_csv(out_path)
    assert list(sub.columns) == ["ID", "class"], f"Expected ['ID','class'], got {list(sub.columns)}"
    assert len(sub) == len(test_ids), f"Row count mismatch: {len(sub)} != {len(test_ids)}"
    assert sub["ID"].nunique() == len(sub), "Duplicate IDs in submission"


# ---------------------------------------------------------------------------
# T018 — kriged output: shape and no NaN
# ---------------------------------------------------------------------------


def test_krige_gis_features_no_nan():
    """Kriging output has shape (n_test, 3) and contains no NaN values."""
    rng = np.random.RandomState(42)
    n_tr = 30
    n_te = 10

    train_df = pd.DataFrame(
        {"x": rng.uniform(0, 100, n_tr), "y": rng.uniform(0, 100, n_tr)}
    )
    test_df = pd.DataFrame(
        {"x": rng.uniform(0, 100, n_te), "y": rng.uniform(0, 100, n_te)}
    )

    slope_tr = rng.uniform(0, 45, n_tr).astype(float)
    bz_tr = rng.randint(0, N_BZ_CLUSTERS, n_tr)
    af_tr = rng.randint(0, N_AF_CLUSTERS, n_tr)
    slope_te_raw = rng.uniform(0, 45, n_te)
    bz_te_raw = rng.randint(0, N_BZ_CLUSTERS, n_te)
    af_te_raw = rng.randint(0, N_AF_CLUSTERS, n_te)

    result = _krige_gis_features(
        train_df, test_df, slope_tr, bz_tr, af_tr, slope_te_raw, bz_te_raw, af_te_raw
    )

    assert result.shape == (n_te, 3), f"Expected ({n_te}, 3), got {result.shape}"
    assert list(result.columns) == ["krige_slope", "krige_bz", "krige_af"]
    assert not result.isna().any().any(), "Kriging result contains NaN values"


# ---------------------------------------------------------------------------
# T024 — run report contains all run names and F1 values
# ---------------------------------------------------------------------------


def test_run_report_contains_all_runs(tmp_path):
    """Report Markdown contains all run names and their F1 values."""
    results = {"baseline": 0.70, "run_a": 0.73, "run_b": 0.75}
    _write_run_report(out_dir=tmp_path, results=results)

    report_path = tmp_path / "run-008-gis-feature-engineering.md"
    assert report_path.exists(), "Report file not created"

    text = report_path.read_text(encoding="utf-8")

    for run_name in ("baseline", "run_a", "run_b"):
        assert run_name in text, f"Run name '{run_name}' not found in report"

    # F1 values formatted as .4f: 0.70 → "0.7000", 0.73 → "0.7300", 0.75 → "0.7500"
    # Substring check works because "0.70" ⊂ "0.7000"
    assert "0.70" in text, "F1 value 0.70 not found in report"
    assert "0.73" in text, "F1 value 0.73 not found in report"
    assert "0.75" in text, "F1 value 0.75 not found in report"
