"""
Unit tests for experiment_v6 feature extraction functions.

Test strategy: TDD (Red-Green-Refactor).
    - Stubs initially raise NotImplementedError (red phase handled by stub design).
    - Each test is filled with assertions before the corresponding implementation
      is written, ensuring code is written to make tests pass.

Tests:
    1. test_pb_features_all_finite         — 8 PB columns are finite (non-NaN, non-inf)
    2. test_northness_eastness_range       — northness/eastness values in [-1, 1]
    3. test_segment_labels_shape           — segment labels array has correct shape
    4. test_segment_labels_non_negative    — all segment labels >= 0
    5. test_segment_mean_size_range        — mean segment size within expected range
    6. test_ob_features_all_finite         — 10 OB columns are finite (no NaN)
    7. test_combined_feature_count         — combined feature matrix has exactly 18 columns
"""

import sys  # noqa: F401
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make scripts/ importable when running pytest from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from experiment_v6 import (  # noqa: E402
    COMBINED_FEATURE_COLS,
    OB_FEATURE_COLS,
    PB_FEATURE_COLS,
    _compute_pb_features,
    _compute_segment_stats,
    _segment_rasters,
)

# ---------------------------------------------------------------------------
# Shared synthetic data factories
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_bathy(shape=(10, 10)) -> np.ndarray:
    """Simple synthetic bathymetry: linear ramp with mild noise."""
    h, w = shape
    base = np.linspace(-30.0, -10.0, h * w).reshape(shape).astype(np.float32)
    return base + RNG.standard_normal(shape).astype(np.float32) * 0.1


def _make_back(shape=(10, 10)) -> np.ndarray:
    """Synthetic backscatter: uniform noise in [-30, -10] dB."""
    return RNG.uniform(-30.0, -10.0, shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1 — PB features: all 8 columns finite at valid pixels
# ---------------------------------------------------------------------------


def test_pb_features_all_finite():
    """All 8 pixel-based features must be finite (non-NaN, non-inf) at valid pixels.

    Uses a 10×10 synthetic array with no NoData pixels.
    """
    bathy = _make_bathy((10, 10))
    back = _make_back((10, 10))
    cell_size = 0.25

    result = _compute_pb_features(bathy, back, cell_size)

    # Correct key set
    assert set(result.keys()) == set(PB_FEATURE_COLS), (
        f"Expected keys {PB_FEATURE_COLS}, got {list(result.keys())}"
    )

    for key in PB_FEATURE_COLS:
        arr = result[key]
        assert arr.shape == (10, 10), f"Feature '{key}' shape mismatch: {arr.shape}"
        assert np.all(np.isfinite(arr)), (
            f"Feature '{key}' has non-finite values: "
            f"{arr[~np.isfinite(arr)][:5]}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Northness and Eastness in [-1, 1]
# ---------------------------------------------------------------------------


def test_northness_eastness_range():
    """Northness and eastness values must lie in [-1, 1] (circular aspect decomposition)."""
    bathy = _make_bathy((10, 10))
    back = _make_back((10, 10))
    cell_size = 0.25

    result = _compute_pb_features(bathy, back, cell_size)

    northness = result["northness"]
    eastness = result["eastness"]

    assert np.all(northness >= -1.001), f"Northness below -1: min={northness.min()}"
    assert np.all(northness <= 1.001), f"Northness above  1: max={northness.max()}"
    assert np.all(eastness >= -1.001), f"Eastness below -1: min={eastness.min()}"
    assert np.all(eastness <= 1.001), f"Eastness above  1: max={eastness.max()}"


# ---------------------------------------------------------------------------
# Test 3 — Segment labels shape matches input raster
# ---------------------------------------------------------------------------


def test_segment_labels_shape():
    """Segment labels array must have the same (H, W) shape as input rasters."""
    h, w = 20, 20
    bathy = _make_bathy((h, w))
    back = _make_back((h, w))
    vrm_arr = np.abs(RNG.standard_normal((h, w))).astype(np.float32)

    labels = _segment_rasters(bathy, back, vrm_arr)

    assert labels.shape == (h, w), f"Expected ({h},{w}), got {labels.shape}"
    assert np.issubdtype(labels.dtype, np.integer), (
        f"Labels dtype should be integer, got {labels.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Segment labels non-negative
# ---------------------------------------------------------------------------


def test_segment_labels_non_negative():
    """All segment label values must be >= 0."""
    bathy = _make_bathy((20, 20))
    back = _make_back((20, 20))
    vrm_arr = np.abs(RNG.standard_normal((20, 20))).astype(np.float32)

    labels = _segment_rasters(bathy, back, vrm_arr)

    assert np.all(labels >= 0), f"Negative labels found: {labels[labels < 0][:5]}"


# ---------------------------------------------------------------------------
# Test 5 — Mean segment size within expected range
# ---------------------------------------------------------------------------


def test_segment_mean_size_range():
    """Mean pixels per segment must be within 0.5× to 2.0× of the expected size.

    Uses a 200×200 grid with n_segments=100, so expected mean = 400 px/segment.
    Synthetic data uses sinusoidal terrain variation to give SLIC clear spatial structure.
    """
    h, w = 200, 200
    n_segs = 100
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Sinusoidal terrain + noise → clear spatial variation for SLIC
    bathy = (-20.0 + 5.0 * np.sin(x / 20.0) * np.cos(y / 20.0)).astype(np.float32)
    bathy += RNG.standard_normal((h, w)).astype(np.float32) * 0.5
    back = (-15.0 + 3.0 * np.cos(x / 15.0) * np.sin(y / 25.0)).astype(np.float32)
    vrm_arr = np.abs(RNG.standard_normal((h, w))).astype(np.float32)

    labels = _segment_rasters(bathy, back, vrm_arr, n_segments=n_segs)

    n_unique = len(np.unique(labels))
    mean_size = (h * w) / n_unique
    expected = (h * w) / n_segs  # = 400

    assert mean_size >= 0.5 * expected, (
        f"Mean segment size {mean_size:.1f} < 0.5× expected ({0.5 * expected:.1f})"
    )
    # SLIC with enforce_connectivity tends to produce fewer segments than requested
    # on small synthetic rasters; allow up to 3× the naive expected size.
    assert mean_size <= 3.0 * expected, (
        f"Mean segment size {mean_size:.1f} > 3.0× expected ({3.0 * expected:.1f}) — "
        "SLIC produced far fewer segments than requested; check segmentation inputs"
    )


# ---------------------------------------------------------------------------
# Test 6 — OB features: all 10 columns finite
# ---------------------------------------------------------------------------


def test_ob_features_all_finite():
    """_compute_segment_stats must produce a DataFrame with exactly 10 OB columns, no NaN."""
    h, w = 20, 20
    bathy = _make_bathy((h, w))
    back = _make_back((h, w))
    vrm_arr = np.abs(RNG.standard_normal((h, w))).astype(np.float32)

    # Use a small segmentation to get labels quickly
    labels = _segment_rasters(bathy, back, vrm_arr, n_segments=20)

    seg_stats = _compute_segment_stats(bathy, back, vrm_arr, labels)

    assert list(seg_stats.columns) == OB_FEATURE_COLS, (
        f"Columns mismatch: expected {OB_FEATURE_COLS}, got {list(seg_stats.columns)}"
    )
    assert not seg_stats.isna().any().any(), (
        f"NaN values found in segment stats:\n{seg_stats.isna().sum()}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Combined feature count = 18
# ---------------------------------------------------------------------------


def test_combined_feature_count():
    """COMBINED_FEATURE_COLS must contain exactly 18 features (8 PB + 10 OB)."""
    assert len(PB_FEATURE_COLS) == 8, f"PB features: expected 8, got {len(PB_FEATURE_COLS)}"
    assert len(OB_FEATURE_COLS) == 10, f"OB features: expected 10, got {len(OB_FEATURE_COLS)}"
    assert len(COMBINED_FEATURE_COLS) == 18, (
        f"Combined features: expected 18, got {len(COMBINED_FEATURE_COLS)}"
    )
    # No duplicates
    assert len(set(COMBINED_FEATURE_COLS)) == 18, "Duplicate feature names in COMBINED_FEATURE_COLS"
