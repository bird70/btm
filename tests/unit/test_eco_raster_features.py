"""Unit tests for eco raster derivative functions (T025).

Tests use synthetic 5×5 (or larger) elevation arrays with known analytical
solutions to verify correctness of:
  - compute_northness_eastness  (Horn 1981 gradient)
  - compute_max_curvature       (Schmidt 2003 / Evans 1980)
  - compute_complexity          (Wilson 2007 slope-of-slope)

All tests work on pure NumPy arrays — no rasterio, no GIS runtime required.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_dem(n: int = 5, value: float = -10.0) -> np.ndarray:
    """A perfectly flat DEM (constant elevation)."""
    return np.full((n, n), value, dtype=float)


def _tilted_dem(n: int = 7, slope_degs: float = 10.0, cell_size: float = 1.0) -> np.ndarray:
    """A planar DEM sloping east (increasing in x / col direction)."""
    slope_rad = np.radians(slope_degs)
    rise_per_cell = np.tan(slope_rad) * cell_size
    cols = np.arange(n, dtype=float)
    return np.tile(-cols * rise_per_cell, (n, 1))  # rows are identical


def _north_sloping_dem(n: int = 7, slope_degs: float = 10.0) -> np.ndarray:
    """A planar DEM sloping north (increasing in y / row direction)."""
    slope_rad = np.radians(slope_degs)
    rise_per_row = np.tan(slope_rad)
    rows = np.arange(n, dtype=float)
    return np.tile((-rows * rise_per_row)[:, np.newaxis], (1, n))


def _bowl_dem(n: int = 11) -> np.ndarray:
    """A concave bowl — z = x²+y² centred at middle."""
    cx, cy = n // 2, n // 2
    y, x = np.mgrid[0:n, 0:n]
    return -(((x - cx) ** 2 + (y - cy) ** 2)).astype(float)


# ---------------------------------------------------------------------------
# T025a: compute_northness_eastness
# ---------------------------------------------------------------------------


def test_northness_eastness_flat_returns_nan() -> None:
    """Flat DEM → all NaN (no aspect direction defined)."""
    from btm.features.extract import compute_northness_eastness

    dem = _flat_dem()
    northness, eastness = compute_northness_eastness(dem, cell_size=1.0)

    assert northness.shape == dem.shape
    assert eastness.shape == dem.shape
    assert np.all(np.isnan(northness)), "Flat DEM northness should be all NaN"
    assert np.all(np.isnan(eastness)), "Flat DEM eastness should be all NaN"


def test_northness_eastness_east_sloping_dem() -> None:
    """East-sloping DEM → eastness close to -1, northness close to 0 at interior."""
    from btm.features.extract import compute_northness_eastness

    dem = _tilted_dem(n=9, slope_degs=20.0, cell_size=1.0)
    northness, eastness = compute_northness_eastness(dem, cell_size=1.0)

    # Interior cells away from edges should all be nearly identical.
    interior_e = eastness[2:-2, 2:-2]
    interior_n = northness[2:-2, 2:-2]

    assert not np.any(np.isnan(interior_e)), "Interior eastness should not be NaN"
    # Slope descends to east: downslope aspect = east (90°)
    # eastness = sin(90°) = 1 → positive
    assert np.all(interior_e > 0), f"Expected eastness > 0 for east-descending slope, got {interior_e}"
    assert np.allclose(interior_n, 0.0, atol=0.1), (
        f"Expected northness ≈ 0 for east slope, got mean={interior_n.mean():.3f}"
    )


def test_northness_eastness_north_sloping_dem() -> None:
    """North-sloping DEM → northness non-zero, eastness close to 0 at interior."""
    from btm.features.extract import compute_northness_eastness

    dem = _north_sloping_dem(n=9, slope_degs=15.0)
    northness, eastness = compute_northness_eastness(dem, cell_size=1.0)

    interior_e = eastness[2:-2, 2:-2]
    interior_n = northness[2:-2, 2:-2]

    assert not np.any(np.isnan(interior_n))
    assert np.allclose(interior_e, 0.0, atol=0.1), (
        f"Expected eastness ≈ 0 for north slope, got mean={interior_e.mean():.3f}"
    )
    # North slope → positive north gradient → northness non-zero
    assert np.all(np.abs(interior_n) > 0.5), (
        f"Expected |northness| > 0.5 for north slope, got {interior_n}"
    )


def test_northness_eastness_output_range() -> None:
    """Northness and eastness values must lie within [-1, 1] for non-NaN cells."""
    from btm.features.extract import compute_northness_eastness

    rng = np.random.default_rng(42)
    dem = rng.normal(-15, 3, (20, 20))
    northness, eastness = compute_northness_eastness(dem, cell_size=0.5)

    valid_n = northness[~np.isnan(northness)]
    valid_e = eastness[~np.isnan(eastness)]

    assert np.all(valid_n >= -1.0 - 1e-9) and np.all(valid_n <= 1.0 + 1e-9), (
        "Northness must be in [-1, 1]"
    )
    assert np.all(valid_e >= -1.0 - 1e-9) and np.all(valid_e <= 1.0 + 1e-9), (
        "Eastness must be in [-1, 1]"
    )


def test_northness_eastness_returns_correct_shape() -> None:
    """Output arrays must have the same shape as the input DEM."""
    from btm.features.extract import compute_northness_eastness

    dem = np.ones((7, 11)) * -5.0
    dem[3, 5] = -6.0  # single perturbation to avoid all-flat
    n, e = compute_northness_eastness(dem, cell_size=2.0)

    assert n.shape == (7, 11)
    assert e.shape == (7, 11)


# ---------------------------------------------------------------------------
# T025b: compute_max_curvature
# ---------------------------------------------------------------------------


def test_max_curvature_flat_returns_zero_or_nan() -> None:
    """Flat DEM → max curvature should be 0 (or NaN for flat cells)."""
    from btm.features.extract import compute_max_curvature

    dem = _flat_dem(n=7)
    curv = compute_max_curvature(dem, cell_size=1.0)

    assert curv.shape == dem.shape
    # For a flat DEM, curvature is effectively 0; NaN is also acceptable.
    non_nan = curv[~np.isnan(curv)]
    assert np.allclose(non_nan, 0.0, atol=1e-6), (
        f"Flat DEM curvature should be 0, got max={np.nanmax(np.abs(curv)):.2e}"
    )


def test_max_curvature_planar_returns_near_zero() -> None:
    """Planar (tilted) DEM → curvature should be near zero for interior cells."""
    from btm.features.extract import compute_max_curvature

    dem = _tilted_dem(n=11, slope_degs=5.0)
    curv = compute_max_curvature(dem, cell_size=1.0)

    interior = curv[2:-2, 2:-2]
    non_nan = interior[~np.isnan(interior)]
    assert np.allclose(non_nan, 0.0, atol=1e-4), (
        f"Planar DEM curvature should be ~0, got max={np.abs(non_nan).max():.2e}"
    )


def test_max_curvature_bowl_is_positive() -> None:
    """Bowl-shaped DEM (concave up) → curvature > 0 at bowl centre.

    The Hessian eigenvalue approach is used, so zero-slope cells (like the
    bowl centre) never produce NaN.
    """
    from btm.features.extract import compute_max_curvature

    dem = _bowl_dem(n=11)
    curv = compute_max_curvature(dem, cell_size=1.0)

    cx = 5
    # Hessian eigenvalues do not require non-zero slope, so bowl centre is defined.
    assert not np.isnan(curv[cx, cx]), (
        f"Bowl centre curvature must not be NaN (got {curv[cx, cx]}). "
        "Ensure Hessian eigenvalue approach is used (not plan/profile which requires non-zero slope)."
    )
    assert curv[cx, cx] > 0, (
        f"Bowl centre should have positive curvature, got {curv[cx, cx]:.4f}"
    )


def test_max_curvature_non_negative() -> None:
    """max_curvature returns max(|plan|, |profile|) — always ≥ 0."""
    from btm.features.extract import compute_max_curvature

    rng = np.random.default_rng(7)
    dem = rng.normal(-10, 2, (15, 15))
    curv = compute_max_curvature(dem, cell_size=1.0)

    non_nan = curv[~np.isnan(curv)]
    assert np.all(non_nan >= 0), "max_curvature must be non-negative"


# ---------------------------------------------------------------------------
# T025c: compute_complexity
# ---------------------------------------------------------------------------


def test_complexity_flat_returns_near_zero() -> None:
    """Flat DEM → complexity (slope of slope) should be near zero."""
    from btm.features.extract import compute_complexity

    dem = _flat_dem(n=9)
    comp = compute_complexity(dem, cell_size=1.0)

    assert comp.shape == dem.shape
    assert np.allclose(comp, 0.0, atol=1e-6), (
        f"Flat DEM complexity should be 0, got max={comp.max():.2e}"
    )


def test_complexity_planar_returns_near_zero_interior() -> None:
    """Planar DEM → complexity near zero in the interior (uniform slope)."""
    from btm.features.extract import compute_complexity

    dem = _tilted_dem(n=11, slope_degs=8.0)
    comp = compute_complexity(dem, cell_size=1.0)

    interior = comp[2:-2, 2:-2]
    assert np.allclose(interior, 0.0, atol=1e-4), (
        f"Planar DEM interior complexity should be ~0, got max={interior.max():.2e}"
    )


def test_complexity_non_negative() -> None:
    """Complexity (absolute slope-of-slope) must be ≥ 0 everywhere."""
    from btm.features.extract import compute_complexity

    rng = np.random.default_rng(13)
    dem = rng.normal(-8, 3, (20, 20))
    comp = compute_complexity(dem, cell_size=1.0)

    assert np.all(comp >= 0), f"Complexity must be ≥ 0, got min={comp.min():.4f}"


def test_complexity_rough_surface_higher_than_flat() -> None:
    """A rough surface has strictly higher mean complexity than a flat surface."""
    from btm.features.extract import compute_complexity

    flat = _flat_dem(n=15)
    rng = np.random.default_rng(21)
    rough = rng.normal(-10, 2, (15, 15))

    comp_flat = compute_complexity(flat, cell_size=1.0)
    comp_rough = compute_complexity(rough, cell_size=1.0)

    assert comp_rough.mean() > comp_flat.mean(), (
        "Rough surface should have higher mean complexity than flat surface"
    )
