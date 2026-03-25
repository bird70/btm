"""Pytest configuration and shared fixtures."""

import pathlib

import pytest

# ---------------------------------------------------------------------------
# Collect-ignore guard for arcgis tests when arcpy is unavailable
# ---------------------------------------------------------------------------
collect_ignore = []

try:
    import arcpy  # noqa: F401
except ImportError:
    collect_ignore.append("arcgis")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "arcgis: requires ArcGIS Pro / arcpy")
    config.addinivalue_line("markers", "qgis: requires QGIS / PyQGIS")


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def bathy_raster_path():
    """Path to the clipped bathymetric test raster."""
    p = DATA_DIR / "bathy5m_clip.tif"
    assert p.exists(), f"Test raster not found: {p}"
    return str(p)


@pytest.fixture
def fagatelebay_csv_path():
    """Path to the Fagatele Bay CSV classification dictionary."""
    p = DATA_DIR / "fagatelebay.csv"
    assert p.exists(), f"Test CSV not found: {p}"
    return str(p)


@pytest.fixture
def fagatelebay_xml_path():
    """Path to the Fagatele Bay XML zone classification dictionary."""
    p = DATA_DIR / "fagatelebay_zone.xml"
    assert p.exists(), f"Test XML not found: {p}"
    return str(p)


@pytest.fixture
def tmp_outdir(tmp_path):
    """Temporary output directory, cleaned up automatically."""
    return str(tmp_path)
