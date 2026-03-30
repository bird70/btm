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
# Collect-ignore guard for benthic_model tests when dependencies unavailable.
# benthic_model requires pyyaml, scikit-learn, xgboost etc. (the [benthic]
# optional group).  When those are absent we skip collection rather than
# fail with ImportError, so the core BTM test suite stays green.
# ---------------------------------------------------------------------------
_BENTHIC_UNIT_TESTS = [
    "unit/test_config_and_metadata.py",
    "unit/test_cv_protocols.py",
    "unit/test_data_use_restrictions.py",
    "unit/test_experiment_registry.py",
    "unit/test_feature_engineering.py",
    "unit/test_metadata_ingestion.py",
    "unit/test_performance_threshold_policy.py",
    "unit/test_raster_extract.py",
]
_BENTHIC_INTEGRATION_TESTS = [
    "integration/test_experiment_reproducibility.py",
    "integration/test_full_pipeline_smoke.py",
    "integration/test_submission_generation_pipeline.py",
    "integration/test_training_evaluation_pipeline.py",
]
_BENTHIC_CONTRACT_TESTS = ["contract"]

try:
    import yaml  # noqa: F401 — installed by pyyaml, canary for [benthic] extras
except ImportError:
    collect_ignore.extend(_BENTHIC_UNIT_TESTS)
    collect_ignore.extend(_BENTHIC_INTEGRATION_TESTS)
    collect_ignore.extend(_BENTHIC_CONTRACT_TESTS)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "arcgis: requires ArcGIS Pro / arcpy")
    config.addinivalue_line("markers", "qgis: requires QGIS / PyQGIS")
    config.addinivalue_line(
        "markers",
        "benthic: benthic_model ML pipeline tests (requires [benthic] extras)",
    )


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
