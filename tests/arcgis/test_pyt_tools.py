"""Stub ArcGIS Pro toolbox tests (require @pytest.mark.arcgis, skipped without arcpy)."""

import pytest


@pytest.mark.arcgis
class TestPytTools:
    def test_placeholder(self):
        """Placeholder — real tests require ArcGIS Pro environment."""
        pytest.skip("ArcGIS Pro required")
