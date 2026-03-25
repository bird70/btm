"""Numerical parity tests against BTM 3.0 reference outputs.

Reference .npy files are expected in tests/data/reference/.
If they do not exist yet, these tests are automatically skipped.
Generate references using T032 (see tasks.md).
"""

import pathlib

import numpy as np
import pytest

from btm.core.model import run_full_model

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
REF_DIR = DATA_DIR / "reference"


def _ref(name: str) -> pathlib.Path:
    return REF_DIR / name


def _skip_if_missing(path: pathlib.Path) -> None:
    if not path.exists():
        pytest.skip(f"Reference file not yet generated: {path.name}")


@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    bathy = str(DATA_DIR / "bathy5m_clip.tif")
    classdict = str(DATA_DIR / "fagatelebay.csv")
    return run_full_model(
        bathy,
        broad_bpi_inner=10, broad_bpi_outer=30,
        fine_bpi_inner=1, fine_bpi_outer=5,
        classification_file=classdict,
        outdir=str(tmp),
    )


class TestNumericalParity:
    def test_broad_bpi_within_tolerance(self, pipeline_outputs):
        import rasterio

        ref_path = _ref("broad_bpi_ref.npy")
        _skip_if_missing(ref_path)
        ref = np.load(str(ref_path))
        with rasterio.open(pipeline_outputs["broad_bpi"]) as src:
            actual = src.read(1)
        np.testing.assert_allclose(actual, ref, atol=1)

    def test_fine_bpi_within_tolerance(self, pipeline_outputs):
        import rasterio

        ref_path = _ref("fine_bpi_ref.npy")
        _skip_if_missing(ref_path)
        ref = np.load(str(ref_path))
        with rasterio.open(pipeline_outputs["fine_bpi"]) as src:
            actual = src.read(1)
        np.testing.assert_allclose(actual, ref, atol=1)

    def test_slope_within_tolerance(self, pipeline_outputs):
        import rasterio

        ref_path = _ref("slope_ref.npy")
        _skip_if_missing(ref_path)
        ref = np.load(str(ref_path))
        with rasterio.open(pipeline_outputs["slope"]) as src:
            actual = src.read(1)
        np.testing.assert_allclose(actual, ref, atol=0.01)

    def test_classified_zones_95pct_agreement(self, pipeline_outputs):
        import rasterio

        ref_path = _ref("classified_zones_ref.npy")
        _skip_if_missing(ref_path)
        ref = np.load(str(ref_path))
        with rasterio.open(pipeline_outputs["classified_zones"]) as src:
            actual = src.read(1)
        agreement = np.mean(actual == ref)
        assert agreement >= 0.95, f"Classification agreement {agreement:.1%} < 95%"
