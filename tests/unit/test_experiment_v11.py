"""Unit tests for experiment_v11 helper functions.

Covers:
- ``_extract_mbes8_features()`` — MBES-8 feature extraction (FR-001, EC-1)
- ``_write_feature_importance()`` — feature importance CSV schema (partial FR-009)
- ``_write_submission()`` — Kaggle submission CSV format (FR-010)
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts"
KNOWN_CLASSES = {"NVB", "FMAT", "SGZ", "ALG", "SGAM"}
SAMPLE_SUBMISSION = (
    Path(__file__).parent.parent.parent / "data" / "sample_submission.csv"
)


def _import_v11():
    spec_path = SCRIPT_DIR / "experiment_v11.py"
    if not spec_path.exists():
        pytest.skip("experiment_v11.py not yet written")
    spec = importlib.util.spec_from_file_location("experiment_v11", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMbes8Extraction:
    """T005: _extract_mbes8_features returns 8 columns, all finite."""

    def test_returns_8_columns_and_correct_rows(self):
        mod = _import_v11()

        rng = np.random.default_rng(42)
        bathy = rng.standard_normal((20, 20)).astype(np.float32) - 50.0
        back = rng.standard_normal((20, 20)).astype(np.float32) * 10.0
        cell_size = 0.25

        # 5 points within bounds (col, row) as xy_coords
        xy_coords = np.array(
            [[5, 5], [10, 10], [15, 15], [3, 7], [12, 18]], dtype=float
        )

        result = mod._extract_mbes8_features(
            bathy, back, cell_size, xy_coords, transform=None
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        expected_cols = [
            "depth",
            "backscatter",
            "slope",
            "vrm",
            "complexity",
            "max_curvature",
            "northness",
            "eastness",
        ]
        assert list(result.columns) == expected_cols
        assert (
            result.isna().sum().sum() == 0
        ), "All values should be finite after NaN fill"

    def test_nan_fill_for_out_of_bounds(self):
        """EC-1: Out-of-bounds points get median-filled values, not NaN."""
        mod = _import_v11()

        rng = np.random.default_rng(42)
        bathy = rng.standard_normal((10, 10)).astype(np.float32) - 50.0
        back = rng.standard_normal((10, 10)).astype(np.float32) * 10.0
        cell_size = 0.25

        # One point in bounds, one out of bounds
        xy_coords = np.array([[5, 5], [100, 100]], dtype=float)

        result = mod._extract_mbes8_features(
            bathy, back, cell_size, xy_coords, transform=None
        )

        assert len(result) == 2
        assert result.isna().sum().sum() == 0, "NaN should be filled with median"


class TestWriteFeatureImportance:
    """T006: _write_feature_importance output schema."""

    def test_output_schema(self, tmp_path):
        mod = _import_v11()

        importances_mean = [0.05, 0.12, 0.0, 0.08]
        importances_std = [0.01, 0.02, 0.0, 0.01]
        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d"]
        out_path = tmp_path / "importance.csv"

        mod._write_feature_importance(
            importances_mean, importances_std, feature_names, str(out_path)
        )

        assert out_path.exists()
        df = pd.read_csv(out_path)
        required_cols = {"feature_name", "importance_mean", "importance_std", "rank"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) == len(feature_names)

    def test_rank_is_sequential(self, tmp_path):
        mod = _import_v11()
        out_path = tmp_path / "imp.csv"
        mod._write_feature_importance(
            [0.1, 0.3, 0.2], [0.0, 0.0, 0.0], ["a", "b", "c"], str(out_path)
        )
        df = pd.read_csv(out_path)
        assert set(df["rank"]) == {1, 2, 3}


class TestWriteSubmission:
    """T007: _write_submission output format matches sample_submission.csv."""

    def _get_sample_columns(self):
        with open(SAMPLE_SUBMISSION) as f:
            reader = csv.reader(f)
            return next(reader)

    def test_columns_match_sample_submission(self, tmp_path):
        mod = _import_v11()
        expected_cols = self._get_sample_columns()
        n = 98
        predictions = pd.DataFrame(
            {expected_cols[0]: range(1, n + 1), expected_cols[1]: ["NVB"] * n}
        )
        out_path = tmp_path / "submission.csv"
        mod._write_submission(predictions, str(out_path))

        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert list(df.columns) == expected_cols

    def test_all_class_values_valid(self, tmp_path):
        mod = _import_v11()
        expected_cols = self._get_sample_columns()
        predictions = pd.DataFrame(
            {
                expected_cols[0]: range(1, 6),
                expected_cols[1]: ["NVB", "FMAT", "SGZ", "ALG", "SGAM"],
            }
        )
        out_path = tmp_path / "sub.csv"
        mod._write_submission(predictions, str(out_path))
        df = pd.read_csv(out_path)
        assert set(df[expected_cols[1]]).issubset(KNOWN_CLASSES)

    def test_row_count_matches_input(self, tmp_path):
        mod = _import_v11()
        expected_cols = self._get_sample_columns()
        n = 98
        predictions = pd.DataFrame(
            {expected_cols[0]: range(1, n + 1), expected_cols[1]: ["FMAT"] * n}
        )
        out_path = tmp_path / "sub98.csv"
        mod._write_submission(predictions, str(out_path))
        df = pd.read_csv(out_path)
        assert len(df) == n
