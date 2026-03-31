"""Unit tests for experiment_v10 helper functions (FR-006, FR-009).

These tests cover the two private helpers:
- ``_write_feature_importance()`` — feature importance CSV output schema
- ``_write_submission()`` — Kaggle submission CSV format validation
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# We import the helper functions from experiment_v10 at test time.
# If the script doesn't exist yet, the tests are skipped gracefully.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts"
KNOWN_CLASSES = {"NVB", "FMAT", "SGZ", "ALG", "SGAM"}
SAMPLE_SUBMISSION = (
    Path(__file__).parent.parent.parent / "data" / "sample_submission.csv"
)


def _import_v10():
    spec_path = SCRIPT_DIR / "experiment_v10.py"
    if not spec_path.exists():
        pytest.skip("experiment_v10.py not yet written")
    spec = importlib.util.spec_from_file_location("experiment_v10", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestWriteFeatureImportance:
    def test_output_schema(self, tmp_path):
        """_write_feature_importance must produce a CSV with the required schema."""
        mod = _import_v10()

        importances_mean = [0.05, 0.12, 0.0, 0.08]
        importances_std = [0.01, 0.02, 0.0, 0.01]
        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d"]
        out_path = tmp_path / "importance.csv"

        mod._write_feature_importance(
            importances_mean, importances_std, feature_names, str(out_path)
        )

        assert out_path.exists()
        df = pd.read_csv(out_path)
        required_cols = {
            "feature_name",
            "importance_mean",
            "importance_std",
            "rank",
            "selected",
        }
        assert required_cols.issubset(set(df.columns))
        assert len(df) == len(feature_names)

    def test_rank_is_sequential(self, tmp_path):
        mod = _import_v10()
        out_path = tmp_path / "imp.csv"
        mod._write_feature_importance(
            [0.1, 0.3, 0.2], [0.0, 0.0, 0.0], ["a", "b", "c"], str(out_path)
        )
        df = pd.read_csv(out_path)
        assert set(df["rank"]) == {1, 2, 3}


class TestWriteSubmission:
    def _get_sample_columns(self):
        with open(SAMPLE_SUBMISSION) as f:
            reader = csv.reader(f)
            return next(reader)

    def test_columns_match_sample_submission(self, tmp_path):
        """_write_submission must produce a CSV with identical column names."""
        mod = _import_v10()
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
        """All predicted classes in submission must belong to the known class set."""
        mod = _import_v10()
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
        """Output row count must equal the number of rows in the predictions DataFrame."""
        mod = _import_v10()
        expected_cols = self._get_sample_columns()
        n = 98
        predictions = pd.DataFrame(
            {expected_cols[0]: range(1, n + 1), expected_cols[1]: ["FMAT"] * n}
        )
        out_path = tmp_path / "sub98.csv"
        mod._write_submission(predictions, str(out_path))
        df = pd.read_csv(out_path)
        assert len(df) == n
