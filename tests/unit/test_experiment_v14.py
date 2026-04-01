"""Unit tests for experiment_v14_seed_ensemble helper functions.

Covers (TDD — F5 remediation):
- ``majority_vote()`` — unanimous 5/5, split 3/2, tiebreaker 2/2/1 using highest-CV seed
- ``build_ensemble()`` — vote matrix to ensemble DataFrame
- ``main()`` — dry-run flag exits 0 and reports file count
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts"
KNOWN_CLASSES = {"NVB", "FMAT", "SGZ", "ALG", "SGAM"}


def _import_v14():
    spec_path = SCRIPT_DIR / "experiment_v14_seed_ensemble.py"
    if not spec_path.exists():
        pytest.skip("experiment_v14_seed_ensemble.py not yet written")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("experiment_v14_seed_ensemble", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEED_CV_FIXTURE = {42: 0.8024, 123: 0.7626, 456: 0.7890, 789: 0.8101, 2026: 0.7815}


class TestMajorityVote:
    """Tests for majority_vote() function."""

    def test_unanimous_five_votes(self):
        mod = _import_v14()
        result = mod.majority_vote(["NVB", "NVB", "NVB", "NVB", "NVB"])
        assert result == "NVB"

    def test_split_three_two(self):
        mod = _import_v14()
        result = mod.majority_vote(["ALG", "ALG", "ALG", "NVB", "NVB"])
        assert result == "ALG"

    def test_split_three_two_minority_wins(self):
        mod = _import_v14()
        result = mod.majority_vote(["NVB", "NVB", "ALG", "ALG", "ALG"])
        assert result == "ALG"

    def test_tiebreaker_uses_highest_cv_seed(self):
        """With votes [ALG, NVB, ALG, NVB, FMAT], top seeds by CV are 789(idx=3)=NVB then 42(idx=0)=ALG.
        ALG and NVB each have 2 votes; FMAT has 1.
        Tiebreaker: seed 789 (CV=0.8101, highest) → NVB wins.
        Seed ordering in SEED_RUN_IDS: 42, 123, 456, 789, 2026 → indices 0-4.
        votes[3] = NVB → NVB wins.
        """
        mod = _import_v14()
        # votes ordered by seed: 42=ALG, 123=NVB, 456=ALG, 789=NVB, 2026=FMAT
        votes = ["ALG", "NVB", "ALG", "NVB", "FMAT"]
        result = mod.majority_vote(votes, seed_cv=SEED_CV_FIXTURE)
        assert result == "NVB"  # seed=789 (idx=3) is highest CV and voted NVB

    def test_all_classes_considered(self):
        """Ensure the function handles all 5 allowed classes."""
        mod = _import_v14()
        for cls in KNOWN_CLASSES:
            result = mod.majority_vote([cls] * 5)
            assert result == cls

    def test_single_vote_wins(self):
        """3-1-1 split: NVB wins with 3 votes."""
        mod = _import_v14()
        result = mod.majority_vote(["NVB", "NVB", "NVB", "ALG", "FMAT"])
        assert result == "NVB"


class TestBuildEnsemble:
    """Tests for build_ensemble() function."""

    def test_unanimous_predictions(self):
        mod = _import_v14()
        vote_matrix = pd.DataFrame(
            {42: ["NVB", "ALG"], 123: ["NVB", "ALG"], 456: ["NVB", "ALG"],
             789: ["NVB", "ALG"], 2026: ["NVB", "ALG"]},
            index=[1, 2],
        )
        vote_matrix.index.name = "ID"
        result = mod.build_ensemble(vote_matrix, SEED_CV_FIXTURE)
        assert result.iloc[0]["class"] == "NVB"
        assert result.iloc[1]["class"] == "ALG"
        assert result.iloc[0]["votes"] == 5
        assert result.iloc[1]["votes"] == 5

    def test_output_has_required_columns(self):
        mod = _import_v14()
        vote_matrix = pd.DataFrame(
            {42: ["NVB"], 123: ["NVB"], 456: ["ALG"], 789: ["ALG"], 2026: ["NVB"]},
            index=[1],
        )
        vote_matrix.index.name = "ID"
        result = mod.build_ensemble(vote_matrix, SEED_CV_FIXTURE)
        assert set(result.columns) >= {"ID", "class", "votes", "total"}

    def test_vote_count_correct(self):
        mod = _import_v14()
        vote_matrix = pd.DataFrame(
            {42: ["NVB"], 123: ["NVB"], 456: ["ALG"], 789: ["NVB"], 2026: ["ALG"]},
            index=[1],
        )
        vote_matrix.index.name = "ID"
        result = mod.build_ensemble(vote_matrix, SEED_CV_FIXTURE)
        assert result.iloc[0]["class"] == "NVB"
        assert result.iloc[0]["votes"] == 3
        assert result.iloc[0]["total"] == 5


class TestDryRun:
    """Tests for main() --dry-run flag."""

    def test_dry_run_exits_zero_when_files_missing(self, tmp_path):
        mod = _import_v14()
        # Pass non-existent predictions dir — dry-run reports MISSING but exits 0
        with patch.object(mod, "PREDICTIONS_DIR", tmp_path):
            exit_code = mod.main(["--dry-run", "--predictions-dir", str(tmp_path)])
        assert exit_code == 0

    def test_dry_run_reports_file_count(self, tmp_path, capsys):
        mod = _import_v14()
        exit_code = mod.main(["--dry-run", "--predictions-dir", str(tmp_path)])
        captured = capsys.readouterr()
        assert "0/" in captured.out or "Found" in captured.out
        assert exit_code == 0
