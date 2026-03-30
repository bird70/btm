from pathlib import Path

import pandas as pd
import pytest

from benthic_model.submission.writer import validate_submission_predictions


def test_validate_submission_predictions_rejects_duplicate_ids(tmp_path: Path) -> None:
    pred = pd.DataFrame({"ID": [1, 1], "class": ["ALG", "NVB"]})
    sample = pd.DataFrame({"ID": [1, 2], "class": ["ALG", "ALG"]})

    with pytest.raises(ValueError, match="Duplicate IDs"):
        validate_submission_predictions(pred, sample, {"ALG", "NVB"})


def test_validate_submission_predictions_rejects_unknown_class(tmp_path: Path) -> None:
    pred = pd.DataFrame({"ID": [1, 2], "class": ["ALG", "UNKNOWN"]})
    sample = pd.DataFrame({"ID": [1, 2], "class": ["ALG", "ALG"]})

    with pytest.raises(ValueError, match="unknown classes"):
        validate_submission_predictions(pred, sample, {"ALG", "NVB"})


def test_validate_submission_predictions_accepts_valid_schema() -> None:
    pred = pd.DataFrame({"ID": [1, 2], "class": ["ALG", "NVB"]})
    sample = pd.DataFrame({"ID": [1, 2], "class": ["ALG", "ALG"]})

    validate_submission_predictions(pred, sample, {"ALG", "NVB"})
