from pathlib import Path

import pytest

from benthic_model.data.ingest import parse_and_validate_metadata


def test_parse_and_validate_metadata_with_required_fields(tmp_path: Path) -> None:
    metadata_file = tmp_path / "METADATA.MD"
    metadata_file.write_text(
        "Title: Benthic Habitat Data\n"
        "Description: MBES-derived predictors\n"
        "Source: Kaggle challenge\n"
        "License: Competition terms\n",
        encoding="utf-8",
    )

    parsed = parse_and_validate_metadata(metadata_file)

    assert parsed["title"] == "Benthic Habitat Data"
    assert parsed["source"] == "Kaggle challenge"


def test_parse_and_validate_metadata_raises_on_missing_required_fields(tmp_path: Path) -> None:
    metadata_file = tmp_path / "METADATA.MD"
    metadata_file.write_text(
        "Title: Benthic Habitat Data\nDescription: MBES-derived predictors\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field"):
        parse_and_validate_metadata(metadata_file)
