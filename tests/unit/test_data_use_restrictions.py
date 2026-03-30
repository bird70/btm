from pathlib import Path

import pytest

from benthic_model.data.ingest import enforce_competition_data_whitelist


def test_whitelist_accepts_allowed_competition_artifact(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    mbes = data_root / "MBES"
    mbes.mkdir(parents=True)
    target = mbes / "bathymetry.tif"
    target.write_text("stub", encoding="utf-8")

    resolved = enforce_competition_data_whitelist(target, data_root)

    assert resolved == target.resolve()


def test_whitelist_rejects_non_competition_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    rogue = data_root / "notes.txt"
    rogue.write_text("not allowed", encoding="utf-8")

    with pytest.raises(ValueError, match="not an allowed competition artifact"):
        enforce_competition_data_whitelist(rogue, data_root)
