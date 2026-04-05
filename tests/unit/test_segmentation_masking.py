import pandas as pd

from benthic_model.segmentation.masking import build_centered_window_mask


def test_centered_window_mask_builds_expected_metadata() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "class": ["ALG", "NVB", "ALG", "SGAM"],
        }
    )
    mask, metadata = build_centered_window_mask(frame, window_size=5)

    assert mask.ndim == 2
    assert metadata["window_size"] == 5
    assert metadata["expansion_rule"] == "centered_5x5"
    assert "ALG" in metadata["class_index_map"]


def test_mask_window_size_must_be_odd() -> None:
    frame = pd.DataFrame({"x": [1.0], "y": [1.0], "class": ["ALG"]})
    try:
        build_centered_window_mask(frame, window_size=4)
    except ValueError as exc:
        assert "odd" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for even window size")
