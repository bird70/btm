import pandas as pd

from benthic_model.segmentation.stacking_features import build_meta_feature_frames


def test_meta_feature_builder_outputs_aligned_frames() -> None:
    train = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [6, 5, 4, 3, 2, 1],
            "class": ["ALG", "NVB", "ALG", "SGAM", "NVB", "ALG"],
        }
    )
    test = pd.DataFrame({"ID": [101, 102], "x": [2.5, 3.5], "y": [4.5, 3.5]})

    val_truth = pd.DataFrame({"location_id": ["1", "2"], "class": ["ALG", "NVB"]})
    seg_val = pd.DataFrame(
        {
            "location_id": ["1", "2"],
            "predicted_class": ["ALG", "NVB"],
            "confidence": [0.7, 0.8],
            "prob_ALG": [0.7, 0.2],
            "prob_FMAT": [0.1, 0.1],
            "prob_NVB": [0.1, 0.6],
            "prob_SGAM": [0.05, 0.05],
            "prob_SGZ": [0.05, 0.05],
        }
    )
    seg_test = pd.DataFrame(
        {
            "location_id": ["101", "102"],
            "predicted_class": ["ALG", "NVB"],
            "confidence": [0.6, 0.6],
            "prob_ALG": [0.6, 0.3],
            "prob_FMAT": [0.1, 0.1],
            "prob_NVB": [0.2, 0.4],
            "prob_SGAM": [0.05, 0.1],
            "prob_SGZ": [0.05, 0.1],
        }
    )

    val_meta, test_meta, y_val = build_meta_feature_frames(
        train,
        test,
        val_truth=val_truth,
        seg_val=seg_val,
        seg_test=seg_test,
    )

    assert len(val_meta) == len(y_val) == 2
    assert len(test_meta) == 2
    assert "seg_confidence" in val_meta.columns
    assert not val_meta.isna().any().any()
