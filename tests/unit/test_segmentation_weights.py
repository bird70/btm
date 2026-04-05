from benthic_model.segmentation.weights import inverse_freq_capped


def test_inverse_frequency_weights_are_capped() -> None:
    labels = ["ALG"] * 20 + ["SGAM"] * 1
    weights = inverse_freq_capped(labels, cap=4.0)

    assert set(weights) == {"ALG", "SGAM"}
    assert weights["SGAM"] <= 4.0
    assert weights["SGAM"] >= weights["ALG"]
