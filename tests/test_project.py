
import numpy as np
import pandas as pd

from features import make_features


def sample_data():
    close = 100 + np.arange(80) * 0.3 + 5 * np.sin(np.arange(80))
    return pd.DataFrame({"Close": close})


def test_feature_columns_exist():
    result = make_features(sample_data())

    expected = [
        "Return", "SMA_5", "SMA_20",
        "RSI_14", "Volatility_20", "Target"
    ]

    for column in expected:
        assert column in result.columns


def test_features_have_no_missing_values():
    result = make_features(sample_data())

    assert not result.isna().any().any()


def test_target_is_next_day_close():
    data = sample_data()
    result = make_features(data)

    first_index = result.index[0]
    assert np.isclose(
        result.iloc[0]["Target"],
        data.iloc[first_index + 1]["Close"]
    )


def test_moving_average_is_correct():
    data = sample_data()
    result = make_features(data)

    first_index = result.index[0]
    expected = data["Close"].iloc[
        first_index - 19:first_index + 1
    ].mean()

    assert np.isclose(result.iloc[0]["SMA_20"], expected)


def test_rsi_is_within_valid_range():
    result = make_features(sample_data())

    assert (result["RSI_14"] >= 0).all()
    assert (result["RSI_14"] <= 100).all()
