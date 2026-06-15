"""Tests for the StockPreprocessor pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from app.ml.preprocessor import StockPreprocessor, load_processed


@pytest.fixture
def sample_df():
    """Generate 200 days of synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    base = np.linspace(100, 200, 200) + np.random.normal(0, 2, 200)
    df = pd.DataFrame(
        {
            "Open": base + np.random.uniform(-1, 1, 200),
            "High": base + np.random.uniform(0.5, 2, 200),
            "Low": base - np.random.uniform(0.5, 2, 200),
            "Close": base,
            "Volume": np.random.randint(1e6, 5e6, 200),
        },
        index=dates,
    )
    return df


class TestLoadAndClean:
    def test_load_from_dataframe(self, sample_df):
        prep = StockPreprocessor()
        result = prep.preprocess(sample_df)
        assert "X_train_lstm" in result

    def test_load_from_csv(self, sample_df, tmp_path):
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path)
        prep = StockPreprocessor()
        result = prep.preprocess(str(csv_path))
        assert len(result["X_train_lstm"]) > 0

    def test_missing_file_raises(self):
        prep = StockPreprocessor()
        with pytest.raises(FileNotFoundError):
            prep.preprocess("nonexistent.csv")

    def test_empty_dataframe_raises(self):
        prep = StockPreprocessor()
        with pytest.raises(ValueError):
            prep.preprocess(pd.DataFrame())

    def test_missing_columns_raises(self, sample_df):
        bad = sample_df.drop(columns=["Volume"])
        prep = StockPreprocessor()
        with pytest.raises(ValueError, match="Missing required columns"):
            prep.preprocess(bad)

    def test_ffill_handles_nan(self, sample_df):
        sample_df.iloc[10:15, 0] = np.nan
        prep = StockPreprocessor()
        result = prep.preprocess(sample_df)
        assert not np.isnan(result["X_train_lstm"]).any()


class TestSequencing:
    def test_sequence_length(self, sample_df):
        prep = StockPreprocessor(sequence_length=30)
        result = prep.preprocess(sample_df)
        assert result["X_train_lstm"].shape[1] == 30

    def test_lstm_3d_shape(self, sample_df):
        prep = StockPreprocessor(sequence_length=60)
        result = prep.preprocess(sample_df)
        assert result["X_train_lstm"].ndim == 3
        # (samples, timesteps, features)
        assert result["X_train_lstm"].shape[2] == 5

    def test_ann_2d_shape(self, sample_df):
        prep = StockPreprocessor(sequence_length=60)
        result = prep.preprocess(sample_df)
        assert result["X_train_ann"].ndim == 2
        assert result["X_train_ann"].shape[1] == 60 * 5

    def test_train_test_sizes(self, sample_df):
        prep = StockPreprocessor(test_size=0.2)
        result = prep.preprocess(sample_df)
        total = len(result["X_train_lstm"]) + len(result["X_test_lstm"])
        assert total > 0
        assert len(result["X_test_lstm"]) / total == pytest.approx(0.2, abs=0.05)


class TestScaling:
    def test_values_in_zero_one_range(self, sample_df):
        prep = StockPreprocessor()
        result = prep.preprocess(sample_df)
        for key in ["X_train_lstm", "X_test_lstm", "y_train", "y_test"]:
            arr = result[key]
            assert arr.min() >= 0.0 or np.isclose(arr.min(), 0.0)
            assert arr.max() <= 1.0 or np.isclose(arr.max(), 1.0)

    def test_scaler_is_fitted(self, sample_df):
        prep = StockPreprocessor()
        result = prep.preprocess(sample_df)
        assert isinstance(result["scaler"], MinMaxScaler)
        assert hasattr(result["scaler"], "data_min_")

    def test_inverse_transform_restores_approximation(self, sample_df):
        prep = StockPreprocessor()
        result = prep.preprocess(sample_df)
        inverted = prep.inverse_transform_price(
            result["scaler"], result["y_test"][:5]
        )
        assert len(inverted) == 5
        assert inverted.dtype in (np.float32, np.float64)


class TestPersistence:
    def test_save_and_load(self, sample_df, tmp_path):
        prep = StockPreprocessor()
        result = prep.preprocess(sample_df, output_dir=str(tmp_path))

        loaded = load_processed(str(tmp_path))
        for key in ["X_train_lstm", "X_test_lstm", "y_train", "y_test"]:
            assert np.allclose(result[key], loaded[key])

        assert loaded["sequence_length"] == 60
        assert "Close" in loaded.get("target_col", "")

    def test_load_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_processed(tmp_path / "nonexistent")
