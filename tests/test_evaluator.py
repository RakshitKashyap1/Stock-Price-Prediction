"""Tests for RegressionMetrics and model comparison."""

import numpy as np
import pandas as pd
import pytest

from app.ml.evaluator import RegressionMetrics, compare_models


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([100, 102, 101, 105])
        m = RegressionMetrics(y, y)
        assert m.rmse == 0.0
        assert m.mae == 0.0
        assert m.mape == 0.0
        assert m.r2 == 1.0

    @pytest.mark.parametrize("offset", [1.0, 5.0, 10.0])
    def test_constant_offset(self, offset):
        y_true = np.array([100, 102, 101, 105])
        y_pred = y_true + offset
        m = RegressionMetrics(y_true, y_pred)
        assert m.rmse == pytest.approx(offset, abs=1e-4)
        assert m.mae == pytest.approx(offset, abs=1e-4)

    def test_r2_of_mean_model_is_zero(self):
        y_true = np.array([100, 102, 101, 105])
        y_pred = np.full_like(y_true, np.mean(y_true))
        m = RegressionMetrics(y_true, y_pred)
        assert m.r2 == pytest.approx(0.0, abs=1e-4)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            RegressionMetrics(np.array([1, 2]), np.array([1, 2, 3]))

    def test_mape_handles_zeros(self):
        y_true = np.array([0, 100, 200])
        y_pred = np.array([0, 102, 198])
        m = RegressionMetrics(y_true, y_pred)
        assert not np.isnan(m.mape)

    def test_to_dict_keys(self):
        y_true = np.array([100, 102, 101])
        y_pred = np.array([99, 103, 100])
        d = RegressionMetrics(y_true, y_pred).to_dict()
        for key in ["RMSE", "MAE", "MAPE", "R2"]:
            assert key in d

    def test_report_contains_metrics(self):
        y_true = np.array([100, 102, 101])
        y_pred = np.array([99, 103, 100])
        report = RegressionMetrics(y_true, y_pred).report()
        assert "RMSE" in report
        assert "MAE" in report
        assert "R^2" in report

    def test_float_inputs(self):
        y_true = [100.0, 102.0, 101.0]
        y_pred = [101.0, 102.0, 100.0]
        m = RegressionMetrics(y_true, y_pred)
        assert m.rmse > 0

    def test_custom_decimals(self):
        y_true = np.array([100.12345, 102.12345])
        y_pred = np.array([100.0, 102.0])
        m = RegressionMetrics(y_true, y_pred, decimals=2)
        assert len(str(m.rmse).split(".")[1]) <= 2


class TestCompareModels:
    def test_returns_dataframe(self):
        y_true = np.array([100, 102, 101, 105])
        results = {
            "ANN": (y_true, np.array([99, 103, 100, 104])),
            "LSTM": (y_true, np.array([100, 101, 102, 105])),
        }
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["ANN", "LSTM"]

    def test_single_model(self):
        y_true = np.array([100, 102, 101])
        results = {"ANN": (y_true, np.array([99, 103, 100]))}
        df = compare_models(results)
        assert len(df) == 1

    def test_empty_dict_returns_empty(self):
        df = compare_models({})
        assert df.empty
