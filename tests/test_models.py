"""Tests for ANN and LSTM model definitions."""

import numpy as np
import pytest
from keras import Model

from app.ml.models import ANN, LSTM


class TestANN:
    def test_build_returns_model(self):
        model = ANN(input_dim=300).build()
        assert isinstance(model, Model)

    def test_output_shape(self):
        model = ANN(input_dim=300).build()
        X = np.random.randn(16, 300)
        y = model.predict(X, verbose=0)
        assert y.shape == (16, 1)

    def test_compile_has_mse_loss(self):
        model = ANN(input_dim=300).build()
        assert model.loss == "mse"

    def test_compile_has_mae_metric(self):
        model = ANN(input_dim=300).build()
        assert "mae" in model.metrics_names

    def test_param_count_reasonable(self):
        model = ANN(input_dim=300).build()
        # ~300*128 + 128*64 + 64*32 + 32*1 + biases ≈ 47k
        params = model.count_params()
        assert 40_000 < params < 60_000

    def test_different_input_dim(self):
        model = ANN(input_dim=150).build()
        X = np.random.randn(8, 150)
        y = model.predict(X, verbose=0)
        assert y.shape == (8, 1)

    def test_custom_learning_rate(self):
        model = ANN(input_dim=300, learning_rate=1e-4).build()
        lr = float(model.optimizer.learning_rate.numpy())
        assert lr == pytest.approx(1e-4)


class TestLSTM:
    def test_build_returns_model(self):
        model = LSTM(timesteps=60, n_features=5).build()
        assert isinstance(model, Model)

    def test_output_shape(self):
        model = LSTM(timesteps=60, n_features=5).build()
        X = np.random.randn(16, 60, 5)
        y = model.predict(X, verbose=0)
        assert y.shape == (16, 1)

    def test_lstm_accepts_3d_input(self):
        model = LSTM(timesteps=60, n_features=5).build()
        X = np.random.randn(4, 60, 5)
        model.predict(X, verbose=0)  # should not raise

    def test_custom_timesteps(self):
        model = LSTM(timesteps=30, n_features=3).build()
        X = np.random.randn(8, 30, 3)
        y = model.predict(X, verbose=0)
        assert y.shape == (8, 1)
