"""Tests for the StockTrainer pipeline."""

import numpy as np
import pytest
from keras import Model

from app.ml.models import ANN
from app.ml.trainer import StockTrainer


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    X = np.random.randn(200, 300).astype(np.float32)
    y = np.random.randn(200).astype(np.float32)
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture
def model():
    return ANN(input_dim=300).build()


class TestTrainerInitialization:
    def test_default_model_dir_created(self, tmp_path):
        trainer = StockTrainer(model_dir=str(tmp_path / "models"))
        assert (tmp_path / "models").exists()

    def test_custom_epochs(self):
        trainer = StockTrainer(epochs=50)
        assert trainer.epochs == 50


class TestTraining:
    def test_train_returns_history(self, model, synthetic_data, tmp_path):
        X_train, y_train, X_val, y_val = synthetic_data
        trainer = StockTrainer(
            model_dir=str(tmp_path / "models"),
            epochs=5,
            batch_size=16,
            verbose=0,
        )
        history = trainer.train(model, X_train, y_train, X_val, y_val, symbol="TEST")
        assert "loss" in history.history
        assert "val_loss" in history.history
        assert len(history.history["loss"]) <= 5

    def test_model_checkpoint_saved(self, model, synthetic_data, tmp_path):
        X_train, y_train, X_val, y_val = synthetic_data
        trainer = StockTrainer(
            model_dir=str(tmp_path / "models"),
            epochs=3,
            batch_size=16,
            verbose=0,
        )
        trainer.train(model, X_train, y_train, X_val, y_val, symbol="TEST")
        best_path = tmp_path / "models" / "TEST_ANN_best.keras"
        final_path = tmp_path / "models" / "TEST_ANN.keras"
        assert best_path.exists() or final_path.exists()

    def test_load_model(self, model, synthetic_data, tmp_path):
        X_train, y_train, X_val, y_val = synthetic_data
        trainer = StockTrainer(
            model_dir=str(tmp_path / "models"),
            epochs=3,
            batch_size=16,
            verbose=0,
        )
        trainer.train(model, X_train, y_train, X_val, y_val, symbol="TEST")
        loaded = trainer.load_model("TEST", "ANN")
        assert isinstance(loaded, Model)

    def test_load_nonexistent_raises(self, tmp_path):
        trainer = StockTrainer(model_dir=str(tmp_path / "empty"))
        with pytest.raises(FileNotFoundError):
            trainer.load_model("NONEXISTENT", "ANN")
