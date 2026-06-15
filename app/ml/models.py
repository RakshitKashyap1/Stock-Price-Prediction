"""
ANN and LSTM model definitions for stock price prediction.

Each model is a self-contained class that builds, compiles, and
returns a Keras Model ready for training.  The caller supplies
the input shape (e.g. 300 for ANN, or (60, 5) for LSTM).

Usage:
    from app.ml.models import ANN, LSTM

    model = ANN(input_dim=300).build()
    model.summary()
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers

logger = logging.getLogger(__name__)


class ANN:
    """
    Feed-forward Artificial Neural Network.

    Architecture:
        Input (300) → Dense(128, ReLU) → Dropout(0.2)
                    → Dense(64, ReLU)  → Dropout(0.2)
                    → Dense(32, ReLU)  → Dropout(0.1)
                    → Dense(1)

    The input is a flattened vector of ``sequence_length × n_features``
    values — i.e. the last 60 trading days of OHLCV data.

    Attributes:
        input_dim: Number of features in the flattened input layer.
        learning_rate: Adam optimizer learning rate.
    """

    def __init__(self, input_dim: int, learning_rate: float = 1e-3):
        """
        Args:
            input_dim: Flattened input size (e.g. 60 × 5 = 300).
            learning_rate: Adam learning rate.
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate

    def build(self) -> keras.Model:
        """
        Construct and compile the ANN.

        Returns:
            Compiled Keras Model (not yet trained).
        """
        model = models.Sequential(name="ANN")

        model.add(layers.Input(shape=(self.input_dim,), name="input"))
        model.add(layers.Dense(128, activation="relu", name="hidden_1"))
        model.add(layers.Dropout(0.2, name="dropout_1"))
        model.add(layers.Dense(64, activation="relu", name="hidden_2"))
        model.add(layers.Dropout(0.2, name="dropout_2"))
        model.add(layers.Dense(32, activation="relu", name="hidden_3"))
        model.add(layers.Dropout(0.1, name="dropout_3"))
        model.add(layers.Dense(1, name="output"))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info(
            "ANN built — input_dim=%d, params=%d",
            self.input_dim,
            model.count_params(),
        )
        return model


class LSTM:
    """
    Long Short-Term Memory network.

    Architecture:
        Input (60, 5) → LSTM(50, return_seq=True) → Dropout(0.2)
                      → LSTM(50)                  → Dropout(0.2)
                      → Dense(1)

    The input is a 3D tensor of ``(samples, timesteps, features)``
    where each sample is a 60-day window of OHLCV data.

    Attributes:
        timesteps: Number of time steps in each sequence.
        n_features: Number of features per time step.
        learning_rate: Adam optimizer learning rate.
    """

    def __init__(
        self,
        timesteps: int = 60,
        n_features: int = 5,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            timesteps: Sequence length (default 60).
            n_features: Number of features per step (default 5).
            learning_rate: Adam learning rate.
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.learning_rate = learning_rate

    def build(self) -> keras.Model:
        """
        Construct and compile the LSTM.

        Returns:
            Compiled Keras Model (not yet trained).
        """
        model = models.Sequential(name="LSTM")

        model.add(
            layers.LSTM(
                50,
                return_sequences=True,
                input_shape=(self.timesteps, self.n_features),
                name="lstm_1",
            )
        )
        model.add(layers.Dropout(0.2, name="dropout_1"))

        model.add(layers.LSTM(50, return_sequences=False, name="lstm_2"))
        model.add(layers.Dropout(0.2, name="dropout_2"))

        model.add(layers.Dense(1, name="output"))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info(
            "LSTM built — timesteps=%d, features=%d, params=%d",
            self.timesteps,
            self.n_features,
            model.count_params(),
        )
        return model
