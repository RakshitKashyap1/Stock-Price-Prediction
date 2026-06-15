"""
Model training pipeline.

Wraps the Keras ``fit()`` call with early stopping, learning-rate
reduction on plateau, model checkpointing, and optional TensorBoard
logging.

Usage:
    from app.ml.models import ANN
    from app.ml.trainer import StockTrainer

    trainer = StockTrainer(model_dir="models_saved")
    history = trainer.train(model, X_train, y_train, X_val, y_val, symbol="AAPL")
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from keras import callbacks as cb

logger = logging.getLogger(__name__)


class StockTrainer:
    """
    Handles Keras model training with industry-standard callbacks.

    Attributes:
        model_dir:    Directory where model checkpoints are saved.
        epochs:       Maximum number of training epochs.
        batch_size:   Batch size for gradient updates.
        patience_early: Epochs to wait before early stopping.
        patience_lr:    Epochs to wait before reducing LR.
        min_lr:        Floor for learning-rate decay.
    """

    def __init__(
        self,
        model_dir: str = "models_saved",
        epochs: int = 100,
        batch_size: int = 32,
        patience_early: int = 10,
        patience_lr: int = 5,
        min_lr: float = 1e-6,
        verbose: int = 1,
    ):
        """
        Args:
            model_dir: Directory to save model checkpoints.
            epochs: Maximum training epochs.
            batch_size: Training batch size.
            patience_early: EarlyStopping patience.
            patience_lr: ReduceLROnPlateau patience.
            min_lr: Minimum learning rate.
            verbose: Keras verbosity (0=silent, 1=progress, 2=one line/epoch).
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience_early = patience_early
        self.patience_lr = patience_lr
        self.min_lr = min_lr
        self.verbose = verbose

    def train(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        symbol: str = "stock",
    ) -> tf.keras.callbacks.History:
        """
        Train the model with automated callbacks.

        Callbacks applied:
            - EarlyStopping       — stops when val_loss stops improving.
            - ReduceLROnPlateau   — decays LR by 0.5 when plateaued.
            - ModelCheckpoint     — saves the best weights to ``model_dir``.
            - TensorBoard         — logs metrics for visualisation (optional).

        Args:
            model:   Compiled Keras model (from ``ANN().build()`` or similar).
            X_train: Training features.
            y_train: Training targets.
            X_val:   Validation features.
            y_val:   Validation targets.
            symbol:  Stock ticker (used in checkpoint filenames).

        Returns:
            Keras History object containing per-epoch metrics.
        """
        model_name = model.name
        filepath = self.model_dir / f"{symbol}_{model_name}.keras"
        best_only = self.model_dir / f"{symbol}_{model_name}_best.keras"

        early_stop = cb.EarlyStopping(
            monitor="val_loss",
            patience=self.patience_early,
            restore_best_weights=True,
            mode="min",
            verbose=self.verbose,
        )

        reduce_lr = cb.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.patience_lr,
            min_lr=self.min_lr,
            mode="min",
            verbose=self.verbose,
        )

        checkpoint = cb.ModelCheckpoint(
            filepath=str(best_only),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=self.verbose,
        )

        callbacks = [early_stop, reduce_lr, checkpoint]

        logger.info(
            "Training %s for %s — samples=%d, val_samples=%d",
            model_name,
            symbol,
            len(X_train),
            len(X_val),
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
        )

        model.save(filepath)
        logger.info(
            "%s/%s training complete — best val_loss=%.6f",
            symbol,
            model_name,
            min(history.history["val_loss"]),
        )

        return history

    def load_model(self, symbol: str, model_name: str) -> tf.keras.Model:
        """
        Load a previously saved model from disk.

        Args:
            symbol: Stock ticker used during save.
            model_name: "ANN" or "LSTM".

        Returns:
            Loaded Keras Model.
        """
        path = self.model_dir / f"{symbol}_{model_name}_best.keras"
        if not path.exists():
            path = self.model_dir / f"{symbol}_{model_name}.keras"
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model found at {self.model_dir.resolve()}/"
                f"{symbol}_{model_name}*.keras"
            )
        logger.info("Loading model from %s", path.resolve())
        return tf.keras.models.load_model(str(path))
