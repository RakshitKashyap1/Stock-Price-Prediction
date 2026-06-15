"""
Data preprocessing pipeline for stock price prediction.

Transforms raw OHLCV data into normalized sequences ready for
ANN and LSTM models. Handles the full pipeline:

    CSV/DataFrame → clean → scale → sequence → split → save

Typical usage:
    preprocessor = StockPreprocessor()
    data = preprocessor.preprocess("data/AAPL.csv")
    # data["X_train_lstm"].shape == (samples, 60, features)
    # data["X_train_ann"].shape  == (samples, 60 * features)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class StockPreprocessor:
    """
    Preprocess raw stock data into train/test sequences.

    Attributes:
        sequence_length: Number of past days used to predict the next day.
        test_size: Fraction of data reserved for testing (chronological).
        target_col: Column name to predict.
        feature_cols: Columns used as model inputs.
    """

    DEFAULT_FEATURES = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(
        self,
        sequence_length: int = 60,
        test_size: float = 0.2,
        target_col: str = "Close",
        feature_cols: Optional[list] = None,
    ):
        """
        Args:
            sequence_length: Sliding window size (default 60 days).
            test_size: Proportion of data for testing (default 0.2).
            target_col: Column to predict.
            feature_cols: Input columns. Defaults to OHLCV.
        """
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.target_col = target_col
        self.feature_cols = feature_cols or self.DEFAULT_FEATURES

        self._scaler: Optional[MinMaxScaler] = None
        self._feature_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> dict:
        """
        Run the full preprocessing pipeline.

        Steps:
            1. Load data from CSV or DataFrame.
            2. Validate and clean missing values.
            3. Normalise all feature columns to [0, 1].
            4. Build sliding-window sequences.
            5. Chronologically split into train/test.
            6. Reshape for both ANN (2D) and LSTM (3D).

        Args:
            data_source: Path to CSV file or a DataFrame.
            output_dir: If provided, save scaler + arrays to disk.

        Returns:
            Dictionary with keys:
                - X_train_lstm, X_test_lstm   (samples, 60, features)
                - X_train_ann,  X_test_ann    (samples, 60 * features)
                - y_train,      y_test        (samples,)
                - scaler                      MinMaxScaler instance
                - feature_cols                list of column names
                - sequence_length             int
                - train_dates, test_dates     DatetimeIndex slices
        """
        df = self._load(data_source)
        df = self._clean(df)
        self._validate_columns(df)

        raw_values = df[self.feature_cols].values.astype(np.float64)
        dates = df.index

        self._scaler = MinMaxScaler()
        scaled = self._scaler.fit_transform(raw_values)

        target_idx = self.feature_cols.index(self.target_col)
        X, y = self._create_sequences(scaled, target_idx)

        split_idx = int(len(X) * (1 - self.test_size))
        train_dates = dates[self.sequence_length : split_idx + self.sequence_length]
        test_dates = dates[split_idx + self.sequence_length :]

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        result = {
            "X_train_lstm": X_train,
            "X_test_lstm": X_test,
            "X_train_ann": X_train.reshape(X_train.shape[0], -1),
            "X_test_ann": X_test.reshape(X_test.shape[0], -1),
            "y_train": y_train,
            "y_test": y_test,
            "scaler": self._scaler,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "sequence_length": self.sequence_length,
            "train_dates": train_dates,
            "test_dates": test_dates,
        }

        if output_dir:
            self._save(result, output_dir, df)

        for stage, count in [("Training", len(X_train)), ("Test", len(X_test))]:
            logger.info("%s samples: %d", stage, count)

        return result

    def inverse_transform_price(
        self, scaler: MinMaxScaler, predictions: np.ndarray
    ) -> np.ndarray:
        """
        Convert normalised predictions back to original price scale.

        Because the scaler was fit on multi-dimensional data, we pad
        with zeros for the other columns and extract only the target.

        Args:
            scaler: The MinMaxScaler used during preprocessing.
            predictions: 1D or 2D array of scaled predictions.

        Returns:
            Prices in the original unit (e.g. USD).
        """
        predictions = np.asarray(predictions).reshape(-1, 1)
        dummy = np.zeros((len(predictions), len(self.feature_cols) - 1))
        padded = np.hstack([predictions, dummy])

        target_idx = self.feature_cols.index(self.target_col)
        if target_idx != 0:
            cols = list(range(len(self.feature_cols)))
            cols.remove(target_idx)
            cols.insert(0, target_idx)
            padded = padded[:, cols]

        inverted = scaler.inverse_transform(
            np.hstack([predictions, dummy])
            if target_idx == 0
            else np.hstack([dummy[:, :target_idx], predictions, dummy[:, target_idx:]])
        )
        return inverted[:, target_idx].flatten()

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data from CSV path or accept a DataFrame directly."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path.resolve()}")
            logger.info("Loading data from %s", path.resolve())
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(
                f"Unsupported data source type: {type(source)}. "
                f"Expected str, Path, or DataFrame."
            )

        if df.empty:
            raise ValueError("Loaded DataFrame is empty.")
        logger.info("Loaded %d rows with columns: %s", len(df), list(df.columns))
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove or fill missing values.

        - Drops rows where all OHLCV columns are NaN.
        - Forward-fills remaining NaN values (e.g. non-trading days).
        - Drops any leftover NaN rows to guarantee a clean dataset.
        """
        initial = len(df)
        df = df.dropna(how="all", subset=self.feature_cols)
        df = df.ffill()
        df = df.dropna(subset=self.feature_cols)
        dropped = initial - len(df)
        if dropped:
            logger.warning("Dropped %d rows with missing values.", dropped)
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Ensure all expected feature columns exist in the DataFrame."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"DataFrame has: {list(df.columns)}"
            )

    def _create_sequences(
        self, data: np.ndarray, target_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sliding-window sequences from scaled data.

        For each time-step ``t``, the input ``X[t]`` is the
        ``sequence_length`` rows ending at ``t-1``, and the target
        ``y[t]`` is the target column at ``t``.

        This produces an autoregressive structure suitable for both
        ANN (flattened later) and LSTM models.
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(data[i, target_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _save(self, data: dict, output_dir: Union[str, Path], df: pd.DataFrame) -> None:
        """
        Persist processed arrays and the scaler to disk.

        Saves:
            - X_train_lstm.npy, X_test_lstm.npy
            - y_train.npy, y_test.npy
            - scaler.pkl
            - processed_info.csv  (metadata for reproducibility)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X_train_lstm.npy", data["X_train_lstm"])
        np.save(output_dir / "X_test_lstm.npy", data["X_test_lstm"])
        np.save(output_dir / "y_train.npy", data["y_train"])
        np.save(output_dir / "y_test.npy", data["y_test"])

        with open(output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

        info = pd.DataFrame(
            {
                "symbol": [getattr(df.index.name, "name", "unknown")],
                "rows_raw": [len(df)],
                "rows_after_clean": [len(df)],
                "train_samples": [len(data["X_train_lstm"])],
                "test_samples": [len(data["X_test_lstm"])],
                "sequence_length": [self.sequence_length],
                "test_size": [self.test_size],
                "features": [", ".join(self.feature_cols)],
                "target": [self.target_col],
            }
        )
        info.to_csv(output_dir / "processed_info.csv", index=False)

        logger.info("Processed data saved to %s", output_dir.resolve())

    # ------------------------------------------------------------------
    # Helpers for downstream use
    # ------------------------------------------------------------------

    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        """The fitted scaler from the last ``preprocess()`` call."""
        return self._scaler


def load_processed(data_dir: Union[str, Path]) -> dict:
    """
    Load previously saved processed data from disk.

    Args:
        data_dir: Directory containing .npy files and scaler.pkl.

    Returns:
        Dictionary with the same structure as ``preprocess()`` output.
    """
    data_dir = Path(data_dir)
    required = ["X_train_lstm.npy", "X_test_lstm.npy", "y_train.npy", "y_test.npy", "scaler.pkl"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {missing}")

    result = {
        "X_train_lstm": np.load(data_dir / "X_train_lstm.npy"),
        "X_test_lstm": np.load(data_dir / "X_test_lstm.npy"),
        "X_train_ann": np.load(data_dir / "X_train_lstm.npy").reshape(-1, np.load(data_dir / "X_train_lstm.npy").shape[1] * np.load(data_dir / "X_train_lstm.npy").shape[2]),
        "X_test_ann": np.load(data_dir / "X_test_lstm.npy").reshape(-1, np.load(data_dir / "X_test_lstm.npy").shape[1] * np.load(data_dir / "X_test_lstm.npy").shape[2]),
        "y_train": np.load(data_dir / "y_train.npy"),
        "y_test": np.load(data_dir / "y_test.npy"),
        "scaler": pickle.load(open(data_dir / "scaler.pkl", "rb")),
    }

    info_path = data_dir / "processed_info.csv"
    if info_path.exists():
        info = pd.read_csv(info_path)
        result["sequence_length"] = int(info["sequence_length"].iloc[0])
        result["test_size"] = float(info["test_size"].iloc[0])
        result["feature_cols"] = info["features"].iloc[0].split(", ")
        result["target_col"] = info["target"].iloc[0]

    logger.info("Loaded processed data from %s", data_dir.resolve())
    return result


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m app.ml.preprocessor <CSV_PATH> [OUTPUT_DIR]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"

    preprocessor = StockPreprocessor()
    result = preprocessor.preprocess(csv_path, output_dir=output_dir)

    print(f"\nPipeline complete.")
    print(f"  Train samples: {len(result['X_train_lstm'])}")
    print(f"  Test samples:  {len(result['X_test_lstm'])}")
    print(f"  LSTM shape:    {result['X_train_lstm'].shape}")
    print(f"  ANN shape:     {result['X_train_ann'].shape}")
    print(f"  Output saved to: {output_dir}")
