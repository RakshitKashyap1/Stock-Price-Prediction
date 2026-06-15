#!/usr/bin/env python3
"""
End-to-end ANN stock price prediction pipeline.

Usage:
    python run_ann.py AAPL
    python run_ann.py RELIANCE.NS --years 3 --epochs 50

Output:
    - models_saved/{SYMBOL}_ANN_best.keras   Trained model
    - data/processed_{SYMBOL}/               Scaler + numpy arrays
    - output/{SYMBOL}_metrics.csv             RMSE, MAE, MAPE, R²
    - output/{SYMBOL}_predictions.png         Actual vs predicted plot
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.ml.data_fetcher import StockDataFetcher
from app.ml.preprocessor import StockPreprocessor
from app.ml.models import ANN
from app.ml.trainer import StockTrainer
from app.ml.evaluator import RegressionMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ann")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANN on stock data")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g. AAPL)")
    parser.add_argument("--years", type=int, default=5, help="Years of history")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return parser.parse_args()


def save_metrics_csv(metrics: RegressionMetrics, symbol: str, output_dir: Path) -> Path:
    """Save metrics to a CSV file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}_metrics.csv"
    row = {"symbol": symbol, **metrics.to_dict()}
    pd.DataFrame([row]).to_csv(path, index=False)
    logger.info("Metrics saved to %s", path.resolve())
    return path


def save_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates,
    symbol: str,
    output_dir: Path,
) -> Path:
    """Generate and save actual vs predicted price plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}_predictions.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y_true, label="Actual", linewidth=1.8, color="#1f77b4")
    ax.plot(dates, y_pred, label="Predicted", linewidth=1.8, color="#ff7f0e", linestyle="--")
    ax.set_title(f"{symbol} — ANN Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", path.resolve())
    return path


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()

    out_dir = Path("output")
    model_dir = Path("models_saved")
    data_dir = Path("data") / f"processed_{symbol}"

    # ----------------------------------------------------------------
    # 1. Fetch
    # ----------------------------------------------------------------
    logger.info("Step 1/6 — Fetching %d years of data for %s", args.years, symbol)
    fetcher = StockDataFetcher(period_years=args.years)
    df = fetcher.fetch(symbol)
    logger.info("  → %d rows fetched", len(df))

    # ----------------------------------------------------------------
    # 2. Preprocess
    # ----------------------------------------------------------------
    logger.info("Step 2/6 — Preprocessing (seq_len=%d, test_size=0.2)", args.seq_len)
    preprocessor = StockPreprocessor(
        sequence_length=args.seq_len,
        test_size=0.2,
    )
    data = preprocessor.preprocess(df, output_dir=str(data_dir))
    logger.info("  → Train: %d  |  Test: %d", len(data["X_train_ann"]), len(data["X_test_ann"]))

    # ----------------------------------------------------------------
    # 3. Build ANN
    # ----------------------------------------------------------------
    logger.info("Step 3/6 — Building ANN (input_dim=%d)", data["X_train_ann"].shape[1])
    model = ANN(input_dim=data["X_train_ann"].shape[1], learning_rate=args.lr).build()
    model.summary()

    # ----------------------------------------------------------------
    # 4. Train
    # ----------------------------------------------------------------
    logger.info("Step 4/6 — Training (max_epochs=%d, batch_size=%d)", args.epochs, args.batch_size)
    trainer = StockTrainer(
        model_dir=str(model_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience_early=args.patience,
        verbose=1,
    )
    history = trainer.train(
        model,
        data["X_train_ann"],
        data["y_train"],
        data["X_test_ann"],
        data["y_test"],
        symbol=symbol,
    )
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_loss = min(history.history["val_loss"])
    logger.info("  → Best val_loss: %.6f at epoch %d", best_loss, best_epoch)

    # ----------------------------------------------------------------
    # 5. Predict & evaluate
    # ----------------------------------------------------------------
    logger.info("Step 5/6 — Evaluating")
    y_pred_scaled = model.predict(data["X_test_ann"], verbose=0).flatten()
    y_true_scaled = data["y_test"]

    y_pred_price = preprocessor.inverse_transform_price(data["scaler"], y_pred_scaled)
    y_true_price = preprocessor.inverse_transform_price(data["scaler"], y_true_scaled)

    metrics = RegressionMetrics(y_true_price, y_pred_price)
    print("\n" + metrics.report())

    metrics_path = save_metrics_csv(metrics, symbol, out_dir)

    # ----------------------------------------------------------------
    # 6. Plot
    # ----------------------------------------------------------------
    logger.info("Step 6/6 — Generating plot")
    plot_path = save_prediction_plot(
        y_true_price,
        y_pred_price,
        data["test_dates"],
        symbol,
        out_dir,
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"  {symbol} — ANN Pipeline Complete")
    print("=" * 55)
    print(f"  Model:     {model_dir / f'{symbol}_ANN_best.keras'}")
    print(f"  Metrics:   {metrics_path}")
    print(f"  Plot:      {plot_path}")
    print(f"  RMSE:      ${metrics.rmse:.2f}")
    print(f"  MAE:       ${metrics.mae:.2f}")
    print(f"  MAPE:      {metrics.mape:.2f}%")
    print(f"  R²:        {metrics.r2:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
