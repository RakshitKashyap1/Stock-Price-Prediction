"""End-to-end test: fetch → preprocess → train ANN → evaluate."""
import sys, pathlib, logging
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

import numpy as np
from app.ml.data_fetcher import StockDataFetcher
from app.ml.preprocessor import StockPreprocessor
from app.ml.models import ANN
from app.ml.trainer import StockTrainer
from app.ml.evaluator import RegressionMetrics

SYMBOL = "MSFT"

# 1. Fetch
fetcher = StockDataFetcher(period_years=2)
df = fetcher.fetch(SYMBOL)
print(f"1. Fetched {len(df)} rows for {SYMBOL}")

# 2. Preprocess
prep = StockPreprocessor(sequence_length=60, test_size=0.2)
data = prep.preprocess(df)
print(f"2. Preprocessed — train: {len(data['X_train_ann'])}, test: {len(data['X_test_ann'])}")

# 3. Build ANN
model = ANN(input_dim=data["X_train_ann"].shape[1]).build()
print(f"3. ANN built — params={model.count_params()}")

# 4. Train
trainer = StockTrainer(
    model_dir="models_saved",
    epochs=20,
    batch_size=32,
    patience_early=5,
    verbose=0,
)
history = trainer.train(
    model,
    data["X_train_ann"],
    data["y_train"],
    data["X_test_ann"],
    data["y_test"],
    symbol=SYMBOL,
)
best_val_loss = min(history.history["val_loss"])
best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
print(f"4. Trained {len(history.history['loss'])} epochs — best val_loss={best_val_loss:.6f} at epoch {best_epoch}")

# 5. Predict
y_pred_scaled = model.predict(data["X_test_ann"], verbose=0).flatten()
y_true_scaled = data["y_test"]

# 6. Inverse transform to original price scale
y_pred_price = prep.inverse_transform_price(data["scaler"], y_pred_scaled)
y_true_price = prep.inverse_transform_price(data["scaler"], y_true_scaled)

# 7. Evaluate
metrics_scaled = RegressionMetrics(y_true_scaled, y_pred_scaled)
metrics_price = RegressionMetrics(y_true_price, y_pred_price)

print("\n5. Scaled metrics:")
print(metrics_scaled.report())
print("\n6. Price metrics (USD):")
print(metrics_price.report())

# Basic sanity checks
assert metrics_scaled.rmse < 0.1, f"Scaled RMSE too high: {metrics_scaled.rmse}"
assert metrics_scaled.r2 > 0.8, f"R² too low: {metrics_scaled.r2}"
print("\n7. All sanity checks passed — ANN pipeline works correctly!")
