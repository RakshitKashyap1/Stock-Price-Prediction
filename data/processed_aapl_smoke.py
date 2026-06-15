"""Smoke test: fetch AAPL data and run full preprocessing pipeline."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.ml.data_fetcher import StockDataFetcher
from app.ml.preprocessor import StockPreprocessor

fetcher = StockDataFetcher(period_years=2)
df = fetcher.fetch("AAPL")
print(f"Fetched {len(df)} rows")

prep = StockPreprocessor(sequence_length=60, test_size=0.2)
result = prep.preprocess(df, output_dir="data/processed_aapl")
print(f"Train LSTM shape: {result['X_train_lstm'].shape}")
print(f"Test LSTM shape:  {result['X_test_lstm'].shape}")
print(f"Train ANN shape:  {result['X_train_ann'].shape}")
print(f"Test ANN shape:   {result['X_test_ann'].shape}")
print(f"y_train range:    [{result['y_train'].min():.4f}, {result['y_train'].max():.4f}]")
print(f"y_test range:     [{result['y_test'].min():.4f}, {result['y_test'].max():.4f}]")
print(f"Scaler min:       {result['scaler'].data_min_}")
print(f"Scaler max:       {result['scaler'].data_max_}")
print(f"Train dates:      {result['train_dates'][0].date()} to {result['train_dates'][-1].date()}")
print(f"Test dates:       {result['test_dates'][0].date()} to {result['test_dates'][-1].date()}")
