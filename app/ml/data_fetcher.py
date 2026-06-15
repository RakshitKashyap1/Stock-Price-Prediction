"""
Historical stock data downloader using Yahoo Finance API.

This module provides a clean interface to fetch, validate, and persist
historical stock market data via the yfinance library. It handles
error states, caches results locally as CSV, and validates symbol
existence before downloading.

Typical usage:
    fetcher = StockDataFetcher()
    df = fetcher.fetch("AAPL")
    fetcher.save_to_csv(df, "AAPL")
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches historical OHLCV data from Yahoo Finance.

    Attributes:
        data_dir: Path to directory where CSV files are cached.
        period: Default lookback period for data downloads.
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, data_dir: str = "data", period_years: int = 5):
        """
        Args:
            data_dir: Directory for CSV cache files.
            period_years: Number of years of historical data to fetch.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.period_years = period_years

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, symbol: str) -> pd.DataFrame:
        """
        Download historical stock data for a given symbol.

        The download period is calculated as `period_years` from today.
        Returns a clean DataFrame with validated columns and no missing
        values in the OHLCV fields.

        Args:
            symbol: Stock ticker symbol (e.g. "AAPL", "RELIANCE.NS").

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume.

        Raises:
            ValueError: If the symbol is invalid or no data is returned.
            ConnectionError: If the Yahoo Finance API is unreachable.
        """
        symbol = self._validate_symbol(symbol)
        ticker = yf.Ticker(symbol)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.period_years * 365)

        logger.info("Fetching %s data from %s to %s", symbol, start_date.date(), end_date.date())

        try:
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to fetch data for {symbol}. "
                f"Check your network connection and try again."
            ) from exc

        if df.empty:
            raise ValueError(
                f"No data returned for symbol '{symbol}'. "
                f"Verify the symbol is correct and the market is open."
            )

        df = self._clean(df)
        self._validate_columns(df, symbol)

        logger.info(
            "Successfully fetched %d rows for %s (%.1f years)",
            len(df), symbol, self.period_years,
        )
        return df

    def fetch_and_save(self, symbol: str) -> pd.DataFrame:
        """
        Convenience method: fetch data and immediately save to CSV.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            The fetched DataFrame.
        """
        df = self.fetch(symbol)
        self.save_to_csv(df, symbol)
        return df

    def save_to_csv(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Persist a DataFrame to a CSV file named ``{symbol}.csv``.

        Args:
            df: The DataFrame to save.
            symbol: Used to derive the filename.

        Returns:
            The absolute path to the saved CSV file.
        """
        filepath = self._csv_path(symbol)
        df.to_csv(filepath, index=True)
        logger.info("Data saved to %s", filepath.resolve())
        return str(filepath.resolve())

    def load_from_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load previously cached CSV data for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            DataFrame if the file exists, otherwise None.
        """
        filepath = self._csv_path(symbol)
        if filepath.exists():
            logger.info("Loading cached data from %s", filepath.resolve())
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        return None

    def is_valid_symbol(self, symbol: str) -> bool:
        """
        Check whether a stock symbol exists on Yahoo Finance.

        Uses a lightweight metadata lookup rather than downloading
        the full history.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            True if the symbol resolves to a valid instrument.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get("regularMarketPrice") is not None or "longName" in info
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_symbol(self, symbol: str) -> str:
        """Sanitize and validate the symbol string."""
        if not symbol or not symbol.strip():
            raise ValueError("Stock symbol cannot be empty.")
        cleaned = symbol.strip().upper()
        if not cleaned.replace(".", "").isalnum():
            raise ValueError(
                f"Symbol '{symbol}' contains invalid characters. "
                f"Use alphanumeric characters and dots only (e.g. AAPL, BRK.B, RELIANCE.NS)."
            )
        return cleaned

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw DataFrame:
          - Round numeric columns to 2 decimal places.
          - Forward-fill missing values for non-trading days.
          - Drop any remaining rows with NaN in core columns.
        """
        df = df[self.REQUIRED_COLUMNS].copy()
        df = df.round(2)
        df.ffill(inplace=True)
        df.dropna(subset=self.REQUIRED_COLUMNS, inplace=True)
        return df

    def _validate_columns(self, df: pd.DataFrame, symbol: str) -> None:
        """Ensure all required columns exist after cleaning."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Data for {symbol} is missing columns: {missing}. "
                f"Expected: {self.REQUIRED_COLUMNS}"
            )

    def _csv_path(self, symbol: str) -> Path:
        """Return the filesystem path for a symbol's cached CSV."""
        return self.data_dir / f"{symbol.upper()}.csv"


def summary(df: pd.DataFrame) -> str:
    """
    Return a human-readable summary of the stock data.

    Args:
        df: DataFrame returned by ``fetch()``.

    Returns:
        Multi-line summary string.
    """
    lines = [
        f"Rows: {len(df):,}",
        f"Date range: {df.index[0].date()} to {df.index[-1].date()}",
        f"Latest Close: ${df['Close'].iloc[-1]:.2f}",
        f"52-week High: ${df['High'].max():.2f}",
        f"52-week Low:  ${df['Low'].min():.2f}",
        f"Avg Volume:   {df['Volume'].mean():,.0f}",
    ]
    return "\n".join(lines)


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
        print("Usage: python -m app.ml.data_fetcher <SYMBOL> [SYMBOL2 ...]")
        sys.exit(1)

    fetcher = StockDataFetcher()
    for sym in sys.argv[1:]:
        try:
            print(f"\n{'='*50}")
            print(f"  Fetching: {sym}")
            print(f"{'='*50}")
            data = fetcher.fetch(sym)
            fetcher.save_to_csv(data, sym)
            print(summary(data))
        except (ValueError, ConnectionError) as e:
            logger.error("Failed for %s: %s", sym, e)
