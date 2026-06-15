"""Unit tests for the StockDataFetcher module."""

from pathlib import Path
import pandas as pd
import pytest

from app.ml.data_fetcher import StockDataFetcher


@pytest.fixture
def fetcher(tmp_path):
    return StockDataFetcher(data_dir=str(tmp_path), period_years=1)


class TestValidation:
    def test_empty_symbol_raises(self, fetcher):
        with pytest.raises(ValueError, match="cannot be empty"):
            fetcher.fetch("")

    def test_whitespace_symbol_raises(self, fetcher):
        with pytest.raises(ValueError, match="cannot be empty"):
            fetcher.fetch("   ")

    def test_invalid_characters_raises(self, fetcher):
        with pytest.raises(ValueError, match="invalid characters"):
            fetcher.fetch("AAPL@#$")

    def test_nonexistent_symbol_raises(self, fetcher):
        with pytest.raises((ValueError, ConnectionError)):
            fetcher.fetch("ZZZZZZXX")


class TestFetch:
    def test_fetch_aapl_returns_dataframe(self, fetcher):
        df = fetcher.fetch("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_required_columns_present(self, fetcher):
        df = fetcher.fetch("MSFT")
        for col in StockDataFetcher.REQUIRED_COLUMNS:
            assert col in df.columns

    def test_no_nan_in_core_columns(self, fetcher):
        df = fetcher.fetch("GOOGL")
        assert df[StockDataFetcher.REQUIRED_COLUMNS].isnull().sum().sum() == 0

    def test_values_rounded_to_two_decimals(self, fetcher):
        df = fetcher.fetch("AAPL")
        for col in ["Open", "High", "Low", "Close"]:
            assert (df[col] * 100 % 1 == 0).all()


class TestPersistence:
    def test_save_and_load_csv(self, fetcher):
        df = fetcher.fetch("AAPL")
        path = fetcher.save_to_csv(df, "AAPL")
        assert Path(path).exists()

        loaded = fetcher.load_from_csv("AAPL")
        assert loaded is not None
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)

    def test_load_nonexistent_returns_none(self, fetcher):
        assert fetcher.load_from_csv("NONEXISTENT") is None


class TestSymbolValidation:
    def test_valid_symbol_returns_true(self, fetcher):
        assert fetcher.is_valid_symbol("AAPL") is True

    def test_invalid_symbol_returns_false(self, fetcher):
        assert fetcher.is_valid_symbol("ZZZZZZTHISISFAKE") is False

    def test_symbol_uppercased(self, fetcher):
        df = fetcher.fetch("aapl")
        # Should not raise — symbol is uppercased internally
        assert not df.empty
