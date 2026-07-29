"""
test_universe.py — Valida el universo canónico (100 S&P 500).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data_client import load_universe

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = REPO_ROOT / "universe.csv"


class TestUniverse:
    def test_csv_has_exactly_100_unique_tickers(self):
        df = pd.read_csv(UNIVERSE_PATH)
        assert len(df) == 100
        assert df["ticker"].nunique() == 100

    def test_required_columns(self):
        df = pd.read_csv(UNIVERSE_PATH)
        assert {"ticker", "asset_type", "sector", "name"}.issubset(df.columns)

    def test_all_stocks_sp500_style(self):
        df = pd.read_csv(UNIVERSE_PATH)
        assert (df["asset_type"] == "stock").all()
        assert df["sector"].nunique() >= 8  # GICS coverage

    def test_load_universe_finds_repo_root(self):
        """src.data_client.load_universe() resolves repo-root universe.csv."""
        df = load_universe()
        assert len(df) == 100
        assert "AAPL" in df["ticker"].values
        assert "BRK-B" in df["ticker"].values
