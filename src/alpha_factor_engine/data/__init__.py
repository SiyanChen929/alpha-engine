"""Data-source adapters."""

from .fred import FredClient
from .marketdata import MarketDataClient
from .sec import SecCompanyFactsClient
from .yahoo import YahooFinanceClient

__all__ = ["FredClient", "MarketDataClient", "SecCompanyFactsClient", "YahooFinanceClient"]

