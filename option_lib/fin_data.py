"""
Unified Financial Data Facade & Provider Injection (option_lib/fin_data.py)
==========================================================================
Serves as the primary public entry point for financial data queries in option_lib.
Wraps vendor-specific providers (Finnhub, Massive, Yahoo Finance) and manages
data provider injection and composite fallback chains.
"""

import os
from option_lib.data_provider import DataProvider, YahooDataProvider, MassiveDataProvider
from option_lib.finnhub_data import FinnhubDataProvider


class CompositeDataProvider(DataProvider):
    """Composite provider routing queries across an injected primary pricing provider and an ordered fallback chain of earnings providers."""

    def __init__(
        self,
        pricing_provider: DataProvider,
        earnings_providers: list[DataProvider] | DataProvider | None = None,
    ):
        self.pricing_provider = pricing_provider
        if earnings_providers is None:
            self.earnings_providers = [pricing_provider]
        elif isinstance(earnings_providers, list):
            self.earnings_providers = earnings_providers
        else:
            self.earnings_providers = [earnings_providers]

    def get_days_to_expiration(self, expiration_date_str: str) -> int:
        return self.pricing_provider.get_days_to_expiration(expiration_date_str)

    def get_years_to_expiration(self, expiration_date_str: str) -> float:
        return self.pricing_provider.get_years_to_expiration(expiration_date_str)

    def get_stock_info(self, ticker: str) -> dict:
        info = self.pricing_provider.get_stock_info(ticker)
        if isinstance(info, dict) and not info.get("earnings_date"):
            ed = self.get_earnings_date(ticker)
            if ed:
                info["earnings_date"] = ed
        return info

    def get_earnings_date(self, ticker: str) -> str | None:
        """Attempts to fetch earnings date in order across configured earnings providers (e.g. Finnhub -> Yahoo Finance fallback)."""
        for ep in self.earnings_providers:
            try:
                ed = ep.get_earnings_date(ticker)
                if ed:
                    return ed
            except Exception:
                continue
        return None

    def get_sector(self, ticker: str) -> str | None:
        """Sector from the pricing provider.

        Not part of the earnings fallback chain: every provider resolves sector
        through the same Yahoo-backed cache, so asking a second one after the
        first returns None would only repeat the identical lookup.  None is a
        real answer here (ETFs have no sector), not a miss to retry.
        """
        return self.pricing_provider.get_sector(ticker)

    def get_stock_data(self, ticker: str) -> dict | None:
        return self.pricing_provider.get_stock_data(ticker)

    def get_dividend_yield(self, ticker: str) -> float:
        return self.pricing_provider.get_dividend_yield(ticker)

    def calculate_historical_volatility(self, ticker: str, period: str = "1y", days: int = None) -> float | None:
        return self.pricing_provider.calculate_historical_volatility(ticker, period=period, days=days)

    def get_option_chain(self, ticker: str) -> dict:
        return self.pricing_provider.get_option_chain(ticker)

    def get_option_chain_next_months(self, ticker: str, months: int = 6) -> dict:
        return self.pricing_provider.get_option_chain_next_months(ticker, months=months)

    def get_expiration_dates(self, ticker: str) -> list:
        return self.pricing_provider.get_expiration_dates(ticker)

    def get_options_for_expiration(self, ticker: str, expiration_date: str) -> dict:
        return self.pricing_provider.get_options_for_expiration(ticker, expiration_date)

    def get_implied_volatility_for_strike(
        self, ticker: str, expiration_date: str, strike: float, option_type: str = "call", S: float = None, r: float = 0.045
    ) -> float | None:
        return self.pricing_provider.get_implied_volatility_for_strike(
            ticker, expiration_date, strike, option_type=option_type, S=S, r=r
        )

    def get_atm_implied_volatility(
        self, ticker: str, expiration_date: str, current_price: float, option_type: str = "call", r: float = 0.045
    ) -> float | None:
        return self.pricing_provider.get_atm_implied_volatility(
            ticker, expiration_date, current_price, option_type=option_type, r=r
        )

    def fetch_option_theoretical_price(
        self, symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045
    ) -> float | None:
        return self.pricing_provider.fetch_option_theoretical_price(symbol, expiration_iso, strike, option_type, r)

    def fetch_option_delta(
        self, symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045
    ) -> float | None:
        return self.pricing_provider.fetch_option_delta(symbol, expiration_iso, strike, option_type, r)

    def fetch_option_theta(
        self, symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045
    ) -> float | None:
        return self.pricing_provider.fetch_option_theta(symbol, expiration_iso, strike, option_type, r)

    def fetch_option_greeks(
        self, symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045, use_extended: bool = False
    ) -> dict:
        return self.pricing_provider.fetch_option_greeks(symbol, expiration_iso, strike, option_type, r, use_extended)

    def search_ticker(self, query: str, max_results: int = 10) -> list:
        return self.pricing_provider.search_ticker(query, max_results=max_results)


_announced_config: str | None = None


def get_provider(
    pricing_provider: DataProvider | None = None,
    earnings_provider: DataProvider | None = None,
) -> DataProvider:
    """Return configured DataProvider instance using dependency injection.

    If providers are not explicitly passed, inspects environment variables:
    - MASSIVE_API_KEY -> MassiveDataProvider (pricing) vs YahooDataProvider
    - FINNHUB_API_KEY -> FinnhubDataProvider (earnings calendar)
    """
    global _announced_config

    if pricing_provider is None:
        if os.environ.get("MASSIVE_API_KEY"):
            pricing_provider = MassiveDataProvider()
            pricing_name = "Massive.com"
        else:
            pricing_provider = YahooDataProvider()
            pricing_name = "Yahoo Finance"
    else:
        pricing_name = pricing_provider.__class__.__name__

    if earnings_provider is None:
        if os.environ.get("FINNHUB_API_KEY"):
            # Hybrid fallback chain: Finnhub -> pricing_provider -> Yahoo Finance fallback
            earnings_chain = [FinnhubDataProvider(), pricing_provider, YahooDataProvider()]
            earnings_name = "Finnhub (fallback to Yahoo Finance)"
        else:
            earnings_chain = [pricing_provider, YahooDataProvider()]
            earnings_name = pricing_name
    elif isinstance(earnings_provider, list):
        earnings_chain = earnings_provider
        earnings_name = " -> ".join([p.__class__.__name__ for p in earnings_chain])
    else:
        earnings_chain = [earnings_provider]
        earnings_name = earnings_provider.__class__.__name__

    cfg_key = f"pricing:{pricing_name}|earnings:{earnings_name}"
    if _announced_config != cfg_key:
        print(f"Data Providers -> Pricing: {pricing_name} | Earnings: {earnings_name}")
        _announced_config = cfg_key

    return CompositeDataProvider(pricing_provider=pricing_provider, earnings_providers=earnings_chain)


# Module-level convenience wrappers matching DataProvider interface
def get_days_to_expiration(expiration_date_str: str) -> int:
    return get_provider().get_days_to_expiration(expiration_date_str)


def get_years_to_expiration(expiration_date_str: str) -> float:
    return get_provider().get_years_to_expiration(expiration_date_str)


def get_stock_info(ticker: str) -> dict:
    return get_provider().get_stock_info(ticker)


def get_earnings_date(ticker: str) -> str | None:
    return get_provider().get_earnings_date(ticker)


def get_sector(ticker: str) -> str | None:
    return get_provider().get_sector(ticker)


def get_stock_data(ticker: str) -> dict | None:
    return get_provider().get_stock_data(ticker)


def get_dividend_yield(ticker: str) -> float:
    return get_provider().get_dividend_yield(ticker)


def calculate_historical_volatility(ticker: str, period: str = "1y", days: int = None) -> float | None:
    return get_provider().calculate_historical_volatility(ticker, period=period, days=days)


def get_option_chain(ticker: str) -> dict:
    return get_provider().get_option_chain(ticker)


def get_option_chain_next_months(ticker: str, months: int = 6) -> dict:
    return get_provider().get_option_chain_next_months(ticker, months=months)


def get_expiration_dates(ticker: str) -> list:
    return get_provider().get_expiration_dates(ticker)


def get_options_for_expiration(ticker: str, expiration_date: str) -> dict:
    return get_provider().get_options_for_expiration(ticker, expiration_date)


def get_implied_volatility_for_strike(ticker: str, expiration_date: str, strike: float, option_type: str = "call", S: float = None, r: float = 0.045) -> float | None:
    return get_provider().get_implied_volatility_for_strike(ticker, expiration_date, strike, option_type=option_type, S=S, r=r)


def get_atm_implied_volatility(ticker: str, expiration_date: str, current_price: float, option_type: str = "call", r: float = 0.045) -> float | None:
    return get_provider().get_atm_implied_volatility(ticker, expiration_date, current_price, option_type=option_type, r=r)


def fetch_option_theoretical_price(symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045) -> float | None:
    return get_provider().fetch_option_theoretical_price(symbol, expiration_iso, strike, option_type, r)


def fetch_option_delta(symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045) -> float | None:
    return get_provider().fetch_option_delta(symbol, expiration_iso, strike, option_type, r)


def fetch_option_theta(symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045) -> float | None:
    return get_provider().fetch_option_theta(symbol, expiration_iso, strike, option_type, r)


def fetch_option_greeks(symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045, use_extended: bool = False) -> dict:
    return get_provider().fetch_option_greeks(symbol, expiration_iso, strike, option_type, r, use_extended=use_extended)


def search_ticker(query: str, max_results: int = 10) -> list:
    return get_provider().search_ticker(query, max_results=max_results)
