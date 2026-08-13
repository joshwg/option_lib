"""
Data Provider abstraction  (option_lib/data_provider.py)
=========================================================
Defines the DataProvider abstract base class and concrete implementations for
Yahoo Finance and Massive.com.  Use get_provider() to obtain the correct
provider based on the runtime environment.

Usage
-----
    from option_lib.data_provider import get_provider

    data = get_provider()
    info = data.get_stock_info("AAPL")
    chain = data.get_option_chain("AAPL")

The factory checks MASSIVE_API_KEY at call time (not import time) so the
provider can be switched between tests or across requests without restarting.
"""

import os
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class DataProvider(ABC):
    """Market data interface.  All concrete providers must implement every method."""

    # ── Pure date-math helpers ────────────────────────────────────────────────

    @abstractmethod
    def get_days_to_expiration(self, expiration_date_str: str) -> int:
        """Calendar days from today to expiration_date_str ('YYYY-MM-DD'), min 0."""

    @abstractmethod
    def get_years_to_expiration(self, expiration_date_str: str) -> float:
        """Years from today to expiration (for Black-Scholes T)."""

    # ── Stock fundamentals ────────────────────────────────────────────────────

    @abstractmethod
    def get_stock_info(self, ticker: str) -> dict:
        """Current stock snapshot.

        Required keys in the returned dict:
            ticker, current_price, company_name, previous_close,
            volume, avg_volume, market_cap, dividend_yield,
            earnings_date, success

        The Yahoo path also carries 'sector', because its .info call already
        has it and get_sector() reuses that to avoid a second fetch.  Massive's
        does not — it has no sector to give.  Do not read info['sector']; call
        get_sector(), which is provider-independent and cached across restarts.
        """

    @abstractmethod
    def get_earnings_date(self, ticker: str) -> str | None:
        """Next upcoming earnings date as 'YYYY-MM-DD', or None."""

    @abstractmethod
    def get_sector(self, ticker: str) -> str | None:
        """Company sector (e.g. 'Technology'), or None for ETFs and the like."""

    @abstractmethod
    def get_company_name(self, ticker: str) -> str | None:
        """Company display name (e.g. 'Apple Inc.'), or None if unknown.

        Like get_sector(), this is cached on disk across restarts, so it is
        cheap enough to call per row of a table.  Callers should fall back to
        showing the ticker itself when it returns None.
        """

    @abstractmethod
    def get_stock_data(self, ticker: str) -> dict | None:
        """Kivy-compatible stock dict, or None on failure."""

    @abstractmethod
    def get_dividend_yield(self, ticker: str) -> float:
        """Dividend yield as a decimal (e.g. 0.02 for 2%)."""

    # ── Historical volatility ─────────────────────────────────────────────────

    @abstractmethod
    def calculate_historical_volatility(
        self,
        ticker: str,
        period: str = "1y",
        days: int = None,
    ) -> float | None:
        """Annualised historical volatility as a decimal, or None on failure."""

    # ── Option chain ──────────────────────────────────────────────────────────

    @abstractmethod
    def get_option_chain(self, ticker: str) -> dict:
        """All available expiration dates.

        Returns: {'success': bool, 'ticker': str, 'expirations': [str, ...]}
        """

    @abstractmethod
    def get_option_chain_next_months(self, ticker: str, months: int = 6) -> dict:
        """Expiration dates within the next *months* months."""

    @abstractmethod
    def get_expiration_dates(self, ticker: str) -> list:
        """Flat list of expiration date strings (Kivy-compatible)."""

    @abstractmethod
    def get_options_for_expiration(self, ticker: str, expiration_date: str) -> dict:
        """Calls and puts for a specific expiration date.

        Returns: {'success': bool, 'expiration': str,
                  'calls': [row, ...], 'puts': [row, ...]}
        Each row: strike, last_price, bid, ask, volume, open_interest,
                  implied_volatility
        """

    # ── Implied volatility ────────────────────────────────────────────────────

    @abstractmethod
    def get_implied_volatility_for_strike(
        self,
        ticker: str,
        expiration_date: str,
        strike: float,
        option_type: str = "call",
        S: float = None,
        r: float = 0.045,
    ) -> float | None:
        """IV for the closest-matching strike/expiration, or None."""

    @abstractmethod
    def get_atm_implied_volatility(
        self,
        ticker: str,
        expiration_date: str,
        current_price: float,
        option_type: str = "call",
        r: float = 0.045,
    ) -> float | None:
        """ATM IV for an expiration, or None."""

    # ── Theoretical price and greeks ──────────────────────────────────────────

    @abstractmethod
    def fetch_option_theoretical_price(
        self,
        symbol: str,
        expiration_iso: str,
        strike: float,
        option_type: str,
        r: float = 0.045,
    ) -> float | None:
        """American-binomial theoretical price, or None."""

    @abstractmethod
    def fetch_option_delta(
        self,
        symbol: str,
        expiration_iso: str,
        strike: float,
        option_type: str,
        r: float = 0.045,
    ) -> float | None:
        """Probability of assignment (absolute American delta), or None."""

    @abstractmethod
    def fetch_option_theta(
        self,
        symbol: str,
        expiration_iso: str,
        strike: float,
        option_type: str,
        r: float = 0.045,
    ) -> float | None:
        """American-binomial theta per share per day (negative = decay), or None."""

    @abstractmethod
    def fetch_option_greeks(
        self,
        symbol: str,
        expiration_iso: str,
        strike: float,
        option_type: str,
        r: float = 0.045,
        use_extended: bool = False,
    ) -> dict:
        """Price, theta, and delta in one round-trip.

        Returns dict with keys 'price', 'theta', 'delta'; any may be None.
        When use_extended=True, uses the extended-hours stock price as S.
        """

    # ── Ticker search ─────────────────────────────────────────────────────────

    @abstractmethod
    def search_ticker(self, query: str, max_results: int = 10) -> list:
        """Search for tickers by name or symbol fragment."""


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: Yahoo Finance
# ─────────────────────────────────────────────────────────────────────────────

class YahooDataProvider(DataProvider):
    """DataProvider backed by Yahoo Finance (via option_lib.yahoo_data)."""

    def __init__(self):
        from option_lib import yahoo_data as _mod
        self._m = _mod

    def get_days_to_expiration(self, expiration_date_str):
        return self._m.get_days_to_expiration(expiration_date_str)

    def get_years_to_expiration(self, expiration_date_str):
        return self._m.get_years_to_expiration(expiration_date_str)

    def get_stock_info(self, ticker):
        return self._m.get_stock_info(ticker)

    def get_earnings_date(self, ticker):
        return self._m.get_earnings_date(ticker)

    def get_sector(self, ticker):
        return self._m.get_sector(ticker)

    def get_company_name(self, ticker):
        return self._m.get_company_name(ticker)

    def get_stock_data(self, ticker):
        return self._m.get_stock_data(ticker)

    def get_dividend_yield(self, ticker):
        return self._m.get_dividend_yield(ticker)

    def calculate_historical_volatility(self, ticker, period="1y", days=None):
        return self._m.calculate_historical_volatility(ticker, period=period, days=days)

    def get_option_chain(self, ticker):
        return self._m.get_option_chain(ticker)

    def get_option_chain_next_months(self, ticker, months=6):
        return self._m.get_option_chain_next_months(ticker, months=months)

    def get_expiration_dates(self, ticker):
        return self._m.get_expiration_dates(ticker)

    def get_options_for_expiration(self, ticker, expiration_date):
        return self._m.get_options_for_expiration(ticker, expiration_date)

    def get_implied_volatility_for_strike(
        self, ticker, expiration_date, strike,
        option_type="call", S=None, r=0.045,
    ):
        return self._m.get_implied_volatility_for_strike(
            ticker, expiration_date, strike, option_type=option_type, S=S, r=r
        )

    def get_atm_implied_volatility(
        self, ticker, expiration_date, current_price,
        option_type="call", r=0.045,
    ):
        return self._m.get_atm_implied_volatility(
            ticker, expiration_date, current_price, option_type=option_type, r=r
        )

    def fetch_option_theoretical_price(
        self, symbol, expiration_iso, strike, option_type, r=0.045
    ):
        return self._m.fetch_option_theoretical_price(
            symbol, expiration_iso, strike, option_type, r
        )

    def fetch_option_delta(self, symbol, expiration_iso, strike, option_type, r=0.045):
        return self._m.fetch_option_delta(symbol, expiration_iso, strike, option_type, r)

    def fetch_option_theta(self, symbol, expiration_iso, strike, option_type, r=0.045):
        return self._m.fetch_option_theta(symbol, expiration_iso, strike, option_type, r)

    def fetch_option_greeks(self, symbol, expiration_iso, strike, option_type,
                            r=0.045, use_extended=False):
        return self._m.fetch_option_greeks(
            symbol, expiration_iso, strike, option_type, r, use_extended=use_extended
        )

    def search_ticker(self, query, max_results=10):
        return self._m.search_ticker(query, max_results=max_results)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: Massive.com
# ─────────────────────────────────────────────────────────────────────────────

class MassiveDataProvider(DataProvider):
    """DataProvider backed by Massive.com (via option_lib.massive_data).

    Massive.com is used for all pricing and options data.  Internally,
    massive_data falls back to yahoo_data for fields Massive does not provide
    (e.g. earnings dates) and on any API error.

    Requires MASSIVE_API_KEY to be set in the environment.
    """

    def __init__(self):
        from option_lib import massive_data as _mod
        self._m = _mod

    def get_days_to_expiration(self, expiration_date_str):
        return self._m.get_days_to_expiration(expiration_date_str)

    def get_years_to_expiration(self, expiration_date_str):
        return self._m.get_years_to_expiration(expiration_date_str)

    def get_stock_info(self, ticker):
        return self._m.get_stock_info(ticker)

    def get_earnings_date(self, ticker):
        return self._m.get_earnings_date(ticker)

    def get_sector(self, ticker):
        return self._m.get_sector(ticker)

    def get_company_name(self, ticker):
        return self._m.get_company_name(ticker)

    def get_stock_data(self, ticker):
        return self._m.get_stock_data(ticker)

    def get_dividend_yield(self, ticker):
        return self._m.get_dividend_yield(ticker)

    def calculate_historical_volatility(self, ticker, period="1y", days=None):
        return self._m.calculate_historical_volatility(ticker, period=period, days=days)

    def get_option_chain(self, ticker):
        return self._m.get_option_chain(ticker)

    def get_option_chain_next_months(self, ticker, months=6):
        return self._m.get_option_chain_next_months(ticker, months=months)

    def get_expiration_dates(self, ticker):
        return self._m.get_expiration_dates(ticker)

    def get_options_for_expiration(self, ticker, expiration_date):
        return self._m.get_options_for_expiration(ticker, expiration_date)

    def get_implied_volatility_for_strike(
        self, ticker, expiration_date, strike,
        option_type="call", S=None, r=0.045,
    ):
        return self._m.get_implied_volatility_for_strike(
            ticker, expiration_date, strike, option_type=option_type, S=S, r=r
        )

    def get_atm_implied_volatility(
        self, ticker, expiration_date, current_price,
        option_type="call", r=0.045,
    ):
        return self._m.get_atm_implied_volatility(
            ticker, expiration_date, current_price, option_type=option_type, r=r
        )

    def fetch_option_theoretical_price(
        self, symbol, expiration_iso, strike, option_type, r=0.045
    ):
        return self._m.fetch_option_theoretical_price(
            symbol, expiration_iso, strike, option_type, r
        )

    def fetch_option_delta(self, symbol, expiration_iso, strike, option_type, r=0.045):
        return self._m.fetch_option_delta(symbol, expiration_iso, strike, option_type, r)

    def fetch_option_theta(self, symbol, expiration_iso, strike, option_type, r=0.045):
        return self._m.fetch_option_theta(symbol, expiration_iso, strike, option_type, r)

    def fetch_option_greeks(self, symbol, expiration_iso, strike, option_type,
                            r=0.045, use_extended=False):
        return self._m.fetch_option_greeks(
            symbol, expiration_iso, strike, option_type, r, use_extended=use_extended
        )

    def search_ticker(self, query, max_results=10):
        return self._m.search_ticker(query, max_results=max_results)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_announced_provider: str | None = None   # tracks last-printed provider name


def get_provider() -> DataProvider:
    """Return the active DataProvider instance via option_lib.fin_data.get_provider()."""
    from option_lib import fin_data
    return fin_data.get_provider()

