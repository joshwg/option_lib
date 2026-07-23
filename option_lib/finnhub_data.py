"""
Finnhub Data Fetcher  (option_lib/finnhub_data.py)
=================================================
Fetches earnings calendar data from Finnhub.io API.
"""

import os
import requests
from datetime import datetime, date, timedelta
from option_lib.math_util import TTLCache

_cache = TTLCache()
_TTL_EARNINGS = int(os.environ.get('CACHE_TTL_EARNINGS', 86400))  # 24 h


def get_earnings_date(ticker: str) -> str | None:
    """Return the next upcoming earnings date as 'YYYY-MM-DD', or None.

    Reads FINNHUB_API_KEY from environment.  If not set or request fails,
    returns None to allow fallback to other providers (e.g. Yahoo Finance).
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return None

    ticker_upper = ticker.strip().upper()
    cache_key = ('finnhub_get_earnings_date', ticker_upper)
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    future_str = (today + timedelta(days=365)).strftime("%Y-%m-%d")

    url = (
        f"https://finnhub.io/api/v1/calendar/earnings"
        f"?symbol={ticker_upper}&from={today_str}&to={future_str}&token={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            _cache.set(cache_key, None, 60)
            return None

        data = resp.json()
        calendar = data.get("earningsCalendar", [])
        if not calendar or not isinstance(calendar, list):
            _cache.set(cache_key, None, 120)  # Short TTL for empty responses
            return None

        today = date.today()
        future_dates = []

        for entry in calendar:
            d_str = entry.get("date")
            if not d_str:
                continue
            try:
                ed = datetime.strptime(d_str[:10], "%Y-%m-%d").date()
                if ed >= today:
                    future_dates.append((ed, d_str[:10]))
            except (ValueError, TypeError):
                continue

        if future_dates:
            future_dates.sort(key=lambda x: x[0])
            result = future_dates[0][1]
            _cache.set(cache_key, result, _TTL_EARNINGS)
            return result

        _cache.set(cache_key, None, 120)  # Short TTL for empty future dates
        return None

    except Exception:
        _cache.set(cache_key, None, 60)
        return None


def clear_cache():
    """Clear all in-memory TTLCache entries for finnhub_data."""
    _cache.clear()



class FinnhubDataProvider:
    """DataProvider implementation backed by Finnhub.io.

    Provides earnings calendar data. For market pricing, options, and volatility
    methods not available on Finnhub's free tier, delegates to an optional fallback_provider.
    """

    def __init__(self, fallback_provider=None):
        self.fallback_provider = fallback_provider

    def get_earnings_date(self, ticker: str) -> str | None:
        try:
            fh_date = get_earnings_date(ticker)
            if fh_date:
                return fh_date
        except Exception:
            pass
        if self.fallback_provider:
            return self.fallback_provider.get_earnings_date(ticker)
        return None

    def get_days_to_expiration(self, expiration_date_str: str) -> int:
        if self.fallback_provider:
            return self.fallback_provider.get_days_to_expiration(expiration_date_str)
        from datetime import datetime, date
        exp = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
        return max(0, (exp - date.today()).days)

    def get_years_to_expiration(self, expiration_date_str: str) -> float:
        return self.get_days_to_expiration(expiration_date_str) / 365.0

    def get_stock_info(self, ticker: str) -> dict:
        if self.fallback_provider:
            return self.fallback_provider.get_stock_info(ticker)
        return {"ticker": ticker, "success": False}

    def get_stock_data(self, ticker: str) -> dict | None:
        if self.fallback_provider:
            return self.fallback_provider.get_stock_data(ticker)
        return None

    def get_dividend_yield(self, ticker: str) -> float:
        if self.fallback_provider:
            return self.fallback_provider.get_dividend_yield(ticker)
        return 0.0

    def calculate_historical_volatility(self, ticker: str, period: str = "1y", days: int = None) -> float | None:
        if self.fallback_provider:
            return self.fallback_provider.calculate_historical_volatility(ticker, period=period, days=days)
        return None

    def get_option_chain(self, ticker: str) -> dict:
        if self.fallback_provider:
            return self.fallback_provider.get_option_chain(ticker)
        return {"success": False, "ticker": ticker, "expirations": []}

    def get_option_chain_next_months(self, ticker: str, months: int = 6) -> dict:
        if self.fallback_provider:
            return self.fallback_provider.get_option_chain_next_months(ticker, months=months)
        return {"success": False, "ticker": ticker, "expirations": []}

    def get_expiration_dates(self, ticker: str) -> list:
        if self.fallback_provider:
            return self.fallback_provider.get_expiration_dates(ticker)
        return []

    def get_options_for_expiration(self, ticker: str, expiration_date: str) -> dict:
        if self.fallback_provider:
            return self.fallback_provider.get_options_for_expiration(ticker, expiration_date)
        return {"success": False, "expiration": expiration_date, "calls": [], "puts": []}

    def get_implied_volatility_for_strike(self, ticker: str, expiration_date: str, strike: float, option_type: str = "call", S: float = None, r: float = 0.045) -> float | None:
        if self.fallback_provider:
            return self.fallback_provider.get_implied_volatility_for_strike(ticker, expiration_date, strike, option_type, S, r)
        return None

    def get_atm_implied_volatility(self, ticker: str, expiration_date: str, current_price: float, option_type: str = "call", r: float = 0.045) -> float | None:
        if self.fallback_provider:
            return self.fallback_provider.get_atm_implied_volatility(ticker, expiration_date, current_price, option_type, r)
        return None

    def fetch_option_theoretical_price(self, symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045) -> float | None:
        if self.fallback_provider:
            return self.fallback_provider.fetch_option_theoretical_price(symbol, expiration_iso, strike, option_type, r)
        return None

    def fetch_option_delta(self, symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045) -> float | None:
        if self.fallback_provider:
            return self.fallback_provider.fetch_option_delta(symbol, expiration_iso, strike, option_type, r)
        return None

    def fetch_option_theta(self, symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045) -> float | None:
        if self.fallback_provider:
            return self.fallback_provider.fetch_option_theta(symbol, expiration_iso, strike, option_type, r)
        return None

    def fetch_option_greeks(self, symbol: str, expiration_iso: str, strike: float, option_type: str, r: float = 0.045, use_extended: bool = False) -> dict:
        if self.fallback_provider:
            return self.fallback_provider.fetch_option_greeks(symbol, expiration_iso, strike, option_type, r, use_extended)
        return {"price": None, "theta": None, "delta": None}

    def search_ticker(self, query: str, max_results: int = 10) -> list:
        if self.fallback_provider:
            return self.fallback_provider.search_ticker(query, max_results=max_results)
        return []

