"""
Test Finnhub Earnings Data Fetcher  (option_lib/tests/test_finnhub.py)
======================================================================
Diagnostic script and test suite to verify whether Finnhub API returns
earnings calendar data for tickers (e.g. AAPL, MSFT, NVDA, TSLA).

Usage:
    export FINNHUB_API_KEY="your_api_key"
    python3 -m unittest option_lib/tests/test_finnhub.py
"""

import os
import sys
import unittest
import requests
from datetime import date, datetime, timedelta

# Ensure option_lib root is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from option_lib import finnhub_data


class TestFinnhubEarnings(unittest.TestCase):
    """Test suite to diagnose and verify Finnhub earnings data fetching."""

    def setUp(self):
        self.api_key = os.environ.get("FINNHUB_API_KEY", "").strip()

    def test_api_key_presence(self):
        """Verify FINNHUB_API_KEY environment variable is set."""
        print("\n--- [1/3] Environment Check ---")
        if not self.api_key:
            print("WARNING: FINNHUB_API_KEY is not set in environment.")
            print("Set FINNHUB_API_KEY before running this test to test live API calls.")
        else:
            print(f"FINNHUB_API_KEY is present (length: {len(self.api_key)})")

    def test_direct_finnhub_api_response(self):
        """Test raw HTTP responses from Finnhub endpoints with and without date ranges."""
        print("\n--- [2/3] Direct Finnhub REST API Diagnostic ---")
        if not self.api_key:
            self.skipTest("FINNHUB_API_KEY not set")

        test_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
        today = date.today()
        today_str = today.strftime("%Y-%m-%d")
        future_str = (today + timedelta(days=365)).strftime("%Y-%m-%d")

        for ticker in test_tickers:
            print(f"\nTesting ticker: {ticker}")

            # Test A: Call without date range (Legacy URL)
            url_no_dates = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&token={self.api_key}"
            try:
                resp = requests.get(url_no_dates, timeout=10)
                status = resp.status_code
                data = resp.json() if status == 200 else {}
                cal_len = len(data.get("earningsCalendar", [])) if isinstance(data, dict) else 0
                print(f"  [No Date Range] Status: {status} | Calendar items returned: {cal_len}")
            except Exception as exc:
                print(f"  [No Date Range] Exception: {exc}")

            # Test B: Call with date range (from=today to=today+365d)
            url_with_dates = (
                f"https://finnhub.io/api/v1/calendar/earnings"
                f"?symbol={ticker}&from={today_str}&to={future_str}&token={self.api_key}"
            )
            try:
                resp = requests.get(url_with_dates, timeout=10)
                status = resp.status_code
                data = resp.json() if status == 200 else {}
                calendar = data.get("earningsCalendar", []) if isinstance(data, dict) else []
                print(f"  [With Date Range: {today_str} to {future_str}] Status: {status} | Calendar items: {len(calendar)}")
                if calendar and isinstance(calendar, list):
                    for entry in calendar[:3]:
                        print(f"    -> Date: {entry.get('date')} | Symbol: {entry.get('symbol')} | EPS Estimate: {entry.get('epsEstimate')}")
            except Exception as exc:
                print(f"  [With Date Range] Exception: {exc}")

    def test_finnhub_data_module_fetcher(self):
        """Test option_lib.finnhub_data.get_earnings_date function."""
        print("\n--- [3/3] option_lib.finnhub_data.get_earnings_date Test ---")
        if not self.api_key:
            self.skipTest("FINNHUB_API_KEY not set")

        test_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        for ticker in test_tickers:
            date_result = finnhub_data.get_earnings_date(ticker)
            print(f"  Ticker {ticker:5s} -> Next Earnings Date: {date_result}")


def run_standalone_diagnostic():
    """CLI runner to print diagnostic details without needing unittest flags."""
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    print("=======================================================")
    print("  FINNHUB EARNINGS DATA DIAGNOSTIC RUNNER")
    print("=======================================================")
    if not key:
        print("ERROR: FINNHUB_API_KEY is not set.")
        print("Run with: FINNHUB_API_KEY='your_key' python3 test_finnhub.py")
        return

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    future_str = (today + timedelta(days=365)).strftime("%Y-%m-%d")

    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    for ticker in tickers:
        url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&from={today_str}&to={future_str}&token={key}"
        try:
            r = requests.get(url, timeout=10)
            print(f"\nTicker: {ticker}")
            print(f"HTTP Status: {r.status_code}")
            if r.status_code == 200:
                body = r.json()
                cal = body.get("earningsCalendar", [])
                print(f"Calendar Count: {len(cal)}")
                if cal:
                    print(f"First Entry: {cal[0]}")
                module_date = finnhub_data.get_earnings_date(ticker)
                print(f"Parsed Date from finnhub_data module: {module_date}")
            else:
                print(f"Error Response: {r.text}")
        except Exception as e:
            print(f"Error for {ticker}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--standalone":
        run_standalone_diagnostic()
    else:
        unittest.main()
