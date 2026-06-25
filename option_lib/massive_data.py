"""
Massive.com Financial Data Provider  (option_lib/massive_data.py)
=================================================================
Drop-in replacement for yahoo_data.py that sources option-chain and pricing
data from the Massive.com REST API (https://api.massive.com).

Set the ``MASSIVE_API_KEY`` environment variable to your Massive API key.
All public functions fall back to yahoo_data transparently when:
  - MASSIVE_API_KEY is not set, or
  - the Massive API call fails (network error, rate-limit, bad data, etc.)

Interface is identical to yahoo_data.py so callers need only swap the import.
"""

import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

import numpy as np
import pytz
import requests

from option_lib import pricing as _pricing
from option_lib import yahoo_data as _yahoo      # fallback
from option_lib.math_util import (
    safe_float as _safe_float,
    TTLCache as _TTLCache,
    iv_from_mid as _iv_from_mid,
    get_days_to_expiration,
    get_years_to_expiration,
)



# ── Config ─────────────────────────────────────────────────────────────────────

MASSIVE_BASE = "https://api.massive.com"

_TTL_STOCK    = int(os.environ.get("CACHE_TTL_STOCK",    300))
_TTL_OPTIONS  = int(os.environ.get("CACHE_TTL_OPTIONS",  300))
_TTL_VOL      = int(os.environ.get("CACHE_TTL_VOL",     1800))
_TTL_EXPIRIES = int(os.environ.get("CACHE_TTL_EXPIRIES", 1800))
_TTL_BARS     = int(os.environ.get("CACHE_TTL_BARS",    1800))


# ── Cache ──────────────────────────────────────────────────────────────────────

_cache = _TTLCache()


# ── HTTP helpers ────────────────────────────────────────────────────────────────

def _api_key() -> str:
    key = os.environ.get("MASSIVE_API_KEY", "")
    if not key:
        raise RuntimeError("MASSIVE_API_KEY environment variable is not set")
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {_api_key()}"}


# ── Rate limiting — ≤50 calls/second (sliding window) ──────────────────────────
# Tracks timestamps of the last 50 calls; if the oldest is < 1 s ago we sleep
# until the window opens.  The lock serialises concurrent background threads.

_rl_lock = threading.Lock()
_rl_times: deque = deque(maxlen=50)   # timestamps of the last 50 calls
_MASSIVE_MAX_PER_SEC = 50


def _massive_rate_limit() -> None:
    """Block until the next Massive API call is within the 50 req/s budget."""
    with _rl_lock:
        now = time.monotonic()
        if len(_rl_times) == _MASSIVE_MAX_PER_SEC:
            wait = 1.0 - (now - _rl_times[0])
            if wait > 0:
                time.sleep(wait)
        _rl_times.append(time.monotonic())


def _get(path: str, params: dict | None = None, *, full_url: str = "") -> dict:
    """Authenticated GET to Massive API.  Use full_url when following next_url."""
    _massive_rate_limit()
    url  = full_url if full_url else (MASSIVE_BASE + path)
    resp = requests.get(url, headers=_headers(), params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()





def _is_key_set() -> bool:
    return bool(os.environ.get("MASSIVE_API_KEY", ""))


# ── Pure-math helpers (identical to yahoo_data) ────────────────────────────────




# ── Internal: daily bars (shared by HV and avg-volume) ────────────────────────

def _get_daily_bars(ticker: str, lookback_days: int = 400) -> list:
    """Fetch lookback_days of adjusted daily bars from Massive.

    Returns list of bar dicts with at least {'c': close, 'v': volume}.
    Cached for _TTL_BARS seconds.  Returns [] on any error.
    """
    cache_key = ("massive_bars", ticker.upper(), lookback_days)
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    today_str = datetime.now().strftime("%Y-%m-%d")
    from_str  = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        resp = _get(
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{from_str}/{today_str}",
            {"adjusted": "true", "sort": "asc", "limit": 500},
        )
        bars = resp.get("results") or []
        _cache.set(cache_key, bars, _TTL_BARS)
        return bars
    except Exception as e:
        print(f"Massive: error fetching daily bars for {ticker}: {e}")
        return []


# ── Stock info ─────────────────────────────────────────────────────────────────

def get_stock_info(ticker: str) -> dict:
    """Current stock information from Massive, falling back to yahoo_data.

    Returned dict keys match yahoo_data.get_stock_info() plus:
      avg_volume  - 63-trading-day (~3 month) average volume
    """
    if not _is_key_set():
        return _yahoo.get_stock_info(ticker)

    cache_key = ("massive_stock_info", ticker.upper())
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    try:
        # 1) Ticker snapshot - price and volume
        snap = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}")
        t = snap.get("ticker") or {}
        if not t:
            raise ValueError("No ticker snapshot results")
        
        day = t.get("day") or {}
        min_bar = t.get("min") or {}
        prev_day = t.get("prevDay") or {}

        current_price = _safe_float(day.get("c") or min_bar.get("c") or prev_day.get("c"))
        volume = _safe_float(day.get("v") or prev_day.get("v"), 0)
        prev_close = _safe_float(prev_day.get("c"))

        # Extended-hours prices — Polygon surfaces these directly on the snapshot.
        # Fall back to the last-minute bar when it differs from the regular close
        # (covers providers that don't populate the explicit fields).
        pre_market_price  = _safe_float(t.get("preMarket"))
        post_market_price = _safe_float(t.get("afterHours"))
        if not pre_market_price and not post_market_price:
            _min_c = _safe_float(min_bar.get("c"))
            _day_c = _safe_float(day.get("c"))
            if _min_c and _day_c and abs(_min_c - _day_c) > 0.001:
                post_market_price = _min_c   # most recent trade; direction unknown

        # 2) Reference data - name, market cap, dividend yield
        market_cap     = None
        company_name   = ticker
        dividend_yield = 0.0
        try:
            ref = _get(f"/v3/reference/tickers/{ticker.upper()}")
            r   = ref.get("results") or {}
            if r:
                company_name = r.get("name", ticker)
                market_cap   = _safe_float(r.get("market_cap"))
                div_raw      = _safe_float(r.get("dividend_yield"))
                if div_raw and div_raw > 0.10:
                    dividend_yield = div_raw / 100   # pct to decimal
                elif div_raw:
                    dividend_yield = div_raw
        except Exception:
            pass  # non-fatal; continue with partial data

        # 3) Average 3-month volume from cached bars (~100 calendar days = ~63 trading days)
        avg_volume = volume  # safe fallback
        try:
            bars = _get_daily_bars(ticker, lookback_days=100)
            if bars:
                vols = [b["v"] for b in bars if b.get("v") is not None]
                if vols:
                    avg_volume = float(np.mean(vols[-63:]))  # last 63 trading days
        except Exception:
            pass

        result = {
            "ticker":             ticker,
            "current_price":      current_price,
            "company_name":       company_name,
            "previous_close":     prev_close,
            "volume":             volume,
            "avg_volume":         avg_volume,
            "market_cap":         market_cap,
            "dividend_yield":     dividend_yield,
            "earnings_date":      None,   # fetched separately via get_earnings_date()
            "pre_market_price":   pre_market_price,
            "post_market_price":  post_market_price,
            "success":            True,
            "_source":            "massive",
        }
        _cache.set(cache_key, result, _TTL_STOCK)
        return result

    except Exception as e:
        # Full fallback to yahoo_data
        result = _yahoo.get_stock_info(ticker)
        if result.get("success"):
            result["_source"] = "yahoo_fallback"
        return result


# ── Earnings date ─────────────────────────────────────────────────────────────

def get_earnings_date(ticker: str) -> str | None:
    """Next upcoming earnings date as 'YYYY-MM-DD', or None.

    Massive/Polygon has no upcoming-earnings endpoint, so this always delegates
    to yahoo_data.get_earnings_date() which uses stock.calendar (lighter than
    stock.info) and caches the result for 24 hours.
    """
    return _yahoo.get_earnings_date(ticker)


# ── Historical volatility ──────────────────────────────────────────────────────

def calculate_historical_volatility(
    ticker: str, period: str = "1y", days: int = None
) -> float | None:
    """Annualised historical volatility from Massive daily bars.

    Parameters and return value identical to yahoo_data.calculate_historical_volatility().
    Falls back to yahoo_data on any error.
    """
    if not _is_key_set():
        return _yahoo.calculate_historical_volatility(ticker, period=period, days=days)

    if days is not None:
        if days <= 30:    period = "1mo"
        elif days <= 90:  period = "3mo"
        elif days <= 180: period = "6mo"
        elif days <= 365: period = "1y"
        elif days <= 730: period = "2y"
        else:             period = "5y"

    cache_key = ("massive_hv", ticker.upper(), period)
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    period_days = {
        "1mo": 40, "3mo": 100, "6mo": 195,
        "1y": 380, "2y": 750, "5y": 1850,
    }
    lookback = period_days.get(period, 380)

    try:
        bars = _get_daily_bars(ticker, lookback_days=lookback)
        if len(bars) < 2:
            return _yahoo.calculate_historical_volatility(ticker, period=period)

        closes      = np.array([b["c"] for b in bars])
        log_returns = np.log(closes[1:] / closes[:-1])
        vol         = float(log_returns.std() * np.sqrt(252))

        _cache.set(cache_key, vol, _TTL_VOL)
        return vol
    except Exception as e:
        print(f"Massive: error computing HV for {ticker}: {e}")
        return _yahoo.calculate_historical_volatility(ticker, period=period)


# ── Option chain snapshot (internal, paginated) ────────────────────────────────

def _fetch_option_snapshot(ticker: str) -> list:
    """Fetch ALL option contracts for ticker from the Massive snapshot endpoint.

    Follows next_url pagination automatically (up to 25 pages).
    Result is cached for _TTL_OPTIONS seconds.
    Returns [] on any error.
    """
    cache_key = ("massive_snapshot", ticker.upper())
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    all_results = []
    path        = f"/v3/snapshot/options/{ticker.upper()}"
    params      = {"limit": 250}
    follow_url  = ""
    MAX_PAGES   = 25

    for _ in range(MAX_PAGES):
        try:
            resp = _get(path, params, full_url=follow_url)
        except Exception as e:
            print(f"Massive: error fetching option snapshot for {ticker}: {e}")
            break

        all_results.extend(resp.get("results") or [])

        next_url = resp.get("next_url", "")
        if not next_url:
            break

        # Follow pagination - strip apiKey, keep cursor
        parsed     = urlparse(next_url)
        path       = parsed.path
        params     = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        follow_url = next_url
        params.pop("apiKey", None)

    _cache.set(cache_key, all_results, _TTL_OPTIONS)
    return all_results


# ── Option chain ───────────────────────────────────────────────────────────────

def get_option_chain(ticker: str) -> dict:
    """Available option expiration dates for ticker from Massive.

    Return format matches yahoo_data.get_option_chain().
    Falls back to yahoo_data on error or missing key.
    """
    if not _is_key_set():
        return _yahoo.get_option_chain(ticker)

    cache_key = ("massive_chain", ticker.upper())
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    try:
        contracts = _fetch_option_snapshot(ticker)
        if not contracts:
            return _yahoo.get_option_chain(ticker)

        expirations = sorted(set(
            c["details"]["expiration_date"]
            for c in contracts
            if c.get("details", {}).get("expiration_date")
        ))

        if not expirations:
            return _yahoo.get_option_chain(ticker)

        result = {"success": True, "ticker": ticker, "expirations": expirations}
        _cache.set(cache_key, result, _TTL_EXPIRIES)
        return result
    except Exception:
        return _yahoo.get_option_chain(ticker)


def get_option_chain_next_months(ticker: str, months: int = 6) -> dict:
    """Option chain filtered to the next months months.

    Excludes today's expiration after 4 PM ET.
    Return format matches yahoo_data.get_option_chain_next_months().
    """
    try:
        chain = get_option_chain(ticker)
        if not chain.get("success"):
            return chain

        eastern = pytz.timezone("US/Eastern")
        now_et  = datetime.now(eastern)
        today   = datetime.now()
        cutoff  = today + timedelta(days=months * 30)

        filtered = []
        for exp_str in chain["expirations"]:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            if exp_date.date() == today.date() and now_et.hour >= 16:
                continue
            if today.date() <= exp_date.date() <= cutoff.date():
                filtered.append(exp_str)

        return {
            "success":         True,
            "ticker":          ticker,
            "expirations":     filtered,
            "all_expirations": chain["expirations"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_options_for_expiration(ticker: str, expiration_date: str) -> dict:
    """Puts and calls for a specific expiration date from the Massive snapshot.

    Return format matches yahoo_data.get_options_for_expiration().
    Falls back to yahoo_data on error or empty results.
    """
    if not _is_key_set():
        return _yahoo.get_options_for_expiration(ticker, expiration_date)

    cache_key = ("massive_opts", ticker.upper(), expiration_date)
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    try:
        contracts = _fetch_option_snapshot(ticker)
        if not contracts:
            return _yahoo.get_options_for_expiration(ticker, expiration_date)

        calls_data = []
        puts_data  = []

        for c in contracts:
            details       = c.get("details") or {}
            if details.get("expiration_date") != expiration_date:
                continue

            strike        = _safe_float(details.get("strike_price"), 0)
            contract_type = (details.get("contract_type") or "").lower()

            # Bid/ask from last_quote (field name varies by API version)
            lq  = c.get("last_quote") or {}
            bid = _safe_float(lq.get("bid_price") or lq.get("bid"), 0)
            ask = _safe_float(lq.get("ask_price") or lq.get("ask"), 0)

            lt         = c.get("last_trade") or {}
            last_price = _safe_float(lt.get("price"), 0)

            day    = c.get("day") or {}
            volume = _safe_float(day.get("volume") or day.get("v"), 0)

            oi = _safe_float(c.get("open_interest"), 0)
            iv = _safe_float(c.get("implied_volatility"), 0)

            row = {
                "strike":             strike,
                "last_price":         last_price,
                "bid":                bid,
                "ask":                ask,
                "volume":             volume,
                "open_interest":      oi,
                "implied_volatility": iv,
            }

            if contract_type == "call":
                calls_data.append(row)
            elif contract_type == "put":
                puts_data.append(row)

        if not calls_data and not puts_data:
            return _yahoo.get_options_for_expiration(ticker, expiration_date)

        result = {
            "success":    True,
            "expiration": expiration_date,
            "calls":      sorted(calls_data, key=lambda x: x["strike"]),
            "puts":       sorted(puts_data,  key=lambda x: x["strike"]),
        }
        _cache.set(cache_key, result, _TTL_OPTIONS)
        return result

    except Exception:
        return _yahoo.get_options_for_expiration(ticker, expiration_date)


# ── Implied volatility helpers ─────────────────────────────────────────────────




def get_implied_volatility_for_strike(
    ticker: str, expiration_date: str, strike: float,
    option_type: str = "call", S: float = None, r: float = 0.045,
) -> float | None:
    """IV for a specific strike/expiration.  Falls back to yahoo_data when key absent."""
    if not _is_key_set():
        return _yahoo.get_implied_volatility_for_strike(
            ticker, expiration_date, strike, option_type=option_type, S=S, r=r
        )
    try:
        options = get_options_for_expiration(ticker, expiration_date)
        if not options.get("success"):
            return None
        chain = options["calls"] if option_type == "call" else options["puts"]
        best  = min(chain, key=lambda o: abs(o["strike"] - strike), default=None)
        if best is None:
            return None
        if S is None:
            info = get_stock_info(ticker)
            S    = info.get("current_price") if info.get("success") else None
        T  = get_years_to_expiration(expiration_date)
        iv = _iv_from_mid(best["bid"], best["ask"], S, best["strike"], T, r, option_type)
        if iv:
            return iv
        return best["implied_volatility"] if best["implied_volatility"] > 0 else None
    except Exception:
        return _yahoo.get_implied_volatility_for_strike(
            ticker, expiration_date, strike, option_type=option_type, S=S, r=r
        )


def get_atm_implied_volatility(
    ticker: str, expiration_date: str, current_price: float,
    option_type: str = "call", r: float = 0.045,
) -> float | None:
    """ATM IV for an expiration.  Falls back to yahoo_data when key absent."""
    if not _is_key_set():
        return _yahoo.get_atm_implied_volatility(
            ticker, expiration_date, current_price, option_type=option_type, r=r
        )
    try:
        options = get_options_for_expiration(ticker, expiration_date)
        if not options.get("success"):
            return None
        T = get_years_to_expiration(expiration_date)

        def _best_iv(chain, side):
            best = min(chain, key=lambda o: abs(o["strike"] - current_price), default=None)
            if best is None:
                return None
            iv = _iv_from_mid(best["bid"], best["ask"], current_price, best["strike"], T, r, side)
            return iv or (best["implied_volatility"] if best["implied_volatility"] > 0 else None)

        call_iv = _best_iv(options["calls"], "call")
        put_iv  = _best_iv(options["puts"],  "put")

        if option_type == "call" and call_iv:
            return call_iv
        if option_type == "put" and put_iv:
            return put_iv
        if call_iv and put_iv:
            return (call_iv + put_iv) / 2
        return call_iv or put_iv
    except Exception:
        return _yahoo.get_atm_implied_volatility(
            ticker, expiration_date, current_price, option_type=option_type, r=r
        )


# ── Theoretical price / delta / theta ─────────────────────────────────────────

def fetch_option_theoretical_price(
    symbol: str, expiration_iso: str, strike: float,
    option_type: str, r: float = 0.045,
) -> float | None:
    """American-binomial theoretical price.  Falls back to yahoo_data when key absent."""
    if not _is_key_set():
        return _yahoo.fetch_option_theoretical_price(
            symbol, expiration_iso, strike, option_type, r
        )
    try:
        side = "call" if option_type.upper() in ("CALL", "STOCK") else "put"
        info = get_stock_info(symbol)
        S    = info.get("current_price") if info.get("success") else None
        if not S:
            return None
        q = info.get("dividend_yield", 0) or 0
        T = get_years_to_expiration(expiration_iso)
        if T <= 0:
            return max(S - strike, 0.0) if side == "call" else max(strike - S, 0.0)
        iv = get_implied_volatility_for_strike(
            symbol, expiration_iso, strike, option_type=side, S=S, r=r
        )
        if not iv:
            return None
        return _pricing.american_option_binomial(S, strike, T, r, iv, q, side)
    except Exception:
        return _yahoo.fetch_option_theoretical_price(
            symbol, expiration_iso, strike, option_type, r
        )


def fetch_option_delta(
    symbol: str, expiration_iso: str, strike: float,
    option_type: str, r: float = 0.045,
) -> float | None:
    """Probability of assignment (American delta).  Falls back to yahoo_data when key absent."""
    if not _is_key_set():
        return _yahoo.fetch_option_delta(symbol, expiration_iso, strike, option_type, r)
    try:
        side = "call" if option_type.upper() in ("CALL", "STOCK") else "put"
        info = get_stock_info(symbol)
        S    = info.get("current_price") if info.get("success") else None
        if not S:
            return None
        q = info.get("dividend_yield", 0) or 0
        T = get_years_to_expiration(expiration_iso)
        if T <= 0:
            return (1.0 if S > strike else 0.0) if side == "call" else (1.0 if S < strike else 0.0)
        iv = get_implied_volatility_for_strike(
            symbol, expiration_iso, strike, option_type=side, S=S, r=r
        )
        if not iv:
            return None
        greeks = _pricing.american_option_greeks(S, strike, T, r, iv, q, option_type=side)
        return abs(greeks["delta"])
    except Exception:
        return _yahoo.fetch_option_delta(symbol, expiration_iso, strike, option_type, r)


def fetch_option_theta(
    symbol: str, expiration_iso: str, strike: float,
    option_type: str, r: float = 0.045,
) -> float | None:
    """American-binomial theta per share per day.  Falls back to yahoo_data when key absent."""
    if not _is_key_set():
        return _yahoo.fetch_option_theta(symbol, expiration_iso, strike, option_type, r)
    try:
        side = "call" if option_type.upper() in ("CALL", "STOCK") else "put"
        info = get_stock_info(symbol)
        S    = info.get("current_price") if info.get("success") else None
        if not S:
            return None
        q = info.get("dividend_yield", 0) or 0
        T = get_years_to_expiration(expiration_iso)
        if T <= 0:
            options = get_options_for_expiration(symbol, expiration_iso)
            if not options.get("success"):
                return None
            chain = options["calls"] if side == "call" else options["puts"]
            best  = min(chain, key=lambda o: abs(o["strike"] - strike), default=None)
            if best is None:
                return None
            bid, ask = best["bid"], best["ask"]
            if bid <= 0 or ask <= 0:
                return None
            mid       = (bid + ask) / 2.0
            intrinsic = max(S - strike, 0.0) if side == "call" else max(strike - S, 0.0)
            return -(mid - intrinsic)
        iv = get_implied_volatility_for_strike(
            symbol, expiration_iso, strike, option_type=side, S=S, r=r
        )
        if not iv:
            return None
        greeks = _pricing.american_option_greeks(S, strike, T, r, iv, q, option_type=side)
        return greeks["theta"]
    except Exception:
        return _yahoo.fetch_option_theta(symbol, expiration_iso, strike, option_type, r)


# ── Compatibility shims ────────────────────────────────────────────────────────

def search_ticker(query: str, max_results: int = 10) -> list:
    """Delegate to yahoo_data (Massive has no ticker-search endpoint)."""
    return _yahoo.search_ticker(query, max_results=max_results)


def get_stock_data(ticker: str) -> dict | None:
    """Kivy-compatible variant — returns raw stock dict or None on failure."""
    result = get_stock_info(ticker)
    if not result.get("success"):
        return None
    return {
        "currentPrice":  result["current_price"],
        "longName":      result.get("company_name", ticker),
        "shortName":     result.get("company_name", ticker),
        "previousClose": result.get("previous_close"),
        "volume":        result.get("volume"),
        "dividendYield": result.get("dividend_yield", 0),
    }


def get_dividend_yield(ticker: str) -> float:
    result = get_stock_info(ticker)
    return result.get("dividend_yield", 0)


def get_expiration_dates(ticker: str) -> list:
    """Kivy-compatible variant — list of expiration date strings."""
    result = get_option_chain_next_months(ticker, months=6)
    return result.get("expirations", [])
