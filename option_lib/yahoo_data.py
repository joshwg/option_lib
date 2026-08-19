"""
Yahoo Finance Data Fetcher
Downloads stock prices, historical data, and option information
"""

import math
import os
import threading
import time
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import requests
import pytz
from option_lib import pricing as _pricing
from option_lib import name_cache as _name_cache
from option_lib import sector_cache as _sector_cache

# Import YFRateLimitError if available (yfinance ≥ 0.2.28)
try:
    from yfinance.exceptions import YFRateLimitError as _YFRateLimitError
except ImportError:
    _YFRateLimitError = None


from option_lib import implied_vol as _implied_vol
from option_lib import iv_cache as _iv_cache
from option_lib.math_util import (
    safe_float as _safe_float,
    TTLCache as _TTLCache,
    iv_from_mid as _iv_from_mid,
    get_days_to_expiration,
    get_years_to_expiration,
    extended_underlying as _extended_underlying,
)

_cache = _TTLCache()

_TTL_STOCK    = int(os.environ.get('CACHE_TTL_STOCK',    300))
_TTL_OPTIONS  = int(os.environ.get('CACHE_TTL_OPTIONS',  300))
_TTL_VOL      = int(os.environ.get('CACHE_TTL_VOL',     1800))
_TTL_EXPIRIES = int(os.environ.get('CACHE_TTL_EXPIRIES', 1800))
_TTL_SEARCH   = int(os.environ.get('CACHE_TTL_SEARCH',  3600))
_TTL_EARNINGS = int(os.environ.get('CACHE_TTL_EARNINGS', 86400))  # 24 h — dates rarely change
# Sectors and company names are cached on disk for a month instead — see
# sector_cache and name_cache.

# ---------------------------------------------------------------------------
# Rate limiting — minimum gap between Yahoo Finance network calls.
# The lock serialises concurrent threads.  _YF_MIN_INTERVAL starts at 250 ms
# and is increased automatically after each 429 (up to _YF_MAX_INTERVAL).
# ---------------------------------------------------------------------------
_yf_lock         = threading.Lock()
_yf_last_call: float = 0.0
_YF_MIN_INTERVAL = 0.250   # seconds — raised on rate-limit events
_YF_MAX_INTERVAL = 3.0     # ceiling after repeated rate limiting
_YF_RATE_SLEEP   = 10.0    # seconds to pause after a 429
_YF_BACKOFF_STEP = 0.25    # how much to add to _YF_MIN_INTERVAL per 429


def _yf_rate_limit() -> None:
    """Sleep if needed to honour the current minimum inter-call interval."""
    global _yf_last_call
    with _yf_lock:
        wait = _YF_MIN_INTERVAL - (time.monotonic() - _yf_last_call)
        if wait > 0:
            time.sleep(wait)
        _yf_last_call = time.monotonic()


def _on_rate_limited() -> None:
    """Called when Yahoo Finance returns a 429.  Sleeps and slows the interval."""
    global _YF_MIN_INTERVAL
    with _yf_lock:
        _YF_MIN_INTERVAL = min(_YF_MIN_INTERVAL + _YF_BACKOFF_STEP, _YF_MAX_INTERVAL)
        new_interval = _YF_MIN_INTERVAL
    print(
        f"[yahoo_data] Rate limited — sleeping {_YF_RATE_SLEEP:.0f}s "
        f"(inter-call interval now {new_interval:.2f}s)",
        flush=True,
    )
    time.sleep(_YF_RATE_SLEEP)



def normalize_implied_volatility(iv_value):
    """
    Normalize implied volatility to decimal format (0.0 to 1.0 range)
    Yahoo Finance sometimes returns IV as decimal, sometimes as percentage

    Parameters:
    iv_value (float): Implied volatility value from Yahoo Finance

    Returns:
    float: Normalized IV as decimal (e.g., 0.25 for 25%)
    """
    if iv_value is None or iv_value == 0:
        return 0

    # Guard against NaN/Infinity coming from Yahoo Finance
    if isinstance(iv_value, float) and (math.isnan(iv_value) or math.isinf(iv_value)):
        return 0

    # If value is greater than 2, it's likely already a percentage (e.g., 25.5 for 25.5%)
    # Convert it to decimal
    if iv_value > 2:
        return iv_value / 100

    # Otherwise it's already in decimal format (e.g., 0.255 for 25.5%)
    return iv_value


def normalize_dividend_yield(div_value):
    """
    Normalize dividend yield to decimal format (0.0 to 1.0 range)
    Yahoo Finance sometimes returns dividend yield as decimal, sometimes as percentage

    Parameters:
    div_value (float): Dividend yield value from Yahoo Finance

    Returns:
    float: Normalized dividend yield as decimal (e.g., 0.0042 for 0.42%)
    """
    if div_value is None or div_value == 0:
        return 0

    # Most dividend yields are 0-10%, rarely exceeding 15%
    # In decimal format, that's 0.00 to 0.15
    # If value is greater than 0.10 (10%), it's likely in "percentage points" format
    # (e.g., 0.42 meaning 0.42%, or 5.5 meaning 5.5%)
    # Convert it to decimal by dividing by 100
    if div_value > 0.10:
        return div_value / 100

    # Otherwise it's already in decimal format (e.g., 0.0042 for 0.42%)
    return div_value


def search_ticker(query, max_results=10):
    """
    Search for stock tickers by company name or partial ticker

    Parameters:
    query (str): Search query (company name or partial ticker)
    max_results (int): Maximum number of results to return (default 10)

    Returns:
    list: List of dictionaries with ticker info [{symbol, name, exchange, type}, ...]
    """
    cache_key = ('search_ticker', query.lower(), max_results)
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    _yf_rate_limit()
    for _attempt in range(3):
        try:
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            headers = {'User-Agent': 'Mozilla/5.0'}
            params = {
                'q': query,
                'quotesCount': max_results,
                'newsCount': 0,
                'enableFuzzyQuery': False,
                'quotesQueryId': 'tss_match_phrase_query'
            }

            response = requests.get(url, headers=headers, params=params, timeout=5)

            if response.status_code == 429:
                _on_rate_limited()
                continue
            if response.status_code != 200:
                return []

            data = response.json()
            quotes = data.get('quotes', [])

            results = []
            for quote in quotes[:max_results]:
                # Filter to only include stocks (equities)
                if quote.get('quoteType') in ['EQUITY', 'ETF']:
                    results.append({
                        'symbol': quote.get('symbol', ''),
                        'name': quote.get('longname') or quote.get('shortname', ''),
                        'exchange': quote.get('exchange', ''),
                        'type': quote.get('quoteType', '')
                    })

            _cache.set(cache_key, results, _TTL_SEARCH)
            return results
        except Exception as e:
            if _YFRateLimitError and isinstance(e, _YFRateLimitError):
                _on_rate_limited()
                continue
            print(f"Error searching tickers: {e}")
    return []


def get_earnings_date(ticker: str) -> str | None:
    """Return the next upcoming earnings date as 'YYYY-MM-DD', or None via Yahoo Finance (stock.calendar). Cached for 24 hours."""
    import datetime as _dt

    cache_key = ('get_earnings_date', ticker.upper())
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    # If a full stock-info fetch already has the date, use it without another call.
    info_cached, info_hit = _cache.get(('get_stock_info', ticker.upper()))
    if info_hit and info_cached.get('success') and info_cached.get('earnings_date'):
        result = info_cached['earnings_date']
        _cache.set(cache_key, result, _TTL_EARNINGS)
        return result

    _yf_rate_limit()
    for _attempt in range(3):
        try:
            cal = yf.Ticker(ticker).calendar
            result = None
            if cal is not None and 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                if dates and len(dates) > 0:
                    ed = dates[0]
                    result = (
                        ed.strftime('%Y-%m-%d') if isinstance(ed, _dt.date) else str(ed)[:10]
                    )
            _cache.set(cache_key, result, _TTL_EARNINGS)
            return result
        except Exception as _exc:
            if _YFRateLimitError and isinstance(_exc, _YFRateLimitError):
                _on_rate_limited()
                continue
            _cache.set(cache_key, None, 60)   # cache failures briefly
            return None
    _cache.set(cache_key, None, 60)
    return None


def get_sector(ticker: str) -> str | None:
    """Return the company's sector (e.g. 'Technology'), or None if it has none.

    Yahoo's sector vocabulary is the coarse eleven-way split ('Technology',
    'Energy', 'Financial Services', …) rather than a granular industry, which is
    what a portfolio column wants.  ETFs, indices, and other non-operating
    issues genuinely have no sector; that None is an answer, not a failure, and
    is cached like any other.

    Backed by sector_cache, which persists to disk with a one-month expiry — a
    sector does not change, and .info is Yahoo's heaviest call, so it is not
    worth re-fetching on every process start.  A *failed* lookup is not written
    to disk; it retries on the next call rather than sticking for a month.
    """
    cached, state = _sector_cache.peek(ticker)
    if state == _sector_cache.FRESH:
        return cached
    if state == _sector_cache.STALE and not _sector_cache.claim(ticker):
        # The month is up, but another app is already re-fetching this symbol —
        # the cache directory is shared, so its answer will be ours too.  Serve
        # the old sector rather than spend a second identical lookup on it.
        return cached

    # A full stock-info fetch already pulled .info; reuse it rather than repeat
    # the call.  A successful fetch is authoritative even when sector is None.
    info_cached, info_hit = _cache.get(('get_stock_info', ticker.upper()))
    if info_hit and info_cached.get('success'):
        result = info_cached.get('sector')
        _sector_cache.put(ticker, result)
        return result

    _yf_rate_limit()
    for _attempt in range(3):
        try:
            result = (yf.Ticker(ticker).info or {}).get('sector') or None
            _sector_cache.put(ticker, result)
            return result
        except Exception as _exc:
            if _YFRateLimitError and isinstance(_exc, _YFRateLimitError):
                _on_rate_limited()
                continue
            return None
    return None


def get_company_name(ticker: str) -> str | None:
    """Return the company's display name (e.g. 'Apple Inc.'), or None.

    Backed by name_cache on the same terms as get_sector: a name is display-only
    reference data, so a month-old answer is fine and a fresh .info call per
    symbol per process start is not.  A *failed* lookup is not written to disk;
    it retries on the next call rather than sticking for a month.

    Prefers longName ('Apple Inc.') over shortName ('Apple'), which is what a
    tooltip on a ticker wants.  None means Yahoo had no name for the symbol —
    callers should fall back to showing the ticker itself.
    """
    cached, state = _name_cache.peek(ticker)
    if state == _name_cache.FRESH:
        return cached
    if state == _name_cache.STALE and not _name_cache.claim(ticker):
        return cached       # another app is refreshing it — see get_sector()

    # A full stock-info fetch already pulled .info; reuse it rather than repeat
    # the call.  get_stock_info() stores longName under 'company_name', defaulted
    # to the ticker — which is not a name, so it is not worth caching as one.
    info_cached, info_hit = _cache.get(('get_stock_info', ticker.upper()))
    if info_hit and info_cached.get('success'):
        result = info_cached.get('company_name') or None
        if result and result.upper() != ticker.upper():
            _name_cache.put(ticker, result)
            return result

    _yf_rate_limit()
    for _attempt in range(3):
        try:
            info   = yf.Ticker(ticker).info or {}
            result = info.get('longName') or info.get('shortName') or None
            _name_cache.put(ticker, result)
            return result
        except Exception as _exc:
            if _YFRateLimitError and isinstance(_exc, _YFRateLimitError):
                _on_rate_limited()
                continue
            return None
    return None


def get_stock_info(ticker):
    """
    Get current stock information

    Parameters:
    ticker (str): Stock ticker symbol

    Returns:
    dict: Dictionary containing stock information
    """
    cache_key = ('get_stock_info', ticker.upper())
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    _yf_rate_limit()
    # Retry up to 3 times on rate-limit; any other exception fails immediately.
    info = None
    for _attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info
            break
        except Exception as _exc:
            if _YFRateLimitError and isinstance(_exc, _YFRateLimitError):
                _on_rate_limited()
                continue   # retry after sleeping
            return {'ticker': ticker, 'success': False, 'error': str(_exc)}
    else:
        return {'ticker': ticker, 'success': False, 'error': 'Rate limit retries exhausted'}

    try:
        # Get current price — discard implausible values from yfinance
        _raw_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        current_price = float(_raw_price) if (_raw_price is not None and 0.01 <= float(_raw_price) <= 1_000_000) else None

        # Get dividend yield (as decimal, e.g., 0.02 for 2%)
        dividend_yield = info.get('dividendYield', 0) or 0
        dividend_yield = normalize_dividend_yield(dividend_yield)

        # Get next earnings date
        earnings_date = None
        try:
            # Try to get earnings dates from the calendar
            calendar = stock.calendar
            if calendar is not None and 'Earnings Date' in calendar:
                earnings_dates = calendar['Earnings Date']
                if earnings_dates is not None and len(earnings_dates) > 0:
                    # Get the first (earliest) earnings date
                    import datetime as dt
                    earnings_dt = earnings_dates[0]
                    # Convert to string if it's a date object
                    if isinstance(earnings_dt, dt.date):
                        earnings_date = earnings_dt.strftime('%Y-%m-%d')
                    elif isinstance(earnings_dt, str):
                        earnings_date = earnings_dt
        except Exception:
            # If calendar fails, try the info dict
            try:
                earnings_timestamp = info.get('earningsTimestamp')
                if earnings_timestamp:
                    import datetime as dt
                    earnings_date = dt.datetime.fromtimestamp(earnings_timestamp).strftime('%Y-%m-%d')
            except Exception:
                pass

        avg_volume = (
            info.get('averageVolume')
            or info.get('averageDailyVolume10Day')
            or info.get('volume')
        )

        # Extended-hours prices — only populated by yfinance when that session
        # is active; None otherwise.
        def _valid_price(v):
            if v is None:
                return None
            try:
                f = float(v)
                return f if (math.isfinite(f) and 0.01 <= f <= 1_000_000) else None
            except (TypeError, ValueError):
                return None

        pre_market_price  = _valid_price(info.get('preMarketPrice'))
        post_market_price = _valid_price(info.get('postMarketPrice'))

        result = {
            'ticker': ticker,
            'current_price': current_price,
            'company_name': info.get('longName', ticker),
            'previous_close': info.get('previousClose'),
            'volume': info.get('volume'),
            'avg_volume': avg_volume,
            'market_cap': info.get('marketCap'),
            'dividend_yield': dividend_yield,
            'earnings_date': earnings_date,
            # None for ETFs and other non-operating issues, which have no sector.
            'sector': info.get('sector') or None,
            'pre_market_price':  pre_market_price,
            'post_market_price': post_market_price,
            'success': True
        }
        _cache.set(cache_key, result, _TTL_STOCK)
        return result
    except Exception as e:
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }


_TTL_BARS = int(os.environ.get('CACHE_TTL_BARS', 900))   # 15 min — intraday bars move


def get_price_bars(ticker, days=7, interval='1h'):
    """Recent OHLC bars for *ticker*, oldest first.

    Returns a list of {'t': epoch_ms, 'o', 'h', 'l', 'c', 'v'} dicts covering
    the last *days* calendar days at *interval* ('1h' or '1d'; anything else
    Yahoo supports passes straight through).  Regular session only.  [] on any
    error.  Cached for _TTL_BARS seconds.
    """
    interval = str(interval).lower()
    cache_key = ('get_price_bars', ticker.upper(), int(days), interval)
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    _yf_rate_limit()
    try:
        hist = yf.Ticker(ticker).history(period=f"{int(days)}d", interval=interval,
                                         prepost=False, auto_adjust=False)
        bars = []
        for ts, row in hist.iterrows():
            c = row.get('Close')
            if c is None or c != c:          # NaN guard
                continue
            bars.append({
                't': int(ts.timestamp() * 1000),
                'o': float(row.get('Open', c)),
                'h': float(row.get('High', c)),
                'l': float(row.get('Low', c)),
                'c': float(c),
                'v': int(row.get('Volume', 0) or 0),
            })
        _cache.set(cache_key, bars, _TTL_BARS)
        return bars
    except Exception as e:
        print(f"Error fetching price bars for {ticker}: {e}")
        return []


def calculate_historical_volatility(ticker, period='1y', days=None):
    """
    Calculate historical volatility from stock price history

    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')

    Returns:
    float: Annualized volatility (as decimal) or None if error
    """
    if days is not None:
        if days <= 30:    period = '1mo'
        elif days <= 90:  period = '3mo'
        elif days <= 180: period = '6mo'
        elif days <= 365: period = '1y'
        elif days <= 730: period = '2y'
        else:             period = '5y'

    cache_key = ('calculate_historical_volatility', ticker.upper(), period)
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    _yf_rate_limit()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if len(hist) < 2:
            return None

        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = float(log_returns.std() * np.sqrt(252))  # 252 trading days per year

        _cache.set(cache_key, volatility, _TTL_VOL)
        return volatility
    except Exception as e:
        print(f"Error calculating volatility: {e}")
        return None


def get_option_chain(ticker):
    """
    Get option chain for a stock

    Parameters:
    ticker (str): Stock ticker symbol

    Returns:
    dict: Dictionary containing option chain data
    """
    cache_key = ('get_option_chain', ticker.upper())
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    _yf_rate_limit()
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            return {
                'success': False,
                'error': 'No options available for this ticker'
            }

        result = {
            'success': True,
            'ticker': ticker,
            'expirations': list(expirations)
        }
        _cache.set(cache_key, result, _TTL_EXPIRIES)
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def get_option_chain_next_months(ticker, months=6):
    """
    Get option chain filtered to next N months
    Excludes today's expiration if after market close (4 PM ET)

    Parameters:
    ticker (str): Stock ticker symbol
    months (int): Number of months to look ahead (default 6)

    Returns:
    dict: Dictionary containing filtered option chain data
    """
    try:
        chain = get_option_chain(ticker)

        if not chain['success']:
            return chain

        # Get current time in Eastern Time (market timezone)
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        market_close_hour = 16  # 4 PM ET

        # Filter to next N months (use ET date throughout for consistency)
        today = now_et
        cutoff_date = today + timedelta(days=months * 30)

        filtered_expirations = []
        for exp_str in chain['expirations']:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d')

            # Exclude today's expiration if after market close
            if exp_date.date() == today.date() and now_et.hour >= market_close_hour:
                continue

            # Compare dates only to include today's options if still trading
            if today.date() <= exp_date.date() <= cutoff_date.date():
                filtered_expirations.append(exp_str)

        return {
            'success': True,
            'ticker': ticker,
            'expirations': filtered_expirations,
            'all_expirations': chain['expirations']
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def get_options_for_expiration(ticker, expiration_date):
    """
    Get options data for a specific expiration date

    Parameters:
    ticker (str): Stock ticker symbol
    expiration_date (str): Expiration date in format 'YYYY-MM-DD'

    Returns:
    dict: Dictionary containing calls and puts data
    """
    cache_key = ('get_options_for_expiration', ticker.upper(), expiration_date)
    cached, hit = _cache.get(cache_key)
    if hit:
        return cached

    _yf_rate_limit()
    try:
        stock = yf.Ticker(ticker)
        opt = stock.option_chain(expiration_date)

        calls = opt.calls
        puts = opt.puts

        # Convert to list of dictionaries for easier processing
        calls_data = []
        for _, row in calls.iterrows():
            iv = normalize_implied_volatility(row.get('impliedVolatility', 0))
            calls_data.append({
                'strike':             _safe_float(row['strike'], 0),
                'last_price':         _safe_float(row['lastPrice'], 0),
                'bid':                _safe_float(row['bid'], 0),
                'ask':                _safe_float(row['ask'], 0),
                'volume':             _safe_float(row.get('volume'), 0),
                'open_interest':      _safe_float(row.get('openInterest'), 0),
                'implied_volatility': iv,
            })

        puts_data = []
        for _, row in puts.iterrows():
            iv = normalize_implied_volatility(row.get('impliedVolatility', 0))
            puts_data.append({
                'strike':             _safe_float(row['strike'], 0),
                'last_price':         _safe_float(row['lastPrice'], 0),
                'bid':                _safe_float(row['bid'], 0),
                'ask':                _safe_float(row['ask'], 0),
                'volume':             _safe_float(row.get('volume'), 0),
                'open_interest':      _safe_float(row.get('openInterest'), 0),
                'implied_volatility': iv,
            })

        result = {
            'success': True,
            'expiration': expiration_date,
            'calls': calls_data,
            'puts': puts_data
        }
        _cache.set(cache_key, result, _TTL_OPTIONS)
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def get_implied_volatility_for_strike(ticker, expiration_date, strike, option_type='call', S=None, r=0.045):
    """
    Get implied volatility for a specific strike and expiration.

    Resolution runs through implied_vol.resolve_iv — the bid/ask mid first, then
    a ladder of fallbacks for the times there is no mid to be had, which is every
    contract outside regular hours and the illiquid strikes at any hour.  See
    that module for the rungs and why they are ordered the way they are.

    Parameters:
    ticker (str): Stock ticker symbol
    expiration_date (str): Expiration date in format 'YYYY-MM-DD'
    strike (float): Strike price
    option_type (str): 'call' or 'put'
    S (float): Current stock price (fetched automatically if omitted)
    r (float): Risk-free rate as decimal (default 0.045)

    Returns:
    float: Implied volatility (as decimal) or None if not found
    """
    try:
        options = get_options_for_expiration(ticker, expiration_date)
        if not options['success']:
            return None

        chain = options['calls'] if option_type == 'call' else options['puts']

        # Resolve current stock price if not supplied
        if S is None:
            info = get_stock_info(ticker)
            S = info.get('current_price') if info.get('success') else None

        T = get_years_to_expiration(expiration_date)
        return _implied_vol.resolve_iv(chain, strike, option_type, S, T, r,
                                       symbol=ticker, expiration=expiration_date)
    except Exception as e:
        print(f"Error getting implied volatility: {e}")
        return None


def get_atm_implied_volatility(ticker, expiration_date, current_price, option_type='call', r=0.045):
    """
    Get at-the-money implied volatility for an expiration date.
    Computes IV from the bid/ask mid price of the closest ATM option;
    falls back to Yahoo's pre-computed field if mid-based computation fails.

    Parameters:
    ticker (str): Stock ticker symbol
    expiration_date (str): Expiration date in format 'YYYY-MM-DD'
    current_price (float): Current stock price
    option_type (str): 'call' or 'put'
    r (float): Risk-free rate as decimal (default 0.045)

    Returns:
    float: Implied volatility (as decimal) or None if not found
    """
    try:
        options = get_options_for_expiration(ticker, expiration_date)
        if not options['success']:
            return None

        T = get_years_to_expiration(expiration_date)

        def _best_iv(chain, side):
            return _implied_vol.resolve_iv(chain, current_price, side, current_price, T, r,
                                           symbol=ticker, expiration=expiration_date)

        call_iv = _best_iv(options['calls'], 'call')
        put_iv  = _best_iv(options['puts'],  'put')

        if option_type == 'call' and call_iv:
            return call_iv
        if option_type == 'put' and put_iv:
            return put_iv
        if call_iv and put_iv:
            return (call_iv + put_iv) / 2
        return call_iv or put_iv
    except Exception as e:
        print(f"Error getting ATM implied volatility: {e}")
        return None





def get_stock_data(ticker):
    """Kivy-compatible variant — returns raw stock dict or None on failure"""
    result = get_stock_info(ticker)
    if not result.get('success'):
        return None
    return {
        'currentPrice': result['current_price'],
        'longName': result.get('company_name', ticker),
        'shortName': result.get('company_name', ticker),
        'previousClose': result.get('previous_close'),
        'volume': result.get('volume'),
        'dividendYield': result.get('dividend_yield', 0),
    }


def get_dividend_yield(ticker):
    result = get_stock_info(ticker)
    return result.get('dividend_yield', 0)


def get_expiration_dates(ticker):
    """Kivy-compatible variant — returns list of expiration date strings"""
    result = get_option_chain_next_months(ticker, months=6)
    return result.get('expirations', [])


def fetch_option_theoretical_price(symbol: str, expiration_iso: str, strike: float,
                                   option_type: str, r: float = 0.045):
    """Return American-binomial theoretical price for the given contract, or None.

    Uses the CRR binomial tree model so that early-exercise value and dividend
    yield (q) are correctly reflected.  The implied volatility is still derived
    from the market bid/ask mid via Black-Scholes (the industry convention for
    quoting IV), then fed into the American model.

    Parameters:
    symbol (str): Ticker symbol
    expiration_iso (str): Expiration date as 'YYYY-MM-DD'
    strike (float): Strike price
    option_type (str): 'call' or 'put' (case-insensitive)
    r (float): Risk-free rate as decimal (default 0.045)

    Returns:
    float: Theoretical option price per share, or None on failure
    """
    try:
        side = 'call' if option_type.upper() in ('CALL', 'STOCK') else 'put'
        info = get_stock_info(symbol)
        S = info.get('current_price') if info.get('success') else None
        if not S:
            return None
        q = info.get('dividend_yield', 0) or 0   # already normalised to decimal
        T = get_years_to_expiration(expiration_iso)
        if T <= 0:
            # Expiration day: theoretical price is intrinsic value
            return max(S - strike, 0.0) if side == 'call' else max(strike - S, 0.0)
        iv = get_implied_volatility_for_strike(symbol, expiration_iso, strike, option_type=side, S=S, r=r)
        if not iv:
            return None
        return _pricing.american_option_binomial(S, strike, T, r, iv, q, side)
    except Exception:
        return None


def fetch_option_delta(symbol: str, expiration_iso: str, strike: float,
                       option_type: str, r: float = 0.045):
    """Return the probability of assignment (0–1) for the given contract, or None.

    Uses the absolute value of the American-binomial delta so that early-exercise
    risk and dividends are reflected.  Returns a value in [0, 1] for both calls
    and puts.

    On expiration day (T == 0) returns 1.0 if ITM, 0.0 if OTM.
    """
    try:
        side = 'call' if option_type.upper() in ('CALL', 'STOCK') else 'put'
        info = get_stock_info(symbol)
        S = info.get('current_price') if info.get('success') else None
        if not S:
            return None
        q = info.get('dividend_yield', 0) or 0
        T = get_years_to_expiration(expiration_iso)
        if T <= 0:
            # Expiration day: binary — ITM = certain assignment
            return (1.0 if S > strike else 0.0) if side == 'call' else (1.0 if S < strike else 0.0)
        iv = get_implied_volatility_for_strike(symbol, expiration_iso, strike, option_type=side, S=S, r=r)
        if not iv:
            return None
        greeks = _pricing.american_option_greeks(S, strike, T, r, iv, q, option_type=side)
        return abs(greeks['delta'])
    except Exception:
        return None


def fetch_option_theta(symbol: str, expiration_iso: str, strike: float,
                       option_type: str, r: float = 0.045):
    """Return American-binomial theta (per share per day) for the given contract, or None.

    Theta is the $ change in the option's value from today to tomorrow, so it is
    negative when the option loses time value (the usual case for short sellers).

    Parameters:
    symbol (str): Ticker symbol
    expiration_iso (str): Expiration date as 'YYYY-MM-DD'
    strike (float): Strike price
    option_type (str): 'call' or 'put' (case-insensitive)
    r (float): Risk-free rate as decimal (default 0.045)

    Returns:
    float: Theta per share per day (negative = time decay), or None on failure
    """
    try:
        side = 'call' if option_type.upper() in ('CALL', 'STOCK') else 'put'
        info = get_stock_info(symbol)
        S = info.get('current_price') if info.get('success') else None
        if not S:
            return None
        q = info.get('dividend_yield', 0) or 0
        T = get_years_to_expiration(expiration_iso)
        if T <= 0:
            # Expiration day: no future time value remains, so theta is 0.
            # The chain is likely stale or empty at this point, and attempting
            # to derive theta from bid/ask mid on expiration day is unreliable.
            return 0.0
        iv = get_implied_volatility_for_strike(symbol, expiration_iso, strike, option_type=side, S=S, r=r)
        if not iv:
            return None
        greeks = _pricing.american_option_greeks(S, strike, T, r, iv, q, option_type=side)
        return greeks['theta']
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Efficient combined fetcher — price + theta + delta in one round-trip
# ---------------------------------------------------------------------------

def _occ_symbol(symbol: str, expiration_iso: str, strike: float, option_type: str) -> str:
    """Return the OCC option ticker, e.g. 'AAPL250718P00200000'.

    Yahoo Finance accepts these as Ticker symbols for individual contract quotes,
    avoiding the need to download the full option chain for an expiration date.
    """
    exp  = expiration_iso.replace('-', '')          # 'YYYYMMDD'
    yy, mm, dd = exp[2:4], exp[4:6], exp[6:8]
    side = 'C' if option_type.upper() in ('CALL', 'STOCK') else 'P'
    return f"{symbol.upper()}{yy}{mm}{dd}{side}{int(round(strike * 1000)):08d}"


def _get_contract_iv(symbol: str, expiration_iso: str, strike: float,
                     option_type: str, S: float, r: float):
    """Return IV for a single contract, using a targeted OCC-ticker quote where possible.

    Queries the specific option ticker (e.g. 'AAPL250718P00200000') for bid/ask
    and derives IV from the mid price.  This fetches only one contract from Yahoo
    instead of the entire expiration's chain.  Falls back to the full chain if
    the targeted quote is unavailable.
    """
    side = 'call' if option_type.upper() in ('CALL', 'STOCK') else 'put'
    T    = get_years_to_expiration(expiration_iso)
    occ  = _occ_symbol(symbol, expiration_iso, strike, option_type)

    # Check cache for the bid/ask we already fetched for this contract.
    cache_key = ('_contract_bid_ask', occ)
    cached_ba, hit = _cache.get(cache_key)
    if hit:
        bid, ask = cached_ba
    else:
        bid, ask = None, None
        try:
            _yf_rate_limit()
            info = yf.Ticker(occ).info
            bid  = _safe_float(info.get('bid'),  None)
            ask  = _safe_float(info.get('ask'),  None)
            if bid and ask and bid > 0 and ask > 0:
                _cache.set(cache_key, (bid, ask), _TTL_OPTIONS)
        except Exception:
            pass

    if bid and ask and bid > 0 and ask > 0:
        iv = _iv_from_mid(bid, ask, S, strike, T, r, side)
        if _implied_vol.plausible_iv(iv):
            # Same live-market rung as resolve_iv's first, reached by a targeted
            # quote instead of the chain — so remember it on the same terms, or
            # a contract priced only down this path would have nothing on disk
            # to fall back to once the quotes go away.
            _iv_cache.put(_iv_cache.contract_key(symbol, expiration_iso, strike, side), iv)
            return iv

    # Fallback: full chain (already cached per (ticker, expiration) pair).
    return get_implied_volatility_for_strike(symbol, expiration_iso, strike,
                                             option_type=side, S=S, r=r)


def fetch_option_greeks(symbol: str, expiration_iso: str, strike: float,
                        option_type: str, r: float = 0.045,
                        use_extended: bool = False) -> dict:
    """Return price, theta, and delta for one contract in a single computation.

    Compared to calling fetch_option_theoretical_price / fetch_option_theta /
    fetch_option_delta separately, this function:
      - Fetches stock info once  (vs 3x)
      - Fetches IV once          (vs 3x, using a targeted single-contract quote)
      - Runs the binomial tree 5 times (vs 11x: 1 + 5 + 5)

    When use_extended=True, S is the price for the session the clock is in —
    pre-market from 04:00, post-market from 16:00 through to 04:00 — so $/shr,
    theta, and delta all reflect the extended-hours underlying rather than the
    regular-session close.  Falls back to the regular price when that session
    has no print of its own.

    Returns:
    dict with keys 'price', 'theta', 'delta'; any value may be None on failure.
    """
    _none = {'price': None, 'theta': None, 'delta': None}
    try:
        side = 'call' if option_type.upper() in ('CALL', 'STOCK') else 'put'
        info = get_stock_info(symbol)
        if not info.get('success'):
            return _none
        S = _extended_underlying(info) if use_extended else info.get('current_price')
        if not S:
            return _none
        q = info.get('dividend_yield', 0) or 0
        T = get_years_to_expiration(expiration_iso)

        if T <= 0:
            # Expiration day: intrinsic value; theta from live market quote.
            intrinsic = max(S - strike, 0.0) if side == 'call' else max(strike - S, 0.0)
            delta     = (1.0 if S > strike else 0.0) if side == 'call' else (1.0 if S < strike else 0.0)
            theta     = None
            occ = _occ_symbol(symbol, expiration_iso, strike, option_type)
            try:
                _yf_rate_limit()
                opt_info = yf.Ticker(occ).info
                bid = _safe_float(opt_info.get('bid'), None)
                ask = _safe_float(opt_info.get('ask'), None)
                if bid and ask and bid > 0 and ask > 0:
                    theta = -((bid + ask) / 2.0 - intrinsic)
            except Exception:
                pass
            return {'price': intrinsic, 'theta': theta, 'delta': delta}

        iv = _get_contract_iv(symbol, expiration_iso, strike, option_type, S, r)
        if not iv:
            return _none

        # One binomial-tree run now returns price alongside the greeks.
        g = _pricing.american_option_greeks(S, strike, T, r, iv, q, option_type=side)
        return {
            'price': g['price'],
            'theta': g['theta'],
            'delta': abs(g['delta']),
        }
    except Exception:
        return _none
