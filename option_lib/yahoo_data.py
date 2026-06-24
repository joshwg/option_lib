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


from option_lib.math_util import (
    safe_float as _safe_float,
    TTLCache as _TTLCache,
    iv_from_mid as _iv_from_mid,
    get_days_to_expiration,
    get_years_to_expiration,
)

_cache = _TTLCache()

_TTL_STOCK    = int(os.environ.get('CACHE_TTL_STOCK',    300))
_TTL_OPTIONS  = int(os.environ.get('CACHE_TTL_OPTIONS',  300))
_TTL_VOL      = int(os.environ.get('CACHE_TTL_VOL',     1800))
_TTL_EXPIRIES = int(os.environ.get('CACHE_TTL_EXPIRIES', 1800))
_TTL_SEARCH   = int(os.environ.get('CACHE_TTL_SEARCH',  3600))

# ---------------------------------------------------------------------------
# Rate limiting — 250 ms minimum gap between Yahoo Finance network calls.
# The lock serialises concurrent threads so bursts don't exceed ~4 req/s.
# ---------------------------------------------------------------------------
_yf_lock = threading.Lock()
_yf_last_call: float = 0.0
_YF_MIN_INTERVAL = 0.250  # seconds


def _yf_rate_limit() -> None:
    """Sleep if needed to maintain ≥250 ms between Yahoo Finance calls."""
    global _yf_last_call
    with _yf_lock:
        wait = _YF_MIN_INTERVAL - (time.monotonic() - _yf_last_call)
        if wait > 0:
            time.sleep(wait)
        _yf_last_call = time.monotonic()



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
        print(f"Error searching tickers: {e}")
        return []


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
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

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
        except:
            # If calendar fails, try the info dict
            try:
                earnings_timestamp = info.get('earningsTimestamp')
                if earnings_timestamp:
                    import datetime as dt
                    earnings_date = dt.datetime.fromtimestamp(earnings_timestamp).strftime('%Y-%m-%d')
            except:
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
        volatility = log_returns.std() * np.sqrt(252)  # 252 trading days per year

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

        # Filter to next N months
        today = datetime.now()
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
    Computes IV from the bid/ask mid price; falls back to Yahoo's pre-computed
    field only if the mid-based Newton-Raphson does not converge.

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

        best_match = None
        min_diff = float('inf')
        for opt in chain:
            diff = abs(opt['strike'] - strike)
            if diff < min_diff:
                min_diff = diff
                best_match = opt

        if best_match is None:
            return None

        # Resolve current stock price if not supplied
        if S is None:
            info = get_stock_info(ticker)
            S = info.get('current_price') if info.get('success') else None

        T = get_years_to_expiration(expiration_date)
        iv = _iv_from_mid(best_match['bid'], best_match['ask'], S, best_match['strike'], T, r, option_type)
        if iv:
            return iv

        # Fallback: Yahoo's pre-computed field
        return best_match['implied_volatility'] if best_match['implied_volatility'] > 0 else None
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
            best_opt = None
            min_diff = float('inf')
            for opt in chain:
                diff = abs(opt['strike'] - current_price)
                if diff < min_diff:
                    min_diff = diff
                    best_opt = opt
            if best_opt is None:
                return None
            iv = _iv_from_mid(best_opt['bid'], best_opt['ask'], current_price, best_opt['strike'], T, r, side)
            if iv:
                return iv
            return best_opt['implied_volatility'] if best_opt['implied_volatility'] > 0 else None

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
            # Expiration day: theta = remaining time value = market_price - intrinsic
            options = get_options_for_expiration(symbol, expiration_iso)
            if not options['success']:
                return None
            chain = options['calls'] if side == 'call' else options['puts']
            best_match = min(chain, key=lambda o: abs(o['strike'] - strike), default=None)
            if best_match is None:
                return None
            bid, ask = best_match['bid'], best_match['ask']
            if bid <= 0 or ask <= 0:
                return None
            mid = (bid + ask) / 2.0
            intrinsic = max(S - strike, 0.0) if side == 'call' else max(strike - S, 0.0)
            return -(mid - intrinsic)
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
        if iv:
            return iv

    # Fallback: full chain (already cached per (ticker, expiration) pair).
    return get_implied_volatility_for_strike(symbol, expiration_iso, strike,
                                             option_type=side, S=S, r=r)


def fetch_option_greeks(symbol: str, expiration_iso: str, strike: float,
                        option_type: str, r: float = 0.045) -> dict:
    """Return price, theta, and delta for one contract in a single computation.

    Compared to calling fetch_option_theoretical_price / fetch_option_theta /
    fetch_option_delta separately, this function:
      - Fetches stock info once  (vs 3x)
      - Fetches IV once          (vs 3x, using a targeted single-contract quote)
      - Runs the binomial tree 5 times (vs 11x: 1 + 5 + 5)

    Returns:
    dict with keys 'price', 'theta', 'delta'; any value may be None on failure.
    """
    _none = {'price': None, 'theta': None, 'delta': None}
    try:
        side = 'call' if option_type.upper() in ('CALL', 'STOCK') else 'put'
        info = get_stock_info(symbol)
        S    = info.get('current_price') if info.get('success') else None
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
