import math
import threading
import time
from datetime import datetime, time as _time
from zoneinfo import ZoneInfo
from option_lib import pricing as _pricing


def safe_float(value, default=None):
    """Return *value* as a float, or *default* if it is None/NaN/non-numeric."""
    if value is None:
        return default
    try:
        f = float(value)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


class TTLCache:
    """Thread-safe in-memory cache with per-entry TTL."""

    def __init__(self):
        self._store: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None, False
            value, expires_at = entry
            if time.monotonic() < expires_at:
                return value, True
            del self._store[key]
            return None, False

    def set(self, key, value, ttl):
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)

    def invalidate(self, key):
        with self._lock:
            self._store.pop(key, None)


def iv_from_mid(bid, ask, S, K, T, r, option_type):
    """Compute implied vol from bid/ask mid price.

    Returns None if inputs are invalid or Newton-Raphson doesn't converge.
    """
    if not (bid > 0 and ask > 0 and S > 0 and K > 0 and T > 0):
        return None
    try:
        return _pricing.implied_volatility(
            (bid + ask) / 2.0, S, K, T, r, option_type=option_type
        )
    except Exception:
        return None


def next_option_friday() -> 'date':
    """Return the default option-expiration Friday.

    Always returns *next* week's Friday so the form never defaults to the
    current week's expiry.

    Mon–Thu  → next week's Friday (skip this week).
    Fri      → next week's Friday (skip today, it's already expiry day).
    Sat      → next week's Friday (this week's has already passed).
    Sun      → next Friday (5 days out, already next week).
    """
    from datetime import date, timedelta
    today = date.today()
    wd    = today.weekday()   # Mon=0 … Sun=6
    if wd == 4:               # Friday
        days = 7
    elif wd == 5:             # Saturday
        days = 6
    elif wd == 6:             # Sunday → next Friday (already next week)
        days = 5
    else:                     # Mon–Thu: skip to next week's Friday
        days = 4 - wd + 7
    return today + timedelta(days=days)


def get_days_to_expiration(expiration_date_str: str) -> int:
    """Days from today to expiration_date_str ('YYYY-MM-DD'), minimum 0."""
    try:
        exp   = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        return max((exp - today).days, 0)
    except Exception as e:
        print(f"Error calculating days to expiration: {e}")
        return 0


def get_years_to_expiration(expiration_date_str: str) -> float:
    """Years from today to expiration (for Black-Scholes T)."""
    return get_days_to_expiration(expiration_date_str) / 365.0


# ── Market sessions ────────────────────────────────────────────────────────────

MARKET_TZ = ZoneInfo("America/New_York")

_PRE_OPEN     = _time(4, 0)
_MARKET_OPEN  = _time(9, 30)
_MARKET_CLOSE = _time(16, 0)

# Session name → the get_stock_info() field holding that session's price.
EXTENDED_PRICE_KEY = {'pre': 'pre_market_price', 'post': 'post_market_price'}


def market_session(now=None) -> str:
    """Which US equity session the clock is in: 'pre', 'regular', 'post', 'closed'.

    Eastern-time windows::

        04:00 – 09:30   'pre'
        09:30 – 16:00   'regular'
        16:00 – 04:00   'post'      (runs past midnight into the next morning)
        weekends        'closed'

    Post runs to 04:00 rather than the exchanges' 20:00 cutoff: after the last
    after-hours print there is no fresher quote, so it stays the right basis for
    pricing right up until pre-market opens.  The 00:00–04:00 stretch belongs to
    the *previous* day's post session and therefore counts only when that day
    was a weekday — Saturday's small hours continue Friday's evening, Sunday's
    and Monday's do not.

    Holidays are not tracked; a holiday reads as whatever window the clock falls
    in, which is harmless because no extended-hours print exists then and
    callers fall back to the regular price.
    """
    now = now or datetime.now(MARKET_TZ)
    hm  = now.time()
    if hm < _PRE_OPEN:                        # after midnight, before 04:00
        return "post" if 1 <= now.weekday() <= 5 else "closed"   # Tue–Sat
    if now.weekday() >= 5:                    # Saturday / Sunday daytime
        return "closed"
    if hm < _MARKET_OPEN:
        return "pre"
    if hm < _MARKET_CLOSE:
        return "regular"
    return "post"


def extended_underlying(info: dict, now=None):
    """S for extended-hours pricing: the current session's price, else the regular one.

    Only the field belonging to the session the clock is in is fresh — providers
    keep reporting a field long after its session ends (Yahoo carries the
    morning's preMarketPrice all evening), so picking whichever one is populated
    prices options off a quote that can be half a day old.
    """
    key = EXTENDED_PRICE_KEY.get(market_session(now))
    return (info.get(key) if key else None) or info.get("current_price")
