import math
import threading
import time
from datetime import datetime
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

    Sun–Thu  → this coming Friday.
    Fri      → next week's Friday (skip today, it's already expiry day).
    Sat      → next week's Friday (this week's has already passed).
    """
    from datetime import date, timedelta
    today = date.today()
    wd    = today.weekday()   # Mon=0 … Sun=6
    if wd == 4:               # Friday
        days = 7
    elif wd == 5:             # Saturday
        days = 6
    else:                     # Sun(6) and Mon–Thu(0–3)
        days = (4 - wd) % 7
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
