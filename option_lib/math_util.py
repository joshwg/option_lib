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


# ── Market sessions ────────────────────────────────────────────────────────────

MARKET_TZ = ZoneInfo("America/New_York")

_PRE_OPEN     = _time(4, 0)
_MARKET_OPEN  = _time(9, 30)
_MARKET_CLOSE = _time(16, 0)

_HOURS_PER_YEAR = 24.0 * 365.0


def get_days_to_expiration(expiration_date_str: str) -> int:
    """Whole calendar days from today to expiration ('YYYY-MM-DD'), minimum 0.

    For display only.  Pricing must use get_years_to_expiration(), which counts
    the hours actually remaining instead of rounding down to whole days.
    """
    try:
        exp   = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
        today = datetime.now(MARKET_TZ).date()
        return max((exp - today).days, 0)
    except Exception as e:
        print(f"Error calculating days to expiration: {e}")
        return 0


def get_hours_to_expiration(expiration_date_str: str) -> float:
    """Hours from now to the 16:00 ET close on expiration day, minimum 0."""
    try:
        exp    = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
        expiry = datetime.combine(exp, _MARKET_CLOSE, tzinfo=MARKET_TZ)
        return max((expiry - datetime.now(MARKET_TZ)).total_seconds() / 3600.0, 0.0)
    except Exception as e:
        print(f"Error calculating hours to expiration: {e}")
        return 0.0


def get_years_to_expiration(expiration_date_str: str) -> float:
    """Years from now to expiration (for Black-Scholes T).

    Counts the remaining *hours*, not whole calendar days.  Options trade until
    16:00 ET on expiry day, so a Thursday-afternoon view of a Friday expiry has
    ~29h left rather than 24h, and a Friday-morning view has ~6h left rather
    than none.  Flooring to whole days understates T by 20%+ inside the last two
    days and collapses expiry-day options to pure intrinsic value with all
    greeks at zero — the error is invisible at 30 DTE and dominant at 1 DTE,
    where price is roughly exponential in sigma*sqrt(T).
    """
    return get_hours_to_expiration(expiration_date_str) / _HOURS_PER_YEAR

# Which price basis each clock session uses: pre-market hours show the
# pre-market print, regular hours the regular one, and every other hour the
# post-market print.  'closed' — the weekend, and Monday before 04:00 — maps to
# 'post' because no trade happens in that window, so Friday's last after-hours
# print is still the freshest quote there is.  Falling back to the regular close
# would discard every after-hours move across the weekend, which is exactly when
# news tends to land.  Mirrored by MarginWatch's market_data_service.
SESSION_BASIS = {'pre': 'pre', 'regular': None, 'post': 'post', 'closed': 'post'}

# Price basis → the get_stock_info() field holding it.
EXTENDED_PRICE_KEY = {'pre': 'pre_market_price', 'post': 'post_market_price'}


def bars_cache_ttl(interval: str, now: float | None = None, grace: int = 120) -> int:
    """Seconds to cache a set of price bars of *interval* ('1h', '30m', '1d', …).

    Bars only change when one completes, so the answer is "until *grace*
    seconds past the next bar boundary" rather than a flat window: a refresh
    at :02 sees the candle that closed at :00, and nothing is re-fetched in
    between.  Intervals that do not parse, and daily-or-longer ones, fall back
    to an hour.  Never less than 60 s.
    """
    now = time.time() if now is None else now
    try:
        n, unit = int(interval[:-1] or 1), interval[-1].lower()
    except (ValueError, IndexError):
        return 3600
    span = n * {"m": 60, "h": 3600}.get(unit, 0)
    if span <= 0 or span > 86400:
        return 3600
    next_boundary = (int(now) // span + 1) * span
    return max(60, int(next_boundary - now) + grace)


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
    """S for extended-hours pricing: the print the clock calls for, else the regular one.

    Only the field belonging to the basis the clock calls for is fresh —
    providers keep reporting a field long after its session ends (Yahoo carries
    the morning's preMarketPrice all evening), so picking whichever one is
    populated prices options off a quote that can be half a day old.

    An empty field means nothing traded in that window — a stock with no
    pre-market interest, or a weekend after a quiet Friday close — and the
    regular price stands in.
    """
    basis = SESSION_BASIS.get(market_session(now))
    key = EXTENDED_PRICE_KEY.get(basis)
    return (info.get(key) if key else None) or info.get("current_price")
