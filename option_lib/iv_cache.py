"""Disk-backed store of the last implied volatility a contract really had.

Unlike the sector and name caches, this is not reference data — it is a quote,
and quotes go stale.  It is kept anyway because of when it is read: outside
regular hours no market maker quotes an option, so every bid and ask in the
chain reads 0.00 and there is no live volatility to be had at any price.  The
underlying, meanwhile, keeps moving.

Volatility is far stickier overnight than the underlying is, so re-pricing a
contract at the volatility it implied when it last had a two-sided market —
usually yesterday's close — tracks the pre-market move honestly.  It is an
approximation, and it is the right one: the alternative on offer is a zero.

Keyed by contract rather than by symbol, because skew means the IV of one
strike says only so much about the next.  See disk_cache for the storage, its
location, and its failure behaviour.
"""

from __future__ import annotations

import os
import threading
import time

from option_lib.disk_cache import DiskCache

# A week.  Long enough to carry a contract across a weekend plus a holiday
# Monday, which is the longest stretch a live market can be shut; short enough
# that a volatility nobody has refreshed since then is dropped rather than
# quietly priced against.
TTL_SECONDS = int(os.environ.get("OPTION_LIB_IV_TTL", 7 * 24 * 3600))

# How often one contract's IV is actually written to disk.  Every write is a
# locked read-modify-write of the whole file, and a portfolio page re-prices
# every position every few seconds — persisting each one each time would turn a
# refresh into dozens of file rewrites to record a number that barely moved.
# Fifteen minutes is invisible against a seven-day expiry, and what matters is
# only that *some* IV from the session survives it.
WRITE_INTERVAL_SECONDS = int(os.environ.get("OPTION_LIB_IV_WRITE_INTERVAL", 900))

_cache = DiskCache("contract_iv.json", field="iv", ttl_seconds=TTL_SECONDS)

# key → wall-clock time this process last persisted it.
_last_written: dict[str, float] = {}
_write_lock = threading.Lock()


def contract_key(symbol: str, expiration: str, strike: float, side: str) -> str:
    """Stable cache key for one contract.

    Built already-uppercase and with the strike normalised, so that 190 and
    190.0 — which arrive interchangeably from the database and from a chain
    row — cannot end up as two separate entries.
    """
    return f"{symbol.strip().upper()}|{expiration}|{float(strike):g}|{side.strip().upper()}"


def get(key: str) -> float | None:
    """Return the remembered IV for *key*, or None if absent or expired."""
    value, hit = _cache.get(key)
    if not hit or not isinstance(value, (int, float)):
        return None
    return float(value)


def put(key: str, iv: float) -> None:
    """Remember *iv* as this contract's last known good volatility.

    Rate-limited per contract — see WRITE_INTERVAL_SECONDS.  Callers may call
    this on every re-price without thinking about the cost.
    """
    now = time.time()
    with _write_lock:
        last = _last_written.get(key)
        if last is not None and now - last < WRITE_INTERVAL_SECONDS:
            return
        _last_written[key] = now
    _cache.put(key, float(iv))


def clear() -> None:
    """Drop every entry, on disk and in memory.  Intended for tests."""
    with _write_lock:
        _last_written.clear()
    _cache.clear()
