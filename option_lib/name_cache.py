"""Disk-backed company-name cache.

Names are reference data with the same shape as sectors: they change at most on
a rebrand, and re-fetching one costs a full .info call.  A name is wanted mostly
for display — a hover on a ticker column — so it must be cheap enough that a
table of thirty tickers does not turn into thirty network round-trips on every
launch.  See disk_cache for the storage, its location, and its failure behaviour.
"""

from __future__ import annotations

import os

from option_lib.disk_cache import FRESH, MISSING, STALE, DiskCache   # noqa: F401 — re-exported

# One month, matching sectors.  A rename is rare, and a stale month of it costs
# nothing more than a tooltip showing the old name.
TTL_SECONDS = int(os.environ.get("OPTION_LIB_NAME_TTL", 30 * 24 * 3600))

_cache = DiskCache("names.json", field="name", ttl_seconds=TTL_SECONDS)


def get(symbol: str) -> tuple[str | None, bool]:
    """Return (name, hit).  A cached None is a hit — some symbols have no name."""
    return _cache.get(symbol)


def peek(symbol: str) -> tuple[str | None, str]:
    """Return (name, state) — FRESH, STALE, or MISSING.  See DiskCache.peek."""
    return _cache.peek(symbol)


def claim(symbol: str) -> bool:
    """True for the one process that should refresh this stale name."""
    return _cache.claim(symbol)


def put(symbol: str, name: str | None) -> None:
    """Record *name* for *symbol* and persist the cache."""
    _cache.put(symbol, name)


def clear() -> None:
    """Drop every entry, on disk and in memory.  Intended for tests."""
    _cache.clear()
