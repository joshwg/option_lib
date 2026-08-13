"""Disk-backed sector cache.

A sector is effectively permanent — a company does not change sector between
restarts, or between months — so it is cached on disk rather than in memory,
which would re-fetch every symbol on every process start.  See disk_cache for
the storage, its location, and its failure behaviour.
"""

from __future__ import annotations

import os

from option_lib.disk_cache import FRESH, MISSING, STALE, DiskCache   # noqa: F401 — re-exported

# One month.  A sector changes at most on a reclassification, which is rare
# enough that a stale month costs nothing and a re-fetch storm costs more.
TTL_SECONDS = int(os.environ.get("OPTION_LIB_SECTOR_TTL", 30 * 24 * 3600))

_cache = DiskCache("sectors.json", field="sector", ttl_seconds=TTL_SECONDS)


def get(symbol: str) -> tuple[str | None, bool]:
    """Return (sector, hit).  A cached None is a hit — ETFs have no sector."""
    return _cache.get(symbol)


def peek(symbol: str) -> tuple[str | None, str]:
    """Return (sector, state) — FRESH, STALE, or MISSING.  See DiskCache.peek."""
    return _cache.peek(symbol)


def claim(symbol: str) -> bool:
    """True for the one process that should refresh this stale sector."""
    return _cache.claim(symbol)


def put(symbol: str, sector: str | None) -> None:
    """Record *sector* for *symbol* and persist the cache."""
    _cache.put(symbol, sector)


def clear() -> None:
    """Drop every entry, on disk and in memory.  Intended for tests."""
    _cache.clear()
