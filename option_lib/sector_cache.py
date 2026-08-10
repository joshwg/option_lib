"""Disk-backed sector cache.

Sectors are the one piece of reference data here that is effectively permanent —
a company does not change sector between restarts, or between months.  Everything
else in option_lib caches in memory (see math_util.TTLCache), which is right for
prices but wrong here: the desktop apps restart constantly, and an in-memory
cache would send a fresh yfinance .info request per symbol on every launch just
to be told 'Technology' again.

So this one lives on disk, keyed by symbol, with a one-month expiry.  Entries are
stamped with wall-clock time rather than the monotonic clock the in-memory cache
uses, because a monotonic stamp is meaningless once the process it was taken in
has exited.

Location, in order of preference:
    $OPTION_LIB_CACHE_DIR/sectors.json
    $XDG_CACHE_HOME/option_lib/sectors.json
    ~/.cache/option_lib/sectors.json

Every failure mode here — unreadable file, corrupt JSON, read-only directory —
degrades to an in-memory cache for the life of the process rather than raising.
A sector column is not worth taking an app down for.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# One month.  A sector changes at most on a reclassification, which is rare
# enough that a stale month costs nothing and a re-fetch storm costs more.
TTL_SECONDS = int(os.environ.get("OPTION_LIB_SECTOR_TTL", 30 * 24 * 3600))

_lock = threading.Lock()
_entries: dict[str, tuple[str | None, float]] | None = None   # symbol -> (sector, fetched_at)
_disk_ok = True     # cleared after a write failure, so we stop retrying it


def _cache_path() -> Path:
    override = os.environ.get("OPTION_LIB_CACHE_DIR")
    if override:
        return Path(override) / "sectors.json"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "option_lib" / "sectors.json"


def _load() -> dict[str, tuple[str | None, float]]:
    """Read the cache file into memory.  Caller holds the lock."""
    global _entries
    if _entries is not None:
        return _entries
    _entries = {}
    try:
        raw = json.loads(_cache_path().read_text(encoding="utf-8"))
        for symbol, rec in raw.items():
            # Skip anything malformed rather than discarding the whole file —
            # one bad row should not cost every other symbol its cache.
            if isinstance(rec, dict) and isinstance(rec.get("fetched"), (int, float)):
                sector = rec.get("sector")
                if sector is None or isinstance(sector, str):
                    _entries[symbol] = (sector, float(rec["fetched"]))
    except (OSError, ValueError, AttributeError):
        pass    # missing or corrupt — start empty
    return _entries


def _flush() -> None:
    """Write the cache out atomically.  Caller holds the lock."""
    global _disk_ok
    if not _disk_ok or _entries is None:
        return
    path = _cache_path()
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {s: {"sector": sec, "fetched": ts} for s, (sec, ts) in _entries.items()}
        tmp.write_text(json.dumps(payload, indent=0, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)   # atomic, so a crash mid-write cannot truncate the cache
    except OSError:
        _disk_ok = False        # read-only home, full disk — carry on in memory
        try:
            tmp.unlink()
        except OSError:
            pass


def get(symbol: str) -> tuple[str | None, bool]:
    """Return (sector, hit).  A cached None is a hit — ETFs have no sector."""
    with _lock:
        entry = _load().get(symbol.upper())
        if entry is None:
            return None, False
        sector, fetched_at = entry
        if time.time() - fetched_at < TTL_SECONDS:
            return sector, True
        return None, False


def put(symbol: str, sector: str | None) -> None:
    """Record *sector* for *symbol* and persist the cache."""
    with _lock:
        _load()[symbol.upper()] = (sector, time.time())
        _flush()


def clear() -> None:
    """Drop every entry, on disk and in memory.  Intended for tests."""
    global _entries
    with _lock:
        _entries = {}
        _flush()
