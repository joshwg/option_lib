"""Ticker reference data — sector and company name — for display in tables.

The values themselves come from the configured provider (get_sector /
get_company_name) and are cached on disk for a month by sector_cache and
name_cache.  What this module adds is the access pattern every UI needs on top
of that cache, in one place instead of one copy per app:

    cached_*()    never touches the network.  Returns what is already known, or
                  None.  Safe to call from a Tk event handler, a paint loop, or
                  a request handler that must not block.
    resolve()     may fetch.  Only ever call it from a worker thread.
    resolve_many()  the same, for a batch, in parallel and bounded.
    prefetch()    warms the cache in the background and calls you back.

The split is the point: a table paints immediately from cached_*(), asks
prefetch() for what it is missing, and repaints when the answers land.  Nothing
in a render path ever waits on a network call.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from option_lib import name_cache as _name_cache
from option_lib import sector_cache as _sector_cache
from option_lib.fin_data import get_provider

# Bounded so a full table's worth of cold symbols cannot fan out into a burst
# the provider will rate-limit us for.
_MAX_WORKERS = 6

_provider = None
_provider_lock = threading.Lock()


def _data():
    """The shared provider, created on first use.

    Deferred rather than created at import time so that merely importing this
    module — which a GUI does at startup — cannot print provider banners or
    touch the network.
    """
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = get_provider()
        return _provider


# ── Cache-only reads (never block) ────────────────────────────────────────────

def cached_sector(ticker: str) -> str | None:
    """Sector for *ticker* if already cached, else None.  Never fetches."""
    if not ticker:
        return None
    value, _hit = _sector_cache.get(ticker)
    return value


def cached_name(ticker: str) -> str | None:
    """Company name for *ticker* if already cached, else None.  Never fetches."""
    if not ticker:
        return None
    value, _hit = _name_cache.get(ticker)
    return value


# ── Resolving reads (may fetch — worker threads only) ─────────────────────────

def resolve(ticker: str) -> dict[str, str | None]:
    """Return {'sector', 'name'} for *ticker*, fetching whatever is not cached.

    Failures degrade to None rather than raising: a missing sector is a blank
    cell, not a broken table.
    """
    sector = name = None
    if ticker:
        try:
            sector = _data().get_sector(ticker)
        except Exception:
            pass
        try:
            name = _data().get_company_name(ticker)
        except Exception:
            pass
    return {"sector": sector, "name": name}


def resolve_many(tickers) -> dict[str, dict[str, str | None]]:
    """Resolve a batch in parallel.  Returns {ticker: {'sector', 'name'}}."""
    unique = list(dict.fromkeys(t for t in tickers if t))
    if not unique:
        return {}
    if len(unique) == 1:
        return {unique[0]: resolve(unique[0])}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(unique))) as pool:
        return dict(zip(unique, pool.map(resolve, unique)))


def prefetch(tickers, on_done=None) -> threading.Thread:
    """Warm the cache for *tickers* on a background thread.

    *on_done* — if given — is called with the resolved {ticker: meta} map once
    the batch finishes.  It runs on the worker thread, so a Tk caller must
    marshal back to the UI thread itself (self.after / a queue).
    """
    def _run() -> None:
        try:
            result = resolve_many(tickers)
        except Exception:
            result = {}
        if on_done is not None:
            try:
                on_done(result)
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True, name="ticker-meta")
    thread.start()
    return thread
