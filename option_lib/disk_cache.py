"""Disk-backed reference-data cache.

Some of what option_lib fetches is effectively permanent: a company does not
change sector between restarts, and it does not change its name either.  Almost
everything else here caches in memory (see math_util.TTLCache), which is right
for prices but wrong for reference data — the desktop apps restart constantly,
and an in-memory cache would send a fresh yfinance .info request per symbol on
every launch just to be told 'Technology' again.

So reference data lives on disk, keyed by symbol, with a long expiry.  Entries
are stamped with wall-clock time rather than the monotonic clock the in-memory
cache uses, because a monotonic stamp is meaningless once the process it was
taken in has exited.

Location, in order of preference:
    $OPTION_LIB_CACHE_DIR/<filename>
    $XDG_CACHE_HOME/option_lib/<filename>
    ~/.cache/option_lib/<filename>

Every failure mode here — unreadable file, corrupt JSON, read-only directory —
degrades to an in-memory cache for the life of the process rather than raising.
A sector or a hover tooltip is not worth taking an app down for.

See sector_cache and name_cache for the concrete caches built on this.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import NamedTuple

try:
    import fcntl     # POSIX only — absent on Windows, where the desktop apps run
except ImportError:
    fcntl = None


# ── Application tag ───────────────────────────────────────────────────────────
# Names the app writing the cache, so its scratch file is its own: BetterBettor
# writes sectors.json.tmp.bettor, MarginWatch sectors.json.tmp.margin, and so on.
# A stable per-app name rather than a per-process one (a pid) on purpose — the
# set of files stays bounded and each run reuses its own, so a crash or a reboot
# leaves nothing to accumulate or clean up.

_DEFAULT_APP_TAG = "app"
_app_tag = ""

# How long one app's claim on a stale symbol keeps the others from re-fetching
# it.  Long enough to cover a slow provider call, short enough that a process
# killed mid-refresh doesn't strand the symbol for long.
CLAIM_LEASE_SECONDS = int(os.environ.get("OPTION_LIB_CLAIM_LEASE", 300))

# Freshness states reported by DiskCache.peek().
FRESH   = "fresh"      # inside the TTL — use it
STALE   = "stale"      # past the TTL, but the old value is still good enough to show
MISSING = "missing"    # never fetched, or the entry was dropped


class _Entry(NamedTuple):
    """One cached symbol: its value, when it was fetched, when a refresh was claimed."""
    value:   object
    fetched: float
    claimed: float = 0.0


def _sanitise_tag(tag: str) -> str:
    """Keep a tag to characters that are safe in a filename on any platform."""
    cleaned = "".join(c for c in str(tag).strip().lower() if c.isalnum() or c in "-_")
    return cleaned[:24]


def set_app_tag(tag: str) -> None:
    """Declare which application this process is, for scratch-file naming.

    Call once at startup, before any cache write.  $OPTION_LIB_APP does the same
    thing for deployments that would rather configure it than code it.
    """
    global _app_tag
    _app_tag = _sanitise_tag(tag)


def app_tag() -> str:
    """The current tag: whatever set_app_tag() or $OPTION_LIB_APP supplied."""
    return _app_tag or _sanitise_tag(os.environ.get("OPTION_LIB_APP", "")) or _DEFAULT_APP_TAG


class DiskCache:
    """A key → (value, fetched_at) map persisted as JSON.

    Values may be strings, numbers, or None — whatever JSON round-trips as a
    scalar.  Anything else is rejected on read rather than trusted, so a file
    hand-edited into a bad shape cannot hand a caller a dict where it expects
    a sector.

    Parameters
    ----------
    filename : str
        Basename of the JSON file inside the cache directory.
    field : str
        Key the value is stored under in each record, so the file stays
        self-describing ({"AAPL": {"sector": "Technology", "fetched": ...}}).
    ttl_seconds : int
        How long an entry stays fresh.
    """

    # One lock per cache file, shared by every DiskCache instance naming that
    # file.  Thread-safety has to be a property of the file, not of the object:
    # two instances for sectors.json in one process (a second construction, a
    # module reloaded under a test) would otherwise hold separate locks and
    # interleave their read-modify-writes with nothing to stop them.
    _locks_guard: threading.Lock = threading.Lock()
    _locks: dict[str, threading.RLock] = {}

    @classmethod
    def _lock_for(cls, filename: str) -> threading.RLock:
        with cls._locks_guard:
            return cls._locks.setdefault(filename, threading.RLock())

    def __init__(self, filename: str, field: str, ttl_seconds: int):
        self._filename    = filename
        self._field       = field
        self._ttl         = ttl_seconds
        self._lock        = self._lock_for(filename)
        self._entries: dict[str, tuple[object, float]] | None = None
        self._disk_ok     = True    # cleared after a write failure, so we stop retrying it
        self._mtime       = -1.0    # mtime of the file our in-memory copy came from

    # ── Location ──────────────────────────────────────────────────────────────

    def _path(self) -> Path:
        override = os.environ.get("OPTION_LIB_CACHE_DIR")
        if override:
            return Path(override) / self._filename
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
        return base / "option_lib" / self._filename

    # ── Persistence ───────────────────────────────────────────────────────────

    def _read_file(self) -> dict[str, _Entry]:
        """Parse the cache file.  Returns {} when missing or corrupt."""
        entries: dict[str, _Entry] = {}
        try:
            raw = json.loads(self._path().read_text(encoding="utf-8"))
            for symbol, rec in raw.items():
                # Skip anything malformed rather than discarding the whole file —
                # one bad row should not cost every other symbol its cache.
                if isinstance(rec, dict) and isinstance(rec.get("fetched"), (int, float)):
                    value = rec.get(self._field)
                    # bool is an int subclass, and no caller stores one; excluding
                    # it keeps a stray `true` from arriving as the number 1.
                    if value is None or isinstance(value, (str, int, float)) and not isinstance(value, bool):
                        claimed = rec.get("claimed")
                        entries[symbol] = _Entry(
                            value,
                            float(rec["fetched"]),
                            float(claimed) if isinstance(claimed, (int, float)) else 0.0,
                        )
        except (OSError, ValueError, AttributeError):
            pass    # missing or corrupt — start empty
        return entries

    def _file_mtime(self) -> float:
        try:
            return self._path().stat().st_mtime
        except OSError:
            return -1.0

    def _load(self) -> dict[str, tuple[object, float]]:
        """Return the in-memory map, re-reading the file if it changed.

        The re-read is what makes a shared cache directory pay off: several
        apps point OPTION_LIB_CACHE_DIR at one place, and a long-running server
        should pick up symbols another process fetched without being restarted.
        The cost is one stat() per lookup, against a network call saved.
        """
        mtime = self._file_mtime()
        if self._entries is not None and mtime == self._mtime:
            return self._entries
        self._entries = self._read_file()
        self._mtime   = mtime
        return self._entries

    @contextlib.contextmanager
    def _exclusive(self):
        """Hold a cross-process lock for the read-modify-write in _flush().

        Without it, two processes each rewrite the whole file from their own
        snapshot and the second silently drops everything the first added — the
        common case when several apps share one cache directory, not a rare
        race.  flock is POSIX; on Windows this degrades to no lock, where the
        merge below still limits the damage to an occasional lost entry.

        The lock lives in its own file so it survives the os.replace() that
        swaps the cache file's inode out from under it.
        """
        if fcntl is None:
            yield
            return
        lock_path = self._path().with_suffix(".json.lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(lock_path, "a+")
        except OSError:
            yield           # unwritable directory — _flush will fail and park anyway
            return
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _flush(self) -> None:
        """Merge this process's entries into the file and write it back.

        Merged rather than overwritten because the file is shared: whoever
        wrote a symbol most recently wins it, and nobody's entries are dropped
        just because another process happened to load the file earlier.
        """
        if not self._disk_ok or self._entries is None:
            return
        with self._exclusive():
            self._write_locked(self._merge(self._read_file(), self._entries))

    def _write_locked(self, entries: dict[str, _Entry]) -> None:
        """Serialise *entries* to disk.  Caller holds the thread and file locks."""
        path = self._path()
        # Scratch file named for the app, not the process: the set of temp files
        # is bounded by the number of apps sharing the directory, each run reuses
        # its own, and a crash leaves at most one stale file that the next write
        # overwrites.  On POSIX the file lock already serialises writers; the
        # per-app name is what keeps Windows — which has no flock here — from
        # letting two apps interleave into one scratch file.
        tmp = path.with_suffix(f".json.tmp.{app_tag()}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            for symbol, entry in entries.items():
                rec = {self._field: entry.value, "fetched": entry.fetched}
                if entry.claimed:
                    rec["claimed"] = entry.claimed
                payload[symbol] = rec
            tmp.write_text(json.dumps(payload, indent=0, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)   # atomic, so a crash mid-write cannot truncate the cache
            self._entries = entries
            self._mtime   = self._file_mtime()
        except OSError:
            self._disk_ok = False   # read-only home, full disk — carry on in memory
            with contextlib.suppress(OSError):
                tmp.unlink()

    @staticmethod
    def _merge(base: dict[str, _Entry], updates: dict[str, _Entry]) -> dict[str, _Entry]:
        """Fold *updates* into *base*: the newer fetch wins the symbol outright.

        Winning outright is what clears a claim — a completed refresh carries a
        newer timestamp and no claim, so recording it releases the symbol.  Only
        when both sides describe the same fetch (a claim laid on a value someone
        else already had) do the claims combine, taking the later one.
        """
        merged = dict(base)
        for symbol, entry in updates.items():
            existing = merged.get(symbol)
            if existing is None or entry.fetched > existing.fetched:
                merged[symbol] = entry
            elif existing.fetched > entry.fetched:
                merged[symbol] = existing
            else:
                merged[symbol] = existing._replace(
                    claimed=max(entry.claimed, existing.claimed))
        return merged

    # ── Public API ────────────────────────────────────────────────────────────

    def peek(self, symbol: str) -> tuple[object, str]:
        """Return (value, state) where state is FRESH, STALE, or MISSING.

        A stale entry still hands back its value: a month-old sector is a far
        better answer than a blank column while a refresh runs.  Callers that
        can only use a current answer should check the state.
        """
        with self._lock:
            entry = self._load().get(symbol.upper())
            if entry is None:
                return None, MISSING
            if time.time() - entry.fetched < self._ttl:
                return entry.value, FRESH
            return entry.value, STALE

    def get(self, symbol: str) -> tuple[object, bool]:
        """Return (value, hit).  A cached None is a hit — ETFs have no sector."""
        value, state = self.peek(symbol)
        return (value, True) if state == FRESH else (None, False)

    def claim(self, symbol: str, lease_seconds: int | None = None) -> bool:
        """Try to become the one process that refreshes *symbol*.

        Returns True to exactly one caller across every process sharing the
        cache directory; everyone else gets False and should serve the stale
        value.  That is what keeps a monthly expiry from turning into the same
        fetch run once per app — the first to notice does the work, and the
        result lands in the shared file for all of them.

        The claim is a lease, not a lock: if the claimer dies mid-fetch, the
        symbol becomes claimable again after *lease_seconds* rather than never.
        A symbol that has never been fetched is not claimable — the others have
        nothing to serve, so they may as well all try.
        """
        lease = CLAIM_LEASE_SECONDS if lease_seconds is None else lease_seconds
        key   = symbol.upper()
        now   = time.time()
        with self._lock:
            # Re-read under the file lock: another process may have claimed it
            # since our copy was loaded, and this decision must be the file's,
            # not ours.
            with self._exclusive():
                current = self._read_file()
                entry   = current.get(key)
                if entry is None:
                    return True                 # nothing cached — no point serialising
                if now - entry.claimed < lease:
                    return False                # someone else is on it
                current[key] = entry._replace(claimed=now)
                self._write_locked(self._merge(current, self._entries or {}))
            return True

    def put(self, symbol: str, value: object) -> None:
        """Record *value* for *symbol* and persist the cache.

        Clears any claim on the symbol: the refresh it was protecting is done.
        """
        with self._lock:
            self._load()[symbol.upper()] = _Entry(value, time.time(), 0.0)
            self._flush()

    def clear(self) -> None:
        """Drop every entry, on disk and in memory.  Intended for tests.

        Writes straight through rather than going via _flush(), because the
        merge _flush() performs exists to stop one process discarding another's
        entries — which is precisely what clearing has to do.  Routed through
        it, this read the file back and folded it into the empty map, so the
        cache came out of clear() exactly as it went in and a test that cleared
        and then asserted a miss was handed the value it had just dropped.

        Being unconditional is the point, so do not call it against a shared
        cache directory that another process is relying on.
        """
        with self._lock:
            if self._disk_ok:
                with self._exclusive():
                    self._write_locked({})
            # In memory regardless: a parked disk must not leave the cache with
            # no way to reset it at all.
            self._entries = {}
            self._mtime   = self._file_mtime()
