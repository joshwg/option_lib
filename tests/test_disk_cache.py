"""
Disk-backed cache  (option_lib/tests/test_disk_cache.py)
========================================================
The cache file is shared: several apps point OPTION_LIB_CACHE_DIR at one
directory, so a write merges rather than overwrites and nobody's entries are
lost to whoever happened to load the file first.

clear() is the one operation that has to do the opposite.  Routed through the
merging write it silently did nothing — it wrote an empty map, read the file
back, folded it in, and left the cache exactly as it was.  Both behaviours are
pinned here, against each other, because the fix for either one is the bug in
the other.

Usage:
    python3 -m unittest option_lib/tests/test_disk_cache.py
"""

import os
import sys
import tempfile
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from option_lib.disk_cache import DiskCache, FRESH, MISSING, STALE


class _CacheTestCase(unittest.TestCase):
    """Points the cache at a scratch directory for the life of one test."""

    FILENAME = "test_cache.json"
    FIELD    = "value"
    TTL      = 3600

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("OPTION_LIB_CACHE_DIR")
        os.environ["OPTION_LIB_CACHE_DIR"] = self._tmp.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("OPTION_LIB_CACHE_DIR", None)
        else:
            os.environ["OPTION_LIB_CACHE_DIR"] = self._prev
        self._tmp.cleanup()

    def cache(self, ttl=None):
        """A cache over the scratch file.

        Each call returns a fresh instance with its own in-memory map, which is
        how a second process sees the same file — the two share only the disk.
        """
        return DiskCache(self.FILENAME, field=self.FIELD,
                         ttl_seconds=self.TTL if ttl is None else ttl)

    def path(self):
        return os.path.join(self._tmp.name, self.FILENAME)


class TestClear(_CacheTestCase):

    def test_clear_drops_the_value_it_just_stored(self):
        c = self.cache()
        c.put("AAPL", "Technology")
        self.assertEqual(c.get("AAPL"), ("Technology", True))

        c.clear()
        self.assertEqual(c.get("AAPL"), (None, False))

    def test_clear_reaches_the_file_not_just_memory(self):
        """The regression: clearing left the file untouched.

        Checked through a second instance, because the one that cleared could
        report a miss from its own empty map while the disk still held
        everything — which is exactly what it used to do.
        """
        self.cache().put("AAPL", "Technology")
        self.cache().clear()

        self.assertEqual(self.cache().get("AAPL"), (None, False))
        self.assertEqual(self.cache().peek("AAPL"), (None, MISSING))

    def test_clear_drops_entries_this_instance_never_loaded(self):
        # Written by "another process" and never read by the one that clears.
        self.cache().put("AAPL", "Technology")
        self.cache().put("XOM", "Energy")

        c = self.cache()          # fresh, empty in-memory map
        c.clear()

        self.assertEqual(self.cache().get("AAPL"), (None, False))
        self.assertEqual(self.cache().get("XOM"), (None, False))

    def test_clear_leaves_a_readable_empty_file(self):
        """Not deleted and not truncated to nothing — still valid JSON."""
        import json
        self.cache().put("AAPL", "Technology")
        self.cache().clear()

        self.assertTrue(os.path.exists(self.path()))
        with open(self.path(), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {})

    def test_the_cache_still_works_after_a_clear(self):
        c = self.cache()
        c.put("AAPL", "Technology")
        c.clear()
        c.put("MSFT", "Technology")

        self.assertEqual(self.cache().get("MSFT"), ("Technology", True))
        self.assertEqual(self.cache().get("AAPL"), (None, False))

    def test_clear_on_an_untouched_cache_is_harmless(self):
        self.cache().clear()
        self.assertEqual(self.cache().get("AAPL"), (None, False))


class TestMergeStillHolds(_CacheTestCase):
    """clear() had to stop merging; put() still must, or this trades one bug for another."""

    def test_a_write_keeps_another_process_entries(self):
        first  = self.cache()
        second = self.cache()
        first.put("AAPL", "Technology")     # after second loaded its (empty) map
        second.put("XOM", "Energy")

        reader = self.cache()
        self.assertEqual(reader.get("AAPL"), ("Technology", True))
        self.assertEqual(reader.get("XOM"), ("Energy", True))

    def test_the_newer_write_of_a_symbol_wins(self):
        self.cache().put("AAPL", "Technology")
        time.sleep(0.01)
        self.cache().put("AAPL", "Consumer Electronics")
        self.assertEqual(self.cache().get("AAPL"), ("Consumer Electronics", True))


class TestValueTypes(_CacheTestCase):
    """Sectors and names are strings; a remembered IV is a float."""

    def test_a_float_round_trips(self):
        self.cache().put("KEY", 0.4237)
        self.assertEqual(self.cache().get("KEY"), (0.4237, True))

    def test_a_cached_none_is_a_hit(self):
        # An ETF has no sector, and that is an answer worth remembering rather
        # than re-fetching every time.
        self.cache().put("SPY", None)
        self.assertEqual(self.cache().get("SPY"), (None, True))
        self.assertEqual(self.cache().peek("SPY"), (None, FRESH))


class TestExpiry(_CacheTestCase):

    def test_a_stale_entry_misses_but_still_offers_its_value(self):
        self.cache(ttl=0).put("AAPL", "Technology")
        c = self.cache(ttl=0)
        self.assertEqual(c.get("AAPL"), (None, False))
        self.assertEqual(c.peek("AAPL"), ("Technology", STALE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
