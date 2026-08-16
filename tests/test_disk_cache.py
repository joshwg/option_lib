"""
Shared disk cache  (option_lib/tests/test_disk_cache.py)
========================================================
Covers what makes sectors.json and names.json safe to share between the apps
that install option_lib.  They all point OPTION_LIB_CACHE_DIR at one directory
on the server, so the file has several writers, and two things have to hold:

  * nobody's entries are lost.  A flush that rewrote the file from its own
    snapshot dropped whatever another process had added since — which is how
    six writers of 900 symbols ended up with 156 of them.
  * a monthly expiry costs one re-fetch, not one per app.  Whoever notices the
    staleness claims the symbol; everyone else serves the old value until the
    new one lands.

The cross-process tests really do fork: an in-process double of the file lock
would prove nothing about the thing being defended against.

Usage:
    python3 -m unittest option_lib/tests/test_disk_cache.py
"""

import json
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from option_lib import disk_cache
from option_lib.disk_cache import FRESH, MISSING, STALE, DiskCache


def _write_batch(cache_dir: str, tag: str, worker: int, count: int) -> None:
    """Child-process body: write *count* symbols of its own into the shared file."""
    os.environ["OPTION_LIB_CACHE_DIR"] = cache_dir
    disk_cache.set_app_tag(tag)
    cache = DiskCache("sectors.json", field="sector", ttl_seconds=3600)
    for i in range(count):
        cache.put(f"W{worker}S{i}", f"Sector{worker}")


class DiskCacheTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["OPTION_LIB_CACHE_DIR"] = self._tmp.name
        self.dir = Path(self._tmp.name)
        disk_cache.set_app_tag("test")

    def tearDown(self):
        os.environ.pop("OPTION_LIB_CACHE_DIR", None)
        self._tmp.cleanup()

    def cache(self, ttl=3600, filename="sectors.json"):
        return DiskCache(filename, field="sector", ttl_seconds=ttl)

    # ── Basic read/write ──────────────────────────────────────────────────────

    def test_roundtrip_and_miss(self):
        cache = self.cache()
        cache.put("AAPL", "Technology")
        self.assertEqual(cache.get("AAPL"), ("Technology", True))
        self.assertEqual(cache.get("NOSUCH"), (None, False))

    def test_symbols_are_case_insensitive(self):
        cache = self.cache()
        cache.put("aapl", "Technology")
        self.assertEqual(cache.get("AAPL"), ("Technology", True))

    def test_cached_none_is_a_hit(self):
        """An ETF has no sector; None is the answer, not a reason to re-ask."""
        cache = self.cache()
        cache.put("SPY", None)
        self.assertEqual(cache.get("SPY"), (None, True))
        self.assertEqual(cache.peek("SPY"), (None, FRESH))

    def test_survives_a_new_instance(self):
        self.cache().put("AAPL", "Technology")
        self.assertEqual(self.cache().get("AAPL"), ("Technology", True))

    # ── Freshness ─────────────────────────────────────────────────────────────

    def test_peek_reports_missing_fresh_and_stale(self):
        cache = self.cache(ttl=0.2)
        self.assertEqual(cache.peek("AAPL"), (None, MISSING))
        cache.put("AAPL", "Technology")
        self.assertEqual(cache.peek("AAPL"), ("Technology", FRESH))
        time.sleep(0.3)
        # Stale still hands back the value — a month-old sector beats a blank cell.
        self.assertEqual(cache.peek("AAPL"), ("Technology", STALE))
        self.assertEqual(cache.get("AAPL"), (None, False))

    # ── Shared refresh ────────────────────────────────────────────────────────

    def test_only_one_app_claims_a_stale_symbol(self):
        apps = [self.cache(ttl=0.2) for _ in range(3)]
        apps[0].put("AAPL", "Technology")
        time.sleep(0.3)
        claims = [app.claim("AAPL") for app in apps]
        self.assertEqual(claims.count(True), 1, claims)

    def test_completed_refresh_releases_the_claim(self):
        winner, other = self.cache(ttl=0.2), self.cache(ttl=0.2)
        winner.put("AAPL", "Technology")
        time.sleep(0.3)
        self.assertTrue(winner.claim("AAPL"))
        winner.put("AAPL", "Technology")          # refresh lands
        record = json.loads((self.dir / "sectors.json").read_text())["AAPL"]
        self.assertNotIn("claimed", record)
        self.assertEqual(other.peek("AAPL"), ("Technology", FRESH))

    def test_claim_expires_with_its_lease(self):
        """A process killed mid-refresh must not strand the symbol forever."""
        cache = self.cache(ttl=0.2)
        cache.put("AAPL", "Technology")
        time.sleep(0.3)
        self.assertTrue(cache.claim("AAPL", lease_seconds=0.2))
        self.assertFalse(cache.claim("AAPL", lease_seconds=0.2))
        time.sleep(0.3)
        self.assertTrue(cache.claim("AAPL", lease_seconds=0.2))

    def test_unfetched_symbol_is_not_claim_gated(self):
        """With nothing to serve, every caller may as well fetch."""
        a, b = self.cache(), self.cache()
        self.assertTrue(a.claim("AAPL"))
        self.assertTrue(b.claim("AAPL"))

    # ── Sharing the file ──────────────────────────────────────────────────────

    def test_picks_up_another_writer_without_restarting(self):
        reader, writer = self.cache(), self.cache()
        reader.put("AAPL", "Technology")          # reader now holds a loaded copy
        writer.put("MSFT", "Technology")
        self.assertEqual(reader.get("MSFT"), ("Technology", True))
        self.assertEqual(reader.get("AAPL"), ("Technology", True))

    def test_concurrent_processes_keep_every_entry(self):
        workers, per_worker = 4, 50
        procs = [
            multiprocessing.Process(
                target=_write_batch,
                args=(str(self.dir), f"app{n}", n, per_worker),
            )
            for n in range(workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)

        written = json.loads((self.dir / "sectors.json").read_text())
        self.assertEqual(len(written), workers * per_worker)

    def test_leaves_no_scratch_litter(self):
        cache = self.cache()
        for i in range(5):
            cache.put(f"SYM{i}", "Technology")
        leftovers = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(leftovers, ["sectors.json", "sectors.json.lock"])

    def test_scratch_file_is_named_for_the_app(self):
        disk_cache.set_app_tag("margin")
        self.assertEqual(disk_cache.app_tag(), "margin")
        disk_cache.set_app_tag("Better Bettor/../")     # path characters stripped
        self.assertEqual(disk_cache.app_tag(), "betterbettor")

    def test_app_tag_falls_back_to_the_environment(self):
        disk_cache.set_app_tag("")
        os.environ["OPTION_LIB_APP"] = "mlscan"
        try:
            self.assertEqual(disk_cache.app_tag(), "mlscan")
        finally:
            os.environ.pop("OPTION_LIB_APP", None)

    # ── Degradation ───────────────────────────────────────────────────────────

    def test_corrupt_file_reads_as_empty_and_recovers(self):
        (self.dir / "sectors.json").write_text("{not json at all")
        cache = self.cache()
        self.assertEqual(cache.get("AAPL"), (None, False))
        cache.put("AAPL", "Technology")
        self.assertEqual(self.cache().get("AAPL"), ("Technology", True))

    def test_malformed_rows_do_not_cost_the_good_ones(self):
        (self.dir / "sectors.json").write_text(json.dumps({
            "AAPL": {"sector": "Technology", "fetched": time.time()},
            "BAD1": {"sector": {"nested": "dict"}, "fetched": time.time()},
            "BAD2": {"sector": "Technology"},              # no timestamp
            "BAD3": "not even a record",
        }))
        cache = self.cache()
        self.assertEqual(cache.get("AAPL"), ("Technology", True))
        for bad in ("BAD1", "BAD2", "BAD3"):
            self.assertEqual(cache.get(bad), (None, False), bad)

    def test_stale_tmp_file_from_a_crash_is_overwritten(self):
        (self.dir / "sectors.json.tmp.test").write_text("half a write")
        cache = self.cache()
        cache.put("AAPL", "Technology")
        self.assertEqual(self.cache().get("AAPL"), ("Technology", True))
        self.assertFalse((self.dir / "sectors.json.tmp.test").exists())

    def test_clear_empties_the_file(self):
        cache = self.cache()
        cache.put("AAPL", "Technology")
        cache.clear()
        self.assertEqual(cache.get("AAPL"), (None, False))
        self.assertEqual(json.loads((self.dir / "sectors.json").read_text()), {})


if __name__ == "__main__":
    unittest.main()
