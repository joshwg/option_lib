"""
Market session windows and extended-hours S selection
=====================================================
Covers math_util.market_session() and math_util.extended_underlying(), which
decide the price basis every provider uses when use_extended=True:

    04:00 – 09:30   pre-market price
    09:30 – 16:00   regular price
    16:00 – 04:00   post-market price (runs past midnight)

No network access — every case is a pure function of an injected clock.

Usage:
    python3 -m unittest option_lib/tests/test_market_session.py
"""

import os
import sys
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

# Ensure option_lib root is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from option_lib.math_util import market_session, extended_underlying

ET = ZoneInfo("America/New_York")

WED = (2025, 11, 12)
FRI = (2025, 11, 14)
SAT = (2025, 11, 15)
SUN = (2025, 11, 16)
MON = (2025, 11, 17)

REGULAR, PRE, POST = 100.0, 101.0, 99.0
INFO = {"current_price": REGULAR, "pre_market_price": PRE, "post_market_price": POST}


def et(h, m, day=WED):
    return datetime(day[0], day[1], day[2], h, m, tzinfo=ET)


class TestMarketSession(unittest.TestCase):

    def test_regular_window(self):
        for h, m in [(9, 30), (12, 0), (15, 59)]:
            self.assertEqual(market_session(et(h, m)), "regular", f"{h:02d}:{m:02d} ET")

    def test_pre_window(self):
        for h, m in [(4, 0), (7, 30), (9, 29)]:
            self.assertEqual(market_session(et(h, m)), "pre", f"{h:02d}:{m:02d} ET")

    def test_post_window_runs_past_midnight(self):
        # Includes 20:00–24:00, which the exchanges call closed but which has no
        # fresher quote than the last after-hours print.
        for h, m in [(16, 0), (18, 0), (20, 0), (23, 59)]:
            self.assertEqual(market_session(et(h, m)), "post", f"{h:02d}:{m:02d} ET")
        for h, m in [(0, 1), (2, 0), (3, 59)]:
            self.assertEqual(market_session(et(h, m)), "post", f"{h:02d}:{m:02d} ET")

    def test_overnight_belongs_to_previous_day(self):
        self.assertEqual(market_session(et(2, 0, SAT)), "post",   "continues Friday evening")
        self.assertEqual(market_session(et(2, 0, SUN)), "closed", "Saturday had no session")
        self.assertEqual(market_session(et(2, 0, MON)), "closed", "Sunday had no session")

    def test_weekend_daytime_closed(self):
        for day in (SAT, SUN):
            self.assertEqual(market_session(et(12, 0, day)), "closed")
        self.assertEqual(market_session(et(16, 30, FRI)), "post", "Friday evening still trades")


class TestExtendedUnderlying(unittest.TestCase):

    def test_uses_current_session_price(self):
        self.assertEqual(extended_underlying(INFO, et(17, 0)), POST)
        self.assertEqual(extended_underlying(INFO, et(2, 0)),  POST)
        self.assertEqual(extended_underlying(INFO, et(6, 0)),  PRE)

    def test_ignores_other_sessions_stale_price(self):
        """After 16:00 the morning's pre price is half a day old — never S."""
        pre_only = {"current_price": REGULAR, "pre_market_price": PRE}
        self.assertEqual(extended_underlying(pre_only, et(17, 0)), REGULAR)
        post_only = {"current_price": REGULAR, "post_market_price": POST}
        self.assertEqual(extended_underlying(post_only, et(6, 0)), REGULAR)

    def test_falls_back_to_regular(self):
        bare = {"current_price": REGULAR}
        self.assertEqual(extended_underlying(bare, et(17, 0)), REGULAR)
        self.assertEqual(extended_underlying(INFO, et(12, 0, SAT)), REGULAR)


if __name__ == "__main__":
    unittest.main()
