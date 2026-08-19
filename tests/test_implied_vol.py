"""
Implied-volatility fallback ladder  (option_lib/tests/test_implied_vol.py)
==========================================================================
Covers the case that motivated implied_vol: outside regular hours no market
maker quotes an option, so every bid and ask in the chain reads 0.00.  With no
mid to solve against, the old code took the provider's own `implied_volatility`
field, which for an unquoted contract is the placeholder 1e-05.  That went into
the binomial tree and came back as an option worth nothing that decays by
nothing — the $/shr and theta columns both printed 0.00.

Every chain here is synthetic, so the tests pin the ladder itself rather than
whatever the market happens to be doing when they run.

Usage:
    python3 -m unittest option_lib/tests/test_implied_vol.py
"""

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from option_lib import implied_vol, iv_cache, pricing


# A quiet, ordinary contract to build chains around: three months out, roughly
# at the money, with a volatility nothing about the ladder should have to guess.
S_UNDER = 100.0
T_YEARS = 0.25
R_RATE  = 0.045
TRUE_IV = 0.40


def _row(strike, *, bid=0.0, ask=0.0, last=0.0, chain_iv=0.0):
    return {'strike': strike, 'bid': bid, 'ask': ask,
            'last_price': last, 'open_interest': 0, 'volume': 0,
            'implied_volatility': chain_iv}


def _fair(strike, side, iv=TRUE_IV, S=S_UNDER):
    """Black-Scholes price of a strike at *iv* — used to build honest quotes."""
    if side == 'call':
        return pricing.black_scholes_call(S, strike, T_YEARS, R_RATE, iv)
    return pricing.black_scholes_put(S, strike, T_YEARS, R_RATE, iv)


def _quoted(strike, side, iv=TRUE_IV, spread=0.02):
    """A row with a tight two-sided market straddling the fair value at *iv*."""
    mid = _fair(strike, side, iv)
    return _row(strike, bid=mid * (1 - spread), ask=mid * (1 + spread),
                last=mid, chain_iv=iv)


def _unquoted(strike, side, *, last=None, chain_iv=1e-05):
    """A row as it looks outside regular hours: no market, placeholder IV."""
    return _row(strike, bid=0.0, ask=0.0,
                last=_fair(strike, side) if last is None else last,
                chain_iv=chain_iv)


class TestPlausibilityBand(unittest.TestCase):
    """The band exists to reject placeholders, not to model anything."""

    def test_rejects_the_yahoo_placeholder(self):
        # 1e-05 is what Yahoo reports for a contract with no market.  Accepting
        # it as a volatility is the whole bug.
        self.assertFalse(implied_vol.plausible_iv(1e-05))

    def test_rejects_zero_and_none(self):
        self.assertFalse(implied_vol.plausible_iv(0))
        self.assertFalse(implied_vol.plausible_iv(None))
        self.assertFalse(implied_vol.plausible_iv('not a number'))

    def test_rejects_a_solve_that_ran_away(self):
        self.assertFalse(implied_vol.plausible_iv(12.0))     # 1200%

    def test_accepts_ordinary_volatilities(self):
        for iv in (0.05, 0.18, 0.40, 1.20, 3.0):
            self.assertTrue(implied_vol.plausible_iv(iv), iv)


class TestLadder(unittest.TestCase):
    """Each rung, reached by removing the ones above it."""

    def setUp(self):
        # Point the disk cache at a scratch directory so a developer's real
        # cache neither leaks into these tests nor gets clobbered by them.
        self._tmp = tempfile.TemporaryDirectory()
        self._prev_dir = os.environ.get('OPTION_LIB_CACHE_DIR')
        os.environ['OPTION_LIB_CACHE_DIR'] = self._tmp.name
        iv_cache.clear()

    def tearDown(self):
        iv_cache.clear()
        if self._prev_dir is None:
            os.environ.pop('OPTION_LIB_CACHE_DIR', None)
        else:
            os.environ['OPTION_LIB_CACHE_DIR'] = self._prev_dir
        self._tmp.cleanup()

    def _resolve(self, rows, strike, side='call', **kw):
        return implied_vol.resolve_iv(rows, strike, side, S_UNDER, T_YEARS, R_RATE, **kw)

    # ── Rung 1: the bid/ask mid ───────────────────────────────────────────────

    def test_live_market_recovers_the_true_iv(self):
        rows = [_quoted(k, 'call') for k in (90, 95, 100, 105, 110)]
        self.assertAlmostEqual(self._resolve(rows, 100), TRUE_IV, places=2)

    def test_a_good_mid_is_remembered_for_later(self):
        rows = [_quoted(100, 'call')]
        self._resolve(rows, 100, symbol='TEST', expiration='2026-09-18')
        key = iv_cache.contract_key('TEST', '2026-09-18', 100, 'call')
        self.assertAlmostEqual(iv_cache.get(key), TRUE_IV, places=2)

    # ── Rung 2: a neighbouring strike ─────────────────────────────────────────

    def test_an_illiquid_strike_borrows_from_its_neighbour(self):
        # Midday, but nobody quotes the 105s.  The strikes either side do.
        rows = [_quoted(95, 'call'), _quoted(100, 'call'),
                _unquoted(105, 'call', last=0), _quoted(110, 'call')]
        self.assertAlmostEqual(self._resolve(rows, 105), TRUE_IV, places=2)

    def test_the_placeholder_never_wins_over_a_neighbour(self):
        # This is the regression: 1e-05 sat on the target row and was taken at
        # face value even though the strike next door had a live market.
        rows = [_quoted(100, 'call'), _unquoted(105, 'call', last=0)]
        iv = self._resolve(rows, 105)
        self.assertGreater(iv, implied_vol.IV_FLOOR)
        self.assertAlmostEqual(iv, TRUE_IV, places=2)

    # ── Rung 3: what the contract implied when it last had a market ───────────

    def test_pre_market_falls_back_to_the_remembered_iv(self):
        """The case the module exists for: a whole chain with no quotes."""
        sym, exp = 'TEST', '2026-09-18'
        # Yesterday, at midday, the chain was quoted and the IV was recorded.
        live = [_quoted(k, 'call') for k in (95, 100, 105)]
        self._resolve(live, 100, symbol=sym, expiration=exp)

        # This morning, before the open: not one bid or ask in the chain.
        dark = [_unquoted(k, 'call', last=0) for k in (95, 100, 105)]
        iv = implied_vol.resolve_iv(dark, 100, 'call', S_UNDER, T_YEARS, R_RATE,
                                    symbol=sym, expiration=exp)
        self.assertAlmostEqual(iv, TRUE_IV, places=2)

    def test_the_remembered_iv_prices_against_the_pre_market_underlying(self):
        """Volatility is sticky overnight; the underlying is not.

        The point of remembering an IV is to re-price at the *new* spot, so an
        overnight move in the underlying shows up in $/shr instead of being
        discarded.
        """
        sym, exp = 'TEST', '2026-09-18'
        self._resolve([_quoted(100, 'call')], 100, symbol=sym, expiration=exp)

        dark = [_unquoted(100, 'call', last=0)]
        moved = S_UNDER * 1.05          # gapped up 5% overnight
        iv = implied_vol.resolve_iv(dark, 100, 'call', moved, T_YEARS, R_RATE,
                                    symbol=sym, expiration=exp)
        self.assertAlmostEqual(iv, TRUE_IV, places=2)

        before = pricing.black_scholes_call(S_UNDER, 100, T_YEARS, R_RATE, iv)
        after  = pricing.black_scholes_call(moved,   100, T_YEARS, R_RATE, iv)
        self.assertGreater(after, before)

    def test_no_memory_without_a_contract_identity(self):
        # Called without symbol/expiration the ladder still runs, just without
        # the disk rungs — nothing should be written under a guessed key.
        self._resolve([_quoted(100, 'call')], 100)
        self.assertEqual(iv_cache.get(iv_cache.contract_key('TEST', '2026-09-18', 100, 'call')),
                         None)

    # ── Rung 4: the last trade ────────────────────────────────────────────────

    def test_falls_back_to_the_last_trade_with_nothing_else_left(self):
        rows = [_unquoted(100, 'call')]      # last_price defaults to fair value
        self.assertAlmostEqual(self._resolve(rows, 100), TRUE_IV, places=2)

    def test_a_stale_trade_loses_to_a_live_neighbour(self):
        # A print from when the stock was much lower implies a nonsense IV.  A
        # quoted neighbour is a worse strike but far better data.
        rows = [_quoted(95, 'call'),
                _unquoted(100, 'call', last=_fair(100, 'call', iv=2.5))]
        self.assertAlmostEqual(self._resolve(rows, 100), TRUE_IV, places=2)

    # ── Rung 5: the provider's own field ──────────────────────────────────────

    def test_the_chain_field_is_the_last_resort_and_only_if_believable(self):
        rows = [_row(100, chain_iv=0.55)]
        self.assertAlmostEqual(self._resolve(rows, 100), 0.55, places=6)

    def test_an_unusable_row_resolves_to_nothing_rather_than_zero(self):
        """No rung produced a volatility, so say so — do not return a number.

        None reaches the UI as '—'.  A zero would reach it as 0.00, which reads
        as a computed fact rather than as missing data, and is exactly what this
        module was written to stop.
        """
        rows = [_unquoted(100, 'call', last=0, chain_iv=1e-05)]
        self.assertIsNone(self._resolve(rows, 100))

    def test_an_empty_chain_resolves_to_nothing(self):
        self.assertIsNone(self._resolve([], 100))

    # ── Strike matching ───────────────────────────────────────────────────────

    def test_an_unlisted_strike_uses_the_nearest_listed_one(self):
        rows = [_quoted(95, 'call'), _quoted(100, 'call')]
        self.assertEqual(implied_vol.nearest_row(rows, 99)['strike'], 100)
        self.assertEqual(implied_vol.nearest_row(rows, 96)['strike'], 95)

    def test_puts_resolve_the_same_way(self):
        rows = [_quoted(k, 'put') for k in (90, 95, 100)]
        self.assertAlmostEqual(self._resolve(rows, 95, side='put'), TRUE_IV, places=2)


class TestPreMarketGreeksEndToEnd(unittest.TestCase):
    """The symptom itself: $/shr and theta reading 0.00 before the open.

    Drives yahoo_data.fetch_option_greeks with a chain that has no quotes in it
    — what every chain looks like at 07:00 — and checks that a real price and a
    real theta come back out.
    """

    SYMBOL, EXPIRY, STRIKE = 'TEST', '2026-09-18', 100.0

    def setUp(self):
        from unittest import mock
        from option_lib import math_util, yahoo_data

        self._tmp = tempfile.TemporaryDirectory()
        self._prev_dir = os.environ.get('OPTION_LIB_CACHE_DIR')
        os.environ['OPTION_LIB_CACHE_DIR'] = self._tmp.name
        iv_cache.clear()

        self.yahoo = yahoo_data
        self.chain = {'success': True, 'expiration': self.EXPIRY,
                      'calls': [], 'puts': []}
        # Pre-market: the underlying has moved and the option chain has not.
        self.info = {'success': True, 'current_price': S_UNDER,
                     'pre_market_price': S_UNDER * 1.03,
                     'post_market_price': None, 'dividend_yield': 0}

        self._patches = [
            # Hold the clock in pre-market, so the test asserts the same thing
            # whatever time of day it is run at.  extended_underlying() reads
            # market_session through math_util's globals.
            mock.patch.object(math_util, 'market_session', lambda *a, **k: 'pre'),
            mock.patch.object(yahoo_data, 'get_options_for_expiration',
                              lambda *a, **k: self.chain),
            mock.patch.object(yahoo_data, 'get_stock_info',
                              lambda *a, **k: self.info),
            mock.patch.object(yahoo_data, 'get_years_to_expiration',
                              lambda *a, **k: T_YEARS),
            # _get_contract_iv quotes the OCC ticker directly before it looks at
            # the chain; before the open that quote is empty too.
            mock.patch.object(yahoo_data, '_yf_rate_limit', lambda: None),
            mock.patch.object(yahoo_data.yf, 'Ticker',
                              lambda *a, **k: mock.Mock(info={'bid': None, 'ask': None})),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        iv_cache.clear()
        if self._prev_dir is None:
            os.environ.pop('OPTION_LIB_CACHE_DIR', None)
        else:
            os.environ['OPTION_LIB_CACHE_DIR'] = self._prev_dir
        self._tmp.cleanup()

    def _greeks(self):
        return self.yahoo.fetch_option_greeks(
            self.SYMBOL, self.EXPIRY, self.STRIKE, 'CALL', use_extended=True)

    def test_a_dark_chain_no_longer_prices_to_zero(self):
        # Yesterday's session recorded an IV for this contract...
        self.chain['calls'] = [_quoted(k, 'call') for k in (95, 100, 105)]
        self._greeks()

        # ...and this morning nothing in the chain is quoted.
        self.chain['calls'] = [_unquoted(k, 'call', last=0) for k in (95, 100, 105)]
        g = self._greeks()

        self.assertIsNotNone(g['price'])
        self.assertIsNotNone(g['theta'])
        self.assertGreater(g['price'], 0.01, "priced to nothing — the original bug")
        self.assertLess(g['theta'], 0.0, "a long option with time left must decay")

    def test_the_placeholder_iv_produces_garbage_in_both_directions(self):
        """Pins why this is a fix and not a coincidence.

        Fed the 1e-05 placeholder, the binomial tree does not merely lose
        accuracy — it stops returning a price at all.  Out of the money every
        greek collapses to exactly 0, which is the zero-valued $/shr and theta
        this module was written for.  In the money it goes the other way: with
        sigma that small the up and down factors are both ~1, the risk-neutral
        probability divides by almost nothing, and the tree returns ~1e+192.

        Both are the same bad input.  Nothing downstream can tell either result
        from a real one, which is why the placeholder has to be rejected at the
        source rather than clamped after the fact.
        """
        def at_placeholder(S, K, side):
            return pricing.american_option_greeks(S, K, T_YEARS, R_RATE, 1e-05, 0,
                                                  option_type=side)

        out_of_the_money = at_placeholder(90.0, 100.0, 'call')
        self.assertEqual(out_of_the_money['price'], 0.0)
        self.assertEqual(out_of_the_money['theta'], 0.0)

        in_the_money = at_placeholder(103.0, 100.0, 'call')
        self.assertGreater(in_the_money['price'], 1e6)

        # The ladder now refuses to reach that rung at all: with nothing usable
        # anywhere it returns no price, which the UI renders as '—'.
        self.chain['calls'] = [_unquoted(100, 'call', last=0)]
        g = self._greeks()
        self.assertIsNone(g['price'])
        self.assertIsNone(g['theta'])

    def test_the_extended_underlying_is_what_gets_priced(self):
        """The recovered IV must be applied to the pre-market spot, not the close."""
        self.chain['calls'] = [_quoted(k, 'call') for k in (95, 100, 105)]
        self._greeks()
        self.chain['calls'] = [_unquoted(k, 'call', last=0) for k in (95, 100, 105)]

        extended = self._greeks()
        regular  = self.yahoo.fetch_option_greeks(
            self.SYMBOL, self.EXPIRY, self.STRIKE, 'CALL', use_extended=False)
        # The pre-market print is 3% above the close, so the call is worth more.
        self.assertGreater(extended['price'], regular['price'])


class TestContractKey(unittest.TestCase):
    """Keys have to be stable across the ways a strike reaches us."""

    def test_strike_shape_does_not_split_the_entry(self):
        self.assertEqual(iv_cache.contract_key('AAPL', '2026-09-18', 190, 'call'),
                         iv_cache.contract_key('AAPL', '2026-09-18', 190.0, 'call'))

    def test_case_and_padding_do_not_split_the_entry(self):
        self.assertEqual(iv_cache.contract_key(' aapl ', '2026-09-18', 190, 'Call'),
                         iv_cache.contract_key('AAPL', '2026-09-18', 190, 'CALL'))

    def test_the_two_sides_are_separate_contracts(self):
        self.assertNotEqual(iv_cache.contract_key('AAPL', '2026-09-18', 190, 'call'),
                            iv_cache.contract_key('AAPL', '2026-09-18', 190, 'put'))


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestBarsCacheTTL(unittest.TestCase):
    """bars_cache_ttl(): cache until just past the next bar boundary."""

    def test_hourly_expires_after_next_hour(self):
        from option_lib.math_util import bars_cache_ttl
        # at :50 → 10 min to the boundary + 2 min grace
        self.assertEqual(bars_cache_ttl("1h", now=10 * 3600 + 50 * 60), 720)
        # one second past the hour → nearly a full hour + grace
        self.assertEqual(bars_cache_ttl("1h", now=10 * 3600 + 1), 3719)

    def test_minute_bars_and_fallbacks(self):
        from option_lib.math_util import bars_cache_ttl
        self.assertEqual(bars_cache_ttl("30m", now=3 * 1800 + 100), 1820)
        self.assertEqual(bars_cache_ttl("1d"), 3600)      # daily: flat hour
        self.assertEqual(bars_cache_ttl("junk"), 3600)    # unparsable: flat hour
        self.assertGreaterEqual(bars_cache_ttl("1h", now=3600 - 1, grace=0), 60)  # floor
