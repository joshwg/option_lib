"""Resolving an implied volatility when the chain will not simply hand one over.

Pricing an option needs a volatility, and the option chain does not always
carry one worth using.  Outside regular hours no market maker quotes, so every
bid and ask in the chain reads 0.00; an illiquid strike looks exactly the same
way at midday.  With no mid there is no IV to solve for, and the providers'
own `implied_volatility` field is not a substitute: for an unquoted contract
Yahoo reports the placeholder 1e-05, and sometimes something merely wrong, like
6% on a fourteen-month LEAP.

Nothing rejected those.  A 1e-05 "volatility" went into the binomial tree and
came back out as an option worth nothing that decays by nothing, so the $/shr
and theta columns both read 0.00.  That is the one answer that is never right,
because a zero reads as a computed fact rather than as missing data.

So a volatility is resolved through a ladder of sources, best first, with every
rung held to the same plausibility band:

    1. the bid/ask mid          — a live two-sided market; the real answer
    2. a neighbouring strike    — live quotes on the same expiration, wrong by
                                  the skew between the two strikes and no more
    3. the last good IV on disk — what this exact contract implied when it last
                                  had a market, normally yesterday's close.
                                  This is the pre-market rung: volatility is
                                  sticky overnight while the underlying moves,
                                  so re-pricing at the remembered IV tracks the
                                  overnight move instead of discarding it
    4. the last traded price    — a real trade, but possibly a stale one on a
                                  strike nobody has touched in weeks
    5. the provider's IV field  — accepted only if it survives the band

Rung 2 beats rung 3 because a live quote one strike away has aged by nothing,
while a remembered IV has aged by a session; rung 3 beats rung 4 because a
stale print can be arbitrarily wrong in a way a remembered IV cannot.

Rung 1 is also the write point: whenever a real mid yields a real IV it is
recorded, so that rung 3 has something to find once the market closes.
"""

from __future__ import annotations

from option_lib import iv_cache
from option_lib import pricing as _pricing
from option_lib.math_util import iv_from_mid

# Plausibility band for an equity option's implied volatility.  The floor is
# there to reject placeholders, not to model anything — no listed single name
# trades at a 1% vol, so a number under it is a data artefact rather than a
# quote.  The ceiling catches the other direction: an IV solved off a stale
# print or a crossed market can come back in the hundreds.
IV_FLOOR   = 0.01     # 1%
IV_CEILING = 5.00     # 500%


def plausible_iv(iv) -> bool:
    """True when *iv* is a number that could be a real volatility."""
    try:
        return IV_FLOOR <= float(iv) <= IV_CEILING
    except (TypeError, ValueError):
        return False


def _iv_from_price(price, S, K, T, r, side) -> float | None:
    """Volatility implied by *price*, or None if it does not solve to a real one."""
    try:
        if not (price and float(price) > 0 and S and float(S) > 0 and K > 0 and T > 0):
            return None
        iv = _pricing.implied_volatility(float(price), float(S), K, T, r, option_type=side)
    except (TypeError, ValueError):
        return None
    return float(iv) if plausible_iv(iv) else None


def _row_iv_from_mid(row, S, T, r, side) -> float | None:
    """Volatility implied by this row's bid/ask mid, or None without a live market."""
    try:
        iv = iv_from_mid(row.get('bid') or 0, row.get('ask') or 0,
                         S or 0, row.get('strike') or 0, T, r, side)
    except (TypeError, ValueError):
        return None
    return float(iv) if plausible_iv(iv) else None


def nearest_row(rows, strike):
    """The chain row closest to *strike*, or None for an empty chain."""
    return min(rows, key=lambda o: abs((o.get('strike') or 0) - strike), default=None)


def _iv_from_neighbours(rows, K, S, T, r, side) -> float | None:
    """IV of the nearest strike on this expiration that still has a live market.

    Only quoted rows are candidates; an unquoted neighbour is the same missing
    data one strike over.  The answer carries the skew between the two strikes
    as its error, which is small beside the alternative of no price at all.
    """
    best_iv, best_gap = None, None
    for row in rows:
        gap = abs((row.get('strike') or 0) - K)
        if best_gap is not None and gap >= best_gap:
            continue        # already have something at least as close
        iv = _row_iv_from_mid(row, S, T, r, side)
        if iv is not None:
            best_iv, best_gap = iv, gap
    return best_iv


def resolve_iv(rows, strike, side, S, T, r=0.045, *,
               symbol=None, expiration=None) -> float | None:
    """Best available implied volatility for one contract, or None.

    *rows* is one side of a chain in the shape both providers return: dicts
    carrying 'strike', 'bid', 'ask', 'last_price' and 'implied_volatility'.
    The row nearest *strike* is the contract being priced, matching how callers
    have always resolved a strike that is not listed verbatim.

    Passing *symbol* and *expiration* enables the disk-backed rungs — the
    remembered IV is read from and written to a per-contract cache.  Without
    them the ladder still works, just without memory across sessions.

    See the module docstring for what each rung is and why they are in this
    order.
    """
    if not rows:
        return None
    row = nearest_row(rows, strike)
    if row is None:
        return None
    K = row.get('strike') or 0

    key = None
    if symbol and expiration:
        key = iv_cache.contract_key(symbol, expiration, K, side)

    # 1. A live two-sided market.  Also the write point for rung 3.
    iv = _row_iv_from_mid(row, S, T, r, side)
    if iv is not None:
        if key:
            iv_cache.put(key, iv)
        return iv

    # 2. The nearest strike that does have one.
    iv = _iv_from_neighbours(rows, K, S, T, r, side)
    if iv is not None:
        return iv

    # 3. What this contract implied when it last had a market.
    if key:
        iv = iv_cache.get(key)
        if plausible_iv(iv):
            return iv

    # 4. The last trade — real, but possibly long stale.
    iv = _iv_from_price(row.get('last_price'), S, K, T, r, side)
    if iv is not None:
        return iv

    # 5. The provider's own field, if it is a volatility at all.
    iv = row.get('implied_volatility')
    return float(iv) if plausible_iv(iv) else None
