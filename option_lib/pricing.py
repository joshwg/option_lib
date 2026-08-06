"""
Option Pricing Module
Calculates option prices and Greeks using multiple models:
- Black-Scholes model for European options
- Cox-Ross-Rubinstein Binomial Tree model for American options
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a European call option

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    sigma (float): Volatility (as decimal)

    Returns:
    float: Call option price
    """
    if T <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a European put option

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    sigma (float): Volatility (as decimal)

    Returns:
    float: Put option price
    """
    if T <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def prob_expire_otm(S, K, T, r, sigma, option_type='put'):
    """
    Risk-neutral probability that an option expires out-of-the-money (worthless).

    For a put  : P(S_T > K) = N(d₂)
    For a call : P(S_T < K) = N(−d₂)

    This is the theoretical probability under the risk-neutral measure derived
    from Black-Scholes and serves as a practical Probability-of-Profit (PoP)
    estimate for short option positions.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    sigma (float): Implied volatility (as decimal)
    option_type (str): 'put' (default) or 'call'

    Returns:
    float: Probability in [0, 1], or float('nan') on degenerate input
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float('nan')
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'put':
        return float(norm.cdf(d2))    # P(S_T > K) — put expires worthless
    else:
        return float(norm.cdf(-d2))   # P(S_T < K) — call expires worthless


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    sigma (float): Volatility (as decimal)
    option_type (str): 'call' or 'put'

    Returns:
    dict: Dictionary containing Greeks (delta, gamma, theta, vega, rho)
    """
    if T <= 0:
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    if option_type == 'call':
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega (same for call and put)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


# Search bracket for implied_volatility(). 600% covers 0-DTE meme-stock and
# earnings-day vols; below 0.01% the option is worth its discounted intrinsic.
IV_SEARCH_MIN = 1e-4
IV_SEARCH_MAX = 6.0


def implied_volatility(option_price, S, K, T, r, option_type='call', max_iterations=100, tolerance=1e-5):
    """
    Calculate implied volatility by bracketed root-finding (Brent's method)

    Black-Scholes price is strictly increasing in sigma, so the root is always
    bracketed by [IV_SEARCH_MIN, IV_SEARCH_MAX] whenever the quote is
    attainable at all.  Brent is used in preference to Newton-Raphson because
    vega collapses towards zero for short-dated out-of-the-money options: from
    a fixed seed the Newton step overshoots into sigma <= 0, and the solve
    fails on exactly the contracts (1-2 DTE, OTM) where an accurate IV matters
    most.  A failed solve is silent at the call sites, which then fall back to
    the data provider's own stored IV field.

    Parameters:
    option_price (float): Market price of the option
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    option_type (str): 'call' or 'put'
    max_iterations (int): Maximum root-finder iterations
    tolerance (float): Price-space tolerance for the zero-vol edge case

    Returns:
    float: Implied volatility (as decimal), or None when the price lies
           outside the attainable range (stale or crossed quote) or the
           inputs are degenerate
    """
    if T <= 0 or option_price is None or option_price <= 0 or S <= 0 or K <= 0:
        return None

    if option_type == 'call':
        def price_at(sigma):
            return black_scholes_call(S, K, T, r, sigma)
    else:
        def price_at(sigma):
            return black_scholes_put(S, K, T, r, sigma)

    f_lo = price_at(IV_SEARCH_MIN) - option_price
    f_hi = price_at(IV_SEARCH_MAX) - option_price

    # Quote sits below the zero-vol floor (under discounted intrinsic) or above
    # the sigma ceiling — no root exists, so let the caller fall back.
    if f_lo > 0 or f_hi < 0:
        return None
    if abs(f_lo) < tolerance:
        return IV_SEARCH_MIN

    try:
        sigma = brentq(lambda s: price_at(s) - option_price,
                       IV_SEARCH_MIN, IV_SEARCH_MAX,
                       xtol=1e-8, maxiter=max_iterations)
    except (ValueError, RuntimeError):
        return None

    return sigma if sigma > 0 else None


def american_option_binomial(S, K, T, r, sigma, q=0, option_type='call', steps=100):
    """
    Calculate American option price using the Cox-Ross-Rubinstein binomial tree model.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    sigma (float): Volatility (as decimal)
    q (float): Continuous dividend yield (as decimal), default 0
    option_type (str): 'call' or 'put'
    steps (int): Number of time steps in the binomial tree (default 100)

    Returns:
    float: American option price
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    # Time step
    dt = T / steps

    # Up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Risk-neutral probability
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Discount factor
    discount = np.exp(-r * dt)

    # Asset prices at maturity (vectorised)
    i = np.arange(steps + 1)
    asset_prices = S * (u ** (steps - i)) * (d ** i)

    # Option payoffs at maturity
    if option_type == 'call':
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    # Step back through the tree (inner loop vectorised per step)
    for step in range(steps - 1, -1, -1):
        idx = np.arange(step + 1)
        asset_at_step = S * (u ** (step - idx)) * (d ** idx)
        held = discount * (p * option_values[:step + 1] + (1 - p) * option_values[1:step + 2])
        if option_type == 'call':
            exercise = np.maximum(asset_at_step - K, 0)
        else:
            exercise = np.maximum(K - asset_at_step, 0)
        option_values[:step + 1] = np.maximum(held, exercise)

    return float(option_values[0])


def american_option_greeks(S, K, T, r, sigma, q=0, option_type='call', steps=100):
    """
    Calculate greeks for an American option via finite differences on the binomial tree.

    Delta and gamma use a central-difference bump on S; theta uses a one-day
    forward difference on T.  Vega uses a forward difference on sigma.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    sigma (float): Volatility (as decimal)
    q (float): Continuous dividend yield (as decimal), default 0
    option_type (str): 'call' or 'put'
    steps (int): Binomial tree steps (default 100)

    Returns:
    dict: {'delta', 'gamma', 'theta', 'vega'}
          delta: sensitivity to $1 move in S (call 0–1, put −1–0)
          theta: $ change in value per calendar day (negative = time decay)
          gamma: rate of change of delta per $1 move in S
          vega:  $ change per 1 percentage-point move in implied vol
    """
    if T <= 0:
        delta = (1.0 if S > K else 0.0) if option_type == 'call' else (-1.0 if S < K else 0.0)
        price = max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
        return {'price': price, 'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

    price_0 = american_option_binomial(S, K, T, r, sigma, q, option_type, steps)

    # Delta and gamma — central difference on S (1% bump)
    dS = S * 0.01
    price_up = american_option_binomial(S + dS, K, T, r, sigma, q, option_type, steps)
    price_dn = american_option_binomial(S - dS, K, T, r, sigma, q, option_type, steps)
    delta = (price_up - price_dn) / (2 * dS)
    gamma = (price_up - 2 * price_0 + price_dn) / (dS ** 2)

    # Theta — one calendar day forward difference
    T_tomorrow = max(T - 1 / 365, 1e-6)
    price_tomorrow = american_option_binomial(S, K, T_tomorrow, r, sigma, q, option_type, steps)
    theta = price_tomorrow - price_0   # negative when option loses value overnight

    # Vega — forward difference on sigma (1 percentage-point = 0.01)
    d_sigma = 0.01
    price_vup = american_option_binomial(S, K, T, r, sigma + d_sigma, q, option_type, steps)
    vega = price_vup - price_0         # $ change per 1 pp rise in IV

    return {'price': price_0, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}
