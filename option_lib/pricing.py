"""
Option Pricing Module
Calculates option prices and Greeks using multiple models:
- Black-Scholes model for European options
- Cox-Ross-Rubinstein Binomial Tree model for American options
"""

import numpy as np
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
        return {
            'delta': 1.0 if S > K else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

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


def implied_volatility(option_price, S, K, T, r, option_type='call', max_iterations=100, tolerance=1e-5):
    """
    Calculate implied volatility using Newton-Raphson method

    Parameters:
    option_price (float): Market price of the option
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    option_type (str): 'call' or 'put'
    max_iterations (int): Maximum number of iterations
    tolerance (float): Convergence tolerance

    Returns:
    float: Implied volatility (as decimal) or None if not converged
    """
    if T <= 0:
        return None

    # Initial guess
    sigma = 0.3

    for i in range(max_iterations):
        if option_type == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)

        diff = price - option_price

        if abs(diff) < tolerance:
            return sigma

        # Vega for Newton-Raphson
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)

        if vega < 1e-10:
            return None

        sigma = sigma - diff / vega

        # Keep sigma positive
        if sigma <= 0:
            sigma = 0.01

    return None  # Did not converge


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
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

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

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}
