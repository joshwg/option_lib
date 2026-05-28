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
    Calculate American option price using the Cox-Ross-Rubinstein binomial tree model

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (as decimal)
    sigma (float): Volatility (as decimal)
    q (float): Dividend yield (as decimal), default 0
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

    # Initialize asset prices at maturity
    asset_prices = np.zeros(steps + 1)
    for i in range(steps + 1):
        asset_prices[i] = S * (u ** (steps - i)) * (d ** i)

    # Initialize option values at maturity
    option_values = np.zeros(steps + 1)
    if option_type == 'call':
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    # Step back through the tree
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            # Calculate the asset price at this node
            asset_price = S * (u ** (step - i)) * (d ** i)

            # Expected option value (discounted)
            option_values[i] = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])

            # Check for early exercise
            if option_type == 'call':
                exercise_value = max(asset_price - K, 0)
            else:
                exercise_value = max(K - asset_price, 0)

            option_values[i] = max(option_values[i], exercise_value)

    return option_values[0]
