"""
Mathematical Core
=================
Exact implementation of the Dirichlet-KL framework using polygamma
functions (not logarithmic surrogates).

Key formulas (ASCII notation):
    theta(S)      = 1/S - lam*(S-K)*psi'(S)        [threshold, Eq. 6]
    h(alpha; eta) = eta/alpha - lam*(alpha-1)*psi'(alpha)  [KKT, Eq. 7]
    g_p(S)        = S - (K-p) - sum alpha_i(theta(S))      [self-consistency, Eq. 13]

Here psi'(x) is the trigamma function (polygamma of order 1).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import polygamma, gammaln, digamma
from typing import Tuple


def theta_exact(S: float, lam: float, K: int) -> float:
    """Threshold function theta(S) from the paper (Eq. 6).

    theta(S) = 1/S - lam * (S - K) * psi'(S),
    where psi'(S) is the trigamma function.

    Parameters
    ----------
    S : float
        Sum of Dirichlet parameters (S > K).
    lam : float
        Regularization parameter (lam > 0).
    K : int
        Number of classes.

    Returns
    -------
    float
        Threshold value theta(S).

    Notes
    -----
    theta(S) is strictly decreasing for S > K (Proposition 2.2).
    theta(S) tends to -lam as S tends to infinity.
    """
    return 1.0 / S - lam * (S - K) * polygamma(1, S)


def theta_prime(S: float, lam: float, K: int) -> float:
    """Derivative of the threshold function theta'(S).

    theta'(S) = -1/S^2 - lam * [psi'(S) + (S - K) * psi''(S)]

    Parameters
    ----------
    S : float
        Sum of Dirichlet parameters.
    lam : float
        Regularization parameter.
    K : int
        Number of classes.

    Returns
    -------
    float
        Derivative theta'(S), which is negative for S > K.
    """
    psi1 = polygamma(1, S)
    psi2 = polygamma(2, S)
    return -1.0 / S**2 - lam * (psi1 + (S - K) * psi2)


def h_exact(alpha: float, eta: float, lam: float) -> float:
    """KKT function h(alpha; eta) from the paper (Eq. 7).

    h(alpha; eta) = eta / alpha - lam * (alpha - 1) * psi'(alpha)

    Parameters
    ----------
    alpha : float
        Dirichlet parameter (alpha >= 1).
    eta : float
        Softmax probability (0 <= eta <= 1).
    lam : float
        Regularization parameter.

    Returns
    -------
    float
        KKT function value h(alpha; eta).

    Notes
    -----
    h is strictly decreasing in alpha for any eta > 0 (Proposition 2.2).
    Boundary value: h(1; eta) = eta.
    Limit: h(alpha; eta) tends to -lam as alpha tends to infinity.
    """
    return eta / alpha - lam * (alpha - 1.0) * polygamma(1, alpha)


def h_prime(alpha: float, eta: float, lam: float) -> float:
    """Derivative of the KKT function with respect to alpha.

    dh/dalpha = -eta / alpha^2 - lam * [psi'(alpha) + (alpha - 1) * psi''(alpha)]

    Parameters
    ----------
    alpha : float
        Dirichlet parameter.
    eta : float
        Softmax probability.
    lam : float
        Regularization parameter.

    Returns
    -------
    float
        Derivative dh/dalpha, which is negative.
    """
    psi1 = polygamma(1, alpha)
    psi2 = polygamma(2, alpha)
    return -eta / alpha**2 - lam * (psi1 + (alpha - 1.0) * psi2)


def solve_alpha_for_theta(
    eta_i: float,
    theta_S: float,
    lam: float,
    tol: float = 1e-14,
) -> float:
    """Solve h(alpha; eta) = theta(S) for alpha > 1.

    Parameters
    ----------
    eta_i : float
        Softmax probability for class i.
    theta_S : float
        Threshold value theta(S).
    lam : float
        Regularization parameter.
    tol : float, optional
        Tolerance for root finding (default: 1e-14).

    Returns
    -------
    float
        Optimal alpha_i > 1 if eta_i > theta(S), else 1.0.

    Notes
    -----
    Uses Brent's method with safeguards. Returns 1.0 (inactive class)
    if eta_i <= theta(S), enforcing the box constraint alpha >= 1.
    """
    # Box constraint: if eta_i <= theta(S), the class is inactive
    if eta_i <= theta_S:
        return 1.0

    # h(alpha) strictly decreases from h(1) = eta_i to h(inf) = -lam.
    # Find an upper bound where h(alpha) < theta(S).
    alpha_lo = 1.0 + 1e-12
    alpha_hi = 2.0
    max_iter = 100
    for _ in range(max_iter):
        if h_exact(alpha_hi, eta_i, lam) < theta_S:
            break
        alpha_hi *= 2.0
        if alpha_hi > 1e6:
            break

    try:
        alpha_root = brentq(
            lambda a: h_exact(a, eta_i, lam) - theta_S,
            alpha_lo,
            alpha_hi,
            xtol=tol,
            rtol=tol,
        )
        return max(alpha_root, 1.0)
    except ValueError:
        # Fallback: class is inactive
        return 1.0


def compute_g_p(
    S: float,
    eta_sorted: np.ndarray,
    p: int,
    lam: float,
    K: int,
) -> float:
    """Self-consistency function g_p(S) from the paper (Eq. 13).

    g_p(S) = S - (K - p) - sum_{i in A} alpha_(i)(theta(S)),
    where A is the active set of size p (top-p classes by eta).

    Parameters
    ----------
    S : float
        Sum of Dirichlet parameters.
    eta_sorted : np.ndarray
        Softmax probabilities sorted in descending order.
    p : int
        Size of the active set (1 <= p <= K).
    lam : float
        Regularization parameter.
    K : int
        Number of classes.

    Returns
    -------
    float
        Self-consistency function value g_p(S).

    Notes
    -----
    Returns a large value (1e9) if any active class violates alpha > 1,
    indicating that this p is invalid for the given S.
    """
    theta_S = theta_exact(S, lam, K)
    alpha_sum = 0.0

    for i in range(p):
        alpha_i = solve_alpha_for_theta(eta_sorted[i], theta_S, lam)
        # Check if the class is actually active
        if alpha_i <= 1.0 + 1e-9:
            return 1e9  # Invalid: this class cannot be active
        alpha_sum += alpha_i

    # Inactive classes: alpha_j = 1
    alpha_sum += (K - p) * 1.0

    return S - alpha_sum


def compute_g_p_prime(
    S: float,
    eta_sorted: np.ndarray,
    p: int,
    lam: float,
    K: int,
) -> float:
    """Analytical derivative g_p'(S) from the paper (Proof of Theorem 2).

    g_p'(S) = 1 + theta'(S) * sum (D_ii)^(-1)
            = 1 - theta'(S) * sum (dh/dalpha)^(-1)

    Parameters
    ----------
    S : float
        Sum of Dirichlet parameters.
    eta_sorted : np.ndarray
        Softmax probabilities sorted in descending order.
    p : int
        Size of the active set.
    lam : float
        Regularization parameter.
    K : int
        Number of classes.

    Returns
    -------
    float
        Derivative g_p'(S).
    """
    theta_S = theta_exact(S, lam, K)
    dtheta_dS = theta_prime(S, lam, K)

    sum_inv_h_prime = 0.0
    for i in range(p):
        alpha_i = solve_alpha_for_theta(eta_sorted[i], theta_S, lam)
        if alpha_i <= 1.0 + 1e-9:
            return 1e9  # Invalid: class cannot be active
        dh_dalpha = h_prime(alpha_i, eta_sorted[i], lam)
        sum_inv_h_prime += 1.0 / dh_dalpha

    return 1.0 - dtheta_dS * sum_inv_h_prime


def kl_dirichlet(alpha: np.ndarray) -> float:
    """KL divergence KL(Dir(alpha) || Dir(1, ..., 1)).

    KL = ln Gamma(S) - sum ln Gamma(alpha_k) - ln Gamma(K)
         + sum (alpha_k - 1) * (psi(alpha_k) - psi(S))

    Parameters
    ----------
    alpha : np.ndarray
        Dirichlet parameters (alpha_k >= 1).

    Returns
    -------
    float
        KL divergence value.
    """
    S = np.sum(alpha)
    K = len(alpha)
    kl = (
        gammaln(S)
        - np.sum(gammaln(alpha))
        - gammaln(K)
        + np.sum((alpha - 1) * (digamma(alpha) - digamma(S)))
    )
    return kl


def compute_loss(alpha: np.ndarray, eta: np.ndarray, lam: float) -> float:
    """Objective function L(alpha; eta, lam) = NLL + lam * KL.

    Parameters
    ----------
    alpha : np.ndarray
        Dirichlet parameters.
    eta : np.ndarray
        Softmax probabilities.
    lam : float
        Regularization parameter.

    Returns
    -------
    float
        Objective function value L(alpha).
    """
    S = np.sum(alpha)
    # Negative log-likelihood: ln(S) - sum eta_k * ln(alpha_k)
    nll = np.log(S) - np.sum(eta * np.log(alpha))
    # KL divergence
    kl = kl_dirichlet(alpha)
    return nll + lam * kl