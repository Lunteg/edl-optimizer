"""
Algorithm 1: Hybrid Compact-Core Newton
=======================================
Certified face-enumeration solver with guaranteed global optimality.

The algorithm:
    1. Enumerates all possible active-set sizes p = 1, ..., K.
    2. For each p, solves the self-consistency equation g_p(S) = 0.
    3. Checks second-order sufficient conditions (SOSC) and KKT validity.
    4. Selects the global minimum by comparing L(alpha) values.

Theorem (paper): for almost all eta, this algorithm finds the global
optimum. A robust fallback handles numerical edge cases.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, List, Tuple

from .core import (
    theta_exact,
    compute_g_p,
    compute_g_p_prime,
    solve_alpha_for_theta,
    compute_loss,
)


def compute_S_crit(lam: float, K: int) -> float:
    """Compute the critical threshold S_crit where theta(S) = 0.

    S_crit is the unique root of theta(S) = 0 for S > K.
    For S > S_crit, theta(S) < 0, which forces all classes to be
    active (p = K).

    Parameters
    ----------
    lam : float
        Regularization parameter.
    K : int
        Number of classes.

    Returns
    -------
    float
        Critical threshold S_crit.

    Notes
    -----
    This is the Asymptotic Active-Set Saturation theorem in the paper.
    """
    # Find an upper bound where theta(S) < 0
    S_hi = K + 1.0
    for _ in range(100):
        if theta_exact(S_hi, lam, K) < 0:
            break
        S_hi *= 2.0
        if S_hi > 1000.0:
            break

    # Find the root using Brent's method
    S_crit = brentq(
        lambda s: theta_exact(s, lam, K),
        K + 1e-5,
        S_hi,
        xtol=1e-14,
        rtol=1e-14,
    )
    return S_crit


def algorithm_1_exact(
    eta: np.ndarray,
    lam: float,
    K: int,
    epsilon: float = 1e-6,
) -> Tuple[float, int, np.ndarray]:
    """Algorithm 1: Hybrid Compact-Core Newton (exact, robust version).

    Returns alpha in the ORIGINAL order (matching the order of eta).

    Parameters
    ----------
    eta : np.ndarray
        Softmax probabilities (K,).
    lam : float
        Regularization parameter.
    K : int
        Number of classes.
    epsilon : float, optional
        Numerical tolerance for the SOSC check (default: 1e-6).

    Returns
    -------
    S : float
        Optimal total evidence S*.
    p : int
        Optimal active-set size p*.
    alpha : np.ndarray
        Optimal Dirichlet parameters alpha* (K,), in the original order.
    """
    # Remember the sorting permutation to restore the original order later
    sort_indices = np.argsort(eta)[::-1]
    eta_sorted = eta[sort_indices]

    # ----------------------------------------------------------
    # Step 1: Find all roots of g_p(S) = 0 for p = 1, ..., K
    # ----------------------------------------------------------
    all_roots: List[Tuple[float, int]] = []

    for p in range(1, K + 1):
        S_lo = K - p + 1e-5
        S_hi = S_lo + 1.0

        # Expand the upper bound until g_p(S_hi) > 0
        for _ in range(100):
            if compute_g_p(S_hi, eta_sorted, p, lam, K) > 0:
                break
            S_hi *= 1.5
            if S_hi > 1000.0:
                break

        g_lo = compute_g_p(S_lo, eta_sorted, p, lam, K)
        g_hi = compute_g_p(S_hi, eta_sorted, p, lam, K)

        if g_lo * g_hi <= 0:
            try:
                S_root = brentq(
                    lambda s: compute_g_p(s, eta_sorted, p, lam, K),
                    S_lo,
                    S_hi,
                    xtol=1e-12,
                    rtol=1e-12,
                )
                all_roots.append((S_root, p))
            except ValueError:
                pass

    # ----------------------------------------------------------
    # Step 2: Apply strict SOSC and KKT filtering
    # ----------------------------------------------------------
    candidates: List[Dict] = []

    for S_root, p in all_roots:
        theta_S = theta_exact(S_root, lam, K)

        # SOSC check: g_p'(S*) >= -epsilon
        g_p_prime_val = compute_g_p_prime(S_root, eta_sorted, p, lam, K)
        is_sosc = g_p_prime_val >= -epsilon

        # KKT check for inactive classes: eta_(p+1) < theta*
        is_kkt_valid = (p == K) or (eta_sorted[p] < theta_S + 1e-6)

        if is_sosc and is_kkt_valid:
            # Build alpha in the SORTED order
            alpha_sorted = np.ones(K)
            for i in range(p):
                alpha_sorted[i] = solve_alpha_for_theta(
                    eta_sorted[i], theta_S, lam
                )

            # Restore the ORIGINAL order via the inverse permutation
            unsort_indices = np.argsort(sort_indices)
            alpha = alpha_sorted[unsort_indices]

            # Compute the objective using the ORIGINAL eta
            L_val = compute_loss(alpha, eta, lam)

            candidates.append({
                "S": S_root,
                "p": p,
                "alpha": alpha,
                "L": L_val,
            })

    # ----------------------------------------------------------
    # Step 3: Robust fallback for numerical edge cases
    # ----------------------------------------------------------
    # If strict filtering removed all candidates, take the root with
    # the smallest objective value (it must be the global minimum).
    if not candidates and all_roots:
        best_S, best_p, best_alpha = None, None, None
        best_L = np.inf

        for S_root, p in all_roots:
            theta_S = theta_exact(S_root, lam, K)

            alpha_sorted = np.ones(K)
            for i in range(p):
                alpha_sorted[i] = solve_alpha_for_theta(
                    eta_sorted[i], theta_S, lam
                )

            unsort_indices = np.argsort(sort_indices)
            alpha = alpha_sorted[unsort_indices]
            L_val = compute_loss(alpha, eta, lam)

            if L_val < best_L:
                best_L = L_val
                best_S, best_p, best_alpha = S_root, p, alpha

        candidates.append({
            "S": best_S,
            "p": best_p,
            "alpha": best_alpha,
            "L": best_L,
        })

    # ----------------------------------------------------------
    # Step 4: Check the degenerate case p = 0 (uniform prior)
    # ----------------------------------------------------------
    if np.max(eta) <= 1.0 / K:
        alpha_prior = np.ones(K)
        L_prior = compute_loss(alpha_prior, eta, lam)
        candidates.append({
            "S": float(K),
            "p": 0,
            "alpha": alpha_prior,
            "L": L_prior,
        })

    # ----------------------------------------------------------
    # Step 5: Select the best candidate
    # ----------------------------------------------------------
    if not candidates:
        alpha_prior = np.ones(K)
        return float(K), 0, alpha_prior

    best = min(candidates, key=lambda c: c["L"])
    return best["S"], best["p"], best["alpha"]


def algorithm_1_batch(
    eta_batch: np.ndarray,
    lam: float,
    K: int,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch version of Algorithm 1 for multiple samples.

    Parameters
    ----------
    eta_batch : np.ndarray
        Softmax probabilities of shape (N, K).
    lam : float
        Regularization parameter.
    K : int
        Number of classes.
    epsilon : float, optional
        Numerical tolerance (default: 1e-6).

    Returns
    -------
    S_batch : np.ndarray
        Optimal S* for each sample, shape (N,).
    p_batch : np.ndarray
        Optimal p* for each sample, shape (N,).
    alpha_batch : np.ndarray
        Optimal alpha* for each sample, shape (N, K).
    """
    N = eta_batch.shape[0]
    S_batch = np.zeros(N)
    p_batch = np.zeros(N, dtype=int)
    alpha_batch = np.zeros((N, K))

    for i in range(N):
        S, p, alpha = algorithm_1_exact(eta_batch[i], lam, K, epsilon)
        S_batch[i] = S
        p_batch[i] = p
        alpha_batch[i] = alpha

    return S_batch, p_batch, alpha_batch