"""
Table 4: Scalability Benchmark
==============================
Compares wall-clock time and active-set identification accuracy of
Algorithm 1 vs. compiled L-BFGS-B for K in {10, 50, 100, 500}.
"""

import numpy as np
from scipy.optimize import minimize
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.solver import algorithm_1_exact
from src.core import compute_loss
from src.utils import set_global_seed


def generate_confident_eta(K: int) -> np.ndarray:
    """Generate a probability vector eta with one dominant class
    (simulating a confident backbone classifier)."""
    dominant = np.random.uniform(0.8, 0.95)
    rest = np.random.dirichlet(np.ones(K - 1)) * (1.0 - dominant)
    return np.concatenate([[dominant], rest])


def lbfgsb_solve(eta: np.ndarray, lam: float, K: int) -> np.ndarray:
    """L-BFGS-B baseline with box constraints alpha >= 1."""
    def objective(alpha):
        alpha = np.maximum(alpha, 1.0 + 1e-12)
        return compute_loss(alpha, eta, lam)

    x0 = np.ones(K) * 1.1
    bounds = [(1.0, None)] * K
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return np.maximum(result.x, 1.0)


def active_set_accuracy(
    alpha_pred: np.ndarray,
    alpha_true: np.ndarray,
    tol: float = 1e-3,
) -> float:
    """Fraction of correctly identified active/inactive classes."""
    active_pred = (alpha_pred > 1.0 + tol).astype(int)
    active_true = (alpha_true > 1.0 + tol).astype(int)
    return np.mean(active_pred == active_true) * 100.0


def run_experiment(
    K_values: list = None,
    lam: float = 0.1,
    n_trials: int = 30,
    seed: int = 42,
):
    """Run scalability benchmark."""
    if K_values is None:
        K_values = [10, 50, 100, 500]

    set_global_seed(seed)

    print("=" * 70)
    print("Table 4: Scalability Benchmark")
    print(f"K in {K_values}, lam={lam}, n_trials={n_trials} per K")
    print("=" * 70)

    print(
        f"\n{'K':<6} {'Alg.1 (ms)':<14} {'L-BFGS-B (ms)':<16} "
        f"{'Time Ratio':<12} {'L-BFGS-B Acc.'}"
    )
    print("-" * 70)

    for K in K_values:
        etas = [generate_confident_eta(K) for _ in range(n_trials)]

        # Time Algorithm 1
        alg1_times = []
        alg1_alphas = []
        for eta in etas:
            t0 = time.perf_counter()
            S, p, alpha = algorithm_1_exact(eta, lam, K)
            alg1_times.append((time.perf_counter() - t0) * 1000)
            alg1_alphas.append(alpha)

        # Time L-BFGS-B
        lbfgs_times = []
        lbfgs_alphas = []
        for eta in etas:
            t0 = time.perf_counter()
            alpha = lbfgsb_solve(eta, lam, K)
            lbfgs_times.append((time.perf_counter() - t0) * 1000)
            lbfgs_alphas.append(alpha)

        # Compute accuracy (Algorithm 1 as ground truth)
        accuracies = []
        for i in range(n_trials):
            accuracies.append(
                active_set_accuracy(lbfgs_alphas[i], alg1_alphas[i])
            )

        t_alg1 = np.median(alg1_times)
        t_lbfgs = np.median(lbfgs_times)
        acc = np.mean(accuracies)
        ratio = t_alg1 / max(t_lbfgs, 1e-6)

        print(
            f"{K:<6} {t_alg1:>8.2f}      {t_lbfgs:>8.2f}        "
            f"{ratio:>5.1f}x       {acc:.1f}%"
        )


if __name__ == "__main__":
    run_experiment()