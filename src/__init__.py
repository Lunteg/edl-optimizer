"""
EDL Optimizer: Certified Face-Enumeration for Regularized Dirichlet Likelihood

This package implements the mathematical framework and Algorithm 1 from:
"Geometry of Regularized Dirichlet Likelihood: Structural Theorem,
 SOSC Characterization, and Certified Face-Enumeration Algorithm"
"""

from .core import (
    theta_exact,
    h_exact,
    compute_g_p,
    kl_dirichlet,
    compute_loss,
)
from .solver import algorithm_1_exact, compute_S_crit
from .metrics import compute_msp, compute_energy, compute_auroc

__version__ = "1.0.0"
__all__ = [
    "theta_exact",
    "h_exact",
    "compute_g_p",
    "kl_dirichlet",
    "compute_loss",
    "algorithm_1_exact",
    "compute_S_crit",
    "compute_msp",
    "compute_energy",
    "compute_auroc",
]