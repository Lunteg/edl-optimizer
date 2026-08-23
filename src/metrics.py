"""
OOD Detection Metrics
=====================
Standard out-of-distribution detection metrics used in the paper:
    - MSP (Maximum Softmax Probability)
    - Energy Score
    - Vacuity (EDL epistemic uncertainty)
    - AUROC and FPR@TPR computation
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple


def compute_msp(eta: np.ndarray) -> np.ndarray:
    """Maximum Softmax Probability (MSP) score.

    MSP(x) = max_k eta_k(x).
    Higher MSP means more confident, hence more likely in-distribution.

    Parameters
    ----------
    eta : np.ndarray
        Softmax probabilities of shape (N, K).

    Returns
    -------
    np.ndarray
        MSP scores of shape (N,).

    References
    ----------
    Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and
    Out-of-Distribution Examples in Neural Networks", ICLR 2017.
    """
    return eta.max(axis=1)


def compute_energy(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy Score: -T * log(sum exp(z_k / T)).

    Lower energy means more likely in-distribution.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits from the neural network, shape (N, K).
    T : float, optional
        Temperature parameter (default: 1.0).

    Returns
    -------
    np.ndarray
        Energy scores of shape (N,).

    References
    ----------
    Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
    """
    # LogSumExp with numerical stability
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted / T), axis=1))
    return -T * log_sum_exp


def compute_vacuity(S: np.ndarray, K: int) -> np.ndarray:
    """EDL Vacuity (epistemic uncertainty): u* = K / S*.

    Higher vacuity means more uncertain, hence more likely OOD.

    Parameters
    ----------
    S : np.ndarray
        Sum of Dirichlet parameters, shape (N,).
    K : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Vacuity scores of shape (N,).
    """
    return K / S


def compute_auroc(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    higher_is_id: bool = True,
) -> float:
    """Compute AUROC for OOD detection.

    Parameters
    ----------
    scores_id : np.ndarray
        Scores for in-distribution samples.
    scores_ood : np.ndarray
        Scores for out-of-distribution samples.
    higher_is_id : bool, optional
        If True, higher scores indicate ID (default: True).
        If False, lower scores indicate ID.

    Returns
    -------
    float
        AUROC score (0.5 = random, 1.0 = perfect).
    """
    # Labels: ID = 1, OOD = 0
    labels = np.concatenate([
        np.ones(len(scores_id)),
        np.zeros(len(scores_ood)),
    ])
    scores = np.concatenate([scores_id, scores_ood])

    # If lower scores indicate ID, invert
    if not higher_is_id:
        scores = -scores

    return roc_auc_score(labels, scores)


def compute_fpr_at_tpr(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    tpr_target: float = 0.95,
    higher_is_id: bool = True,
) -> float:
    """Compute FPR at a given TPR (False Positive Rate at True Positive Rate).

    Parameters
    ----------
    scores_id : np.ndarray
        Scores for in-distribution samples.
    scores_ood : np.ndarray
        Scores for out-of-distribution samples.
    tpr_target : float, optional
        Target TPR (default: 0.95 for FPR@95).
    higher_is_id : bool, optional
        If True, higher scores indicate ID.

    Returns
    -------
    float
        FPR at the target TPR (lower is better).
    """
    # Find the threshold that achieves the target TPR on ID
    if higher_is_id:
        threshold = np.percentile(scores_id, 100 * (1 - tpr_target))
        fpr = np.mean(scores_ood > threshold)
    else:
        threshold = np.percentile(scores_id, 100 * tpr_target)
        fpr = np.mean(scores_ood < threshold)

    return fpr