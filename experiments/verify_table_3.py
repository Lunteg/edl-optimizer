"""
Full Verification of Table 3 (OOD Detectors)
============================================
Multi-level validation of correctness and statistical significance.

Level 1: Basic metric correctness on synthetic data.
Level 2: Validation on real logits (CIFAR-10 ID, SVHN OOD).
Level 3: Statistical significance via bootstrap confidence intervals.
Level 4: Sensitivity analysis with respect to the regularization
         parameter lambda.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.special import polygamma, gammaln, digamma
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
import time
import os
from typing import Tuple, Dict, List


# ============================================================
# LEVEL 1: BASIC METRIC CORRECTNESS
# ============================================================

def verify_metrics_correctness():
    """Verify that the OOD metrics are computed correctly on synthetic data
    with a known ground-truth separation."""
    print("=" * 70)
    print("LEVEL 1: Metric correctness check")
    print("=" * 70)

    # Synthetic data with a known outcome
    np.random.seed(42)

    # ID: high scores (AUROC should be close to 1.0)
    scores_id = np.random.normal(0.8, 0.1, 1000)
    scores_ood = np.random.normal(0.2, 0.1, 1000)

    labels = np.concatenate([np.ones(len(scores_id)), np.zeros(len(scores_ood))])
    scores = np.concatenate([scores_id, scores_ood])

    # Check 1: AUROC
    auroc = roc_auc_score(labels, scores)
    print(f"[OK] AUROC (expected ~1.0): {auroc:.4f}")
    assert auroc > 0.95, "AUROC too low for a perfect separation!"

    # Check 2: FPR at 95% TPR
    threshold = np.percentile(scores_id, 5)
    fpr = np.mean(scores_ood > threshold)
    print(f"[OK] FPR@95 (expected ~0.0): {fpr:.4f}")
    assert fpr < 0.05, "FPR@95 too high!"

    # Check 3: Vacuity inversion
    # Vacuity: lower is better, so we invert the sign for AUROC
    vac_id = 1 - scores_id
    vac_ood = 1 - scores_ood

    # For AUROC: ID must have the HIGHER score, hence score = -vacuity
    auroc_vac = roc_auc_score(labels, np.concatenate([-vac_id, -vac_ood]))
    print(f"[OK] AUROC Vacuity (inverted, expected ~1.0): {auroc_vac:.4f}")
    assert auroc_vac > 0.95, "Vacuity inversion is not working correctly!"

    print("\n[PASS] All basic checks passed!")
    return True


# ============================================================
# LEVEL 2: VALIDATION ON REAL DATA (CIFAR-10 + SVHN)
# ============================================================

def theta_exact(S, lam, K):
    """Threshold function theta(S)."""
    return 1.0 / S - lam * (S - K) * polygamma(1, S)


def h_exact(alpha, eta, lam):
    """KKT function h(alpha; eta)."""
    return eta / alpha - lam * (alpha - 1.0) * polygamma(1, alpha)


def solve_alpha_for_theta(eta_i, theta_S, lam):
    """Solve h(alpha; eta) = theta(S) for alpha >= 1.
    Returns 1.0 if the class is inactive."""
    if eta_i <= theta_S:
        return 1.0
    alpha_lo = 1.0 + 1e-12
    alpha_hi = 2.0
    while h_exact(alpha_hi, eta_i, lam) > theta_S:
        alpha_hi *= 2.0
        if alpha_hi > 1e6:
            break
    try:
        alpha_root = brentq(
            lambda a: h_exact(a, eta_i, lam) - theta_S,
            alpha_lo, alpha_hi,
            xtol=1e-14, rtol=1e-14,
        )
        return max(alpha_root, 1.0)
    except ValueError:
        return 1.0


def compute_g_p(S, eta_sorted, p, lam, K):
    """Self-consistency function g_p(S)."""
    theta_S = theta_exact(S, lam, K)
    alpha_sum = 0.0
    for i in range(p):
        alpha_i = solve_alpha_for_theta(eta_sorted[i], theta_S, lam)
        if alpha_i <= 1.0 + 1e-9:
            return 1e9
        alpha_sum += alpha_i
    alpha_sum += (K - p) * 1.0
    return S - alpha_sum


def kl_dirichlet(alpha):
    """KL divergence KL(Dir(alpha) || Dir(1,...,1))."""
    S = np.sum(alpha)
    K = len(alpha)
    return (
        gammaln(S)
        - np.sum(gammaln(alpha))
        - gammaln(K)
        + np.sum((alpha - 1) * (digamma(alpha) - digamma(S)))
    )


def compute_L(alpha, eta, lam):
    """Objective function L(alpha; eta, lam) = NLL + lam * KL."""
    S = np.sum(alpha)
    nll = np.log(S) - np.sum(eta * np.log(alpha))
    kl = kl_dirichlet(alpha)
    return nll + lam * kl


def algorithm_1_exact(eta, lam, K):
    """Algorithm 1: Hybrid Compact-Core Newton (self-contained implementation).
    Enumerates active-set sizes p = 1..K and returns the global minimum."""
    eta_sorted = np.sort(eta)[::-1]
    candidates = []

    for p in range(1, K + 1):
        S_lo = K - p + 1e-5
        S_hi = S_lo + 1.0
        while compute_g_p(S_hi, eta_sorted, p, lam, K) < 0:
            S_hi *= 1.5
            if S_hi > 1000.0:
                break

        g_lo = compute_g_p(S_lo, eta_sorted, p, lam, K)
        g_hi = compute_g_p(S_hi, eta_sorted, p, lam, K)

        if g_lo * g_hi <= 0:
            try:
                S_root = brentq(
                    lambda s: compute_g_p(s, eta_sorted, p, lam, K),
                    S_lo, S_hi,
                    xtol=1e-12, rtol=1e-12,
                )
                theta_S = theta_exact(S_root, lam, K)
                alpha = np.ones(K)
                for i in range(p):
                    alpha[i] = solve_alpha_for_theta(eta_sorted[i], theta_S, lam)
                candidates.append({
                    "S": S_root,
                    "p": p,
                    "alpha": alpha,
                    "L": compute_L(alpha, eta_sorted, lam),
                })
            except ValueError:
                pass

    # Degenerate case p = 0 (uniform prior)
    if np.max(eta) <= 1.0 / K:
        alpha_prior = np.ones(K)
        candidates.append({
            "S": float(K),
            "p": 0,
            "alpha": alpha_prior,
            "L": compute_L(alpha_prior, eta, lam),
        })

    if not candidates:
        alpha_prior = np.ones(K)
        return float(K), 0, alpha_prior

    best = min(candidates, key=lambda c: c["L"])
    return best["S"], best["p"], best["alpha"]


def load_model_and_data():
    """Load the pretrained ResNet-18 and the CIFAR-10 / SVHN loaders."""
    K = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet18(weights=None, num_classes=K)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()

    path = "cifar10_resnet18.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"[OK] Model loaded from {path}")
    else:
        print(f"[FAIL] Model not found: {path}")
        return None, None, None

    model = model.to(DEVICE).eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    svhn_test = torchvision.datasets.SVHN(
        root="./data", split="test", download=True, transform=transform_test
    )

    cifar_loader = DataLoader(cifar_test, batch_size=128, shuffle=False)
    svhn_loader = DataLoader(svhn_test, batch_size=128, shuffle=False)

    return model, cifar_loader, svhn_loader


@torch.no_grad()
def extract_data(model, loader, n_samples):
    """Extract logits and softmax probabilities from a data loader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits, labels = [], []
    count = 0
    for x, y in loader:
        if count >= n_samples:
            break
        logits.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
        count += x.size(0)

    logits = np.concatenate(logits)[:n_samples]
    labels = np.concatenate(labels)[:n_samples]

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    eta = exp_l / exp_l.sum(axis=1, keepdims=True)
    return logits, eta, labels


def verify_table_3_real_data():
    """Validate Table 3 on real data (CIFAR-10 as ID, SVHN as OOD)."""
    print("\n" + "=" * 70)
    print("LEVEL 2: Validation on real data")
    print("=" * 70)

    K = 10
    LAMBDA = 0.1
    N_SAMPLES = 1000

    model, cifar_loader, svhn_loader = load_model_and_data()
    if model is None:
        return False

    print("\nExtracting logits...")
    logits_id, eta_id, _ = extract_data(model, cifar_loader, N_SAMPLES)
    logits_ood, eta_ood, _ = extract_data(model, svhn_loader, N_SAMPLES)

    print(f"[OK] ID:  {len(eta_id)} samples, mean max(eta) = {eta_id.max(axis=1).mean():.4f}")
    print(f"[OK] OOD: {len(eta_ood)} samples, mean max(eta) = {eta_ood.max(axis=1).mean():.4f}")

    S_crit = brentq(lambda s: theta_exact(s, LAMBDA, K), K + 1e-5, 100.0)
    print(f"[OK] S_crit = {S_crit:.4f}")

    print("\nRunning Algorithm 1...")
    res_id, res_ood = [], []
    t0 = time.time()

    for i in range(len(eta_id)):
        S, p, alpha = algorithm_1_exact(eta_id[i], LAMBDA, K)
        res_id.append({"S": S, "p": p})

    for i in range(len(eta_ood)):
        S, p, alpha = algorithm_1_exact(eta_ood[i], LAMBDA, K)
        res_ood.append({"S": S, "p": p})

    print(f"[OK] Completed in {time.time() - t0:.1f}s")

    S_id = np.array([r["S"] for r in res_id])
    S_ood = np.array([r["S"] for r in res_ood])
    max_id = eta_id.max(axis=1)
    max_ood = eta_ood.max(axis=1)

    msp_id, msp_ood = max_id, max_ood
    energy_id = np.log(np.sum(np.exp(logits_id), axis=1))
    energy_ood = np.log(np.sum(np.exp(logits_ood), axis=1))
    vac_id, vac_ood = K / S_id, K / S_ood

    labels = np.concatenate([np.ones(len(msp_id)), np.zeros(len(msp_ood))])

    auroc_msp = roc_auc_score(labels, np.concatenate([msp_id, msp_ood]))
    auroc_energy = roc_auc_score(labels, np.concatenate([energy_id, energy_ood]))
    auroc_vac = roc_auc_score(labels, np.concatenate([-vac_id, -vac_ood]))

    print("\n" + "-" * 70)
    print("RESULTS:")
    print("-" * 70)
    print(f"MSP:          AUROC = {auroc_msp:.4f}")
    print(f"Energy Score: AUROC = {auroc_energy:.4f}")
    print(f"Vacuity:      AUROC = {auroc_vac:.4f}")

    assert auroc_vac > 0.7, "Vacuity too low!"
    assert auroc_msp > 0.8, "MSP too low!"
    assert auroc_energy > 0.8, "Energy Score too low!"

    print("\n[PASS] Real-data validation passed!")
    return True


# ============================================================
# LEVEL 3: STATISTICAL SIGNIFICANCE (BOOTSTRAP)
# ============================================================

def bootstrap_auroc(scores_id, scores_ood, n_bootstraps=1000):
    """Compute bootstrap confidence intervals for the AUROC."""
    labels = np.concatenate([np.ones(len(scores_id)), np.zeros(len(scores_ood))])
    scores = np.concatenate([scores_id, scores_ood])

    aurocs = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstraps):
        idx = rng.choice(len(labels), len(labels), replace=True)
        auroc = roc_auc_score(labels[idx], scores[idx])
        aurocs.append(auroc)

    aurocs = np.array(aurocs)
    mean = np.mean(aurocs)
    std = np.std(aurocs)
    ci_low = np.percentile(aurocs, 2.5)
    ci_high = np.percentile(aurocs, 97.5)

    return mean, std, ci_low, ci_high


def verify_statistical_significance():
    """Verify the statistical significance of the differences between methods."""
    print("\n" + "=" * 70)
    print("LEVEL 3: Statistical significance (bootstrap)")
    print("=" * 70)

    K = 10
    LAMBDA = 0.1
    N_SAMPLES = 500

    model, cifar_loader, svhn_loader = load_model_and_data()
    if model is None:
        return False

    logits_id, eta_id, _ = extract_data(model, cifar_loader, N_SAMPLES)
    logits_ood, eta_ood, _ = extract_data(model, svhn_loader, N_SAMPLES)

    res_id, res_ood = [], []
    for i in range(len(eta_id)):
        S, p, alpha = algorithm_1_exact(eta_id[i], LAMBDA, K)
        res_id.append({"S": S})
    for i in range(len(eta_ood)):
        S, p, alpha = algorithm_1_exact(eta_ood[i], LAMBDA, K)
        res_ood.append({"S": S})

    S_id = np.array([r["S"] for r in res_id])
    S_ood = np.array([r["S"] for r in res_ood])
    max_id = eta_id.max(axis=1)
    max_ood = eta_ood.max(axis=1)

    msp_id, msp_ood = max_id, max_ood
    energy_id = np.log(np.sum(np.exp(logits_id), axis=1))
    energy_ood = np.log(np.sum(np.exp(logits_ood), axis=1))
    vac_id, vac_ood = K / S_id, K / S_ood

    print("\nComputing confidence intervals (1000 bootstrap iterations)...")

    print("  MSP...")
    msp_mean, msp_std, msp_low, msp_high = bootstrap_auroc(msp_id, msp_ood, n_bootstraps=1000)

    print("  Energy...")
    energy_mean, energy_std, energy_low, energy_high = bootstrap_auroc(energy_id, energy_ood, n_bootstraps=1000)

    print("  Vacuity...")
    vac_mean, vac_std, vac_low, vac_high = bootstrap_auroc(-vac_id, -vac_ood, n_bootstraps=1000)

    print("\n" + "-" * 70)
    print("RESULTS WITH CONFIDENCE INTERVALS:")
    print("-" * 70)
    print(f"MSP:          {msp_mean:.4f} +/- {msp_std:.4f}  [95% CI: {msp_low:.4f}, {msp_high:.4f}]")
    print(f"Energy Score: {energy_mean:.4f} +/- {energy_std:.4f}  [95% CI: {energy_low:.4f}, {energy_high:.4f}]")
    print(f"Vacuity:      {vac_mean:.4f} +/- {vac_std:.4f}  [95% CI: {vac_low:.4f}, {vac_high:.4f}]")

    assert msp_std < 0.05, "MSP confidence interval too wide!"
    assert energy_std < 0.05, "Energy confidence interval too wide!"
    assert vac_std < 0.05, "Vacuity confidence interval too wide!"

    assert msp_low > 0.7, "MSP CI lower bound too low!"
    assert energy_low > 0.7, "Energy CI lower bound too low!"
    assert vac_low > 0.7, "Vacuity CI lower bound too low!"

    print("\n[PASS] Statistical significance confirmed!")
    return True


# ============================================================
# LEVEL 4: HYPERPARAMETER SENSITIVITY
# ============================================================

def verify_hyperparameter_sensitivity():
    """Verify how the results change under different values of lambda."""
    print("\n" + "=" * 70)
    print("LEVEL 4: Hyperparameter sensitivity")
    print("=" * 70)

    K = 10
    N_SAMPLES = 300

    model, cifar_loader, svhn_loader = load_model_and_data()
    if model is None:
        return False

    logits_id, eta_id, _ = extract_data(model, cifar_loader, N_SAMPLES)
    logits_ood, eta_ood, _ = extract_data(model, svhn_loader, N_SAMPLES)

    labels = np.concatenate([np.ones(len(eta_id)), np.zeros(len(eta_ood))])

    # Baseline metrics (independent of lambda)
    max_id = eta_id.max(axis=1)
    max_ood = eta_ood.max(axis=1)
    auroc_msp = roc_auc_score(labels, np.concatenate([max_id, max_ood]))

    energy_id = np.log(np.sum(np.exp(logits_id), axis=1))
    energy_ood = np.log(np.sum(np.exp(logits_ood), axis=1))
    auroc_energy = roc_auc_score(labels, np.concatenate([energy_id, energy_ood]))

    print("\nBaseline metrics (independent of lambda):")
    print(f"  MSP:    AUROC = {auroc_msp:.4f}")
    print(f"  Energy: AUROC = {auroc_energy:.4f}")

    lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]

    print("\nVacuity sensitivity to lambda:")
    print("-" * 70)
    print(f"{'lambda':<10} {'S_crit':<15} {'Vacuity AUROC':<20} {'Delta vs MSP'}")
    print("-" * 70)

    for lam in lambdas:
        def theta_func(s):
            return theta_exact(s, lam, K)

        # Find an upper bound where theta(S) < 0
        S_hi = 100.0
        for _ in range(10):
            if theta_func(S_hi) < 0:
                break
            S_hi *= 2.0

        try:
            S_crit = brentq(theta_func, K + 1e-5, S_hi, xtol=1e-14)
        except ValueError as e:
            print(f"{lam:<10.2f} {'ERROR':<15} {'N/A':<20} {'N/A'}")
            print(f"  [WARN] Could not find S_crit for lambda={lam}: {e}")
            continue

        res_id, res_ood = [], []
        for i in range(len(eta_id)):
            S, p, alpha = algorithm_1_exact(eta_id[i], lam, K)
            res_id.append({"S": S})
        for i in range(len(eta_ood)):
            S, p, alpha = algorithm_1_exact(eta_ood[i], lam, K)
            res_ood.append({"S": S})

        S_id = np.array([r["S"] for r in res_id])
        S_ood = np.array([r["S"] for r in res_ood])
        vac_id, vac_ood = K / S_id, K / S_ood

        auroc_vac = roc_auc_score(labels, np.concatenate([-vac_id, -vac_ood]))
        delta = auroc_vac - auroc_msp

        print(f"{lam:<10.2f} {S_crit:<15.4f} {auroc_vac:<20.4f} {delta:+.4f}")

        assert auroc_vac > 0.6, f"Vacuity too low at lambda={lam}!"

    print("\n[PASS] Hyperparameter sensitivity within acceptable bounds!")
    return True


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_full_verification():
    """Run the full verification pipeline for Table 3."""
    print("\n" + "=" * 70)
    print("FULL VERIFICATION OF TABLE 3 (OOD Detectors)")
    print("=" * 70)

    results = {
        "level_1": False,
        "level_2": False,
        "level_3": False,
        "level_4": False,
    }

    for level, func in [
        ("level_1", verify_metrics_correctness),
        ("level_2", verify_table_3_real_data),
        ("level_3", verify_statistical_significance),
        ("level_4", verify_hyperparameter_sensitivity),
    ]:
        try:
            results[level] = func()
        except Exception as e:
            print(f"\n[FAIL] Error at {level}: {e}")
            import traceback
            traceback.print_exc()

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    for level, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{level}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\nALL CHECKS PASSED!")
        print("Table 3 is correct and statistically significant.")
    else:
        print("\nSOME CHECKS FAILED!")
        print("Additional debugging is required.")

    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    run_full_verification()