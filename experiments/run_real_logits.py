"""
Tables 2 & 3: Real-World Logits
===============================
Verifies the Evidence Ceiling effect and OOD detection metrics on real logits.
Strictly matches Tables 2 and 3 in the LaTeX manuscript.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torchattacks
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.solver import algorithm_1_exact, compute_S_crit
from src.metrics import compute_msp, compute_energy, compute_vacuity, compute_auroc
from src.utils import set_global_seed


def get_loaders(batch_size: int = 128):
    """Load the ID (CIFAR-10) and multiple OOD datasets."""
    # CIFAR-10 and CIFAR-100 images are already 32x32
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # SVHN and DTD have other sizes, so resize to 32x32
    transform_resized = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # In-distribution dataset
    cifar_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_cifar
    )

    # Out-of-distribution datasets
    svhn_test = torchvision.datasets.SVHN(
        root="./data", split="test", download=True, transform=transform_resized
    )
    cifar100_test = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_cifar
    )
    dtd_test = torchvision.datasets.DTD(
        root="./data", split="test", download=True, transform=transform_resized
    )

    loaders = {
        "ID": DataLoader(cifar_test, batch_size=batch_size, shuffle=False),
        "SVHN": DataLoader(svhn_test, batch_size=batch_size, shuffle=False),
        "CIFAR-100": DataLoader(cifar100_test, batch_size=batch_size, shuffle=False),
        "Textures": DataLoader(dtd_test, batch_size=batch_size, shuffle=False),
    }
    return loaders


def generate_adversarial_data(model, loader, n_samples, device, epsilon=8 / 255):
    """Generate PGD adversarial examples using torchattacks."""
    model.eval()
    atk = torchattacks.PGD(model, eps=epsilon, alpha=2 / 255, steps=10)

    logits_list = []
    count = 0
    for x, y in loader:
        if count >= n_samples:
            break
        x, y = x.to(device), y.to(device)

        # Generating adversarial examples requires gradients
        adv_x = atk(x, y)

        # Disable gradients only for the forward pass to obtain logits
        with torch.no_grad():
            adv_logits = model(adv_x).cpu().numpy()

        logits_list.append(adv_logits)
        count += x.size(0)

    logits = np.concatenate(logits_list)[:n_samples]
    return logits


def load_model(K: int = 10, device: str = "cpu") -> nn.Module:
    """Load the pretrained ResNet-18 for CIFAR-10."""
    model = torchvision.models.resnet18(weights=None, num_classes=K)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    path = "cifar10_resnet18.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError(
            f"Pretrained model not found at {path}. Please train or download it."
        )
    return model.to(device).eval()


@torch.no_grad()
def extract_data(model, loader, n_samples, device):
    """Extract logits and softmax probabilities from a data loader."""
    logits_list, labels_list = [], []
    count = 0
    for x, y in loader:
        if count >= n_samples:
            break
        logits = model(x.to(device)).cpu().numpy()
        logits_list.append(logits)
        labels_list.append(y.numpy())
        count += x.size(0)

    logits = np.concatenate(logits_list)[:n_samples]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    eta = exp_l / exp_l.sum(axis=1, keepdims=True)
    return logits, eta


def run_experiment(
    K: int = 10,
    lam: float = 0.1,
    n_samples: int = 2000,
    device: str = "cuda",
    seed: int = 42,
):
    """Run the real-logits experiment."""
    set_global_seed(seed)

    print("\n" + "=" * 70)
    print("Tables 2 & 3: Real-World Logits Verification")
    print("=" * 70)

    loaders = get_loaders()
    model = load_model(K, device)
    S_crit = compute_S_crit(lam, K)
    print(f"[INFO] S_crit = {S_crit:.4f}")

    # Extract data for all static datasets
    data_dict = {}
    for name, loader in loaders.items():
        print(f"[INFO] Extracting data for {name}...")
        logits, eta = extract_data(model, loader, n_samples, device)
        data_dict[name] = {"logits": logits, "eta": eta}

    # ----------------------------------------------------------
    # Table 2: Evidence Ceiling statistics
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLE 2: Statistics of Algorithm 1 solutions")
    print("=" * 70)

    bins = [(0.0, 0.6), (0.6, 0.9), (0.9, 1.001)]
    names = ["[0.0, 0.6)", "[0.6, 0.9)", "[0.9, 1.0]"]

    print(
        f"\n{'Type':<5} {'Bin max(eta)':<13} {'Fraction':<9} "
        f"{'Mean S*':<9} {'Mean p*':<9} {'Frac S*>S_crit':<15}"
    )
    print("-" * 65)

    for dtype in ["ID", "SVHN"]:
        eta = data_dict[dtype]["eta"]
        max_arr = eta.max(axis=1)

        S_arr, p_arr = [], []
        for i in range(len(eta)):
            S, p, _ = algorithm_1_exact(eta[i], lam, K)
            S_arr.append(S)
            p_arr.append(p)
        S_arr, p_arr = np.array(S_arr), np.array(p_arr)

        for (lo, hi), name in zip(bins, names):
            mask = (max_arr >= lo) & (max_arr < hi)
            if mask.sum() == 0:
                continue
            frac = mask.mean() * 100
            mean_S = S_arr[mask].mean()
            mean_p = p_arr[mask].mean()
            frac_violation = (S_arr[mask] > S_crit).mean() * 100
            print(
                f"{dtype:<5} {name:<13} {frac:.0f}%     "
                f"{mean_S:<9.2f} {mean_p:<9.2f} {frac_violation:.1f}%"
            )

    # ----------------------------------------------------------
    # Table 3: OOD detectors (AUROC)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLE 3: OOD Detectors (AUROC)")
    print("=" * 70)

    id_logits = data_dict["ID"]["logits"]
    id_eta = data_dict["ID"]["eta"]

    # Pre-compute ID metrics once
    print("[INFO] Computing ID metrics...")
    id_S = np.array([algorithm_1_exact(eta, lam, K)[0] for eta in id_eta])

    msp_id = compute_msp(id_eta)
    energy_id = compute_energy(id_logits)
    vac_id = compute_vacuity(id_S, K)

    print(f"\n{'OOD Dataset':<20} {'MSP':<9} {'Energy':<9} {'Vacuity':<9} {'Delta vs Best'}")
    print("-" * 60)

    ood_datasets = ["SVHN", "CIFAR-100", "Textures"]

    for ood_name in ood_datasets:
        ood_logits = data_dict[ood_name]["logits"]
        ood_eta = data_dict[ood_name]["eta"]

        print(f"[INFO] Processing {ood_name}...")
        ood_S = np.array([algorithm_1_exact(eta, lam, K)[0] for eta in ood_eta])

        msp_ood = compute_msp(ood_eta)
        energy_ood = compute_energy(ood_logits)
        vac_ood = compute_vacuity(ood_S, K)

        auroc_msp = compute_auroc(msp_id, msp_ood, higher_is_id=True)
        auroc_energy = compute_auroc(energy_id, energy_ood, higher_is_id=True)
        auroc_vac = compute_auroc(vac_id, vac_ood, higher_is_id=False)

        best_baseline = max(auroc_msp, auroc_energy)
        delta = auroc_vac - best_baseline

        print(
            f"{ood_name:<20} {auroc_msp:<9.3f} {auroc_energy:<9.3f} "
            f"{auroc_vac:<9.3f} {delta:+.3f}"
        )

    # ----------------------------------------------------------
    # Adversarial examples (PGD)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Adversarial Examples (PGD) Verification")
    print("=" * 70)

    print("[INFO] Generating Adversarial Examples (PGD)...")
    set_global_seed(seed)

    id_loader_for_adv = loaders["ID"]
    logits_adv = generate_adversarial_data(model, id_loader_for_adv, n_samples, device)

    shifted_adv = logits_adv - logits_adv.max(axis=1, keepdims=True)
    exp_l_adv = np.exp(shifted_adv)
    eta_adv = exp_l_adv / exp_l_adv.sum(axis=1, keepdims=True)

    print("[INFO] Running Algorithm 1 on Adversarial data...")
    S_adv = np.array([algorithm_1_exact(eta, lam, K)[0] for eta in eta_adv])
    max_adv = eta_adv.max(axis=1)
    vac_adv = compute_vacuity(S_adv, K)
    energy_adv_ood = compute_energy(logits_adv)

    auroc_msp_adv = compute_auroc(msp_id, max_adv, higher_is_id=True)
    auroc_energy_adv = compute_auroc(energy_id, energy_adv_ood, higher_is_id=True)
    auroc_vac_adv = compute_auroc(vac_id, vac_adv, higher_is_id=False)

    best_auroc_adv = max(auroc_msp_adv, auroc_energy_adv, auroc_vac_adv)
    delta_adv = auroc_vac_adv - best_auroc_adv

    print(f"\n{'OOD Dataset':<20} {'MSP':<9} {'Energy':<9} {'Vacuity':<9} {'Delta vs Best'}")
    print("-" * 60)
    print(
        f"{'Adversarial (PGD)':<20} {auroc_msp_adv:<9.3f} {auroc_energy_adv:<9.3f} "
        f"{auroc_vac_adv:<9.3f} {delta_adv:+.3f}"
    )

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment(device=device)