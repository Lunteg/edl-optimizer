# EDL Optimizer: Certified Face-Enumeration for Regularized Dirichlet Likelihood

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22075827.svg)](https://doi.org/10.5281/zenodo.22075827)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:

> **"Geometry of Regularized Dirichlet Likelihood: Structural Theorem, SOSC Characterization, and Certified Face-Enumeration Algorithm"**
> Danila Ozerov, Alexey Bogomolov, Vadim  Kushnikov
> (submitted to *Machine Learning*, 2026)

## What is this?

A certified solver for the box-constrained optimization problem arising in Evidential Deep Learning (EDL):

$$
\min_{\alpha} \; L(\alpha) \;=\; \underbrace{\ln S - \sum_{k=1}^{K} \eta_k \ln \alpha_k}_{\text{NLL}(\alpha;\,\eta)} \;+\; \lambda \cdot \mathrm{KL}\big(\mathrm{Dir}(\alpha) \,\|\, \mathrm{Dir}(\mathbf{1})\big)
$$

subject to the box constraints

$$
\alpha_k \ge 1, \qquad S = \sum_{k=1}^{K} \alpha_k,
$$

where $\eta$ is the probability vector (soft labels from the backbone classifier) and $\lambda > 0$ is the regularization coefficient.

Unlike standard solvers (L-BFGS-B, PGD), which achieve only ~64–67% accuracy in active-set identification due to ill-conditioned geometry near the box constraints, our **Hybrid Compact-Core Newton** algorithm guarantees 100% accuracy via exhaustive face enumeration with certified SOSC checks. Remarkably, it is also **faster** than the highly optimized compiled L-BFGS-B for large $K$ (up to $2.1\times$ speedup at $K=500$).

## Installation

Requires Python ≥ 3.9. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.solver import algorithm_1_exact
import numpy as np

# Example: η = [0.7, 0.2, 0.05, 0.03, 0.02]
eta = np.array([0.7, 0.2, 0.05, 0.03, 0.02])
S, p, alpha = algorithm_1_exact(eta, lam=0.1, K=5)

print(f"Optimal S* = {S:.2f}")
print(f"Active set size p* = {p}")
print(f"Alpha = {alpha}")
print(f"Vacuity = {5 / S:.4f}")
```

The solver returns the optimal total evidence $S^*$, the active-set size $p^*$, and the Dirichlet parameters $\alpha^*$ that minimize $L(\alpha)$.

## Reproducing the Paper's Tables

### Table 1: Algorithm Validation (Synthetic Data)

```bash
python experiments/run_synthetic.py
```

Compares Algorithm 1 vs L-BFGS-B vs PGD on 1000 random configurations ($K=10$, $\lambda=0.1$).

**Expected output:**

```text
Method          Active Set Acc.      Mean ||alpha-alpha*||  Boundary dev.
----------------------------------------------------------------------
Alg. 1 (ours)   100.0 +/- 0.0     0.00e+00            Analytical
L-BFGS-B        63.8 +/- 10.5     8.05e-01            1.35e-01
PGD             67.4 +/- 13.1     8.21e-01            1.14e-01
```

### Tables 2 & 3: Real-World Logits (Evidence Ceiling)

```bash
python experiments/run_real_logits.py
```

Downloads CIFAR-10 (ID), SVHN, CIFAR-100, Describable Textures (OOD); loads a pretrained ResNet-18 checkpoint; generates adversarial examples via a PGD attack.

**Requirements:** The pretrained checkpoint `cifar10_resnet18.pth` must be present in the repository root. If missing, download it from the Zenodo archive (DOI in the paper).

**This will:**

1. Download OOD datasets (~1 GB total) on first run
2. Load the fixed pretrained ResNet-18 weights
3. Extract logits and run Algorithm 1 on 2000 samples per dataset
4. Generate Table 2 (Evidence Ceiling statistics) and Table 3 (OOD AUROC)

**Expected output (Table 2):**

```text
Type  Bin max(eta)  Fraction  Mean S*   Mean p*   Frac S*>S_crit
-----------------------------------------------------------------
ID    [0.0, 0.6)    4%        16.29     2.08      0.0%
ID    [0.6, 0.9)    9%        17.19     1.55      0.0%
ID    [0.9, 1.0]    87%       19.14     1.00      0.0%
SVHN  [0.0, 0.6)    18%       16.57     2.15      0.0%
SVHN  [0.6, 0.9)    41%       17.10     1.59      0.0%
SVHN  [0.9, 1.0]    42%       18.48     1.00      0.0%
```

Across all bins, $S^*$ never crosses the theoretical threshold $S_{\text{crit}} \approx 19.75$ (0% violation rate), while the active-set size sparsifies as $p^* \to 1$.

**Expected output (Table 3):**

```text
OOD Dataset          MSP       Energy    Vacuity   Delta vs Best
------------------------------------------------------------
SVHN                 0.880     0.880     0.878     -0.001
CIFAR-100            0.865     0.865     0.871     +0.006
Textures             0.882     0.882     0.888     +0.007
Adversarial (PGD)    0.710     0.710     0.713     +0.000
```

### Table 4: Scalability Benchmark

```bash
python experiments/run_scalability.py
```

Wall-clock time comparison (median of 30 runs) for $K \in \{10, 50, 100, 500\}$.

**Expected output:**

```text
K      Alg.1 (ms)     L-BFGS-B (ms)    Speedup   L-BFGS-B Acc.
----------------------------------------------------------------------
10         4.53          5.79          1.3x       65.3%
50        18.91         19.72          1.0x       70.5%
100       36.88         40.40          1.1x       74.0%
500      183.05        372.39          2.0x       86.0%
```

### Statistical Verification (Bootstrap + Sensitivity)

```bash
python experiments/verify_table_3.py
```

Multi-level verification of Table 3 results:

- **Level 1:** Metric correctness on synthetic data with known ground truth
- **Level 2:** Validation on real data (CIFAR-10 ID vs SVHN OOD)
- **Level 3:** Bootstrap confidence intervals (1000 iterations, 95% CI)
- **Level 4:** Hyperparameter sensitivity ($\lambda \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$)

**Expected output:**

```text
level_1: [PASS]
level_2: [PASS]
level_3: [PASS]
level_4: [PASS]

ALL CHECKS PASSED!
Table 3 is correct and statistically significant.
```

### Symbolic Verification of $E \equiv 0$

```bash
python experiments/verify_E_zero.py
```

Symbolically verifies the algebraic cancellation of the coefficient $E$ at $1/S$ in the asymptotic expansion of $g_p'(S)$ (Lemma: Asymptotic Strict Monotonicity), confirming the $O(1/S^2)$ convergence rate.

## Repository Structure

```text
edl-optimizer/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation metadata
├── requirements.txt             # Pinned Python dependencies
├── cifar10_resnet18.pth         # Pretrained ResNet-18 checkpoint
├── src/
│   ├── __init__.py
│   ├── core.py                  # Mathematical core (theta, h, g_p, KL)
│   ├── solver.py                # Algorithm 1: Hybrid Compact-Core Newton
│   ├── metrics.py               # OOD metrics (MSP, Energy, Vacuity, AUROC)
│   └── utils.py                 # Reproducibility utilities (seed fixing)
├── experiments/
│   ├── run_synthetic.py         # Table 1: Algorithm validation
│   ├── run_real_logits.py       # Tables 2 & 3: Evidence Ceiling + OOD
│   ├── run_scalability.py       # Table 4: Scalability benchmark
│   ├── verify_table_3.py        # Multi-level statistical verification
│   └── verify_E_zero.py         # SymPy symbolic check (E = 0)
└── data/                        # Created automatically (datasets)
```

## Key Mathematical Functions

All functions below use the **exact polygamma formulation** (no logarithmic surrogates). Here $\psi(x)$ denotes the digamma function and $\psi^{(n)}(x)$ the polygamma functions.

### Threshold Function $\theta(S)$

$$
\theta(S) = \frac{1}{S} - \lambda\,(S-K)\,\psi'(S)
$$

Strictly decreasing for $S > K$; satisfies $\theta(S) \to -\lambda$ as $S \to \infty$. The critical threshold $S_{\text{crit}}$ is the unique root of $\theta(S) = 0$.

```python
from src.core import theta_exact
theta = theta_exact(S=15.0, lam=0.1, K=10)
```

### KKT Function $h(\alpha;\,\eta)$

$$
h(\alpha;\,\eta) = \frac{\eta}{\alpha} - \lambda\,(\alpha-1)\,\psi'(\alpha)
$$

Strictly decreasing in $\alpha$ for any $\eta > 0$, with boundary value $h(1;\,\eta) = \eta$. The KKT conditions read $h(\alpha_j;\eta_j) = \theta(S)$ for active classes ($\alpha_j > 1$) and $\eta_j \le \theta(S)$ for inactive ones.

```python
from src.core import h_exact
h_val = h_exact(alpha=2.5, eta=0.7, lam=0.1)
```

### Self-Consistency Function $g_p(S)$

$$
g_p(S) = S - (K-p) - \sum_{i \in \mathcal{A}} \alpha_{(i)}\big(\theta(S)\big),
$$

where $\alpha_{(i)}(\theta)$ is the unique root of $h(\alpha;\eta_{(i)}) = \theta$ on $[1,\infty)$ and $\mathcal{A}$ is the active set of size $p$. The optimal $S^*$ on a given face satisfies $g_p(S^*) = 0$.

```python
from src.core import compute_g_p
import numpy as np
eta_sorted = np.array([0.7, 0.2, 0.05, 0.03, 0.02])
g_val = compute_g_p(S=10.0, eta_sorted=eta_sorted, p=3, lam=0.1, K=5)
```

### Derivatives $\theta'(S)$ and $h'_\alpha$

$$
\theta'(S) = -\frac{1}{S^2} - \lambda\big[\psi'(S) + (S-K)\,\psi''(S)\big]
$$

$$
h'_\alpha(\alpha;\,\eta) = -\frac{\eta}{\alpha^2} - \lambda\big[\psi'(\alpha) + (\alpha-1)\,\psi''(\alpha)\big] < 0
$$

These drive the reduced Hessian $H_{\text{act}} = D + \theta'(S)\,\mathbf{1}\mathbf{1}^T$ and the SOSC check $g_p'(S^*) > 0$.

```python
from src.core import theta_prime, h_prime
dtheta = theta_prime(S=15.0, lam=0.1, K=10)
dh = h_prime(alpha=2.5, eta=0.7, lam=0.1)
```

### KL Divergence $\mathrm{KL}(\mathrm{Dir}(\alpha) \| \mathrm{Dir}(\mathbf{1}))$

$$
\mathrm{KL}\big(\mathrm{Dir}(\alpha) \,\|\, \mathrm{Dir}(\mathbf{1})\big) = \ln\Gamma(S) - \sum_{k=1}^{K} \ln\Gamma(\alpha_k) - \ln\Gamma(K) + \sum_{k=1}^{K} (\alpha_k - 1)\big(\psi(\alpha_k) - \psi(S)\big)
$$

```python
from src.core import kl_dirichlet
kl = kl_dirichlet(alpha)
```

### Vacuity (Epistemic Uncertainty)

$$
u(\eta) = \frac{K}{S^*(\eta)}
$$

Higher vacuity indicates greater epistemic uncertainty (more likely OOD).

```python
from src.metrics import compute_vacuity
vac = compute_vacuity(S, K)
```

## Reproducibility

All experiments are fully reproducible with a fixed random seed (`seed=42`) across Python, NumPy, and PyTorch, with deterministic cuDNN operations enabled (`torch.backends.cudnn.deterministic=True`). The ResNet-18 backbone weights are fixed and provided as a checkpoint; no on-the-fly training is performed during evaluation. Adversarial examples are generated with a deterministic PGD start. All data loaders use deterministic ordering (`shuffle=False`).

Wall-clock timings may vary slightly across hardware runs; accuracy and speedup trends are stable.

### Reproducing the Paper's Figures

```bash
python experiments/generate_figures.py
```
Generates publication-quality vector graphics (PDF) for Figures 1, 2, and 3.

- Fig. 1: Self-consistency functions $g_p(S)$ illustrating the thresholding structure.
- Fig. 2: Evidence Ceiling effect on 5000 synthetic logits.
- Fig. 3: Wall-clock time scalability benchmark.

```bash
python experiments/analyze_figures.py
```

Prints detailed numerical data, KKT admissibility checks for Fig. 1 roots, and exact median timings used in Table 4.




## Citation

If you use this code, please cite:

```bibtex
@article{Ozerov2026EDLGeometry,
  title   = {Geometry of Regularized Dirichlet Likelihood:
             Structural Theorem, {SOSC} Characterization,
             and Certified Face-Enumeration Algorithm},
  author  = {Ozerov, Danila and Bogomolov, Alexey and Kushnikov, Vadim },
  journal = {Machine Learning},
  year    = {2026},
  note    = {to appear}
}

@dataset{Ozerov2026EDLData,
  author       = {Ozerov, Danila and Bogomolov, Aleksey and Kushnikov, Vadim},
  title        = {{Supporting Data and Code for: Geometry of Regularized Dirichlet Likelihood}},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22075826},
  url          = {https://doi.org/10.5281/zenodo.22075826}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Questions? Open an issue or email ozerov-danil@yandex.ru.

## Troubleshooting

### Pretrained weights not found

If you see `Reproducibility Error: Pretrained weights 'cifar10_resnet18.pth' not found`, download the checkpoint from the Zenodo archive linked in the paper and place it in the repository root.

### CUDA out of memory

Reduce batch size in `run_real_logits.py`:

```python
BATCH_SIZE = 64  # instead of 128
```

### Import errors

Make sure you run scripts from the repository root:

```bash
cd edl-optimizer
python experiments/run_synthetic.py
```