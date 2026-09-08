"""
Generate publication-quality figures for the paper.
Run: python experiments/generate_figures.py
Output: figures/Fig1.pdf, figures/Fig2.pdf, figures/Fig3.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import time
from scipy.optimize import brentq, minimize
from scipy.special import polygamma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.solver import algorithm_1_exact, compute_S_crit
from src.core import compute_g_p, theta_exact, compute_loss
from src.utils import set_global_seed

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
})

os.makedirs('figures', exist_ok=True)
set_global_seed(42)

def fig1_landscape():
    K, lam = 10, 0.1
    eta = np.array([0.65, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03])
    eta_sorted = np.sort(eta)[::-1]
    S_crit = compute_S_crit(lam, K)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    S_range = np.linspace(K + 0.01, 25, 500)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, K))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))] 
    
    p_to_plot = [1, 3, 5, 10]
    for p in range(1, K + 1):
        g_vals = [compute_g_p(s, eta_sorted, p, lam, K) for s in S_range]
        g_vals = np.array([v if v < 1e8 and np.isfinite(v) else np.nan for v in g_vals])
        
        valid = ~np.isnan(g_vals)
        if valid.sum() > 10:
            ls = linestyles[p % len(linestyles)]
            label = f'$p={p}$' if p in p_to_plot else ''
            ax.plot(S_range[valid], g_vals[valid], color=colors[p-1], 
                   linestyle=ls, linewidth=1.5, label=label)

    ax.axvline(x=S_crit, color='red', linestyle=':', linewidth=2.0,
               label=f'$S_{{\\mathrm{{crit}}}} \\approx {S_crit:.2f}$')

    # Mark roots robustly
    for p in p_to_plot:
        S_scan = np.linspace(K - p + 1e-3, 25, 1000)
        g_scan = np.array([compute_g_p(s, eta_sorted, p, lam, K) for s in S_scan])
        g_scan = np.where(g_scan > 1e6, np.nan, g_scan)
        
        for i in range(len(g_scan)-1):
            if np.isnan(g_scan[i]) or np.isnan(g_scan[i+1]): continue
            if g_scan[i] <= 0 and g_scan[i+1] > 0:
                try:
                    root = brentq(lambda s: compute_g_p(s, eta_sorted, p, lam, K), 
                                  S_scan[i], S_scan[i+1], xtol=1e-12, rtol=1e-12)
                    ax.plot(root, 0, 'o', color=colors[p-1], markersize=8, 
                            markeredgecolor='black', markeredgewidth=1.0, zorder=5)
                except ValueError: pass

    ax.axhline(y=0, color='black', linewidth=1.0)
    ax.set_xlabel('$S$ (total evidence)')
    ax.set_ylabel('$g_p(S)$')
    ax.set_xlim(K, 25)
    ax.set_ylim(-3, 3)
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.5, linestyle=':', linewidth=0.5)
    fig.savefig('figures/Fig1.pdf')
    plt.close(fig)
    print("[OK] Fig 1: Fig1.pdf")

def fig2_evidence_ceiling():
    K, lam = 10, 0.1
    S_crit = compute_S_crit(lam, K)
    n = 5000

    max_etas, S_stars, p_stars = [], [], []
    print(f"Generating Fig 2 data ({n} samples)...")
    for i in range(n):
        if i % 1000 == 0 and i > 0: print(f"  Processed {i}/{n}...")
        dominant = np.random.beta(8, 2)
        rest = np.random.dirichlet(np.ones(K-1) * 0.5) * (1 - dominant)
        eta = np.concatenate([[dominant], rest])
        np.random.shuffle(eta)
        S, p, _ = algorithm_1_exact(eta, lam, K)
        max_etas.append(eta.max())
        S_stars.append(S)
        p_stars.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}
    
    for p_val in sorted(markers.keys()):
        mask = np.array(p_stars) == p_val
        if np.any(mask):
            ax.scatter(np.array(max_etas)[mask], np.array(S_stars)[mask], 
                       c=colors[p_val], marker=markers[p_val], s=30, alpha=0.8, 
                       edgecolors='none', label=f'$p^*={p_val}$', zorder=3)

    ax.axhline(y=S_crit, color='red', linestyle='--', linewidth=2.0,
               label=f'$S_{{\\mathrm{{crit}}}} \\approx {S_crit:.2f}$')
    ax.set_xlabel('$\\max_k \\eta_k$ (backbone confidence)')
    ax.set_ylabel('$S^*$ (total evidence)')
    ax.legend(loc='lower right', framealpha=0.9, title='Active set size $p^*$')
    ax.grid(True, alpha=0.5, linestyle=':', linewidth=0.5)
    fig.savefig('figures/Fig2.pdf')
    plt.close(fig)
    print("[OK] Fig 2: Fig2.pdf")

def fig3_scalability():
    K_values = [10, 25, 50, 75, 100, 200, 300, 500]
    lam = 0.1
    n_trials = 30

    alg_times, lbfgs_times = [], []

    def lbfgsb_solve(eta, lam, K):
        def obj(a):
            a = np.maximum(a, 1.0 + 1e-12)
            return compute_loss(a, eta, lam)
        res = minimize(obj, np.ones(K)*1.1, method='L-BFGS-B',
                       bounds=[(1.0, None)]*K, 
                       options={'maxiter': 500, 'ftol': 1e-12})
        return res.x

    eta_warm = np.random.dirichlet(np.ones(10))
    algorithm_1_exact(eta_warm, lam, 10)
    lbfgsb_solve(eta_warm, lam, 10)

    for K in K_values:
        print(f"Benchmarking K={K} ({n_trials} trials)...")
        etas = []
        for _ in range(n_trials):
            d = np.random.uniform(0.8, 0.95)
            r = np.random.dirichlet(np.ones(K-1)) * (1-d)
            etas.append(np.concatenate([[d], r]))

        t1_list, t2_list = [], []
        for eta in etas:
            t0 = time.perf_counter()
            algorithm_1_exact(eta, lam, K)
            t1_list.append((time.perf_counter()-t0)*1000)

            t0 = time.perf_counter()
            lbfgsb_solve(eta, lam, K)
            t2_list.append((time.perf_counter()-t0)*1000)

        alg_times.append(np.median(t1_list))
        lbfgs_times.append(np.median(t2_list))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(K_values, alg_times, 'o-', color='#2196F3', linewidth=2.0,
            markersize=8, label='Algorithm 1 (ours)')
    ax.plot(K_values, lbfgs_times, 's--', color='#FF5722', linewidth=2.0,
            markersize=8, label='L-BFGS-B (SciPy)')
    ax.set_xlabel('Number of classes $K$')
    ax.set_ylabel('Median time (ms)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(framealpha=0.9, loc='upper left')
    ax.grid(True, alpha=0.5, linestyle=':', linewidth=0.5, which='both')
    fig.savefig('figures/Fig3.pdf')
    plt.close(fig)
    print("[OK] Fig 3: Fig3.pdf")

if __name__ == "__main__":
    print("Generating figures...")
    fig1_landscape()
    fig2_evidence_ceiling()
    fig3_scalability()
    print("\n All figures saved to figures/")