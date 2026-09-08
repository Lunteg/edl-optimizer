"""
Analyze and print numerical data used in figures.
Run: python experiments/analyze_figures.py
"""
import numpy as np
import pandas as pd
import os
import sys
import time
from scipy.optimize import brentq, minimize
from scipy.special import polygamma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.solver import algorithm_1_exact, compute_S_crit
from src.core import compute_g_p, theta_exact, compute_loss
from src.utils import set_global_seed

os.makedirs('figures/data', exist_ok=True)
set_global_seed(42)

SEP = "=" * 70

def analyze_fig1_landscape():
    K, lam = 10, 0.1
    eta = np.array([0.65, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03])
    eta_sorted = np.sort(eta)[::-1]
    S_crit = compute_S_crit(lam, K)

    print(f"\n{SEP}\nFIGURE 1 ANALYSIS\n{SEP}")
    
    p_values = [1, 2, 3, 5, 10]
    S_scan = np.linspace(K + 0.01, 25, 2000)
    
    print(f"\nRoots of g_p(S) = 0 (KKT stationary points):")
    for p in p_values:
        g_scan = np.array([compute_g_p(s, eta_sorted, p, lam, K) for s in S_scan])
        g_scan = np.where(g_scan > 1e6, np.nan, g_scan)
        
        roots_found = False
        for i in range(len(g_scan)-1):
            if np.isnan(g_scan[i]) or np.isnan(g_scan[i+1]): continue
            if g_scan[i] <= 0 and g_scan[i+1] > 0:
                try:
                    root = brentq(lambda s: compute_g_p(s, eta_sorted, p, lam, K), 
                                  S_scan[i], S_scan[i+1], xtol=1e-12, rtol=1e-12)
                    print(f"  p={p:<2}: S* = {root:.4f}  (θ(S*) = {theta_exact(root, lam, K):+.4f}) [ADMISIBLE if η_{{p+1}} < θ*]")
                    roots_found = True
                except ValueError: pass
        if not roots_found:
            print(f"  p={p:<2}: no root in admissible interval")

def analyze_fig2_evidence_ceiling():
    K, lam = 10, 0.1
    S_crit = compute_S_crit(lam, K)
    n = 5000

    print(f"\n{SEP}\nFIGURE 2 ANALYSIS\n{SEP}")
    
    try:
        df = pd.read_csv('figures/data/fig2_raw.csv')
        max_etas = df['max_eta'].values
        S_stars = df['S_star'].values
        p_stars = df['p_star'].values
        print(f"Loaded existing data from fig2_raw.csv (N={len(df)})")
    except FileNotFoundError:
        print(f"Data not found. Generating {n} samples...")
        max_etas, S_stars, p_stars = [], [], []
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
        max_etas, S_stars, p_stars = np.array(max_etas), np.array(S_stars), np.array(p_stars)
        pd.DataFrame({'max_eta': max_etas, 'S_star': S_stars, 'p_star': p_stars}).to_csv('figures/data/fig2_raw.csv', index=False)

    print(f"S* > S_crit count: {(S_stars > S_crit).sum()} / {n}  ({(S_stars > S_crit).mean()*100:.2f}% violation)")
    print(f"Max observed S* = {S_stars.max():.4f}  (gap to S_crit: {S_crit - S_stars.max():.4f})")

def analyze_fig3_scalability():
    K_values = [10, 25, 50, 75, 100, 200, 300, 500]
    lam = 0.1
    n_trials = 30

    print(f"\n{SEP}\nFIGURE 3 & TABLE 4 ANALYSIS\n{SEP}")

    def lbfgsb_solve(eta, lam, K):
        def obj(a):
            a = np.maximum(a, 1.0 + 1e-12)
            return compute_loss(a, eta, lam)
        res = minimize(obj, np.ones(K)*1.1, method='L-BFGS-B',
                       bounds=[(1.0, None)]*K,
                       options={'maxiter': 500, 'ftol': 1e-12})
        return res.x

    results = []
    for K in K_values:
        print(f"Benchmarking K={K} ({n_trials} trials)... This may take a moment for large K.")
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

        t1_med = np.median(t1_list)
        t2_med = np.median(t2_list)
        speedup = t2_med / max(t1_med, 1e-6)

        results.append({
            'K': K,
            'Alg1_ms': f"{t1_med:.2f}",
            'LBFGSB_ms': f"{t2_med:.2f}",
            'Speedup': f"{speedup:.1f}x",
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    df.to_csv('figures/data/fig3_scalability.csv', index=False)
    
    print(f"\n{SEP}")
    print(SEP)
    for _, row in df.iterrows():
        print(f"K = {row['K']:<3} | Alg 1: {row['Alg1_ms']:<6} ms | L-BFGS-B: {row['LBFGSB_ms']:<6} ms | Speedup: {row['Speedup']}")
    print(SEP)

if __name__ == "__main__":
    print(SEP)
    print("NUMERICAL DATA ANALYSIS")
    print(SEP)
    analyze_fig1_landscape()
    analyze_fig2_evidence_ceiling()
    analyze_fig3_scalability()
    print("\nANALYSIS COMPLETE")