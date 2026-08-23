"""
Symbolic verification that the coefficient E in the asymptotic expansion
of the derivative g_p'(S) is identically zero.

This confirms the Lemma on Asymptotic Strict Monotonicity:
the coefficient at 1/S vanishes for any eta, lambda, K, so g_p'(S)
converges to its asymptote at a rate of O(1/S^2).

Run: python experiments/verify_E_zero.py
"""

import sympy as sp


def main():
    print("Symbolic verification: coefficient E is identically zero")
    print("in the asymptotic expansion of the derivative of g_p(S)")
    print("=" * 70)

    # Symbolic variables
    lam, K = sp.symbols("lam K", positive=True)
    eta_i = sp.symbols("eta_i", positive=True)

    # Constant Lambda = 1 + lambda * (K - 1/2)
    Lambda = 1 + lam * (K - sp.Rational(1, 2))

    # Coefficient c_i = (eta_i + lambda/2) / Lambda
    c_i = (eta_i + lam / 2) / Lambda

    # Coefficient d_i from Step 1 of the proof
    d_i = lam / (3 * c_i * Lambda) - lam * (3 * K - 1) * c_i / (6 * Lambda)

    # Coefficient delta_i from Step 2 of the proof
    delta_i = 2 * lam / (3 * c_i ** 2 * Lambda) - 2 * d_i / c_i

    # Product c_i * delta_i
    ci_di = sp.simplify(c_i * delta_i)
    print(f"c_i * delta_i = {ci_di}")

    # Per-term coefficient of E:
    # E = sum(c_i * delta_i) - [lambda*(K - 1/3)/Lambda] * sum(c_i)
    E_coeff = sp.simplify(
        ci_di - lam * (K - sp.Rational(1, 3)) * c_i / Lambda
    )
    print(f"\nCoefficient E per term = {sp.simplify(E_coeff)}")

    # Check whether E is identically zero
    if sp.simplify(E_coeff) == 0:
        print("\n[OK] VERIFIED: E is identically zero for all eta, lambda, K.")
        print("     This confirms the O(1/S^2) convergence rate of g_p'(S).")
    else:
        print(f"\n[FAIL] E is not zero. Got {E_coeff}")


if __name__ == "__main__":
    main()