#!/usr/bin/env python3
"""Randomized & adversarial counterexample verification suite for LogGeo-HyperBC (Theorems 1-3).
Checks Theorem 1 (Barrier finiteness, anti-undercoverage, zero-atom slack),
Theorem 2 (Exact s_eff pinball cancellation & IQR breakdown),
and Theorem 3 (Shifted log-geometric AM-GM inequality, eps-scale equivariance, & post-drop tail bound)
across hundreds of randomized and boundary trials (0 violations allowed).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loggeo_hyperbc as rc


def _solve_barrier_u(target: float, sigma_min: float = 1e-4) -> float:
    if target <= 0.0:
        return 0.0
    lo, hi = 0.0, max(35.0, ((target / (sigma_min * math.cosh(30.0))) ** (1.0 / 3.0)) * 1.05)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = sigma_min * (mid ** 3) * math.cosh(min(mid, 30.0))
        if val >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _suffix_zero_excess(s: np.ndarray, alpha: float) -> float:
    T = int(s.size)
    if T == 0:
        return 0.0
    increments = (s == 0.0).astype(np.float64) - (1.0 - alpha)
    rev_cum = np.cumsum(increments[::-1]) / float(T)
    return float(max(0.0, float(np.max(rev_cum))))


def verify_theorem1_barrier_and_zero_atom(seed: int = 20261119) -> None:
    rng = np.random.default_rng(seed)

    # Direct exact witness confirming suffix-zero excess Delta_0(s) is sharp and non-vacuous:
    # alpha=0.50, c0=0.01, sigma_min=1.0, q_cal=1.0, T=100, s[:50]=0.5, s[50:]=0.0 -> p0_hat = 0.50 = 1 - alpha
    T_w, alpha_w, c0_w = 100, 0.50, 0.01
    s_w = np.concatenate([np.full(50, 0.5), np.zeros(50)])
    E_w = 0.0
    errs_w = []
    for t_idx in range(1, T_w + 1):
        h_w = max(c0_w * ((t_idx - 1) ** (2.0 / 3.0)), 1.0)
        u_w = E_w / h_w
        psi_w = (u_w ** 3) * math.cosh(min(abs(u_w), 30.0))
        q_w = max(0.0, 1.0 + psi_w)
        e_w = 1.0 if s_w[t_idx - 1] > q_w else 0.0
        errs_w.append(e_w)
        E_w += e_w - alpha_w
    cov_w = 1.0 - float(np.mean(errs_w))
    slack_w = (max(c0_w * ((T_w - 1) ** (2.0 / 3.0)), 1.0) * max(_solve_barrier_u(0.5, 1.0), _solve_barrier_u(1.0, 1.0)) + 1.0) / T_w
    delta0_w = _suffix_zero_excess(s_w, alpha_w)
    assert slack_w < 0.025 and abs(cov_w - 0.50) > 0.20, "Suffix-zero witness must exceed naive zero-atom bound"
    assert abs(cov_w - (1.0 - alpha_w)) <= slack_w + delta0_w + 1e-9

    # Randomized inline-barrier trials (sigma_min = 1, c0 = 0.01, q_cal = 1): the slack is below alpha,
    # so these exercise the over-coverage side of Theorem 1(iii) non-vacuously.
    n_nonvac = 0
    for _ in range(200):
        a_r = float(rng.choice([0.3, 0.5]))
        T_r = int(rng.integers(100, 401))
        n_pos = int(rng.integers(1, T_r))
        s_r = np.zeros(T_r, dtype=np.float64)
        s_r[:n_pos] = rng.uniform(1e-3, 0.9, size=n_pos)
        E_r, errs_r = 0.0, []
        for t_idx in range(1, T_r + 1):
            u_r = E_r / max(0.01 * ((t_idx - 1) ** (2.0 / 3.0)), 1.0)
            q_r = max(0.0, 1.0 + (u_r ** 3) * math.cosh(min(abs(u_r), 30.0)))
            e_r = 1.0 if s_r[t_idx - 1] > q_r else 0.0
            errs_r.append(e_r)
            E_r += e_r - a_r
        cov_r = 1.0 - float(np.mean(errs_r))
        B_r = float(np.max(s_r))
        slack_r = (max(0.01 * ((T_r - 1) ** (2.0 / 3.0)), 1.0) * max(_solve_barrier_u(B_r, 1.0), _solve_barrier_u(1.0, 1.0)) + 1.0) / T_r
        d0_r = _suffix_zero_excess(s_r, a_r)
        assert slack_r < a_r, f"inline trial vacuous: slack={slack_r}"
        assert abs(cov_r - (1.0 - a_r)) <= slack_r + d0_r + 1e-9, (
            f"Theorem 1(iii) violated (inline): err={abs(cov_r - (1.0 - a_r))}, rhs={slack_r + d0_r}"
        )
        n_nonvac += int(abs(cov_r - (1.0 - a_r)) > slack_r)
    assert n_nonvac > 0, "no inline trial needed the Delta_0 term"

    # Check Theorem 1(ii) two-regime analytical bound on u^+(B_T)
    clamp_thresh = (30.0 ** 3) * math.cosh(30.0)
    for ratio in [0.01, 1.0, 1e3, 1e8, 1e15, clamp_thresh * 0.99, clamp_thresh * 10.0, 1e22]:
        u_exact = _solve_barrier_u(ratio * 1e-4, sigma_min=1e-4)
        if ratio <= clamp_thresh:
            assert u_exact <= math.log(2.0 + 2.0 * ratio) + 1e-9
        else:
            u_clamp = (ratio / math.cosh(30.0)) ** (1.0 / 3.0)
            assert abs(u_exact - u_clamp) / max(u_clamp, 1e-12) < 1e-6

    # Adversarial bounded, heavy-tailed, and suffix-zero streams: 0% +infty, 0% empty, and exact bounds
    for trial in range(150):
        T = int(rng.integers(550, 1250))
        alpha = float(rng.choice([0.05, 0.10, 0.20]))
        if trial == 0:
            s = np.zeros(T, dtype=np.float64)
        elif trial == 1:
            # 95% i.i.d. zero-atom mass + 5% spikes
            s = np.where(rng.uniform(size=T) < 0.95, 0.0, rng.uniform(5.0, 20.0, size=T))
        elif trial == 2:
            # Deterministic suffix-zero adversarial witness (p0_hat = 0.80 <= 1 - alpha = 0.90 at alpha = 0.10)
            alpha = 0.10
            T = 1500
            s = np.zeros(T, dtype=np.float64)
            s[: T // 5] = 1.5
        elif trial == 3:
            # Alternating zero-blocks and positive bursts
            s = np.zeros(T, dtype=np.float64)
            s[: T // 3] = rng.uniform(0.5, 3.0, size=T // 3)
            s[(2 * T) // 3 :] = 0.0
        elif trial % 3 == 0:
            s = np.minimum(np.abs(rng.standard_cauchy(size=T)), 500.0)
        else:
            s = np.abs(rng.normal(loc=1.5, scale=1.0, size=T))
            s[T // 2 :] *= 4.0

        out = rc.run_loggeo_hyperbc(s, alpha=alpha)
        q = out["q"]
        err = out["err"]
        assert np.all(np.isfinite(q)), f"Theorem 1(i) violated: non-finite q_t on trial {trial}"
        assert np.all(q >= 0.0), f"Theorem 1(i) violated: negative q_t on trial {trial}"
        assert not np.any(out["inf"]) and not np.any(out["empty"])

        B_T = float(np.max(s))
        # Upper bound on q_cal,t <= exp(0.60) * max(1.0, B_T) * 4.0
        Q_max_T = float(math.exp(0.60) * max(1.0, B_T) * 4.0)
        h_T = max(0.95 * ((T - 1) ** (2.0 / 3.0)), 1.0)
        p0_hat = float(np.mean(s == 0.0))
        delta0_T = _suffix_zero_excess(s, alpha)
        assert max(0.0, p0_hat - (1.0 - alpha)) - 1e-12 <= delta0_T <= alpha * p0_hat + 1e-12

        cov_T = float(np.mean(1.0 - err))
        u_plus = _solve_barrier_u(B_T, sigma_min=1e-4)
        u_minus = _solve_barrier_u(Q_max_T, sigma_min=1e-4)
        lower_slack_T = (h_T * u_plus + (1.0 - alpha)) / float(T)
        twosided_slack_T = (h_T * max(u_plus, u_minus) + 1.0) / float(T)
        # Full-algorithm trials exercise (i)-(ii); their two-sided slack is often above alpha, so
        # (iii) is exercised non-vacuously by the inline witnesses above, not here.
        assert cov_T >= (1.0 - alpha) - lower_slack_T - 1e-9, f"Theorem 1(ii) violated: {cov_T}"
        assert abs(cov_T - (1.0 - alpha)) <= twosided_slack_T + delta0_T + 1e-9, (
            f"Theorem 1(iii) violated on trial {trial}: err={abs(cov_T - (1.0 - alpha))}, "
            f"rhs={twosided_slack_T + delta0_T}"
        )

    print("PASS: Theorem 1 (Clamped cubic-hyperbolic barrier finiteness, anti-undercoverage, & suffix zero-excess) — 0 counterexamples")


def verify_theorem2_seff_pinball_cancellation(seed: int = 20261120) -> None:
    rng = np.random.default_rng(seed)

    # Exact algebraic & IEEE-754 identity across 500 random candidate vectors, alphas, and s_t from -100 to +1e300
    for _ in range(500):
        M = int(rng.integers(2, 10))
        cands = np.abs(rng.normal(scale=10.0, size=M))
        alpha = float(rng.uniform(0.01, 0.49))
        c_max = float(np.max(cands))

        # Test both moderate s_t < c_max, moderate s_t >= c_max, and extreme 1e300!
        for st in [float(rng.uniform(-5.0, c_max + 5.0)), c_max + float(rng.exponential(20.0)), 1e300]:
            s_eff = min(st, c_max)
            d_eff = s_eff - cands
            loss_eff = np.where(d_eff >= 0.0, (1.0 - alpha) * d_eff, -alpha * d_eff)

            if st < 1e8:
                d_raw = st - cands
                loss_raw = np.where(d_raw >= 0.0, (1.0 - alpha) * d_raw, -alpha * d_raw)
                diff_mat_raw = loss_raw[:, None] - loss_raw[None, :]
                diff_mat_eff = loss_eff[:, None] - loss_eff[None, :]
                assert float(np.max(np.abs(diff_mat_raw - diff_mat_eff))) < 1e-10
            else:
                # For any st >= c_max, exact real-arithmetic pairwise difference is -(1 - alpha) * (c_j - c_k)
                exact_real_diff = -(1.0 - alpha) * (cands[:, None] - cands[None, :])
                diff_mat_eff = loss_eff[:, None] - loss_eff[None, :]
                assert float(np.max(np.abs(diff_mat_eff - exact_real_diff))) < 1e-13

    # Verify IQR upper breakdown >= 0.25 vs 0% max-window breakdown
    W = 80
    clean = np.abs(rng.normal(size=W))
    corrupted = clean.copy()
    n_bad = int(math.floor(0.24 * W))
    corrupted[:n_bad] = 1e12
    iqr_corrupted = float(np.percentile(corrupted, 75) - np.percentile(corrupted, 25))
    max_corrupted = float(np.max(corrupted))
    assert iqr_corrupted < 50.0 and max_corrupted >= 1e12

    print("PASS: Theorem 2 (Exact s_eff pinball cancellation up to 1e300 & local-IQR breakdown) — 0 counterexamples")


def verify_theorem3_loggeo_and_drop_tail(seed: int = 20261121) -> None:
    rng = np.random.default_rng(seed)
    eps = 1e-4

    for _ in range(500):
        M = int(rng.integers(2, 8))
        c = np.abs(rng.normal(scale=5.0, size=M))
        raw_w = rng.uniform(0.05, 1.0, size=M)
        w = raw_w / np.sum(raw_w)

        # 1. Exact zero preservation
        g_zero = math.exp(float(np.dot(w, np.log(np.zeros(M) + eps)))) - eps
        assert abs(g_zero) < 1e-15

        # 2. AM-GM anti-overcoverage: G_eps(c; w) <= sum_m w_m c_m (strict when max(c) - min(c) > 0)
        g_eps = math.exp(float(np.dot(w, np.log(c + eps)))) - eps
        lin_mean = float(np.dot(w, c))
        assert g_eps <= lin_mean + 1e-12
        if float(np.max(c) - np.min(c)) > 1e-2:
            assert g_eps < lin_mean - 1e-7

        # 3. Exact sharp Mahler-GM-HM scale-equivariance bound:
        #    0 <= G_eps(lam * c; w) - lam * G_0(c; w) <= eps * (G_0(c; w) / H_0(c; w) - 1) for all lam > 0!
        c_pos = c + 1e-3
        g_0 = math.exp(float(np.dot(w, np.log(c_pos))))
        h_0 = 1.0 / float(np.dot(w, 1.0 / c_pos))
        rhs = eps * ((g_0 / h_0) - 1.0)
        for lam in [1e-3, 0.1, 1.0, 10.0, 1e4]:
            g_scaled = math.exp(float(np.dot(w, np.log(lam * c_pos + eps)))) - eps
            diff = g_scaled - lam * g_0
            assert diff >= -1e-11 * max(1.0, lam), f"Theorem 3(iii) Mahler lower bound violated: {diff}"
            assert diff <= rhs + 1e-10 * max(1.0, lam), f"Theorem 3(iii) GM-HM upper bound violated: {diff} > {rhs}"

    # 4. Exact closed-form post-drop empirical quantile contamination duration:
    #    N_contam(w_keep, alpha) = ceil(((1 - alpha) / alpha) * (w_keep + 1)) steps (e.g., 333 -> 99 at alpha=0.10)
    for alpha in (0.05, 0.10, 0.20):
        for w_keep in (6, 10, 28, 36):
            pre = np.full(w_keep, 10.0)
            contaminated_steps = 0
            for n_new in range(0, 900):
                seg = np.concatenate([pre, np.ones(n_new)])
                N = len(seg)
                k_ord = min(max(int(math.ceil((1.0 - alpha) * (N + 1))) - 1, 0), N - 1)
                q_emp = float(np.sort(seg)[k_ord])
                if q_emp >= 10.0:
                    contaminated_steps += 1
            exact_formula = int(math.ceil(((1.0 - alpha) / alpha) * (w_keep + 1) - 1e-12))
            assert contaminated_steps == exact_formula, f"Mismatch at alpha={alpha}, w={w_keep}: {contaminated_steps} != {exact_formula}"

    print("PASS: Theorem 3 (Shifted log-geometric AM-GM, uniform eps-equivariance, & post-drop tail bound) — 0 counterexamples")


if __name__ == "__main__":
    verify_theorem1_barrier_and_zero_atom()
    verify_theorem2_seff_pinball_cancellation()
    verify_theorem3_loggeo_and_drop_tail()
    print("ALL_PROOFS_VERIFIED_ZERO_COUNTEREXAMPLES")
