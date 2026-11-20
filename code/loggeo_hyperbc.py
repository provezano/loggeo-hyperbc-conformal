#!/usr/bin/env python3
"""Reference implementation of LogGeo-HyperBC (Algorithm 1)."""
from __future__ import annotations

import math
import numpy as np
from scipy import stats


def causal_iqr_scale(win: np.ndarray, sigma_min: float = 1e-4) -> float:
    n = len(win)
    if n >= 8:
        idx25 = max(0, int(0.25 * (n - 1)))
        idx75 = min(n - 1, int(round(0.75 * (n - 1))))
        part = np.partition(win, (idx25, idx75))
        return max(float(part[idx75] - part[idx25]) / 1.349, sigma_min)
    if n >= 2:
        med = float(np.median(win))
        return max(float(np.median(np.abs(win - med))) * 1.4826, sigma_min)
    return max(float(abs(win[0])) if n == 1 else 1.0, sigma_min)


def _hypergeom_min_k(N: int, K: int, n: int, alpha_tail: float) -> int:
    lo = max(1, int(math.ceil(n * (K / max(N, 1)))))
    hi = min(K, n)
    if lo > hi or float(stats.hypergeom.sf(hi - 1, N, K, n)) > alpha_tail:
        return hi + 1
    ans = hi + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if float(stats.hypergeom.sf(mid - 1, N, K, n)) <= alpha_tail:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def detect_hyperbc_signed(seg_clean: np.ndarray) -> tuple[int, int]:
    N = len(seg_clean)
    if N < 24 or (N % 2) != 0:
        return 0, 0
    w_fast = max(6, int(math.ceil(1.20 * math.log(N + 2.0))))
    w_med = max(20, 2 * w_fast + 4)
    w_grad = min(120, max(48, N // 3))
    specs = (
        (w_fast, 0.14, 0.20 / (N ** 2.10)),
        (w_med, 0.28, 0.15 / (N ** 2.10)),
        (w_grad, 0.50, 0.15 / (N ** 2.10)),
    )
    for w, beta, a_each in specs:
        if N < 2 * w:
            continue
        m = int(math.floor(beta * N))
        if m < 2:
            continue
        k_crit = _hypergeom_min_k(N, m, w, a_each)
        if k_crit > w:
            continue
        idx_hi = N - m - 1
        idx_lo = m
        part = np.partition(seg_clean, (idx_lo, idx_hi))
        tail_seg = seg_clean[-w:]
        if int(np.sum(tail_seg > float(part[idx_hi]))) >= k_crit:
            return w, +1
        if int(np.sum(tail_seg < float(part[idx_lo]))) >= k_crit:
            return w, -1
    return 0, 0


def run_loggeo_hyperbc(scores: np.ndarray, alpha: float = 0.10, c0_bar: float = 0.95) -> dict[str, np.ndarray]:
    T = len(scores)
    q_out = np.zeros(T, dtype=np.float64)
    err_out = np.zeros(T, dtype=np.float64)
    inf_out = np.zeros(T, dtype=bool)
    empty_out = np.zeros(T, dtype=bool)

    pin_specs = ((0.03, 180), (0.08, 80), (0.16, 36), (0.30, 16))
    m_pin = len(pin_specs)
    q_pin = np.ones(m_pin, dtype=np.float64)
    n_cands = 1 + m_pin
    weights = np.full(n_cands, 1.0 / n_cands, dtype=np.float64)
    sigma_share = 0.006
    eta_hedge = 2.20
    prior_share = np.array([0.24, 0.19, 0.19, 0.19, 0.19], dtype=np.float64)
    eps_log = 1e-4

    tau = 0
    tau_clean = 0
    E_overhang = 0.0
    theta_cal = 0.0

    for t_0 in range(T):
        if t_0 == 0:
            q_emp = 1.0
            sig_loc = 1.0
            n_seg = 1
        else:
            seg = scores[tau:t_0]
            n_seg = len(seg)
            k_emp = min(n_seg - 1, max(0, int(math.ceil((1.0 - alpha) * (n_seg + 1))) - 1))
            if k_emp >= n_seg - 1 and n_seg < 15:
                k_emp = max(0, int(math.floor((1.0 - alpha) * (n_seg - 1))))
            q_emp = float(np.partition(seg, k_emp)[k_emp])
            sig_loc = causal_iqr_scale(scores[max(0, t_0 - 32) : t_0], sigma_min=1e-4)

        cands = np.empty(n_cands, dtype=np.float64)
        cands[0] = max(0.0, q_emp)
        cands[1:] = np.maximum(0.0, q_pin)

        q_meta = float(math.exp(float(np.dot(weights, np.log(cands + eps_log)))) - eps_log)
        if t_0 > 0:
            q_cal = float(math.exp(theta_cal) * q_meta)
            h_t = max(c0_bar * (t_0 ** (2.0 / 3.0)), 1.0)
            u_norm = E_overhang / h_t
            u_clamp = min(abs(u_norm), 30.0)
            psi_val = sig_loc * (u_norm ** 3) * math.cosh(u_clamp)
            q_pred = max(0.0, q_cal + psi_val)
        else:
            q_pred = max(0.0, q_meta)

        q_out[t_0] = q_pred
        st = float(scores[t_0])
        e_t = 1.0 if st > q_pred else 0.0
        err_out[t_0] = e_t
        E_overhang += e_t - alpha

        eta_th = 0.38 / ((n_seg + 12.0) ** 0.55)
        theta_cal = float(np.clip(theta_cal + eta_th * (e_t - alpha), -0.60, 0.60))

        s_eff = min(st, float(np.max(cands)))
        diff_vec = s_eff - cands
        pin_losses = np.where(diff_vec >= 0.0, (1.0 - alpha) * diff_vec, -alpha * diff_vec)
        rel_losses = pin_losses - float(np.min(pin_losses))
        scale_norm = max(sig_loc, 0.25 * float(np.median(cands)), 1e-4)
        norm_losses = np.clip(rel_losses / scale_norm, 0.0, 6.0)
        w_tilde = weights * np.exp(-eta_hedge * norm_losses)
        s_w = float(np.sum(w_tilde))
        w_tilde = (w_tilde / s_w) if s_w > 0.0 else np.full(n_cands, 1.0 / n_cands)
        weights = (1.0 - sigma_share) * w_tilde + sigma_share * prior_share

        for m_idx, (eta_m, w_m) in enumerate(pin_specs):
            win_m = scores[max(0, t_0 - w_m + 1) : t_0 + 1]
            sig_m = causal_iqr_scale(win_m, sigma_min=1e-4)
            if t_0 == 0:
                q_pin[m_idx] = max(0.0, st)
            elif t_0 == 4:
                q_pin[m_idx] = max(0.0, float(np.quantile(scores[:5], 1.0 - alpha)))
            else:
                miss_m = 1.0 if st > q_pin[m_idx] else 0.0
                q_pin[m_idx] = max(0.0, q_pin[m_idx] + eta_m * sig_m * (miss_m - alpha))

        w_det, dir_det = detect_hyperbc_signed(scores[tau_clean : t_0 + 1])
        if w_det > 0:
            w_keep = min(w_det, 28) if dir_det > 0 else min(w_det, 10)
            tau = max(0, (t_0 + 1) - w_keep)
            tau_clean = t_0 + 1
            theta_cal = 0.0

    return {"q": q_out, "err": err_out, "inf": inf_out, "empty": empty_out}
