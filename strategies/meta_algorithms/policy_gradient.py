from __future__ import annotations

import numpy as np
import pandas as pd

from walk_forward import segmented_wealth

# r[i, t] = gross return of expert i in period t, q = weights on the
# experts, phi = per-expert features of the recent past, theta/c = the
# policy's shared feature weights and per-expert biases


def features(r: np.ndarray, L: int, ref: int) -> np.ndarray:
    # phi[t, i] is built from r[:, :t] only. Three features per expert over
    # the last L periods -- volatility, drawdown of log wealth, and log
    # wealth relative to expert ref -- each standardized across experts so
    # the scale of the universe does not matter.
    K, T = r.shape
    logr = np.log(r)
    phi = np.zeros((T, K, 3))
    for t in range(1, T):
        past = logr[:, max(0, t - L) : t]
        level = np.concatenate([np.zeros((K, 1)), np.cumsum(past, axis=1)], axis=1)
        phi[t, :, 0] = past.std(axis=1)
        phi[t, :, 1] = level.max(axis=1) - level[:, -1]
        phi[t, :, 2] = past.sum(axis=1) - past[ref].sum()
        phi[t] = (phi[t] - phi[t].mean(axis=0)) / (phi[t].std(axis=0) + 1e-8)
    return phi


def _softmax(s: np.ndarray) -> np.ndarray:
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _fit(phi, r, p, rng, steps, batch, lr, l2):
    # The action (q) does not change the next state -- the experts' returns
    # are what they are -- so the policy gradient of the expected log
    # return has no sampling noise: it is the exact gradient of
    # E[log q . r] through the softmax, estimated on a minibatch of past
    # periods. Adam ascent.
    F = phi.shape[2]
    m, v = np.zeros_like(p), np.zeros_like(p)
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(phi), batch)
        phi_b, r_b = phi[idx], r[:, idx].T
        q = _softmax(phi_b @ p[:F] + p[F:])
        g_s = q * (r_b / (q * r_b).sum(axis=1, keepdims=True) - 1.0)
        g = np.concatenate([np.einsum("bk,bkf->f", g_s, phi_b), g_s.sum(axis=0)]) / batch - l2 * p
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        p = p + lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return p


def policy_experts(
    r: np.ndarray,
    dates: pd.DatetimeIndex,
    seeds: list[int],
    ref: int,
    L: int = 60,
    warmup_years: int = 3,
    steps: int = 300,
    batch: int = 1024,
    lr: float = 0.02,
    l2: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    # Linear softmax policy over K experts, refit every 1 January on the
    # periods before it, like walk_forward.py (expanding window, first
    # warmup_years at equal weights). The previous year's parameters start
    # the next fit. The optimizer settings were fixed on synthetic experts
    # with one slightly better than the rest, not on market data (an L2
    # penalty of 1e-3 kept the weights near equal and hid that edge). Each
    # seed differs only in its initial theta and its minibatches. Returns
    # the wealth of the seed-averaged weights and that of each seed.
    K, T = r.shape
    phi = features(r, L, ref)
    F = phi.shape[2]
    q = np.full((len(seeds), T, K), 1.0 / K)
    for s, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        p = np.concatenate([rng.normal(0.0, 0.1, F), np.zeros(K)])
        for year in range(dates[0].year + warmup_years, dates[-1].year + 1):
            d = pd.Timestamp(year, 1, 1)
            k = dates.searchsorted(d)
            p = _fit(phi[1:k], r[:, 1:k], p, rng, steps, batch, lr, l2)
            during = (dates >= d) & (dates < pd.Timestamp(year + 1, 1, 1))
            q[s, during] = _softmax(phi[during] @ p[:F] + p[F:])
    return _wealth(q.mean(axis=0), r), [_wealth(q[s], r) for s in range(len(seeds))]


def policy_experts_wealth(
    experts: list[list[np.ndarray]],
    segments: list[np.ndarray],
    dates: pd.DatetimeIndex,
    seeds: list[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    # b_t . x_t = sum_i q_i (b_ti . x_t), so the combiner only needs each
    # expert's period returns, which chain across segments even though the
    # asset columns change. Expert 0 (BAH) is the reference for the
    # relative-wealth feature.
    S = [segmented_wealth(e, segments) for e in experts]
    r = np.array([w[1:] / w[:-1] for w in S])
    return policy_experts(r, dates, seeds, ref=0)


def _wealth(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.concatenate([[1.0], np.cumprod((q * r.T).sum(axis=1))])
