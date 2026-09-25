"""Paired calendar-block statistics; independent daily-sum check in verify.py."""
import numpy as np
import pandas as pd
from core import DESIGN


def day_weights(frame, block_days):
    nboot = DESIGN['uncertainty']['repetitions']
    rng = np.random.default_rng(DESIGN['uncertainty']['seed_base_plus_block_days'] + block_days)
    d = frame.reset_index(drop=True)
    weights = np.zeros((nboot, len(d)), dtype=np.int16)
    for _, g in d.groupby('fold', sort=True):
        days = pd.date_range(g.origin.dt.normalize().min(), g.origin.dt.normalize().max(), freq='D')
        starts = rng.integers(0, len(days), size=(nboot, int(np.ceil(len(days) / block_days))))
        draw = ((starts[:, :, None] + np.arange(block_days)) % len(days)).reshape(nboot, -1)[:, :len(days)]
        counts = np.zeros((nboot, len(days)), dtype=np.int16)
        for i in range(nboot):
            counts[i] = np.bincount(draw[i], minlength=len(days))
        j = days.get_indexer(g.origin.dt.normalize())
        weights[:, g.index] = counts[:, j]
    if (weights.sum(axis=1) == 0).any():
        raise ValueError('Empty bootstrap replicate')
    return weights


def bootstrap_mean(weights, values):
    return weights @ np.asarray(values, float) / weights.sum(axis=1)


def effect(reference, candidate, weights):
    ref, new = np.asarray(reference, float), np.asarray(candidate, float)
    delta = ref - new
    point = float(delta.mean())
    draws = bootstrap_mean(weights, delta)
    gains = 100 * draws / bootstrap_mean(weights, ref)
    lo, hi = np.quantile(gains, [.025, .975])
    p = (1 + np.count_nonzero(np.abs(draws - point) >= abs(point))) / (len(draws) + 1)
    return dict(difference_bp=point, improvement_pct=100 * (1-new.mean()/ref.mean()),
                improvement_lo=float(lo), improvement_hi=float(hi), p_centered_boot=float(p))


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues, kind='stable')
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    return adjusted
