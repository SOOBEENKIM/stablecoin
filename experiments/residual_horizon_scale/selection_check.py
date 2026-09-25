"""Verify both independent loss arithmetic and the frozen numerical tie rule."""
import numpy as np


def verify_selection(inner, ids, saved_candidate):
    independent = inner.groupby('candidate').checkloss.mean()
    replay = {int(i): float(g.checkloss.to_numpy().mean()) for i,g in inner.groupby('candidate')}
    # pandas uses a different reduction; this is a round-off check, not a
    # statistical tolerance or an outer-performance acceptance threshold.
    for i in ids:
        tolerance = 16*np.finfo(float).eps*max(1.,abs(independent.loc[i]))
        if abs(independent.loc[i]-replay[i]) > tolerance:
            raise AssertionError('Independent score exceeds floating-point error bound')
    chosen = min(ids,key=lambda i:(replay[i],i))
    if int(saved_candidate) != chosen:
        raise AssertionError('Saved choice differs from frozen NumPy score/tie rule')
    alt = min(ids,key=lambda i:(independent.loc[i],i))
    gap = float(independent.loc[chosen]-independent.loc[alt])
    if gap > 16*np.finfo(float).eps*max(1.,abs(independent.loc[alt])):
        raise AssertionError('Saved choice is not a numerical minimizer independently')
    return dict(saved_candidate=chosen,independent_reduction_candidate=int(alt),
                independent_score_gap_bp=gap,roundoff_changed_tie=(chosen!=alt))
