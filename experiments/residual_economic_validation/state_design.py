"""Outcome-blind, within-month/session matching of observed exposures."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

EXPOSURES = {'account': 'account_change24', 'funding': 'funding_change8', 'oi': 'oi_change24'}
MATCH_COLS = ['e_now', 'log_vol', 'log_downside', 'e_change1']


def covariates(d):
    return pd.DataFrame(dict(e_now=d.e_now, log_vol=np.log1p(d.btc_vol24),
        log_downside=np.log1p(d.downside24), e_change1=d.e_change1), index=d.index)


def match_month(train, sample, exposure):
    """Maximum-cardinality then minimum-distance assignment; no replacement.

    Inputs needed: past thresholds and origin-time exposures/covariates only.
    No target, future attribution or forecast column is accessed.
    """
    column = EXPOSURES[exposure]
    low, high = train[column].quantile([.25, .75])
    scaler = StandardScaler().fit(covariates(train))
    z = pd.DataFrame(scaler.transform(covariates(sample)), index=sample.index, columns=MATCH_COLS)
    hi = sample.index[sample[column] > high]
    lo = sample.index[sample[column] < low]
    rows = []
    sessions = pd.Series(sample.index.tz_convert('Asia/Seoul').hour // 6, index=sample.index)
    for session in range(4):
        h = hi[sessions.loc[hi].to_numpy() == session]
        l = lo[sessions.loc[lo].to_numpy() == session]
        if not len(h) or not len(l):
            continue
        differences = z.loc[h].to_numpy()[:, None, :] - z.loc[l].to_numpy()[None, :, :]
        valid = (np.abs(differences) <= np.array([.5, .5, .5, 1.])).all(axis=2)
        distance = np.square(differences).sum(axis=2)
        costs = np.column_stack([np.where(valid, distance, 1e9), np.full((len(h), len(h)), 1e6)])
        ii, jj = linear_sum_assignment(costs)
        for i, j in zip(ii, jj):
            if j >= len(l) or not valid[i, j]:
                continue
            rows.append(dict(high_origin=h[i], low_origin=l[j], session=session,
                distance=distance[i, j], low_cut=low, high_cut=high,
                **{'difference_' + c: differences[i, j, k] for k, c in enumerate(MATCH_COLS)}))
    pairs = pd.DataFrame(rows)
    if len(pairs):
        assert pairs.high_origin.is_unique and pairs.low_origin.is_unique
        assert not set(pairs.high_origin).intersection(pairs.low_origin)
    return pairs, z, scaler, dict(low_cut=float(low), high_cut=float(high),
                                high_n=len(hi), low_n=len(lo), pairs=len(pairs))
