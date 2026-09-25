"""Post-result, bounded horizon and residual-scale development experiment."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
OLD = HERE.parent / 'residual_factorial_execution'
sys.path.insert(0, str(OLD))
import core as old
from execution_guard import sha, now, write_new, verify_lock as verify_old_lock

HORIZONS = [1, 6, 12]
VARIANTS = ['original', 'own_scale']
INFOS = ['R', 'B', 'F']
STRATEGIES = ['linear', 'qrf', 'boosting', 'threshold', 'ml_selected']
SEED = 20260925
START = pd.Timestamp('2025-12-01', tz='UTC')
END = pd.Timestamp('2026-03-20', tz='UTC')


def own_scale(e):
    if not e.index.is_monotonic_increasing or e.index.has_duplicates:
        raise ValueError('Unordered or repeated residual timestamps')
    de = e - old.legacy.a.lead(e, -1)
    return de.pow(2).rolling('72h', min_periods=12).mean().pow(.5).rename('e_rms72')


def columns(info, variant):
    if variant not in VARIANTS or info not in INFOS:
        raise ValueError('Unknown information set')
    return old.feature_sets('available_macro')[info] + (['e_rms72'] if variant == 'own_scale' else [])


def make_case(panel, cutoff, h):
    r = old.legacy.a.Residualizer('EQ').fit(panel, cutoff)
    e, _ = r.transform(panel)
    d = old.with_calendar(old.legacy.a.frame(panel, r, 'available_macro', h)).join(own_scale(e))
    train = d[d.target_time < cutoff]
    test = d[(d.index >= cutoff) & (d.index < cutoff + pd.offsets.MonthBegin(1)) & (d.target_time < END)]
    for x in [train, test]:
        if len(x) == 0 or not np.isfinite(x[columns('F', 'own_scale') + ['target']].to_numpy()).all():
            raise ValueError('Missing observations; no sample-changing fallback')
        assert ((x.target_time - x.index) == pd.Timedelta(hours=h)).all()
    assert train.target_time.max() < cutoff and r.fit_end < cutoff
    if h == 1:
        tr, te, _ = old.data_case('available_macro', 'EQ', cutoff, panel)
        pd.testing.assert_frame_equal(train[tr.columns], tr)
        pd.testing.assert_frame_equal(test[te.columns], te)
    return train, test, r


def calibrate(rows, h):
    d = rows.sort_values('origin').reset_index(drop=True).copy()
    origins, labels = pd.DatetimeIndex(d.origin), pd.DatetimeIndex(d.target_time)
    for key in ['horizon', 'variant', 'information', 'candidate', 'seed']:
        if key in d and d[key].nunique(dropna=False) != 1:
            raise ValueError('Mixed candidate stream: ' + key)
    if (origins.has_duplicates or not labels.is_monotonic_increasing or
            not ((labels - origins) == pd.Timedelta(hours=h)).all()):
        raise ValueError('Wrong horizon or repeated origin')
    errors = (d.target - d.pred_raw).to_numpy()
    if not np.isfinite(errors).all():
        raise ValueError('Nonfinite errors')
    correction, counts, latest = [], [], []
    for t in origins:
        stop = labels.searchsorted(t, side='left')
        start = max(labels.searchsorted(t-pd.Timedelta(days=90), side='left'), stop-60)
        n = stop-start
        correction.append(float(np.quantile(errors[start:stop], .1, method='inverted_cdf')) if n >= 30 else 0.)
        counts.append(n)
        latest.append(labels[stop-1] if n else pd.NaT)
    d['correction'], d['calibration_n'], d['latest_calibration_label'] = correction, counts, latest
    d['pred_calibrated'] = d.pred_raw + d.correction
    scored = d.origin >= pd.Timestamp('2025-09-01', tz='UTC')
    if (d.loc[scored, 'calibration_n'] < 30).any():
        raise ValueError('Insufficient calibration')
    return d


def choose(stream, cutoff, strategy, h):
    for key in ['horizon', 'variant', 'information', 'seed']:
        if stream[key].nunique() != 1:
            raise ValueError('Mixed selection input')
    if not ((stream.target_time-stream.origin) == pd.Timedelta(hours=h)).all():
        raise ValueError('Wrong selection horizon')
    inner = old.purged_inner(stream, cutoff)
    expected = {m.strftime('%Y-%m') for m in old.inner_months(cutoff)}
    if set(inner.origin.dt.strftime('%Y-%m')) != expected or set(inner.candidate) != set(range(60)):
        raise ValueError('Missing month/candidate')
    ref, scores = None, []
    for candidate, g in inner.groupby('candidate', sort=True):
        z = g.sort_values('origin')[['origin', 'target_time', 'target']].reset_index(drop=True)
        if ref is None:
            ref = z
        else:
            pd.testing.assert_frame_equal(z, ref)
        if (g.calibration_n < 30).any() or not np.isfinite(g.pred_calibrated).all():
            raise ValueError('Invalid scored candidate')
        scores.append(dict(candidate=int(candidate), kind=old.SPECS[candidate]['kind'],
                           score=float(old.loss(g.target, g.pred_calibrated).mean()), n=len(g)))
    kinds = ['qrf', 'boosting'] if strategy == 'ml_selected' else [strategy]
    score, cid = min((x['score'], x['candidate']) for x in scores if x['kind'] in kinds)
    return dict(**old.SPECS[cid], inner_score=score, inner_months=sorted(expected)), scores


def read(path):
    d = pd.read_csv(path, float_precision='round_trip')
    for c in ['origin', 'target_time', 'latest_calibration_label']:
        if c in d:
            d[c] = pd.to_datetime(d[c], utc=True)
    return d


def save(d, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0} if path.name.endswith('.gz') else None)
