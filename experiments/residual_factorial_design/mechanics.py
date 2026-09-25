"""Design primitives only. No forecast model is fitted or experiment scored here."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE / 'design.json').read_text())


def candidate_specs():
    specs = []
    for kind in DESIGN['families']:
        for window in DESIGN['windows']:
            for form in DESIGN['target_forms']:
                for params in DESIGN['grids'][kind]:
                    specs.append(dict(candidate=len(specs), kind=kind,
                                      window=window, form=form,
                                      params=copy.deepcopy(params)))
    if len(specs) != DESIGN['candidate_count_per_information_set']:
        raise ValueError('Candidate count differs from protocol')
    return specs


def feature_sets(mode):
    if mode not in ['available_macro', 'strict']:
        raise ValueError(mode)
    r = DESIGN['residual_features'] + DESIGN['calendar_in_all_cells_and_cases']
    b = r + DESIGN['market_features']
    if mode == 'available_macro':
        b += DESIGN['available_macro_extra']
    return dict(R=r, B=b, F=b + DESIGN['position_features'])


def with_calendar(frame):
    d = frame.copy()
    kst = d.index.tz_convert('Asia/Seoul')
    d['hour_sin'] = np.sin(kst.hour * np.pi / 12)
    d['hour_cos'] = np.cos(kst.hour * np.pi / 12)
    d['weekend'] = (kst.dayofweek >= 5).astype(float)
    return d


def loss(y, pred):
    error = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    q = DESIGN['quantile']
    return np.where(error >= 0, q * error, (q - 1) * error)


def calibrate_candidate(rows):
    """Exactly one candidate/info/case/seed stream, including August warm-up."""
    for key in DESIGN['candidate_stream_identity']:
        if key in rows and rows[key].nunique(dropna=False) != 1:
            raise ValueError('Mixed calibration stream: ' + key)
    d = rows.sort_values('origin').reset_index(drop=True).copy()
    origins, labels = pd.DatetimeIndex(d.origin), pd.DatetimeIndex(d.target_time)
    if (len(d) == 0 or origins.has_duplicates or not labels.is_monotonic_increasing
            or not ((labels - origins) == pd.Timedelta(hours=1)).all()):
        raise ValueError('Invalid one-hour candidate stream')
    errors = d.target.to_numpy(float) - d.pred_raw.to_numpy(float)
    if not np.isfinite(errors).all():
        raise ValueError('Nonfinite candidate error')
    rule = DESIGN['calibration']
    corrections, counts, latest = [], [], []
    for t in origins:
        stop = labels.searchsorted(t, side='left')
        start = max(labels.searchsorted(t - pd.Timedelta(days=rule['maximum_age_days']),
                                       side='left'), stop - rule['past_errors'])
        count = stop - start
        correction = (np.quantile(errors[start:stop], DESIGN['quantile'],
                                 method=rule['quantile_method'])
                      if count >= rule['minimum'] else 0.)
        corrections.append(float(correction))
        counts.append(count)
        latest.append(labels[stop - 1] if count else pd.NaT)
    d['correction'] = corrections
    d['calibration_n'] = counts
    d['latest_calibration_label'] = latest
    d['pred_calibrated'] = d.pred_raw + d.correction
    return d


def inner_months(cutoff):
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or cutoff != cutoff.normalize() or cutoff.day != 1:
        raise ValueError('Expected timezone-aware month boundary')
    return [cutoff - pd.offsets.MonthBegin(i) for i in [3, 2, 1]]


def choose_past(rows, cutoff, strategy):
    """Receive only purged prior-three-month rows for one info/case/seed.

    All 60 candidate streams must cover the same origins. No fallback or
    candidate-specific sample reduction is allowed. No outer rows accepted.
    """
    cutoff = pd.Timestamp(cutoff)
    expected_months = [m.strftime('%Y-%m') for m in inner_months(cutoff)]
    d = rows.copy()
    for key in ['mode', 'definition', 'information', 'seed']:
        if key in d and d[key].nunique(dropna=False) != 1:
            raise ValueError('Mixed selection context: ' + key)
    if 'seed' in d and int(d.seed.iloc[0]) != DESIGN['candidate_selection_seed']:
        raise ValueError('Selection must use the fixed first seed')
    origins = pd.DatetimeIndex(d.origin)
    labels = pd.DatetimeIndex(d.target_time)
    months = origins.strftime('%Y-%m')
    ends = origins.normalize().map(lambda t: t.replace(day=1) + pd.offsets.MonthBegin(1))
    if (set(months) != set(expected_months) or not (labels < cutoff).all()
            or not (labels < ends).all()
            or not ((labels - origins) == pd.Timedelta(hours=1)).all()):
        raise ValueError('Future, unpurged, or incomplete inner months')
    if d.duplicated(['candidate', 'origin']).any():
        raise ValueError('Duplicate candidate/origin')
    specs = candidate_specs()
    if set(d.candidate) != set(s['candidate'] for s in specs):
        raise ValueError('Missing/extra candidate')
    if (d.calibration_n < DESIGN['calibration']['minimum']).any():
        raise ValueError('Insufficient calibration at a scored origin')
    if not np.isfinite(d[['target', 'pred_calibrated']].to_numpy(float)).all():
        raise ValueError('Nonfinite selection values')
    paired = None
    scores = []
    for candidate, g in d.groupby('candidate', sort=True):
        g = g.sort_values('origin')
        comparison = g[['origin', 'target_time', 'target']].reset_index(drop=True)
        if paired is None:
            paired = comparison
        elif not comparison.equals(paired):
            raise ValueError('Candidates do not share origins and targets')
        scores.append((float(loss(g.target, g.pred_calibrated).mean()), int(candidate)))
    if strategy == 'ml_selected':
        kinds = DESIGN['ml_selection_pool']
    elif strategy in DESIGN['families']:
        kinds = [strategy]
    else:
        raise ValueError(strategy)
    score, candidate = min((v, c) for v, c in scores if specs[c]['kind'] in kinds)
    return dict(**specs[candidate], inner_score=score, inner_months=expected_months)
