"""Execute the separately frozen factorial design without changing its grid."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
DESIGN_DIR = HERE.parent / 'residual_factorial_design'
sys.path.insert(0, str(DESIGN_DIR))
from mechanics import (DESIGN, candidate_specs, feature_sets, with_calendar,
                       calibrate_candidate, choose_past, inner_months, loss)
sys.path.insert(0, str(HERE.parent / 'residual_nested_validation'))
import nested as legacy

SPECS = candidate_specs()
CASES = [tuple(x) for x in DESIGN['cases']]
MONTHS = pd.date_range(DESIGN['candidate_stream_start'],
    pd.Timestamp(DESIGN['evaluation_end_exclusive']) - pd.offsets.MonthBegin(1), freq='MS')
OUTER_MONTHS = MONTHS[MONTHS >= pd.Timestamp(DESIGN['evaluation_start'])]
IDENTITY = DESIGN['candidate_stream_identity']


def seeds_for(mode, definition):
    return DESIGN['seeds_primary'] if (mode, definition) == tuple(DESIGN['primary_case']) else DESIGN['seeds_supplementary']


def data_case(mode, definition, cutoff, panel=None):
    if panel is None:
        panel = legacy.a.panel(mode)
    train, test, residualizer = legacy.outer_sample(panel, mode, definition, cutoff)
    train, test = with_calendar(train), with_calendar(test)
    cols = feature_sets(mode)['F']
    assert train.target_time.max() < cutoff and residualizer.fit_end < cutoff
    assert np.isfinite(train[cols + ['target']].to_numpy()).all()
    assert np.isfinite(test[cols + ['target']].to_numpy()).all()
    return train, test, residualizer


def fit_predict(spec, train, test, cols, seed):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = legacy.fit_model(spec, train, cols, seed)
        pred = np.asarray(model.predict(test[cols]), dtype=float)
    if caught:
        raise RuntimeError('Unresolved fit warning: ' + '; '.join(str(w.message) for w in caught))
    if pred.shape != (len(test),) or not np.isfinite(pred).all():
        raise RuntimeError('Invalid prediction vector')
    return model, pred


def read_predictions(path):
    d = pd.read_csv(path, float_precision='round_trip')
    for c in ['origin', 'target_time', 'latest_calibration_label']:
        if c in d:
            d[c] = pd.to_datetime(d[c], utc=True)
    return d


def purged_inner(stream, cutoff):
    starts = inner_months(cutoff)
    d = stream[(stream.origin >= starts[0]) & (stream.origin < cutoff)].copy()
    next_month = d.origin.dt.normalize().map(lambda t: t.replace(day=1) + pd.offsets.MonthBegin(1))
    return d[(d.target_time < next_month) & (d.target_time < cutoff)]


def comparison_specs():
    result = [dict(x, primary=True) for x in DESIGN['primary_comparisons']]
    for info in DESIGN['information_sets']:
        for family in ['qrf', 'boosting']:
            result.append(dict(id=family+'_vs_linear_'+info,
                reference=[info, 'linear'], candidate=[info, family], primary=False))
    result.append(dict(id='nonlinear_B', reference=['B', 'linear'],
                       candidate=['B', 'ml_selected'], primary=False))
    for name, ref, new in [('market', 'R', 'B'), ('position', 'B', 'F'), ('total', 'R', 'F')]:
        result.append(dict(id='matched_'+name, reference=[ref, 'matched_full_spec'],
                           candidate=[new, 'matched_full_spec'], primary=False))
    return result
