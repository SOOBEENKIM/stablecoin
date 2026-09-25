"""Bounded stage-one benchmarks; prior ML predictions are immutable inputs."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'residual_nested_validation'))
import nested as n
a = n.a
DESIGN = json.loads((HERE / 'design.json').read_text())
OUT = HERE / 'results'
LEGACY = HERE.parent / 'residual_nested_validation'
CASES = [tuple(x) for x in DESIGN['cases']]


def candidate_specs():
    rows = []
    for strategy in DESIGN['new_strategies']:
        for window in DESIGN['windows']:
            forms = ['change'] if strategy == 'persistence' else DESIGN['forms']
            alphas = [None] if strategy == 'persistence' else DESIGN['alphas']
            for form in forms:
                for alpha in alphas:
                    rows.append(dict(candidate=len(rows), strategy=strategy,
                                     window=window, form=form, alpha=alpha))
    assert len(rows) == 26
    return rows


SPECS = candidate_specs()


class Benchmark:
    def __init__(self, spec):
        self.spec = spec
        self.columns = DESIGN['features'][spec['strategy']]

    def fit(self, train):
        if self.spec['strategy'] == 'persistence':
            self.offset = float(np.quantile(train.target - train.e_now, .1))
        else:
            self.model = a.DynamicModel('linear', self.spec['form'],
                {'alpha': self.spec['alpha']}).fit(train, self.columns)
        return self

    def predict(self, features):
        # Callers pass features only; no target/outcome columns enter models.
        if self.spec['strategy'] == 'persistence':
            return features.e_now.to_numpy() + self.offset
        return self.model.predict(features[self.columns])

    def coefficients(self):
        if self.spec['strategy'] == 'persistence':
            return dict(intercept=self.offset, e_now=1.)
        m = self.model.model
        values = m.y_std * m.model.coef_ / m.scaler.scale_
        result = dict(zip(self.columns, map(float, values)))
        result['intercept'] = float(m.y_mean + m.y_std * m.model.intercept_ - values @ m.scaler.mean_)
        if self.spec['form'] == 'change':
            result['e_now'] += 1.
        return result


def choose(scores, cutoff, strategy):
    expected = {x.strftime('%Y-%m') for x in n.inner_months(cutoff)}
    if not set(scores.fold).issubset(expected):
        raise ValueError('Selection received an outer/future month')
    if scores.duplicated(['candidate', 'fold']).any():
        raise ValueError('Duplicate candidate/month')
    options = []
    for candidate, g in scores[scores.strategy == strategy].groupby('candidate'):
        if set(g.fold) != expected or not g.valid.all():
            continue
        if (g.n <= 0).any() or not np.isfinite(g.loss_sum).all():
            continue
        options.append((float(g.loss_sum.sum() / g.n.sum()), int(candidate)))
    if not options:
        raise ValueError('No candidate valid in all three earlier months')
    score, candidate = min(options)
    return dict(**SPECS[candidate], inner_score=score, inner_months=sorted(expected))


def read_legacy():
    d = pd.read_csv(LEGACY / 'results/predictions.csv.gz')
    for c in ['origin', 'target_time', 'latest_calibration_label']:
        d[c] = pd.to_datetime(d[c], utc=True)
    return d[(d['info'] == DESIGN['legacy_info']) &
             d.strategy.isin(DESIGN['legacy_strategies'])].copy()
