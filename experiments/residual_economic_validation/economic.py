"""Frozen group ablations and a train-only regime quantile benchmark."""
from pathlib import Path
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor

HERE = Path(__file__).resolve().parent
ADAPTIVE = HERE.parent / 'residual_dynamics_adaptive'
sys.path.insert(0, str(ADAPTIVE))
import adaptive as a

GROUPS = {
    'account': ['account_log', 'account_change1', 'account_change24'],
    'funding': ['funding_bp', 'funding_change8'],
    'oi': ['oi_ret1', 'oi_surprise', 'oi_change24'],
}
CASES = [('available_macro', x) for x in ['EQ', 'CAP', 'PCA']] + [('strict', 'EQ')]
SEEDS = [20260925, 20260926, 20260927, 20260928]


def features(mode, info):
    if info.startswith('without_'):
        removed = GROUPS[info[len('without_'):]]
        return [c for c in a.columns(mode, 'history') if c not in removed]
    return a.columns(mode, info)


def threshold_candidates():
    return [(days, form, dict(band=band, alpha=alpha))
        for days in ['all', 90] for form in ['level', 'change']
        for band in [[.2, .8], [.35, .65]] for alpha in [0., .002, .01]]


class ThresholdQuantile:
    """Three residual regimes with distinct slopes and intercepts, q=.10.

    This is an adapted forecasting benchmark, not a replication of the
    mean-threshold Kimchi-premium paper or a structural error-correction model.
    """
    def __init__(self, form, params):
        self.form, self.params = form, params

    def design(self, d, fit=False):
        if fit:
            self.cuts = np.quantile(d.e_now, self.params['band'])
            self.scaler = StandardScaler().fit(d[self.cols])
        regime = np.searchsorted(self.cuts, d.e_now.to_numpy(), side='right')
        if fit:
            self.counts = np.bincount(regime, minlength=3)
            if self.counts.min() < 40:
                raise ValueError('Insufficient training observations in a regime')
        z = self.scaler.transform(d[self.cols])
        mat = np.column_stack([(regime == k)[:, None] * z for k in range(3)] +
                              [(regime == 1).astype(float), (regime == 2).astype(float)])
        if fit:
            self.basis_scaler = StandardScaler().fit(mat)
        return self.basis_scaler.transform(mat)

    def fit(self, d, cols):
        self.cols = list(cols)
        y = d.target.to_numpy() - (d.e_now.to_numpy() if self.form == 'change' else 0)
        self.y_mean, self.y_std = y.mean(), max(y.std(ddof=1), 1e-8)
        self.model = QuantileRegressor(quantile=.1, alpha=self.params['alpha'], solver='highs',
            solver_options={'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8})
        self.model.fit(self.design(d, fit=True), (y - self.y_mean) / self.y_std)
        return self

    def predict(self, d):
        pred = self.y_mean + self.y_std * self.model.predict(self.design(d))
        return pred + (d.e_now.to_numpy() if self.form == 'change' else 0)


def make_model(kind, form, params, seed):
    a.SEED = seed
    return ThresholdQuantile(form, params) if kind == 'threshold' else a.DynamicModel('qrf', form, params)
