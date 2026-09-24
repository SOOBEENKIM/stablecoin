"""Small, fixed candidate sets; all transformations learned on training only."""
import numpy as np
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

SEED = 20260925
GRIDS = {
    'linear': [{'alpha': a} for a in [0., .01, .05]],
    'interaction': [{'alpha': a} for a in [0., .01, .05]],
    'spline': [{'alpha': a} for a in [.01, .05, .1]],
    'boosting': [dict(max_depth=d, n_estimators=n, min_samples_leaf=l)
                 for d, n, l in [(1, 150, 20), (2, 150, 20), (2, 250, 40)]],
    'constant': [{}], 'persistence': [{}],
}


def pinball(y, pred, q):
    err = np.asarray(y) - np.asarray(pred)
    return np.maximum(q * err, (q - 1) * err)


class ForecastModel:
    def __init__(self, kind, q, params):
        self.kind, self.q, self.params = kind, q, params

    def prepare(self, x, fit=False):
        if fit:
            self.scaler = StandardScaler().fit(x)
            self.cols = list(x.columns)
        z = self.scaler.transform(x)
        if self.kind == 'interaction':
            lookup = {c: z[:, j] for j, c in enumerate(self.cols)}
            extra = [lookup['downside24'] * lookup['btc_vol24'],
                     lookup['e_now'] * lookup['downside24']]
            for c in ['account_log', 'funding_bp', 'oi_ret1', 'oi_surprise']:
                if c in lookup:
                    extra.append(lookup['downside24'] * lookup[c])
            if 'account_log' in lookup:
                extra.append(lookup['e_now'] * lookup['account_log'])
            z = np.column_stack([z] + extra)
            if fit:
                self.second_scaler = StandardScaler().fit(z)
            z = self.second_scaler.transform(z)
        if self.kind == 'spline':
            if fit:
                self.spline = SplineTransformer(n_knots=3, degree=2,
                    knots='quantile', extrapolation='linear', include_bias=False).fit(z)
            z = self.spline.transform(z)
        return z

    def fit(self, x, y):
        y = np.asarray(y)
        if self.kind in ['constant', 'persistence']:
            delta = y - (x.e_now.to_numpy() if self.kind == 'persistence' else 0)
            self.offset = np.quantile(delta, self.q)
            return self
        self.y_mean, self.y_std = y.mean(), max(y.std(ddof=1), 1e-8)
        z = self.prepare(x, fit=True)
        ys = (y - self.y_mean) / self.y_std
        if self.kind == 'boosting':
            self.model = GradientBoostingRegressor(loss='quantile', alpha=self.q,
                learning_rate=.03, subsample=1., random_state=SEED, **self.params)
        else:
            self.model = QuantileRegressor(quantile=self.q, fit_intercept=True,
                solver='highs', solver_options={'primal_feasibility_tolerance': 1e-8,
                    'dual_feasibility_tolerance': 1e-8}, **self.params)
        self.model.fit(z, ys)
        return self

    def predict(self, x):
        if self.kind == 'constant':
            return np.repeat(self.offset, len(x))
        if self.kind == 'persistence':
            return x.e_now.to_numpy() + self.offset
        return self.y_mean + self.y_std * self.model.predict(self.prepare(x))
