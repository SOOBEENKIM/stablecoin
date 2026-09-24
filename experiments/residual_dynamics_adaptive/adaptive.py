"""Post-result, bounded extensions of the existing residual forecasting study."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'residual_dynamics_ml'
sys.path.insert(0, str(LEGACY))
from data import (BASE, POSITION, ROOT, INPUT_PATHS, Residualizer, build_panel,
                  make_design, macro, past_observation, lead, sha256)
from models import ForecastModel, GRIDS, pinball, SEED

HISTORY = ['account_change1', 'account_change24', 'funding_change8', 'oi_change24']
MACRO_FEATURES = ['vix_known_log', 'vix_known_age_hours', 'dxy_known_age_hours',
                  'hour_sin', 'hour_cos', 'weekend']
QRF_GRID = [dict(max_depth=5, min_samples_leaf=20, max_features=1.),
            dict(max_depth=None, min_samples_leaf=20, max_features=.7),
            dict(max_depth=None, min_samples_leaf=40, max_features=1.)]
FOLDS = pd.date_range('2025-11-01', '2026-03-01', freq='MS', tz='UTC')
EVALUATION_START = pd.Timestamp('2025-12-01', tz='UTC')


def panel(mode):
    p = build_panel('NY_delay1')
    p['account_change1'] = p.account_log - lead(p.account_log, -1)
    p['account_change24'] = p.account_log - lead(p.account_log, -24)
    p['funding_change8'] = p.funding_bp - lead(p.funding_bp, -8)
    loi = np.log(p.oi.where(p.oi > 0))
    p['oi_change24'] = (loi - lead(loi, -24)) * 1e4
    if mode == 'available_macro':
        for col, file in [('vix', 'VIXY.csv'), ('dxy', 'DXY.csv')]:
            known, age = past_observation(macro(file, 'America/New_York', 1), p.index, '96h')
            p[col + '_ret1'] = np.log(known).diff() * 1e4
            p[col + '_known_age_hours'] = age / 60
            if col == 'vix':
                p['vix_known_log'] = np.log(known)
        kst = p.index.tz_convert('Asia/Seoul')
        p['hour_sin'] = np.sin(kst.hour * np.pi / 12)
        p['hour_cos'] = np.cos(kst.hour * np.pi / 12)
        p['weekend'] = (kst.dayofweek >= 5).astype(float)
    elif mode != 'strict':
        raise ValueError(mode)
    return p.replace([np.inf, -np.inf], np.nan)


def frame(p, r, mode, h=1):
    d = make_design(p, r, h)
    extra = HISTORY + (MACRO_FEATURES if mode == 'available_macro' else [])
    d = d.join(p[extra])
    return d.dropna(subset=extra)


def columns(mode, info):
    return (BASE + (MACRO_FEATURES if mode == 'available_macro' else []) +
            ([] if info == 'base' else POSITION) + (HISTORY if info == 'history' else []))


def splits(p, mode, definition, cutoff):
    vstart = cutoff - pd.offsets.MonthBegin(1)
    ri = Residualizer(definition).fit(p, vstart)
    ro = Residualizer(definition).fit(p, cutoff)
    di, do = frame(p, ri, mode), frame(p, ro, mode)
    ti = di.loc[di.target_time < vstart]
    va = di.loc[(di.index >= vstart) & (di.target_time < cutoff)]
    tr = do.loc[do.target_time < cutoff]
    te = do.loc[(do.index >= cutoff) & (do.index < cutoff + pd.offsets.MonthBegin(1))]
    assert ti.target_time.max() < va.index.min()
    assert tr.target_time.max() < te.index.min()
    assert ri.fit_end < vstart and ro.fit_end < cutoff
    return ti, va, tr, te, ri, ro


def window(d, cutoff, days):
    return d if days == 'all' else d.loc[d.index >= cutoff - pd.Timedelta(days=int(days))]


class QuantileForest:
    """Leaf-neighbour weighted empirical CDF, not a quantile of tree means.

    Trees are grown on bootstrap samples; terminal-leaf distributions contain
    all original training observations routed through the fitted partition.
    """
    def __init__(self, q=.1, **params):
        self.q = q
        self.forest = RandomForestRegressor(n_estimators=150, bootstrap=True,
            random_state=SEED, n_jobs=1, **params)

    def fit(self, x, y):
        a = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.order = np.argsort(self.y, kind='stable')
        self.forest.fit(a, self.y)
        leaves = self.forest.apply(a)
        self.members = [{int(leaf): np.flatnonzero(leaves[:, k] == leaf)
            for leaf in np.unique(leaves[:, k])} for k in range(leaves.shape[1])]
        return self

    def weights(self, x):
        leaves = self.forest.apply(np.asarray(x, dtype=float))
        w = np.zeros((len(leaves), len(self.y)))
        for k, members in enumerate(self.members):
            for leaf in np.unique(leaves[:, k]):
                ii = np.flatnonzero(leaves[:, k] == leaf)
                jj = members[int(leaf)]
                w[np.ix_(ii, jj)] += 1. / (len(jj) * len(self.members))
        np.testing.assert_allclose(w.sum(axis=1), 1., atol=1e-12)
        return w

    def predict(self, x):
        w = self.weights(x)[:, self.order]
        cdf = np.cumsum(w, axis=1)
        # Preserve equality at an exact empirical-quantile boundary despite
        # floating-point accumulation across trees and training observations.
        idx = (cdf >= self.q - 1e-12).argmax(axis=1)
        return self.y[self.order[idx]]


class DynamicModel:
    def __init__(self, kind, target_form, params):
        self.kind, self.target_form, self.params = kind, target_form, params

    def fit(self, d, cols):
        self.cols = list(cols)
        y = d.target.to_numpy() - (d.e_now.to_numpy() if self.target_form == 'change' else 0)
        if self.kind == 'qrf':
            self.model = QuantileForest(q=.1, **self.params).fit(d[cols], y)
        else:
            self.model = ForecastModel(self.kind, .1, self.params).fit(d[cols], y)
        return self

    def predict(self, d):
        y = self.model.predict(d[self.cols])
        return y + (d.e_now.to_numpy() if self.target_form == 'change' else 0)


def candidates(kind):
    grid = QRF_GRID if kind == 'qrf' else GRIDS[kind]
    return [(days, form, params) for days in ['all', 90]
            for form in ['level', 'change'] for params in grid]


def calibrate(g, size=60, min_size=30, max_days=90):
    """Use only labels that arrived strictly before the forecast origin.

    The history contains RAW forecast errors. Current/future outcomes cannot
    affect the current correction; no exchangeability guarantee is claimed.
    """
    g = g.sort_values('origin').reset_index(drop=True).copy()
    origins = pd.DatetimeIndex(g.origin)
    targets = pd.DatetimeIndex(g.target_time)
    error = g.target.to_numpy() - g.pred_raw.to_numpy()
    correction, ns, latest = [], [], []
    j = 0
    for i, t in enumerate(origins):
        while j < len(g) and targets[j] < t:
            j += 1
        eligible = np.arange(max(0, j - size), j)
        eligible = eligible[targets[eligible] >= t - pd.Timedelta(days=max_days)]
        assert (eligible < i).all()
        n = len(eligible)
        correction.append(float(np.quantile(error[eligible], .1, method='inverted_cdf'))
                          if n >= min_size else 0.)
        ns.append(n)
        latest.append(targets[eligible[-1]] if n else pd.NaT)
    g['correction'] = correction
    g['calibration_n'] = ns
    g['latest_calibration_label'] = latest
    g['pred_calibrated'] = g.pred_raw + g.correction
    if len(g[g.calibration_n >= min_size]):
        mask = g.calibration_n >= min_size
        assert (g.loc[mask, 'latest_calibration_label'] < g.loc[mask, 'origin']).all()
    return g
