"""A5 shared data, chronology, fixed estimators and paired comparisons."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'latent_factors'))
import models
study = models.study
SOURCE = HERE.parent / 'latent_factors' / 'stabilized'
OUT = HERE / 'results'
METHODS = ['sequential_ols', 'pca_1', 'pca_2', 'ae_selected', 'joint_ols']
PRIMARY = METHODS[:4]
MONTHS = study.pilot.MONTHS
SEED = 20260927


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression={'method':'gzip', 'mtime':0} if str(path).endswith('.gz') else None)


def load(scenario):
    panel, _ = study.build_panel(scenario)
    pred = pd.read_csv(SOURCE/scenario/'predictions.csv.gz', parse_dates=['time'])
    pred = pred.loc[pred.model.isin(PRIMARY)].copy()
    pairs = pd.read_csv(SOURCE/scenario/'residual_pairs.csv.gz', parse_dates=['time','target_time','fit_cutoff'])
    pairs = pairs.loc[pairs.method.isin(PRIMARY)].copy()
    data = panel.dropna(subset=['y','m','g']+study.KCOLS)
    new_pred, new_pairs = [], []
    for month in study.MONTHS:
        cutoff = pd.Timestamp(month+'-01', tz='UTC')
        train = data.loc[data.index < cutoff]
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(train)),train[['m','g']]]), train.y, rcond=None)[0]
        fitted = np.column_stack([np.ones(len(data)),data[['m','g']]])@coef
        e = pd.Series(data.y.to_numpy()-fitted, index=data.index)
        p = pred.loc[(pred.model=='sequential_ols') & (pred.evaluation_month==month)].copy()
        p['model'] = 'joint_ols'; p['candidate_id'] = 'joint_ols'
        p['residual_bp'] = e.reindex(p.time).to_numpy()
        p['fitted_bp'] = p.observed_bp-p.residual_bp
        p['reference_reconstruction_mse'] = np.nan
        new_pred.append(p)
        f = pairs.loc[(pairs.method=='sequential_ols') & (pairs.month==month)].copy()
        f['method'] = 'joint_ols'
        f['e_origin'] = e.reindex(f.time).to_numpy()
        f['target'] = e.reindex(f.target_time).to_numpy()
        new_pairs.append(f)
    return panel, pd.concat([pred]+new_pred, ignore_index=True), pd.concat([pairs]+new_pairs, ignore_index=True)


def quantile_loss(y, p, q):
    e = np.asarray(y)-np.asarray(p)
    return np.maximum(q*e, (q-1)*e)


def estimator(algorithm, q=None):
    if algorithm=='ridge': return make_pipeline(StandardScaler(), Ridge(alpha=1.))
    if algorithm=='linear_qr':
        return make_pipeline(StandardScaler(), QuantileRegressor(alpha=.01, quantile=q, solver='highs'))
    params = dict(n_estimators=100, learning_rate=.03, reg_lambda=1., random_state=SEED,
                  n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)
    if q is None: return lgb.LGBMRegressor(num_leaves=4, max_depth=2, min_child_samples=100, **params)
    return lgb.LGBMRegressor(objective='quantile', alpha=q, num_leaves=7, max_depth=3, min_child_samples=50, **params)


def paired(a, b, value='loss'):
    d = a[['time',value]].merge(b[['time',value]], on='time', validate='one_to_one', suffixes=('_a','_b')).sort_values('time')
    assert len(d)==len(a)==len(b) and d.notna().all().all()
    d['delta'] = d[value+'_a']-d[value+'_b']
    index = pd.DatetimeIndex(d.time)
    result = []
    for block in [3,7]:
        calendar, weights = study.calendar_counts(index,block,np.random.default_rng(SEED+block),2000)
        count = pd.Series(1.,index=index).groupby(index.normalize()).sum().reindex(calendar,fill_value=0).to_numpy()
        sums = pd.Series(d.delta.to_numpy(),index=index).groupby(index.normalize()).sum().reindex(calendar,fill_value=0).to_numpy()
        draws = (weights@sums)/(weights@count)
        result.append(dict(block_days=block,n=len(d),delta=float(d.delta.mean()),
            relative_pct=100*float(d.delta.mean())/float(d[value+'_b'].mean()),
            ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975))))
    return result
