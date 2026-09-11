"""Explicit quote units, exact clock alignment, and a common quantile solver.

The primary risk definition uses hourly USD-converted BTC log returns. Archived
USDT-return risk measures are available ONLY as a labelled diagnostic option.
No interpolation or forward filling is performed.
"""
import hashlib
import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack

COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']
DERIVATIVES = ['FUNDING', 'OI', 'OI_VALUE', 'TOPTRADER_ACCOUNT_LS',
               'TOPTRADER_POSITION_LS', 'ACCOUNT_LS', 'TAKER_LONG_SHORT_VOL_RATIO']
MACRO = ['USDKRW', 'DXY', 'VIX', 'SPY', 'XAUUSD']
BX = ['RESIDUAL_FINAL_lag1', 'BTC_DOWNSIDE_24_lag1', 'DXY_ret_lag1', 'USDKRW_ret_lag1']
GLOBAL = ['BTC_VOL_24_lag1', 'BTC_RET_lag1', 'VIX_ret_lag1']
SUPPLY = {'BTC': 19.9e6, 'ETH': 120.5e6, 'XRP': 59.5e9, 'SOL': 590e6, 'DOGE': 149e9}


def check_time(index):
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError('Timezone-aware timestamps are required; identify the raw timezone first.')
    if index.has_duplicates or index.hasnans:
        raise ValueError('Duplicate or missing timestamps are not permitted.')
    if not (index == index.floor('h')).all():
        raise ValueError('Input rows must identify exact hourly timestamps.')


def at_hour(series, hours):
    """At row t, return the observation at t+hours; unmatched timestamps are NaN."""
    check_time(series.index)
    out = series.reindex(series.index + pd.Timedelta(hours=hours)).copy()
    out.index = series.index
    return out


def correct_quotes(raw):
    d = raw.copy()
    required = ['USDT_BINANCE_CLOSE', 'USDT_UPBIT_CLOSE', 'USDKRW']
    required += [c+s for c in COINS for s in ['_BINANCE_CLOSE', '_UPBIT_CLOSE']]
    missing = set(required) - set(d.columns)
    if missing:
        raise ValueError('Missing raw price columns: ' + ', '.join(sorted(missing)))
    if ((d[required] <= 0) & d[required].notna()).any().any():
        raise ValueError('Observed prices and exchange rates must be strictly positive.')
    # q: USDT per USDC; a: USD per USDC (default assumption documented in metadata).
    q = d['USDT_BINANCE_CLOSE']
    a = d['USDC_USD_CLOSE'] if 'USDC_USD_CLOSE' in d else pd.Series(1., index=d.index)
    if ((a <= 0) & a.notna()).any():
        raise ValueError('USDC/USD prices must be positive.')
    d['USDT_USD_CLOSE'] = a / q
    for c in COINS:
        d[c+'_USD_CLOSE'] = d[c+'_BINANCE_CLOSE'] * a / q
        d[c+'_KP'] = d[c+'_UPBIT_CLOSE'] / (d['USDKRW'] * d[c+'_USD_CLOSE']) - 1
    d['USDT_KP'] = d.USDT_UPBIT_CLOSE / (d.USDKRW * d.USDT_USD_CLOSE) - 1
    # Require all five assets; an unannounced changing basket is not equal weight.
    d['MKT_KP_EQ'] = d[[c+'_KP' for c in COINS]].mean(axis=1, skipna=False)
    d['DEPEG_GLOBAL'] = (q-1).abs()  # Relative stablecoin-pair proxy, not all USD depeg risk.
    for c in COINS + ['USDT']:
        d[c+'_KP_pct'] = d[c+'_KP']*100
    d['MKT_KP_EQ_pct'] = d.MKT_KP_EQ*100
    return d


def residualize(y, x):
    x = x.to_frame() if isinstance(x, pd.Series) else x
    joined = pd.concat([y.rename('__y'), x], axis=1).dropna()
    result = pd.Series(np.nan, index=y.index, dtype=float)
    if len(joined) <= x.shape[1] + 1:
        return result
    X = np.column_stack([np.ones(len(joined)), joined.iloc[:, 1:].to_numpy()])
    yv = joined.iloc[:, 0].to_numpy()
    result.loc[joined.index] = yv-X@np.linalg.lstsq(X, yv, rcond=None)[0]
    return result


def prepare_data(raw, risk_source='hourly_usd', projection='sequential', market='EQ'):
    check_time(raw.index)
    raw = raw.sort_index().copy()
    raw.index = raw.index.tz_convert('UTC')
    if risk_source not in ['hourly_usd', 'archived_usdt']:
        raise ValueError('Unknown risk source')
    if projection not in ['sequential', 'joint'] or market not in ['EQ', 'CAP', 'PCA']:
        raise ValueError('Unknown residual or market definition')
    # Whitelist source observations. Never let old premiums, lags, or returns survive.
    observed = [c+s for c in COINS+['USDT'] for s in ['_BINANCE_CLOSE', '_UPBIT_CLOSE']]
    observed += MACRO + DERIVATIVES + ['USDC_USD_CLOSE']
    d = correct_quotes(raw[[c for c in observed if c in raw]])
    kpcols = [c+'_KP' for c in COINS]
    market_col = 'MKT_KP_EQ'
    pca_share = None
    if market == 'CAP':
        cap = pd.DataFrame({c:d[c+'_USD_CLOSE']*SUPPLY[c] for c in COINS})
        weights = cap.div(cap.sum(axis=1, skipna=False), axis=0)
        d['MKT_KP_CAP'] = (weights.to_numpy()*d[kpcols].to_numpy()).sum(axis=1)
        market_col = 'MKT_KP_CAP'
    if market == 'PCA':
        k = d[kpcols].dropna()
        z = (k-k.mean())/k.std()
        ev, vectors = np.linalg.eigh(z.corr().to_numpy())
        v = vectors[:, -1]
        if v.sum() < 0:
            v = -v
        d.loc[k.index, 'MKT_KP_PCA'] = z.to_numpy()@v
        pca_share = float(ev[-1]/ev.sum())
        market_col = 'MKT_KP_PCA'
    d['RESIDUAL_RAW'] = d.USDT_KP-d[market_col]
    d['RESIDUAL_RAW_pct'] = d.RESIDUAL_RAW*100
    d['RESIDUAL_OLS'] = residualize(d.USDT_KP, d[market_col])
    if projection == 'sequential':
        d['RESIDUAL_FINAL'] = residualize(d.RESIDUAL_OLS, d.DEPEG_GLOBAL)
    else:
        d['RESIDUAL_FINAL'] = residualize(d.USDT_KP, d[[market_col, 'DEPEG_GLOBAL']])

    for c in MACRO + ['OI', 'OI_VALUE']:
        if c in d:
            d[c+'_ret' if c in MACRO else c+'_RET'] = np.log(d[c].where(d[c] > 0))-np.log(at_hour(d[c], -1).where(lambda x:x > 0))
    if risk_source == 'hourly_usd':
        full_index = pd.date_range(d.index.min(), d.index.max(), freq='h')
        prices = d.BTC_USD_CLOSE.reindex(full_index)
        ret = np.log(prices).diff()
        d['BTC_RET'] = ret.reindex(d.index)
        d['BTC_VOL_24'] = ret.rolling(24, min_periods=24).std(ddof=1).reindex(d.index)
        d['BTC_DOWNSIDE_24'] = ret.clip(upper=0).pow(2).rolling(24, min_periods=24).mean().reindex(d.index)
    else:
        for c in ['BTC_VOL_24', 'BTC_DOWNSIDE_24']:
            if c not in raw:
                raise ValueError('Archived USDT-risk diagnostic needs '+c)
            d[c] = raw[c]
        d['BTC_RET'] = np.log(d.BTC_BINANCE_CLOSE)-np.log(at_hour(d.BTC_BINANCE_CLOSE, -1))
    lagged = ['RESIDUAL_FINAL', 'BTC_RET', 'BTC_VOL_24', 'BTC_DOWNSIDE_24']
    lagged += [c+'_ret' for c in MACRO]+DERIVATIVES+['OI_RET', 'OI_VALUE_RET']
    for c in lagged:
        if c in d:
            d[c+'_lag1'] = at_hour(d[c], -1)
    threshold = d.BTC_DOWNSIDE_24.quantile(.90)
    d['DOWNSIDE'] = (d.BTC_DOWNSIDE_24 >= threshold).where(d.BTC_DOWNSIDE_24.notna()).astype('Float64')
    d['tick_krw'] = tick_krw(d.USDT_UPBIT_CLOSE)
    d['day'] = d.index.normalize()
    metadata = {
        'n_input':len(raw), 'n_days':raw.index.normalize().nunique(),
        'risk_source':risk_source, 'projection':projection, 'market':market,
        'pca_explained_share':pca_share,
        'usd_conversion':'observed USDC/USD' if 'USDC_USD_CLOSE' in raw else 'USDC=1 USD assumption',
        'valid_24h_risk_rows':int(d.BTC_DOWNSIDE_24.notna().sum()),
        'lag_rule':'exact UTC timestamp t-1 hour; no interpolation',
        'lp_rule':'predictors measured at s=t-1 hour; target residual at s+h hours',
        'macro_provenance':'inherited archive labels/timezones; VIX versus VIXY unresolved',
        'cap_supply':'unverified fixed mid-2025 reference supplies' if market=='CAP' else None,
        'publication_ready':False,
    }
    return d, metadata


def tick_krw(price):
    """Verified sample price band only; uncertain transition dates remain missing."""
    dates = price.index.tz_convert('Asia/Seoul').strftime('%Y-%m-%d')
    out = pd.Series(np.nan, index=price.index)
    old = (dates > '2025-03-21') & (dates < '2025-07-31') & price.between(1000, 9999.999)
    new = (dates > '2025-07-31') & price.between(1000, 4999.999)
    out.loc[old] = .5
    out.loc[new] = 1.
    return out


def quantile_fit(X, y, tau):
    """Solve check loss by scaled linear programming; first column is intercept."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if not 0 < tau < 1 or len(y) <= X.shape[1]:
        raise ValueError('Invalid quantile or insufficient observations')
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.allclose(X[:,0], 1):
        raise ValueError('Finite design with intercept in first column required')
    xm, xs = X[:,1:].mean(0), X[:,1:].std(0, ddof=1)
    if (xs == 0).any():
        raise ValueError('Constant explanatory variable')
    z = np.column_stack([np.ones(len(y)), (X[:,1:]-xm)/xs])
    if np.linalg.matrix_rank(z) != z.shape[1]:
        raise ValueError('Rank-deficient quantile design')
    ym, ys = y.mean(), y.std(ddof=1)
    if ys == 0:
        return np.r_[ym, np.zeros(X.shape[1]-1)]
    n, k = z.shape
    fit = linprog(np.r_[np.zeros(k), np.full(n,tau), np.full(n,1-tau)],
                  A_eq=hstack([csr_matrix(z),eye(n),-eye(n)],format='csr'),
                  b_eq=(y-ym)/ys, bounds=[(None,None)]*k+[(0,None)]*(2*n),
                  method='highs', options={'primal_feasibility_tolerance':1e-9,
                                          'dual_feasibility_tolerance':1e-9})
    if not fit.success:
        raise RuntimeError('Quantile optimization failed: '+fit.message)
    slopes = fit.x[1:k]*ys/xs
    return np.r_[ym+ys*fit.x[0]-xm@slopes, slopes]


def bootstrap_model(d, yc, xc, q, targets, B, block_days, seed, model_id, standardized=False):
    cols = [yc]+xc
    sub = d[cols].replace([np.inf,-np.inf],np.nan).dropna()
    if len(sub) < max(40, 5*(len(xc)+1)) or sub.index.normalize().nunique() < 10:
        return [], [], {'model':model_id,'q':q,'n':len(sub),'status':'insufficient_data'}
    z = (sub-sub.mean())/sub.std() if standardized else sub
    X = np.column_stack([np.ones(len(z)),z[xc].to_numpy()])
    y = z[yc].to_numpy()
    point = quantile_fit(X,y,q)
    days = sub.index.normalize()
    unique = days.unique()
    groups = [np.flatnonzero(days==day) for day in unique]
    # Stable per-model stream; adding/reordering other models cannot change this model.
    key = '{}:{}:{}:{}'.format(seed,model_id,q,block_days).encode()
    rng = np.random.RandomState(int(hashlib.sha256(key).hexdigest()[:8],16))
    draws, errors = [], []
    for rep in range(B):
        starts = rng.randint(0,len(groups),int(math.ceil(len(groups)/block_days)))
        chosen = np.concatenate([(s+np.arange(block_days))%len(groups) for s in starts])[:len(groups)]
        rows = np.concatenate([groups[i] for i in chosen])
        try:
            b = quantile_fit(X[rows],y[rows],q)
            draws.append((rep,b))
        except (ValueError,RuntimeError) as exc:
            errors.append({'rep':rep,'error':str(exc)})
    if len(draws) < .95*B:
        raise RuntimeError('{}: more than 5% of bootstrap fits failed: {}'.format(model_id,errors[:3]))
    rows, saved = [], []
    for target in targets:
        j = xc.index(target)+1
        a = np.array([b[j] for _,b in draws])
        low90,high90 = np.quantile(a,[.05,.95])
        low95,high95 = np.quantile(a,[.025,.975])
        # Count exact zeros in both tails; never turn all-zero draws into p=0.
        p = min(1.,2*(min((a<=0).sum(),(a>=0).sum())+1)/(len(a)+1))
        rows.append({'model':model_id,'q':q,'target':target,'n':len(sub),'beta':point[j],
                     'p_sign_plus_one':p,'ci90_low':low90,'ci90_high':high90,
                     'ci95_low':low95,'ci95_high':high95,'B_requested':B,'B_success':len(a),
                     'block_days':block_days,'standardized':standardized})
        saved.extend({'model':model_id,'q':q,'target':target,'rep':rep,'beta':b[j]}
                     for rep,b in draws)
    return rows,saved,{'model':model_id,'q':q,'n':len(sub),'status':'estimated',
                       'bootstrap_failures':errors,'standardization':'fixed original sample' if standardized else 'original units',
                       'inference_scope':'conditional on estimated residuals and risk features; first stages not bootstrapped'}
