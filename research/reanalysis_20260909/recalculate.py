"""Controlled quote/clock sensitivity; all timezone scenarios are conditional.

No original files are modified. Clock scenarios use crypto candle END times,
backward-only macro observations younger than one hour, and exact clock leads.
Original sequential residual definition and BTC/USDT downside risk are retained.
Bootstrap refits both residual stages using calendar-block frequency weights;
clock links are formed before resampling, never across artificial block joins.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / 'data/raw data'
COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']
BASE = ['e_origin', 'downside', 'dxy_ret', 'fx_ret']
GLOBAL = ['btc_vol', 'btc_ret', 'vix_ret']
EXTRA = ['account_ls']
MODELS = {'M0': BASE, 'M1': BASE + GLOBAL, 'M2': BASE + GLOBAL + EXTRA}
SCENARIOS = {
    'archived': None,
    'quote_only': None,
    'clock_UTC_oldquote': ('UTC', 'UTC', 0, False),
    'clock_UTC_quote': ('UTC', 'UTC', 0, True),
    'clock_NY_quote': ('America/New_York', 'America/New_York', 0, True),
    'clock_FXUTC_USNY_quote': ('UTC', 'America/New_York', 0, True),
    'clock_FXKST_USNY_quote': ('Asia/Seoul', 'America/New_York', 0, True),
    'clock_NY_macro_end1h_quote': ('America/New_York', 'America/New_York', 1, True),
}


def read_panel(path):
    d = pd.read_csv(path)
    d['datetime_utc'] = pd.to_datetime(d['datetime_utc'], utc=True, format='mixed')
    return d.set_index('datetime_utc').sort_index()


def past_observation(series, grid, tolerance):
    """Return last actually observed value; no backfill from the future."""
    s = series.dropna().sort_index()
    if s.index.has_duplicates:
        raise ValueError('Duplicate source timestamps')
    left = pd.DataFrame({'t': grid})
    right = pd.DataFrame({'observed_at': s.index, 'value': s.values})
    m = pd.merge_asof(left, right, left_on='t', right_on='observed_at',
                      direction='backward', tolerance=pd.Timedelta(tolerance))
    age = (m.t - m.observed_at).dt.total_seconds().to_numpy() / 60
    assert np.all(age[np.isfinite(age)] >= 0)
    return pd.Series(m.value.to_numpy(), index=grid), pd.Series(age, index=grid)


def macro(name, zone, end_delay):
    d = pd.read_csv(RAW/name, header=None, names=['time', 'price'])
    d['time'] = pd.to_datetime(d.time, errors='coerce')
    d['price'] = pd.to_numeric(d.price, errors='coerce')
    d = d.dropna()
    idx = pd.DatetimeIndex(d.time).tz_localize(zone, ambiguous='raise', nonexistent='raise').tz_convert('UTC')
    return pd.Series(d.price.to_numpy(), index=idx + pd.Timedelta(hours=end_delay))


def premiums(d, corrected):
    q, fx = d.USDT_BINANCE_CLOSE, d.USDKRW
    if corrected:
        y = d.USDT_UPBIT_CLOSE*q/fx - 1
        kp = pd.DataFrame({c: d[c+'_UPBIT_CLOSE']*q/(fx*d[c+'_BINANCE_CLOSE'])-1 for c in COINS})
    else:
        y = d.USDT_UPBIT_CLOSE/(fx*q)-1
        kp = pd.DataFrame({c: d[c+'_UPBIT_CLOSE']/(fx*d[c+'_BINANCE_CLOSE'])-1 for c in COINS})
    return y, kp.mean(axis=1, skipna=False), (q-1).abs()


def weighted_ols(y, X, w):
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1) & (w > 0)
    z = X[valid]
    sw = np.sqrt(w[valid])
    if len(z) <= z.shape[1]:
        raise ValueError('Insufficient residualization data')
    b = np.linalg.lstsq(z*sw[:,None], y[valid]*sw, rcond=None)[0]
    return b


def residuals(panel, weights=None, joint=False):
    y = panel['y'].to_numpy()
    m = panel['m'].to_numpy()
    g = panel['g'].to_numpy()
    w = np.ones(len(panel)) if weights is None else weights
    fit = np.isfinite(y) & np.isfinite(m) & np.isfinite(g)
    w = w * fit
    if joint:
        X = np.column_stack([np.ones(len(panel)), m, g])
        b = weighted_ols(y, X, w)
        return y-X@b
    X = np.column_stack([np.ones(len(panel)), m])
    b = weighted_ols(y, X, w)
    u = y-X@b
    G = np.column_stack([np.ones(len(panel)), g])
    c = weighted_ols(u, G, w)
    return u-G@c


def build(name):
    cfg = SCENARIOS[name]
    if cfg is None:
        d = read_panel(ROOT/'corrected_outputs/baseline_dataset_with_residual_final.csv')
        y, m, g = premiums(d, name == 'quote_only')
        p = pd.DataFrame({'y': y, 'm': m, 'g': g,
            'downside': d.BTC_DOWNSIDE_24, 'btc_vol': d.BTC_VOL_24,
            'btc_ret': d.BTC_RET, 'dxy_ret': d.DXY_ret,
            'fx_ret': d.USDKRW_ret, 'vix_ret': d.VIX_ret,
            'account_ls': d.ACCOUNT_LS}, index=d.index)
        meta = {'time_rule': 'archived candle labels and irregular row lags',
                'quote_corrected': name == 'quote_only', 'timezone_confirmed': False}
    else:
        fxzone, uszone, delay, corrected = cfg
        b = read_panel(RAW/'binance_1h_2025-06-01_2026-03-19.csv')
        u = read_panel(RAW/'upbit_1h_2025-06-01_2026-03-19.csv')
        d = b.join(u)
        d.index += pd.Timedelta(hours=1)
        assert (d.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
        ages = {}
        for col, file, zone in [('USDKRW','USDKRW.csv',fxzone), ('DXY','DXY.csv',uszone), ('VIX','VIXY.csv',uszone)]:
            d[col], age = past_observation(macro(file, zone, delay), d.index, '59min59s')
            ages[col] = {'n': int(d[col].notna().sum()), 'max_age_minutes': float(age.max())}
        deriv = read_panel(ROOT/'corrected_outputs/binance_btcusdt_data_vision_metrics.csv')
        # Conservative availability delay for a five-minute metric record.
        ls = deriv.ACCOUNT_LS.copy()
        ls.index += pd.Timedelta(minutes=5)
        d['ACCOUNT_LS'], _ = past_observation(ls, d.index, '10min')
        ret = np.log(d.BTC_BINANCE_CLOSE).diff()
        y, m, g = premiums(d, corrected)
        p = pd.DataFrame({'y': y, 'm': m, 'g': g,
            'downside': ret.clip(upper=0).pow(2).rolling(24, min_periods=24).mean(),
            'btc_vol': ret.rolling(24, min_periods=24).std(ddof=1), 'btc_ret': ret,
            'dxy_ret': np.log(d.DXY).diff(), 'fx_ret': np.log(d.USDKRW).diff(),
            'vix_ret': np.log(d.VIX).diff(), 'account_ls': d.ACCOUNT_LS}, index=d.index)
        meta = {'time_rule': 'crypto candle ends, backward macro age <60min, exact t+h targets',
                'fx_timezone_assumption': fxzone, 'vix_dxy_timezone_assumption': uszone,
                'macro_additional_availability_delay_hours': delay, 'macro_ages': ages,
                'metric_availability_delay_minutes': 5, 'quote_corrected': corrected,
                'timezone_confirmed': False}
    p['e'] = residuals(p)
    valid = p[['y','m','g']].dropna()
    X = np.column_stack([np.ones(len(valid)), valid.m])
    b = np.linalg.lstsq(X, valid.y, rcond=None)[0]
    r2 = 1 - np.square(valid.y-X@b).sum()/np.square(valid.y-valid.y.mean()).sum()
    meta.update({'scenario':name,'rows':len(p),'valid_residuals':int(p.e.notna().sum()),
                 'residual_std_bp':float(p.e.std()*1e4), 'market_r2':float(r2),
                 'residual_market_corr':float(p.e.corr(p.m)),
                 'risk_definition':'24 consecutive hourly BTC/USDT log returns, downside mean squared negative returns',
                 'projection':'original sequential OLS', 'usdc_usd_assumption':1.0,
                 'publication_ready':False})
    return p, meta


def links(p, name, h):
    n = len(p)
    if SCENARIOS[name] is None:
        # h=1 is original contemporaneous outcome on lagged predictors.
        # h>1 reproduces the paper's LP shift(-h), NOT h elapsed hours.
        origins = np.arange(n)-1
        targets = np.arange(n)+(0 if h == 1 else h)
    else:
        origins = np.arange(n)
        targets = p.index.get_indexer(p.index + pd.Timedelta(hours=h))
    mask = (origins >= 0) & (origins < n) & (targets >= 0) & (targets < n)
    return np.flatnonzero(mask), origins[mask], targets[mask]


def design(p, name, h, model, e=None, common=True):
    rows, oi, ti = links(p,name,h)
    ev = p.e.to_numpy() if e is None else e
    z = p.iloc[oi][BASE[1:]+GLOBAL+EXTRA].reset_index(drop=True)
    z.insert(0,'e_origin',ev[oi])
    z['target'] = ev[ti]
    needed = BASE+GLOBAL+EXTRA if common else MODELS[model]
    good = np.isfinite(z[['target']+needed]).all(axis=1).to_numpy()
    z = z.loc[good]
    return z[['target']+MODELS[model]].to_numpy(), rows[good], oi[good], ti[good]


def quantile_fit(X, y, q, weights=None):
    w = np.ones(len(y)) if weights is None else weights
    keep = w > 0
    X,y,w = X[keep],y[keep],w[keep]
    xm, xs = X[:,1:].mean(0), X[:,1:].std(0,ddof=1)
    if np.any(xs < 1e-16):
        raise ValueError('Constant predictor')
    ym, ys = y.mean(), y.std(ddof=1)
    if ys < 1e-16:
        raise ValueError('Constant target')
    Z = np.column_stack([np.ones(len(y)), (X[:,1:]-xm)/xs])
    n,k = Z.shape
    fit = linprog(np.r_[np.zeros(k),q*w,(1-q)*w],
        A_eq=hstack([csr_matrix(Z),eye(n),-eye(n)],format='csr'), b_eq=(y-ym)/ys,
        bounds=[(None,None)]*k+[(0,None)]*(2*n), method='highs',
        options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not fit.success:
        raise RuntimeError(fit.message)
    slopes = fit.x[1:k]*ys/xs
    return np.r_[ym+ys*fit.x[0]-xm@slopes,slopes]


def fit_design(a, q, weights=None, standard=None):
    mu, sd = (a.mean(0), a.std(0,ddof=1)) if standard is None else standard
    z = (a-mu)/sd
    b = quantile_fit(np.column_stack([np.ones(len(z)),z[:,1:]]),z[:,0],q,weights)
    return b, (mu,sd)


def point_results():
    out, metas, elapsed = [], [], []
    for name in SCENARIOS:
        p,meta = build(name)
        p.to_csv(HERE/(name+'_panel.csv'))
        metas.append(meta)
        for h in [1,3,6,12]:
            for model in MODELS:
                a,rr,oi,ti = design(p,name,h,model)
                if len(a)<max(40,5*(a.shape[1])):
                    out.append({'scenario':name,'h':h,'model':model,'n':len(a),'status':'insufficient'})
                    continue
                gap = (p.index[ti]-p.index[oi]).total_seconds()/3600
                elapsed.append({'scenario':name,'h':h,'model':model,'n':len(a),
                    'min_hours':min(gap),'median_hours':np.median(gap),'max_hours':max(gap)})
                if SCENARIOS[name] is not None:
                    assert np.all(gap == h)
                for q in [.1,.25,.5,.75,.9] if h==1 else [.1,.5,.9]:
                    b,_ = fit_design(a,q)
                    for j,c in enumerate(MODELS[model],1):
                        if c not in ['downside','btc_vol','account_ls','e_origin']:
                            continue
                        raw = b[j]*a[:,0].std(ddof=1)/a[:,j].std(ddof=1)
                        out.append({'scenario':name,'h':h,'model':model,'n':len(a),
                            'q':q,'target':c,'beta_standardized':b[j], 'beta_raw':raw,
                            'effect_bp_per_predictor_sd':b[j]*a[:,0].std(ddof=1)*1e4,
                            'status':'estimated','h_is_elapsed_hours':SCENARIOS[name] is not None})
        print(name,meta['valid_residuals'],meta['residual_std_bp'],flush=True)
    pd.DataFrame(out).to_csv(HERE/'point_estimates.csv',index=False)
    pd.DataFrame(elapsed).to_csv(HERE/'elapsed_and_sample_counts.csv',index=False)
    (HERE/'scenario_metadata.json').write_text(json.dumps(metas,indent=2))
    files = list(RAW.glob('*.csv'))+[ROOT/'corrected_outputs/baseline_dataset_with_residual_final.csv',ROOT/'corrected_outputs/binance_btcusdt_data_vision_metrics.csv']
    (HERE/'input_manifest.json').write_text(json.dumps({'python':platform.python_version(),
        'numpy':np.__version__,'pandas':pd.__version__,
        'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2))


def bootstrap(name, B, block_days, horizons, refit=True, joint=False):
    p, meta = build(name)
    if joint:
        p['e'] = residuals(p,joint=True)
    grid = pd.date_range(p.index.min().normalize(),p.index.max().normalize(),freq='D')
    day_i = grid.get_indexer(p.index.normalize())
    rng = np.random.RandomState(20260909)
    cases = {}
    for h in horizons:
        for model in MODELS:
            a, rr, oi, ti = design(p,name,h,model)
            if len(a) < max(60,8*a.shape[1]):
                continue
            b, standard = fit_design(a,.1)
            cases[(h,model)] = {'a':a,'rows':rr,'oi':oi,'ti':ti,'point':b,
                                 'standard':standard,'draws':[],'errors':[]}
    start = time.time()
    for rep in range(B):
        starts = rng.randint(0,len(grid),int(np.ceil(len(grid)/block_days)))
        selected = np.concatenate([(s+np.arange(block_days))%len(grid) for s in starts])[:len(grid)]
        counts = np.bincount(selected,minlength=len(grid))
        pw = counts[day_i].astype(float)
        try:
            e = residuals(p,pw,joint=joint) if refit else p.e.to_numpy()
        except Exception as exc:
            for c in cases.values():c['errors'].append({'rep':rep,'error':str(exc)})
            continue
        for (h,model),c in cases.items():
            a = c['a'].copy()
            a[:,0] = e[c['ti']]
            a[:,1] = e[c['oi']]
            try:
                b,_ = fit_design(a,.1,pw[c['rows']],c['standard'])
                c['draws'].append((rep,b))
            except Exception as exc:
                c['errors'].append({'rep':rep,'error':str(exc)})
        if (rep+1)%25==0:
            print(name,block_days,rep+1,'elapsed',round(time.time()-start,1),flush=True)
    records, draws, failures = [], [], []
    for (h,model),c in cases.items():
        if len(c['draws']) < .95*B:
            raise RuntimeError('Excess bootstrap failures: '+str(c['errors'][:3]))
        arr = np.array([b for _,b in c['draws']])
        for j,col in enumerate(MODELS[model],1):
            if col not in ['downside','btc_vol','account_ls','e_origin']:continue
            v = arr[:,j]
            lo,hi = np.quantile(v,[.025,.975])
            p_sign = min(1.,2*(min((v<=0).sum(),(v>=0).sum())+1)/(len(v)+1))
            records.append({'scenario':name,'h':h,'model':model,'q':.1,'target':col,
                'n':len(c['a']),'beta_standardized':c['point'][j],'ci95_low':lo,'ci95_high':hi,
                'bootstrap_sign_tail':p_sign,'B':B,'successful':len(v),'block_calendar_days':block_days,
                'refit_residual_stages':refit,'projection':'joint' if joint else 'sequential',
                'note':'percentile sensitivity intervals; sign-tail statistic is not a calibrated null-imposed p-value'})
            draws.extend({'h':h,'model':model,'target':col,'rep':rep,'beta':b[j]} for rep,b in c['draws'])
        failures.extend({'h':h,'model':model,**err} for err in c['errors'])
    suffix='{}_B{}_d{}_{}_{}'.format(name,B,block_days,'refit' if refit else 'fixed','joint' if joint else 'seq')
    pd.DataFrame(records).to_csv(HERE/('bootstrap_'+suffix+'.csv'),index=False)
    pd.DataFrame(draws).to_csv(HERE/('draws_'+suffix+'.csv'),index=False)
    (HERE/('failures_'+suffix+'.json')).write_text(json.dumps(failures,indent=2))
    print(pd.DataFrame(records).query("target == 'account_ls'").to_string(index=False),flush=True)


def selfcheck():
    # A sparse row successor must never become a one-hour successor.
    idx = pd.to_datetime(['2025-01-01 10:00Z','2025-01-01 11:00Z','2025-01-02 10:00Z'])
    p = pd.DataFrame(index=idx)
    _,oi,ti = links(p,'clock_UTC_quote',1)
    assert oi.tolist()==[0] and ti.tolist()==[1]
    # Quote-unit identity from a synthetic, zero-premium market.
    q,fx=1.02,1400.
    d=pd.DataFrame({'USDT_BINANCE_CLOSE':[q],'USDT_UPBIT_CLOSE':[fx/q],'USDKRW':[fx]})
    for c in COINS:d[c+'_BINANCE_CLOSE']=10.;d[c+'_UPBIT_CLOSE']=10*fx/q
    y,m,_=premiums(d,True)
    assert abs(y.iloc[0])<1e-12 and abs(m.iloc[0])<1e-12
    # A later observation must not leak into the current information set.
    s=pd.Series([3.,9.],index=pd.to_datetime(['2025-01-01 10:15Z','2025-01-01 11:15Z']))
    v,age=past_observation(s,pd.to_datetime(['2025-01-01 10:00Z','2025-01-01 11:00Z']),'59min59s')
    assert pd.isna(v.iloc[0]) and v.iloc[1]==3. and age.iloc[1]==45.
    # Frequency weighting must agree with literal replication in quantile fitting.
    rng=np.random.RandomState(7);X=np.column_stack([np.ones(25),rng.normal(size=(25,2))]);y=X@np.array([1.,2.,-1.])+rng.normal(size=25)
    w=rng.randint(0,4,25);ii=np.repeat(np.arange(25),w)
    assert np.allclose(quantile_fit(X,y,.1,w),quantile_fit(X[ii],y[ii],.1),atol=1e-6)
    a,_=build('archived');b,_=build('quote_only')
    z,_,_,_=design(a,'archived',1,'M2');zz,_,_,_=design(b,'quote_only',1,'M2')
    ba,_=fit_design(z,.1);bb,_=fit_design(zz,.1)
    assert len(z)==714 and len(zz)==714
    assert abs(ba[-1]-(-.150522))<2e-5 and abs(bb[-1]-.032811)<2e-5
    print('Self-checks passed: units, clock gaps, availability, weighted QR, archived/quote-only point replication.',flush=True)


def matched_horizons():
    """Compare horizons on identical origin dates; report bp per same X SD."""
    records, counts = [], []
    for name in ['clock_UTC_quote','clock_NY_quote','clock_FXUTC_USNY_quote','clock_FXKST_USNY_quote']:
        p,_ = build(name)
        designs = {h:design(p,name,h,'M2') for h in [1,3,6,12]}
        common = set.intersection(*[set(v[2]) for v in designs.values()])
        counts.append({'scenario':name,'common_origins':len(common)})
        if len(common)<60:
            continue
        for h,(a,rr,oi,ti) in designs.items():
            a = a[np.array([i in common for i in oi])]
            for model in MODELS:
                cols = [0]+[1+MODELS['M2'].index(c) for c in MODELS[model]]
                z = a[:,cols]
                b,_ = fit_design(z,.1)
                for j,c in enumerate(MODELS[model],1):
                    if c in ['downside','account_ls']:
                        records.append(dict(scenario=name,h=h,model=model,target=c,n=len(z),
                            beta=b[j],effect_bp_per_predictor_sd=b[j]*z[:,0].std(ddof=1)*1e4))
    pd.DataFrame(records).to_csv(HERE/'matched_origin_horizons.csv',index=False)
    pd.DataFrame(counts).to_csv(HERE/'matched_origin_counts.csv',index=False)
    print(pd.DataFrame(counts).to_string(index=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['check','points','bootstrap','matched'])
    ap.add_argument('--scenario',choices=list(SCENARIOS),default='clock_NY_quote')
    ap.add_argument('--B',type=int,default=499);ap.add_argument('--block-days',type=int,default=5)
    ap.add_argument('--horizons',type=int,nargs='+',default=[1,6,12])
    ap.add_argument('--fixed-residuals',action='store_true');ap.add_argument('--joint',action='store_true')
    args=ap.parse_args()
    if args.mode=='check':selfcheck()
    elif args.mode=='points':point_results()
    elif args.mode=='matched':matched_horizons()
    else:bootstrap(args.scenario,args.B,args.block_days,args.horizons,not args.fixed_residuals,args.joint)
