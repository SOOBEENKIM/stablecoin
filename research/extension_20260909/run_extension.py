"""Chronological, bounded extension. Specifications in PROTOCOL_KO.md."""
from pathlib import Path
import sys, json, hashlib, argparse, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.linear_model import QuantileRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'reanalysis_20260909'))
from recalculate import read_panel, past_observation, COINS
QS = np.array([.1,.5,.9])
START = pd.Timestamp('2025-11-01', tz='UTC')
TEST = pd.Timestamp('2026-01-01', tz='UTC')
SEED = 20260909
INPUTS = [ROOT/'data/raw data/binance_1h_2025-06-01_2026-03-19.csv',
          ROOT/'data/raw data/upbit_1h_2025-06-01_2026-03-19.csv',
          ROOT/'corrected_outputs/binance_btcusdt_data_vision_metrics.csv',
          ROOT/'corrected_outputs/binance_btcusdt_funding_rate.csv']


def safe_log(v):
    return np.log(v.where(v > 0))


def construct(target='mean5', h=6):
    d = read_panel(INPUTS[0]).join(read_panel(INPUTS[1]))
    d.index += pd.Timedelta(hours=1)
    assert not d.index.has_duplicates
    assert (d.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    imp = pd.DataFrame({c:safe_log(d[c+'_UPBIT_CLOSE']/d[c+'_BINANCE_CLOSE']) for c in COINS})
    if target == 'mean5':
        reference = imp.mean(axis=1, skipna=False)
    elif target == 'median5':
        reference = imp.median(axis=1, skipna=False)
    else:
        reference = imp.BTC
    b = (safe_log(d.USDT_UPBIT_CLOSE)-reference)*1e4
    x = pd.DataFrame(index=d.index)
    x['basis'] = b
    x['change1'], x['change6'] = b.diff(), b.diff(6)
    x['mean24'], x['std24'] = b.rolling(24).mean(), b.rolling(24).std()
    r = safe_log(d.BTC_BINANCE_CLOSE).diff()
    x['btc_ret1'], x['btc_ret6'] = r, r.rolling(6).sum()
    x['btc_vol24'] = r.rolling(24).std()
    x['btc_downside24'] = r.clip(upper=0).pow(2).rolling(24).mean()
    x['hour_sin'] = np.sin(2*np.pi*d.index.hour/24)
    x['hour_cos'] = np.cos(2*np.pi*d.index.hour/24)
    x['weekend'] = (d.index.dayofweek >= 5).astype(float)
    x['q10_7d'] = b.rolling('7D', min_periods=120).quantile(.1)
    x['q90_7d'] = b.rolling('7D', min_periods=120).quantile(.9)
    groups = {'price': x.columns.tolist()}
    x['dispersion'] = imp.std(axis=1, skipna=False)*1e4
    x['usdt_range'] = safe_log(d.USDT_UPBIT_HIGH/d.USDT_UPBIT_LOW)
    v = pd.DataFrame({c:safe_log(d[c+'_UPBIT_VOLUME']/d[c+'_BINANCE_VOLUME']) for c in COINS})
    x['venue_volume'] = v.mean(axis=1, skipna=False)
    lv = safe_log(d.USDT_UPBIT_VOLUME)
    x['usdt_volume_surprise'] = lv-lv.shift().rolling(24).mean()
    x['usdt_unchanged'] = d.USDT_UPBIT_CLOSE.eq(d.USDT_UPBIT_CLOSE.shift()).astype(float)
    groups['local'] = x.columns.tolist()
    met = read_panel(INPUTS[2])
    met.index += pd.Timedelta(minutes=5)
    for col,new in [('ACCOUNT_LS','account_ls'), ('TOPTRADER_POSITION_LS','top_position_ls'), ('OI','oi')]:
        x[new],age = past_observation(safe_log(met[col]), d.index, '10min')
        assert age.dropna().between(0,10).all()
    x['account_ls_change6'] = x.account_ls.diff(6)
    x['oi_change1'],x['oi_change6'] = x.oi.diff(),x.oi.diff(6)
    x = x.drop(columns='oi')
    taker = safe_log(met.TAKER_LONG_SHORT_VOL_RATIO).rolling('60min',min_periods=10).mean()
    x['taker'],_ = past_observation(taker,d.index,'10min')
    funding = read_panel(INPUTS[3])
    funding.index += pd.Timedelta(minutes=5)
    x['funding'],_ = past_observation(funding.FUNDING,d.index,'9h')
    groups['full'] = x.columns.tolist()
    x['lag1'] = b.shift()
    x['abs_basis'] = b.abs()
    groups['qar'] = ['basis','lag1','abs_basis','std24','q10_7d','q90_7d']
    x['target_time'] = d.index+pd.Timedelta(hours=h)
    x['target'] = b.reindex(pd.DatetimeIndex(x.target_time)).to_numpy()
    for days in [7,28]:
        for q in QS:
            x['hist%d_q%d' % (days,round(q*100))] = b.rolling('%dD'%days,min_periods=120).quantile(q)
    # Historical h-hour changes are observed at the END of the change interval.
    delta = b-b.shift(h)
    for q in QS:
        x['delta_q%d'%round(q*100)] = b+delta.rolling('28D',min_periods=120).quantile(q)
    numerical = x.select_dtypes(include='number')
    keep = np.isfinite(numerical).all(axis=1)
    x = x.loc[keep].copy()
    assert ((x.target_time-x.index) == pd.Timedelta(hours=h)).all()
    # States defined on pre-2026 information, never on future test quantiles.
    pre = x.loc[x.index < TEST]
    states = {'high_downside': ('btc_downside24',.75,'high'),
              'high_dispersion': ('dispersion',.75,'high'),
              'low_usdt_volume': ('usdt_volume_surprise',.25,'low')}
    state_thresholds = {}
    for name,(col,q,side) in states.items():
        cut = pre[col].quantile(q)
        x[name] = x[col].ge(cut) if side == 'high' else x[col].le(cut)
        state_thresholds[name] = float(cut)
    x['negative_basis'] = x.basis < 0
    return x,groups,state_thresholds


def matrix(frame, cols, threshold=None):
    z = frame[cols].copy()
    if threshold is not None:
        z['positive_hinge'] = (frame.basis-threshold).clip(lower=0)
        z['negative_hinge'] = (frame.basis+threshold).clip(upper=0)
        z['interaction'] = frame.basis*frame.btc_downside24
    return z.to_numpy(dtype=float)


def fit_model(fit, cols, family):
    threshold = fit.basis.abs().quantile(.75) if family == 'threshold' else None
    X = matrix(fit,cols,threshold)
    mean,sd = X.mean(axis=0),X.std(axis=0,ddof=1)
    sd[sd < 1e-14] = 1
    X = (X-mean)/sd
    models=[]
    for q in QS:
        if family == 'lgbm':
            m = LGBMRegressor(objective='quantile',alpha=q,num_leaves=7,n_estimators=150,
                learning_rate=.03,min_child_samples=100,reg_lambda=10,
                n_jobs=1,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
        else:
            m = QuantileRegressor(quantile=q,alpha=.01,solver='highs')
        m.fit(X,fit.target.to_numpy())
        models.append(m)
    return models,cols,threshold,mean,sd


def predict(fitted,frame):
    models,cols,threshold,mean,sd = fitted
    z = (matrix(frame,cols,threshold)-mean)/sd
    p = np.column_stack([m.predict(z) for m in models])
    assert np.isfinite(p).all()
    return np.sort(p,axis=1)


def calibrate(frame,pred,days=28):
    origins = frame.index.asi8
    observed = pd.DatetimeIndex(frame.target_time).asi8
    residual = frame.target.to_numpy()[:,None]-pred
    adjusted = pred.copy()
    ncal = np.zeros(len(frame),dtype=int)
    window = pd.Timedelta(days=days).value
    for i,t in enumerate(origins):
        # Strict inequality: observations at t are not used in correction at t.
        right = np.searchsorted(observed,t,side='left')
        left = np.searchsorted(observed,t-window,side='left')
        assert right <= i
        ncal[i] = right-left
        if right-left >= 120:
            correction = [np.quantile(residual[left:right,j],q) for j,q in enumerate(QS)]
            adjusted[i] += correction
    return np.sort(adjusted,axis=1),ncal


def score(y,p):
    e = y[:,None]-p
    losses = np.maximum(QS*e,(QS-1)*e)
    return {'n':len(y),'tail_loss':float(losses[:,[0,2]].mean()),
        'q10_loss':float(losses[:,0].mean()),'q50_loss':float(losses[:,1].mean()),'q90_loss':float(losses[:,2].mean()),
        'below10':float((y<p[:,0]).mean()),'above90':float((y>p[:,2]).mean()),
        'coverage80':float(((y>=p[:,0])&(y<=p[:,2])).mean()),'width80':float((p[:,2]-p[:,0]).mean())}


def run_cell(target,h):
    start=time.time()
    label='%s_h%d'%(target,h)
    data,groups,thresholds = construct(target,h)
    f = data.loc[data.index>=START]
    months = sorted(set(f.index.strftime('%Y-%m')))
    specs = [(family,g) for family in ['linear','lgbm'] for g in ['price','local','full']]
    specs += [('threshold','local'),('qar','qar')]
    predictions = {}
    meta=[]
    freeze = (target=='mean5' and h==6)
    for family,group in specs:
        name=family+'_'+group
        updated=np.zeros((len(f),3))
        frozen=np.zeros((len(f),3)) if freeze else None
        frozen_fit=None
        for month in months:
            cut=pd.Timestamp(month+'-01',tz='UTC')
            mask=f.index.strftime('%Y-%m')==month
            training=data.loc[data.target_time<cut]
            evaluation=f.loc[mask]
            assert training.target_time.max()<evaluation.index.min()
            fitted=fit_model(training,groups[group],family)
            updated[mask]=predict(fitted,evaluation)
            if freeze:
                if cut==TEST:
                    frozen_fit=fitted
                frozen[mask]=predict(frozen_fit,evaluation) if cut>=TEST else updated[mask]
            meta.append({'target':target,'h':h,'model':name,'month':month,'nfit':len(training),
                         'last_training_target':str(training.target_time.max()),'first_prediction':str(evaluation.index.min())})
        for policy,raw in [('update',updated)]+([('freeze',frozen)] if freeze else []):
            predictions[policy+'_'+name+'_raw']=raw
            adj,ncal=calibrate(f,raw,28)
            assert (ncal[f.index>=TEST] >= 120).all()
            predictions[policy+'_'+name+'_cal28']=adj
            if target=='mean5' and h==6 and policy=='update' and name in ['linear_local','lgbm_local','lgbm_full']:
                for days in [14,56]:
                    predictions[policy+'_'+name+'_cal%d'%days]=calibrate(f,raw,days)[0]
        print(label,name,'done',round(time.time()-start,1),'sec',flush=True)
    for days in [7,28]:
        predictions['hist%d'%days]=f[['hist%d_q%d'%(days,round(q*100)) for q in QS]].to_numpy()
    predictions['delta28']=f[['delta_q%d'%round(q*100) for q in QS]].to_numpy()
    states=['high_downside','high_dispersion','low_usdt_volume','negative_basis']
    testmask=f.index>=TEST
    tests=[]; monthly=[]; conditional=[]; exports=[]
    for name,p in predictions.items():
        assert (np.diff(p,axis=1)>=0).all()
        y=f.target.to_numpy()
        tests.append({'target':target,'h':h,'model':name,**score(y[testmask],p[testmask])})
        for month in months:
            keep=f.index.strftime('%Y-%m')==month
            monthly.append({'target':target,'h':h,'model':name,'month':month,**score(y[keep],p[keep])})
        for state in states:
            for yes in [0,1]:
                keep=testmask & f[state].eq(bool(yes)).to_numpy()
                conditional.append({'target':target,'h':h,'model':name,'state':state,'value':yes,
                    'days':f.index[keep].normalize().nunique(),**score(y[keep],p[keep])})
        out=f[['target','target_time','basis']+states].copy()
        out[['q10','q50','q90']]=p
        out['model']=name
        out['origin']=out.index
        exports.append(out.reset_index(drop=True))
    pd.DataFrame(tests).to_csv(HERE/(label+'_scores.csv'),index=False)
    pd.DataFrame(monthly).to_csv(HERE/(label+'_monthly.csv'),index=False)
    pd.DataFrame(conditional).to_csv(HERE/(label+'_states.csv'),index=False)
    pd.DataFrame(meta).to_csv(HERE/(label+'_training.csv'),index=False)
    pd.concat(exports,ignore_index=True).to_csv(HERE/(label+'_forecasts.csv.gz'),index=False,compression='gzip')
    (HERE/(label+'_metadata.json')).write_text(json.dumps({'features':groups,'state_thresholds':thresholds,
        'n_total':len(data),'n_test':int(testmask.sum()),'test_days':int(f.index[testmask].normalize().nunique()),
        'first_origin':str(f.index[testmask].min()),'last_target':str(f.target_time[testmask].max())},indent=2))
    return label,round(time.time()-start,1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    files=INPUTS+[HERE/'PROTOCOL_KO.md',Path(__file__).resolve(),ROOT/'reanalysis_20260909/recalculate.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest={'started_utc':str(pd.Timestamp.now(tz='UTC')),'sha256':hashes,
              'lightgbm':lightgbm.__version__,'numpy':np.__version__,'pandas':pd.__version__}
    (HERE/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    cells=[('mean5',6),('mean5',1),('mean5',12),('btc',6),('median5',6)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_cell,*cell):cell for cell in cells}
        for future in as_completed(futures):
            print('COMPLETED',future.result(),flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    print('ALL CELLS COMPLETE; inputs, code, protocol unchanged',flush=True)


if __name__=='__main__':
    main()
