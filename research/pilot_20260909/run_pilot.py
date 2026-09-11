"""Frozen, small forecast feasibility experiment. See PROTOCOL_KO.md."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import QuantileRegressor
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'reanalysis_20260909'))
from recalculate import read_panel, past_observation, COINS

QUANTILES = [.1,.5,.9]
H = 6
VAL_START = pd.Timestamp('2025-11-01',tz='UTC')
TEST_START = pd.Timestamp('2026-01-01',tz='UTC')
SEED = 20260909


def construct():
    b = read_panel(ROOT/'data/raw data/binance_1h_2025-06-01_2026-03-19.csv')
    u = read_panel(ROOT/'data/raw data/upbit_1h_2025-06-01_2026-03-19.csv')
    d = b.join(u)
    d.index += pd.Timedelta(hours=1)
    assert (d.index.to_series().diff().dropna()==pd.Timedelta(hours=1)).all()
    implied = pd.DataFrame({c:np.log(d[c+'_UPBIT_CLOSE']/d[c+'_BINANCE_CLOSE']) for c in COINS})
    basis = (np.log(d.USDT_UPBIT_CLOSE)-implied.mean(axis=1,skipna=False))*1e4
    r = np.log(d.BTC_BINANCE_CLOSE).diff()
    x = pd.DataFrame(index=d.index)
    x['basis'] = basis
    x['basis_change1'] = basis.diff()
    x['basis_change6'] = basis.diff(6)
    x['basis_mean24'] = basis.rolling(24,min_periods=24).mean()
    x['basis_std24'] = basis.rolling(24,min_periods=24).std()
    x['cross_coin_dispersion'] = implied.std(axis=1,skipna=False)*1e4
    x['btc_ret1'] = r
    x['btc_ret6'] = r.rolling(6,min_periods=6).sum()
    x['btc_vol24'] = r.rolling(24,min_periods=24).std()
    x['btc_downside24'] = r.clip(upper=0).pow(2).rolling(24,min_periods=24).mean()
    x['usdt_range'] = np.log(d.USDT_UPBIT_HIGH/d.USDT_UPBIT_LOW)
    hour = d.index.hour.to_numpy()
    x['hour_sin'] = np.sin(2*np.pi*hour/24)
    x['hour_cos'] = np.cos(2*np.pi*hour/24)
    x['weekend'] = (d.index.dayofweek>=5).astype(float)
    volumes = pd.DataFrame({c:np.log(d[c+'_UPBIT_VOLUME']/d[c+'_BINANCE_VOLUME']) for c in COINS})
    x['venue_volume_ratio'] = volumes.mean(axis=1,skipna=False)
    lv = np.log(d.USDT_UPBIT_VOLUME)
    x['usdt_volume_surprise'] = lv-lv.shift(1).rolling(24,min_periods=24).mean()
    base_cols = x.columns.tolist()
    metrics = read_panel(ROOT/'corrected_outputs/binance_btcusdt_data_vision_metrics.csv')
    assert not metrics.index.has_duplicates
    metrics.index += pd.Timedelta(minutes=5)
    for raw, col in [('ACCOUNT_LS','account_ls'),('TOPTRADER_POSITION_LS','top_position_ls'),('OI','oi')]:
        x[col],age = past_observation(np.log(metrics[raw]),d.index,'10min')
        assert (age.dropna()>=0).all()
    x['account_ls_change6'] = x.account_ls.diff(6)
    x['oi_change1'] = x.oi.diff()
    x['oi_change6'] = x.oi.diff(6)
    x = x.drop(columns=['oi'])
    taker = np.log(metrics.TAKER_LONG_SHORT_VOL_RATIO).rolling('60min',min_periods=10).mean()
    x['taker_1h'],_ = past_observation(taker,d.index,'10min')
    full_cols = x.columns.tolist()
    # Form the exact calendar lead before dropping any missing data.
    target_times = d.index+pd.Timedelta(hours=H)
    future = pd.Series(basis.reindex(target_times).to_numpy(),index=d.index)
    x['target'] = future
    x['target_time'] = target_times
    numeric = x[full_cols+['target']].replace([np.inf,-np.inf],np.nan)
    complete = numeric.notna().all(axis=1)
    x = x.loc[complete].copy()
    assert ((x.target_time-x.index)==pd.Timedelta(hours=H)).all()
    return x,base_cols,full_cols


def feature_matrix(frame, columns, threshold=None):
    z = frame[columns].copy()
    if threshold is not None:
        b = frame.basis
        z['positive_hinge'] = (b-threshold).clip(lower=0)
        z['negative_hinge'] = (b+threshold).clip(upper=0)
        z['basis_downside_interaction'] = b*frame.btc_downside24
    return z.to_numpy(dtype=float)


def forecast(fit, evaluation, columns, family, config):
    if family=='unconditional':
        pred = np.tile(np.quantile(fit.target,QUANTILES),(len(evaluation),1))
    elif family=='persistence':
        error = fit.target-fit.basis
        pred = evaluation.basis.to_numpy()[:,None]+np.quantile(error,QUANTILES)
    else:
        threshold = np.quantile(abs(fit.basis),.75) if family=='threshold' else None
        X = feature_matrix(fit,columns,threshold)
        Z = feature_matrix(evaluation,columns,threshold)
        mean,sd = X.mean(axis=0),X.std(axis=0,ddof=1)
        sd[sd<1e-14] = 1.
        X,Z = (X-mean)/sd,(Z-mean)/sd
        y = (fit.target-fit.basis).to_numpy()
        results = []
        for q in QUANTILES:
            if family in ['linear','threshold']:
                m = QuantileRegressor(quantile=q,alpha=config['alpha'],solver='highs')
            else:
                m = LGBMRegressor(objective='quantile',alpha=q,learning_rate=.03,
                    num_leaves=config['leaves'],n_estimators=config['trees'],
                    min_child_samples=100,reg_lambda=10,n_jobs=1,
                    random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
            m.fit(X,y)
            results.append(m.predict(Z)+evaluation.basis.to_numpy())
        pred = np.column_stack(results)
    crossing = ((pred[:,0]>pred[:,1])|(pred[:,1]>pred[:,2])).mean()
    return np.sort(pred,axis=1),float(crossing)


def loss(y,pred,q):
    e = y-pred
    return np.maximum(q*e,(q-1)*e)


def scores(frame,pred):
    y = frame.target.to_numpy()
    ls = np.column_stack([loss(y,pred[:,j],q) for j,q in enumerate(QUANTILES)])
    return {'n':len(y),'tail_loss':float(ls[:,[0,2]].mean()),
        'loss_q10':float(ls[:,0].mean()),'loss_q50':float(ls[:,1].mean()),'loss_q90':float(ls[:,2].mean()),
        'below_q10':float((y<pred[:,0]).mean()),'above_q90':float((y>pred[:,2]).mean()),
        'coverage80':float(((y>=pred[:,0])&(y<=pred[:,2])).mean()),
        'width80_bp':float((pred[:,2]-pred[:,0]).mean())}


def compare(predictions,frame):
    pairs = [('lgbm_full','lgbm_base'),('lgbm_full','linear_full'),
             ('lgbm_full','threshold_full'),('lgbm_base','threshold_base'),
             ('linear_full','linear_base'),('threshold_full','threshold_base'),
             ('lgbm_full','unconditional'),('lgbm_full','persistence')]
    y = frame.target.to_numpy()
    losses = {name:(loss(y,p[:,0],.1)+loss(y,p[:,2],.9))/2 for name,p in predictions.items()}
    days = pd.date_range(frame.index.min().normalize(),frame.index.max().normalize(),freq='D')
    di = days.get_indexer(frame.index.normalize())
    rng = np.random.RandomState(SEED)
    draws = {pair:[] for pair in pairs}
    for _ in range(999):
        starts = rng.randint(0,len(days),int(np.ceil(len(days)/5)))
        sampled = np.concatenate([(s+np.arange(5))%len(days) for s in starts])[:len(days)]
        w = np.bincount(sampled,minlength=len(days))[di]
        for pair in pairs:
            candidate,reference = pair
            draws[pair].append(np.average(losses[reference]-losses[candidate],weights=w))
    records=[]
    for pair in pairs:
        candidate,reference = pair
        delta = (losses[reference]-losses[candidate]).mean()
        lo,hi = np.quantile(draws[pair],[.025,.975])
        records.append({'candidate':candidate,'reference':reference,'mean_loss_reduction_bp':delta,
            'relative_reduction_pct':100*delta/losses[reference].mean(),
            'ci95_low':lo,'ci95_high':hi,'B':999,'block_calendar_days':5,
            'note':'positive means lower candidate tail loss; conditional pilot percentile interval, no multiplicity correction'})
    pd.DataFrame(records).to_csv(HERE/'paired_comparisons.csv',index=False)


def main():
    input_files = [ROOT/'data/raw data/binance_1h_2025-06-01_2026-03-19.csv',
        ROOT/'data/raw data/upbit_1h_2025-06-01_2026-03-19.csv',
        ROOT/'corrected_outputs/binance_btcusdt_data_vision_metrics.csv',HERE/'PROTOCOL_KO.md']
    manifest = {'started_utc':str(pd.Timestamp.now(tz='UTC')),
        'input_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in input_files},
        'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
    (HERE/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    data,base,full = construct()
    train = data[(data.index<VAL_START)&(data.target_time<VAL_START)]
    val = data[(data.index>=VAL_START)&(data.index<TEST_START)&(data.target_time<TEST_START)]
    combined = data[data.target_time<TEST_START]
    test = data[data.index>=TEST_START]
    assert train.target_time.max()<val.index.min()
    assert combined.target_time.max()<test.index.min()
    splits = {'train':train,'validation':val,'refit':combined,'test':test}
    meta = {'horizon_hours':H,'base_features':base,'full_features':full,
        'splits':{k:{'n':len(v),'start':str(v.index.min()),'end':str(v.index.max()),
            'target_end':str(v.target_time.max()),'days':v.index.normalize().nunique()} for k,v in splits.items()},
        'limitations':'exploratory design after full-period descriptive analysis; not independent external validation; quote-basis target differs from submitted OLS residual'}
    (HERE/'sample_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta['splits'],indent=2),flush=True)
    grids = {'unconditional':[{}],'persistence':[{}],
        'linear':[{'alpha':.01},{'alpha':.1}],
        'threshold':[{'alpha':.01},{'alpha':.1}],
        'lgbm':[{'leaves':7,'trees':150},{'leaves':7,'trees':250},{'leaves':15,'trees':150}]}
    predictions,validation_records,selections,test_records,monthly = {},[],{},[],[]
    start = time.time()
    for family,grid in grids.items():
        infos = [('base',base),('full',full)] if family in ['linear','threshold','lgbm'] else [('',base)]
        for info,cols in infos:
            name = family+('_'+info if info else '')
            candidates = []
            for config in grid:
                vp,cross = forecast(train,val,cols,family,config)
                m = scores(val,vp)
                validation_records.append({'model':name,'config':json.dumps(config),'crossing_before_sort':cross,**m})
                candidates.append((m['tail_loss'],config))
            best = min(candidates,key=lambda c:c[0])[1]
            selections[name] = best
            # Hyperparameters are chosen from validation only.
            tp,cross = forecast(combined,test,cols,family,best)
            predictions[name] = tp
            test_records.append({'model':name,'crossing_before_sort':cross,**scores(test,tp)})
            for month in sorted(set(test.index.strftime('%Y-%m'))):
                keep = test.index.strftime('%Y-%m')==month
                monthly.append({'model':name,'month':month,**scores(test.loc[keep],tp[keep])})
            print('completed',name,'selected',best,'seconds',round(time.time()-start,1),flush=True)
    # Report test scores together, after every model specification is fixed.
    pd.DataFrame(validation_records).to_csv(HERE/'validation_scores.csv',index=False)
    (HERE/'selected_models.json').write_text(json.dumps(selections,indent=2))
    out = pd.DataFrame(test_records)
    out.to_csv(HERE/'test_scores.csv',index=False)
    pd.DataFrame(monthly).to_csv(HERE/'monthly_scores.csv',index=False)
    for name,p in predictions.items():
        f = test[['basis','target','target_time']].copy()
        f[['q10','q50','q90']] = p
        assert (f.q10<=f.q50).all() and (f.q50<=f.q90).all()
        f.to_csv(HERE/('predictions_'+name+'.csv'))
    compare(predictions,test)
    print(out.to_string(index=False),flush=True)
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in manifest['input_sha256'].items())
    print('Completed. Input/protocol hashes unchanged.',flush=True)


if __name__=='__main__':
    main()
