"""Build complete hourly histories and refit tabular controls on matched rows."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from run_pilot import construct,read_panel,INPUTS,panel,labels,COINS,future_any,model,intercept_calibration
HERE=Path(__file__).resolve().parent

z=panel();lab,thresholds=labels(z);st=lab.btc_eth_state
# h=0 keeps features independent of future availability; targets are rebuilt below.
x,g,_=construct('mean5',0)
cols=g['local'].copy()
for c in COINS:
    x['basis_'+c]=z[c];cols.append('basis_'+c)
margin=z[['BTC','ETH']]-thresholds[['BTC','ETH']]
x['max_margin']=margin.max(axis=1,skipna=False)
x['min_margin']=margin.min(axis=1,skipna=False)
x['num_lower']=z[COINS].lt(thresholds[COINS]).sum(axis=1)
x['consensus_count6']=st.rolling(6,min_periods=6).sum()
x['consensus_count24']=st.rolling(24,min_periods=24).sum()
x['mean_minus_median']=z.mean5-z.median5
x['mean_minus_btc']=z.mean5-z.BTC
d=read_panel(INPUTS[1]);d.index+=pd.Timedelta(hours=1)
x['doge_one_won_bp']=2000*np.log1p(1/d.DOGE_UPBIT_CLOSE)
x['usdt_one_won_bp']=10000*np.log1p(1/d.USDT_UPBIT_CLOSE)
cols+=['max_margin','min_margin','num_lower','consensus_count6','consensus_count24',
       'mean_minus_median','mean_minus_btc','doge_one_won_bp','usdt_one_won_bp']
x=x.reindex(z.index)
y=future_any(st)
old=pd.read_csv(HERE/'onset_forecasts.csv.gz',index_col=0,parse_dates=True)
old.index=pd.to_datetime(old.index,utc=True)
arrays=[];origins=[];ys=[]
for i in range(23,len(x)):
    t=x.index[i];win=x.iloc[i-23:i+1][cols].to_numpy(dtype=float)
    if st.loc[t]!=0 or not np.isfinite(y.loc[t]) or not np.isfinite(win).all():
        continue
    # Comparing future forecasts only on the first pilot's known evaluation origins.
    if t>=pd.Timestamp('2025-11-01',tz='UTC') and t not in old.index:
        continue
    assert (x.index[i]-x.index[i-23])==pd.Timedelta(hours=23)
    arrays.append(win);origins.append(t);ys.append(y.loc[t])
X=np.stack(arrays);origin=pd.DatetimeIndex(origins);Y=np.asarray(ys)
observed=origin+pd.Timedelta(hours=6)
frame=pd.DataFrame({'target':Y,'target_time':observed},index=origin)
f=frame.loc[frame.index>=pd.Timestamp('2025-11-01',tz='UTC')].copy()
preds={};records=[]
for fam,view in [('logistic','current'),('logistic','flat'),('tree','current')]:
    name=fam+'_'+view;p=np.zeros(len(f))
    for month in sorted(set(f.index.strftime('%Y-%m'))):
        cut=pd.Timestamp(month+'-01',tz='UTC')
        train=observed<cut;test=origin.strftime('%Y-%m')==month
        dest=f.index.strftime('%Y-%m')==month
        a=X[:,-1,:] if view=='current' else X.reshape(len(X),-1)
        m=model(fam);m.fit(a[train],Y[train].astype(int));p[dest]=m.predict_proba(a[test])[:,1]
        records.append(dict(model=name,month=month,n_train=int(train.sum()),n_test=int(test.sum()),
                            last_training_target=str(observed[train].max())))
    preds[name+'_raw']=p;preds[name+'_cal']=intercept_calibration(f,p)[0]
for name,p in preds.items():
    f[name]=p
for name in ['recent28','recent28_state']:
    f[name]=old.reindex(f.index)[name]
assert f.notna().all().all()
f.to_csv(HERE/'sequence_controls.csv.gz',compression='gzip')
pd.DataFrame(records).to_csv(HERE/'sequence_control_training.csv',index=False)
np.savez_compressed(HERE/'sequence_data.npz',X=X.astype(np.float32),y=Y.astype(np.float32),origin_ns=origin.asi8,feature_names=np.asarray(cols))
(HERE/'SEQUENCE_DATA_METADATA.json').write_text(json.dumps(dict(features=cols,n=len(X),lookback_hours=24,
    n_test=int((origin>=pd.Timestamp('2026-01-01',tz='UTC')).sum()),
    test_positives=int(Y[origin>=pd.Timestamp('2026-01-01',tz='UTC')].sum()),
    max_history_offset_hours=23,inputs_include_past_consensus=True,
    protocol_sha256=hashlib.sha256((HERE/'SEQUENCE_PROTOCOL_KO.md').read_bytes()).hexdigest()),indent=2))
print('Sequence shape',X.shape,'test',(origin>=pd.Timestamp('2026-01-01',tz='UTC')).sum(),flush=True)
