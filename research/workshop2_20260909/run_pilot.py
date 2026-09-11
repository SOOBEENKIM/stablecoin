"""Bounded onset-probability pilot; see frozen PILOT_PROTOCOL_KO.md."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import brentq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,average_precision_score,roc_auc_score
from lightgbm import LGBMClassifier
from feasibility import panel,labels,COINS,TRAIN_END,TEST_START

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'extension_20260909'))
from run_extension import construct,read_panel,INPUTS
SEED=20260909


def future_any(state,h=6):
    f=pd.concat([state.shift(-j) for j in range(1,h+1)],axis=1)
    return f.max(axis=1).where(f.notna().all(axis=1))


def dataset():
    z=panel();lab,thresholds=labels(z);st=lab.btc_eth_state
    x,oldgroups,_=construct('mean5',6)
    groups={'local':oldgroups['local'].copy()}
    extra=[]
    for c in COINS:
        name='basis_'+c; x[name]=z[c];extra.append(name)
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
    extra+=['max_margin','min_margin','num_lower','consensus_count6','consensus_count24',
            'mean_minus_median','mean_minus_btc','doge_one_won_bp','usdt_one_won_bp']
    groups['reference']=groups['local']+extra
    groups['full']=groups['reference']+[c for c in oldgroups['full'] if c not in oldgroups['local']]
    x['target']=future_any(st).reindex(x.index)
    x['state']=st.reindex(x.index)
    # Retain all valid, currently non-consensus timestamps for the frequency baseline.
    history=pd.DataFrame({'y':future_any(st),'state':st,
                          'recent_state':st.rolling(24,min_periods=24).sum().clip(upper=2)},index=z.index)
    history['target_time']=history.index+pd.Timedelta(hours=6)
    history=history.loc[history.state.eq(0)&history.y.notna()]
    x=x.loc[x.state.eq(0)&x.target.notna()&np.isfinite(x[groups['full']]).all(axis=1)].copy()
    for col in ['recent28','recent28_state']:
        x[col]=np.nan
    for t in x.index:
        h=history.loc[(history.target_time<t)&(history.target_time>=t-pd.Timedelta(days=28))]
        rate=(h.y.sum()+1)/(len(h)+2)
        match=h.loc[h.recent_state.eq(min(x.loc[t,'consensus_count24'],2))]
        x.loc[t,'recent28']=rate
        x.loc[t,'recent28_state']=(match.y.sum()+1)/(len(match)+2) if len(match)>=30 else rate
    assert ((x.target_time-x.index)==pd.Timedelta(hours=6)).all()
    return x,groups


def model(family):
    if family=='tree':
        return LGBMClassifier(objective='binary',num_leaves=7,n_estimators=150,
            learning_rate=.03,min_child_samples=100,reg_lambda=10,n_jobs=1,
            random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
    lr=LogisticRegression(C=.1,max_iter=2000,random_state=SEED)
    if family=='spline':
        return make_pipeline(SplineTransformer(n_knots=3,degree=2,knots='quantile',include_bias=False),StandardScaler(),lr)
    return make_pipeline(StandardScaler(),lr)


def intercept_calibration(frame,raw):
    p=raw.copy(); records=[]
    for month in sorted(set(frame.index.strftime('%Y-%m'))):
        cut=pd.Timestamp(month+'-01',tz='UTC')
        mask=frame.index.strftime('%Y-%m')==month
        hist=(frame.target_time<cut)&(frame.target_time>=cut-pd.Timedelta(days=28))
        offset=0.
        if hist.sum()>=100:
            target=(frame.loc[hist,'target'].sum()+1)/(hist.sum()+2)
            lp=logit(np.clip(raw[hist],1e-6,1-1e-6))
            offset=brentq(lambda a:expit(lp+a).mean()-target,-20,20)
            p[mask]=expit(logit(np.clip(raw[mask],1e-6,1-1e-6))+offset)
        records.append({'month':month,'ncal':int(hist.sum()),'offset':float(offset),
                        'last_available_target':str(frame.loc[hist,'target_time'].max())})
    return p,records


def scores(y,p):
    return dict(n=len(y),positive=int(np.sum(y)),prevalence=float(np.mean(y)),
        mean_probability=float(np.mean(p)),brier=brier_score_loss(y,p),
        logloss=log_loss(y,p,labels=[0,1]),ap=average_precision_score(y,p),auc=roc_auc_score(y,p))


def main():
    frozen=[HERE/'PILOT_PROTOCOL_KO.md',HERE/'run_pilot.py',HERE/'feasibility.py']+INPUTS+[
            HERE.parent/'extension_20260909/run_extension.py',HERE.parent/'extension_20260909/basis_definitions.csv']
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    (HERE/'PILOT_RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    x,groups=dataset();f=x.loc[x.index>=TRAIN_END]
    preds={};fits=[];calrecords=[]
    for fam,group in [(fam,g) for fam in ['logistic','tree'] for g in groups]+[('spline','reference')]:
        p=np.zeros(len(f));name=fam+'_'+group
        for month in sorted(set(f.index.strftime('%Y-%m'))):
            cut=pd.Timestamp(month+'-01',tz='UTC');mask=f.index.strftime('%Y-%m')==month
            train=x.loc[x.target_time<cut];valid=f.loc[mask]
            assert train.target_time.max()<valid.index.min()
            m=model(fam);m.fit(train[groups[group]],train.target.astype(int))
            p[mask]=m.predict_proba(valid[groups[group]])[:,1]
            fits.append(dict(model=name,month=month,nfit=len(train),positive=int(train.target.sum()),
                last_training_target=str(train.target_time.max()),first_origin=str(valid.index.min())))
        preds[name+'_raw']=p
        preds[name+'_cal'],records=intercept_calibration(f,p)
        calrecords.extend([dict(model=name,**r) for r in records])
        print(name,'done',flush=True)
    for name in ['recent28','recent28_state']:
        preds[name]=f[name].to_numpy()
    result=f[['target','target_time','state','basis','consensus_count24']].copy()
    table=[];monthly=[]
    test=f.index>=TEST_START
    for name,p in preds.items():
        result[name]=p
        table.append(dict(model=name,**scores(f.loc[test,'target'].to_numpy(),p[test])))
        for month in sorted(set(f.index.strftime('%Y-%m'))):
            mask=f.index.strftime('%Y-%m')==month
            monthly.append(dict(model=name,month=month,**scores(f.loc[mask,'target'].to_numpy(),p[mask])))
    result.to_csv(HERE/'onset_forecasts.csv.gz',compression='gzip')
    pd.DataFrame(table).to_csv(HERE/'pilot_scores.csv',index=False)
    pd.DataFrame(monthly).to_csv(HERE/'pilot_monthly.csv',index=False)
    pd.DataFrame(fits).to_csv(HERE/'pilot_training.csv',index=False)
    pd.DataFrame(calrecords).to_csv(HERE/'pilot_calibration.csv',index=False)
    (HERE/'PILOT_METADATA.json').write_text(json.dumps({'features':groups,'n_model_eligible':len(x),
        'test_days':int(f.index[test].normalize().nunique()),'n_test':int(test.sum()),
        'first_test_origin':str(f.index[test].min()),'last_test_origin':str(f.index[test].max()),
        'freeze_checks':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items()}},indent=2))
    print(pd.DataFrame(table).round(5).to_string(index=False))


if __name__=='__main__':
    main()
