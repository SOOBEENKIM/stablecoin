"""Residual adequacy and equal-target downstream evaluation; no model search."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
import numpy as np
import pandas as pd
from common import *


def probes(panel, pred):
    records = []
    for method in METHODS:
        d = pred.loc[pred.model==method].set_index('time')[['residual_bp']].join(panel[study.KCOLS+['m','g']]).dropna()
        for month in MONTHS:
            start = pd.Timestamp(month+'-01',tz='UTC'); end = start+pd.offsets.MonthBegin(1)
            train, test = d.loc[d.index<start], d.loc[(d.index>=start)&(d.index<end)]
            baseline = np.full(len(test), train.residual_bp.mean())
            for scope, cols in [('mean_market',['m','g']),('all_references',study.KCOLS+['g'])]:
                for alg in ['ridge','lightgbm']:
                    fit = estimator(alg).fit(train[cols].to_numpy(), train.residual_bp.to_numpy())
                    p = fit.predict(test[cols].to_numpy())
                    records.append(pd.DataFrame(dict(time=test.index,month=month,method=method,scope=scope,
                        algorithm=alg,target=test.residual_bp.to_numpy(),prediction=p,baseline=baseline,
                        loss=(test.residual_bp.to_numpy()-p)**2,
                        baseline_loss=(test.residual_bp.to_numpy()-baseline)**2,
                        train_last=str(train.index.max()),cutoff=str(start))))
    p = pd.concat(records,ignore_index=True)
    scores, intervals = [], []
    for key,d in p.groupby(['method','scope','algorithm'],sort=False):
        fields = dict(zip(['method','scope','algorithm'],key))
        for period,f in [('all',d)]+list(d.groupby('month')):
            scores.append(dict(fields,period=period,n=len(f),mse=f.loss.mean(),baseline_mse=f.baseline_loss.mean(),
                skill=1-f.loss.mean()/f.baseline_loss.mean()))
        b = d[['time','baseline_loss']].rename(columns={'baseline_loss':'loss'})
        intervals.extend([dict(fields,**r) for r in paired(d,b)])
    return p,pd.DataFrame(scores),pd.DataFrame(intervals)


def forecasts(panel, pairs):
    pairs = pairs.loc[pairs.h==6].copy()
    pairs = pairs.set_index('time').join(panel[['y']+study.KCOLS]).reset_index()
    pairs['downside_x_ls'] = pairs.downside*pairs.account_ls
    base = ['y']+study.KCOLS+['g','downside','btc_vol','btc_ret']
    residual_base = ['e_origin','m','g','downside','btc_vol','btc_ret']
    extra = ['account_ls','downside_x_ls']
    pairs = pairs.dropna(subset=base+extra+['target','y_change'])
    sets = [set(pairs.loc[pairs.method==m,'time']) for m in METHODS]
    assert all(x==sets[0] for x in sets)
    records, fits = [], []
    for task in ['observed_change','own_residual']:
        variants = ['baseline']+METHODS if task=='observed_change' else METHODS
        for method in variants:
            d = pairs.loc[pairs.method==('sequential_ols' if method=='baseline' else method)].copy()
            d['outcome'] = d.y_change if task=='observed_change' else d.target
            cols = (base+([] if method=='baseline' else ['e_origin'])) if task=='observed_change' else residual_base
            for month in MONTHS:
                start = pd.Timestamp(month+'-01',tz='UTC'); end=start+pd.offsets.MonthBegin(1)
                train = d.loc[(d.time<start)&(d.target_time<start)]
                test = d.loc[(d.time>=start)&(d.time<end)]
                assert len(train)>100 and len(test)>20 and train.target_time.max()<start
                for info, columns in [('without_positioning',cols),('with_positioning',cols+extra)]:
                    for alg in ['linear_qr','quantile_lightgbm']:
                        for q in [.1,.5]:
                            model=estimator(alg,q).fit(train[columns].to_numpy(),train.outcome.to_numpy())
                            prediction=model.predict(test[columns].to_numpy())
                            null=np.full(len(test),np.quantile(train.outcome,q))
                            records.append(pd.DataFrame(dict(time=test.time.to_numpy(),target_time=test.target_time.to_numpy(),
                                month=month,task=task,method=method,info=info,algorithm=alg,q=q,
                                target=test.outcome.to_numpy(),prediction=prediction,baseline=null,
                                loss=quantile_loss(test.outcome,prediction,q),
                                baseline_loss=quantile_loss(test.outcome,null,q))))
                            fits.append(dict(task=task,method=method,month=month,info=info,algorithm=alg,q=q,
                                columns=columns,train_n=len(train),test_n=len(test),
                                train_target_last=str(train.target_time.max()),cutoff=str(start)))
    p=pd.concat(records,ignore_index=True)
    scores,intervals=[],[]
    keys=['task','method','info','algorithm','q']
    for key,d in p.groupby(keys,sort=False):
        fields=dict(zip(keys,key))
        for period,f in [('all',d)]+list(d.groupby('month')):
            scores.append(dict(fields,period=period,n=len(f),pinball=f.loss.mean(),baseline_pinball=f.baseline_loss.mean(),
                skill=1-f.loss.mean()/f.baseline_loss.mean(),below_rate=float((f.target<f.prediction).mean())))
        if fields['task']=='observed_change' and fields['method']!='baseline':
            b=p.loc[(p.task=='observed_change')&(p.method=='baseline')&(p['info']==fields['info'])&
                    (p.algorithm==fields['algorithm'])&(p.q==fields['q'])]
            intervals.extend([dict(fields,contrast='add_residual',**r) for r in paired(d,b)])
        if fields['info']=='with_positioning':
            b=p.loc[(p.task==fields['task'])&(p.method==fields['method'])&(p['info']=='without_positioning')&
                    (p.algorithm==fields['algorithm'])&(p.q==fields['q'])]
            intervals.extend([dict(fields,contrast='add_positioning',**r) for r in paired(d,b)])
        if fields['task']=='observed_change' and fields['method']=='ae_selected':
            for control in ['sequential_ols','pca_2']:
                b=p.loc[(p.task==fields['task'])&(p.method==control)&(p['info']==fields['info'])&
                        (p.algorithm==fields['algorithm'])&(p.q==fields['q'])]
                intervals.extend([dict(fields,contrast='AE_minus_'+control,**r) for r in paired(d,b)])
    return p,pd.DataFrame(scores),pd.DataFrame(intervals),fits


def run(scenario):
    panel,pred,pairs=load(scenario)
    out=OUT/scenario;out.mkdir(parents=True,exist_ok=True)
    for name,f in [('predictions',pred),('pairs',pairs)]: save(f,out/(name+'.csv.gz'))
    p,s,c=probes(panel,pred)
    for name,f in [('probe_predictions',p),('probe_scores',s),('probe_intervals',c)]:
        save(f,out/(name+('.csv.gz' if name.endswith('predictions') else '.csv')))
    print('Residual probes complete: '+scenario,flush=True)
    p,s,c,fits=forecasts(panel,pairs)
    for name,f in [('forecast_predictions',p),('forecast_scores',s),('forecast_intervals',c)]:
        save(f,out/(name+('.csv.gz' if name.endswith('predictions') else '.csv')))
    (out/'forecast_fits.json').write_text(json.dumps(fits,indent=2)+'\n')
    print('Equal-target and within-residual forecasts complete: '+scenario,flush=True)


def main():
    started=time.monotonic();assert not OUT.exists(),'Preserve results'
    before=study.pilot.audit_archive()
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(run,s) for s in study.pilot.SCENARIOS]
        for f in as_completed(futures):f.result()
    for name in ['probe_scores','probe_intervals','forecast_scores','forecast_intervals']:
        save(pd.concat([pd.read_csv(OUT/s/(name+'.csv')).assign(scenario=s) for s in study.pilot.SCENARIOS]),OUT/(name+'.csv'))
    assert before==study.pilot.audit_archive()
    (OUT/'METRICS_MANIFEST.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
        source_manifest_sha256=study.pilot.sha(SOURCE/'RUN_MANIFEST.json'),
        input_sha256={p.name:study.pilot.sha(p) for p in [HERE/'PROTOCOL_KO.md',HERE/'common.py',Path(__file__).resolve()]},
        output_sha256={str(p.relative_to(OUT)):study.pilot.sha(p) for p in OUT.rglob('*') if p.is_file()},
        archive_check=before),indent=2)+'\n')
    print('A5 metrics completed in %.1fs'%(time.monotonic()-started),flush=True)


if __name__=='__main__':main()
