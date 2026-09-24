"""Complete all frozen nested-selection forecasts before exposing any scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from guard import HERE,verify_lock,write_new,sha,now
from nested import a,DESIGN,CASES,SEEDS,FOLDS,SPECS,inner_months,fit_model,choose,outer_sample,purge_inner

OUT=HERE/'results'


def inner_job(task):
    mode,definition,fold,info,kind=task
    cutoff=pd.Timestamp(fold+'-01',tz='UTC')
    panel=a.panel(mode)
    train,test,r=outer_sample(panel,mode,definition,cutoff)
    # An inner validation label on the next month's cutoff is not yet known
    # when that month's selection is made. Outer samples keep their original
    # clock labels, but inner scoring purges this boundary explicitly.
    test=purge_inner(test,cutoff)
    if len(test)==0:raise ValueError('Empty purged inner validation month')
    columns=a.columns(mode,info)
    rows=[]
    for spec in [s for s in SPECS if s['kind']==kind]:
        fit=a.window(train,cutoff,spec['window'])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                model=fit_model(spec,fit,columns,SEEDS[0])
                pred=model.predict(test[columns])
                losses=a.pinball(test.target,pred,.1)
                if not np.isfinite(losses).all():raise RuntimeError('Nonfinite candidate loss')
                valid,loss_sum,reason=True,float(losses.sum()),''
            except ValueError as e:
                if kind!='threshold' or str(e)!='Insufficient training observations in a regime':raise
                valid,loss_sum,reason=False,np.nan,str(e)
        rows.append(dict(mode=mode,definition=definition,fold=fold,info=info,kind=kind,
            candidate=spec['candidate'],valid=valid,loss_sum=loss_sum,n=len(test),reason=reason,
            train_n=len(fit),fit_end=r.fit_end.isoformat(),train_last_target=fit.target_time.max().isoformat(),
            first_validation_origin=test.index.min().isoformat(),last_validation_target=test.target_time.max().isoformat(),
            warnings=[str(w.message) for w in captured]))
    return rows


def outer_job(task):
    mode,definition,fold,info,rows=task
    cutoff=pd.Timestamp(fold+'-01',tz='UTC')
    scores=pd.DataFrame(rows)
    selections={strategy:choose(scores,cutoff,None if strategy=='selected' else strategy)
                for strategy in DESIGN['strategies']}
    train,test,r=outer_sample(a.panel(mode),mode,definition,cutoff)
    columns=a.columns(mode,info)
    seeds=SEEDS if (mode,definition)==tuple(DESIGN['primary_case']) else SEEDS[:1]
    predictions,metadata=[],[]
    cache={}
    for strategy,spec in selections.items():
        for seed in seeds:
            key=spec['candidate'],seed
            if key not in cache:
                fit=a.window(train,cutoff,spec['window'])
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    model=fit_model(spec,fit,columns,seed)
                    raw=model.predict(test[columns])
                assert np.isfinite(raw).all()
                cache[key]=(raw,[str(w.message) for w in captured],len(fit))
            raw,warning_list,train_n=cache[key]
            d=test[['target_time','target','e_now','downside24','account_log']].reset_index()
            for k,v in dict(mode=mode,definition=definition,fold=fold,info=info,strategy=strategy,
                           seed=seed,kind=spec['kind'],candidate=spec['candidate']).items():d[k]=v
            d['pred_raw']=raw;predictions.append(d)
            metadata.append(dict(mode=mode,definition=definition,fold=fold,info=info,strategy=strategy,
                seed=seed,**spec,train_n=train_n,test_n=len(test),train_last_target=train.target_time.max().isoformat(),
                first_test_origin=test.index.min().isoformat(),residualizer=r.metadata(),features=columns,warnings=warning_list))
    return pd.concat(predictions,ignore_index=True),metadata,len(cache)


def main(workers):
    stamp=verify_lock()
    if (OUT/'SCORES_OPENED.json').exists():raise RuntimeError('Scores already opened; do not rerun selection as a new experiment')
    write_new(OUT/'RUN_STARTED.json',dict(**stamp,started_utc=now(),data_scope='existing_data_only'))
    start=time.monotonic();checkpoints=OUT/'checkpoints';checkpoints.mkdir()
    validation_months=sorted({d.strftime('%Y-%m') for cutoff in FOLDS for d in inner_months(cutoff)})
    tasks=[(m,d,f,i,k) for m,d in CASES for f in validation_months for i in DESIGN['information_sets'] for k in DESIGN['algorithms']]
    inner=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(inner_job,t):t for t in tasks}
        for n,future in enumerate(as_completed(pending),1):
            rows=future.result();inner.extend(rows)
            name='__'.join(pending[future])
            (checkpoints/(name+'.json')).write_text(json.dumps(rows,indent=2))
            if n%12==0 or n==len(tasks):print('INNER',n,'/',len(tasks),'elapsed',round(time.monotonic()-start),flush=True)
    scores=pd.DataFrame(inner).sort_values(['mode','definition','fold','info','candidate'])
    scores.to_csv(OUT/'inner_scores.csv',index=False)
    tasks=[]
    for mode,definition in CASES:
        for cutoff in FOLDS:
            months=[x.strftime('%Y-%m') for x in inner_months(cutoff)]
            for info in DESIGN['information_sets']:
                s=scores[(scores['mode']==mode)&(scores.definition==definition)&(scores['info']==info)&scores.fold.isin(months)]
                tasks.append((mode,definition,cutoff.strftime('%Y-%m'),info,s.to_dict('records')))
    frames,metadata=[],[];outer_fits=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(outer_job,t) for t in tasks]
        for n,future in enumerate(as_completed(futures),1):
            frame,meta,count=future.result();frames.append(frame);metadata.extend(meta);outer_fits+=count
            if n%8==0 or n==len(tasks):print('OUTER',n,'/',len(tasks),'elapsed',round(time.monotonic()-start),flush=True)
    raw=pd.concat(frames,ignore_index=True)
    group=['mode','definition','info','strategy','seed']
    pred=pd.concat([a.calibrate(g) for _,g in raw.groupby(group,sort=True)],ignore_index=True)
    pred['evaluation']=pred.origin>=pd.Timestamp(DESIGN['evaluation_start'])
    assert (pred.loc[pred.evaluation,'calibration_n']>=30).all()
    # Prove that changes to the selection rule have not changed the target/sample.
    old=pd.read_csv(HERE.parent/'residual_dynamics_adaptive/results/predictions.csv.gz')
    old.origin=pd.to_datetime(old.origin,utc=True)
    for mode,definition in CASES:
        got=pred[(pred['mode']==mode)&(pred.definition==definition)&(pred['info']=='history')&
                 (pred.strategy=='selected')&(pred.seed==SEEDS[0])&pred.evaluation].sort_values('origin')
        ref=old[(old['mode']==mode)&(old.definition==definition)&(old['info']=='history')&
                (old.model=='qrf')&old.evaluation].sort_values('origin')
        assert list(got.origin)==list(ref.origin)
        np.testing.assert_allclose(got.target,ref.target,atol=1e-10,rtol=0)
    pred.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (OUT/'selected_models.json').write_text(json.dumps(metadata,indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['residualizer','features','warnings']} for r in metadata]).to_csv(OUT/'selected_models.csv',index=False)
    invalid=int((~scores.valid).sum())
    warning_n=sum(len(x) for x in scores.warnings)+sum(len(x['warnings']) for x in metadata)
    write_new(OUT/'PREDICTIONS_SEALED.json',dict(**stamp,sealed_utc=now(),
        prediction_sha256=sha(OUT/'predictions.csv.gz'),inner_scores_sha256=sha(OUT/'inner_scores.csv'),
        selections_sha256=sha(OUT/'selected_models.json'),inner_candidates=len(scores),invalid_candidates=invalid,
        outer_fits=outer_fits,warnings=warning_n,elapsed_seconds=round(time.monotonic()-start,2),
        same_origins_and_targets_as_previous=True,outer_scores_not_used_by_selection=True,
        independent_confirmation=False))
    print('SEALED forecasts. No outer performance scores printed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args();main(args.workers)
