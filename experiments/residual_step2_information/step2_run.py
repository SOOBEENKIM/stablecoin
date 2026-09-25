"""Train all prespecified ablations before exposing outer evaluation scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import warnings
import numpy as np
import pandas as pd
from step2_guard import verify_lock,write_new,sha,now
from step2_core import (OUT,DESIGN,CASES,GROUPS,PROCEDURES,features,interaction_pairs,
                        fit_model,choose,full_spec,read_full,a,n)


def inner_job(task):
    mode,definition,fold,group,kind=task
    cutoff=pd.Timestamp(fold+'-01',tz='UTC')
    train,test,r=n.outer_sample(a.panel(mode),mode,definition,cutoff)
    test=n.purge_inner(test,cutoff)
    columns=features(mode,group)
    rows=[]
    for spec in [s for s in n.SPECS if s['kind']==kind]:
        fit=a.window(train,cutoff,spec['window'])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                model=fit_model(spec,fit,columns,n.SEEDS[0])
                pred=model.predict(test[columns])
                loss=a.pinball(test.target,pred,.1)
                assert np.isfinite(loss).all()
                valid,loss_sum,reason=True,float(loss.sum()),''
            except ValueError as exc:
                if kind!='threshold' or str(exc)!='Insufficient training observations in a regime':raise
                valid,loss_sum,reason=False,np.nan,str(exc)
        rows.append(dict(mode=mode,definition=definition,fold=fold,removed_group=group,
            kind=kind,candidate=spec['candidate'],valid=valid,loss_sum=loss_sum,n=len(test),
            reason=reason,train_n=len(fit),features=columns,
            train_last_label=fit.target_time.max().isoformat(),residualizer_fit_end=r.fit_end.isoformat(),
            first_origin=test.index.min().isoformat(),last_label=test.target_time.max().isoformat(),
            warnings=[str(w.message) for w in captured]))
    return rows


def outer_job(task):
    mode,definition,fold,group,rows=task
    cutoff=pd.Timestamp(fold+'-01',tz='UTC')
    selections={'retuned':choose(pd.DataFrame(rows),cutoff,group),
                'fixed_full_spec':full_spec(mode,definition,fold)}
    train,test,r=n.outer_sample(a.panel(mode),mode,definition,cutoff)
    columns=features(mode,group)
    seeds=n.SEEDS if (mode,definition)==tuple(DESIGN['primary_case']) else n.SEEDS[:1]
    frames,metadata,cache=[],[],{}
    for procedure,spec in selections.items():
        for seed in seeds:
            key=spec['candidate'],seed
            if key not in cache:
                fit=a.window(train,cutoff,spec['window'])
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    model=fit_model(spec,fit,columns,seed)
                    raw=model.predict(test[columns])
                assert np.isfinite(raw).all()
                cache[key]=(raw,len(fit),[str(w.message) for w in captured])
            raw,train_n,warning_list=cache[key]
            d=test[['target_time','target','e_now']].reset_index()
            for k,v in dict(mode=mode,definition=definition,fold=fold,removed_group=group,
                procedure=procedure,seed=seed,kind=spec['kind'],candidate=spec['candidate']).items():d[k]=v
            d['pred_raw']=raw;frames.append(d)
            metadata.append(dict(mode=mode,definition=definition,fold=fold,removed_group=group,
                procedure=procedure,seed=seed,**spec,train_n=train_n,test_n=len(test),features=columns,
                interaction_pairs=interaction_pairs(columns) if spec['kind']=='interaction' else [],
                train_last_label=a.window(train,cutoff,spec['window']).target_time.max().isoformat(),
                first_origin=test.index.min().isoformat(),residualizer=r.metadata(),warnings=warning_list))
    return pd.concat(frames,ignore_index=True),metadata,len(cache)


def main(workers):
    stamp=verify_lock()
    write_new(OUT/'RUN_STARTED.json',dict(**stamp,started_utc=now()))
    start=time.monotonic();checkpoints=OUT/'checkpoints';checkpoints.mkdir()
    months=sorted({m.strftime('%Y-%m') for c in n.FOLDS for m in n.inner_months(c)})
    tasks=[(m,d,f,g,k) for m,d in CASES for f in months for g in GROUPS for k in n.DESIGN['algorithms']]
    inner=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(inner_job,t):t for t in tasks}
        for i,future in enumerate(as_completed(pending),1):
            rows=future.result();inner.extend(rows)
            write_new(checkpoints/('__'.join(pending[future])+'.json'),rows)
            if i%24==0 or i==len(tasks):
                print('INNER',i,'/',len(tasks),'elapsed',round(time.monotonic()-start),flush=True)
    scores=pd.DataFrame(inner).sort_values(['mode','definition','removed_group','fold','candidate'])
    assert len(scores)==7056
    scores.to_csv(OUT/'inner_scores.csv',index=False)
    tasks=[]
    for mode,definition in CASES:
        for cutoff in n.FOLDS:
            months=[x.strftime('%Y-%m') for x in n.inner_months(cutoff)]
            for group in GROUPS:
                s=scores[(scores['mode']==mode)&(scores.definition==definition)&
                    (scores.removed_group==group)&scores.fold.isin(months)]
                tasks.append((mode,definition,cutoff.strftime('%Y-%m'),group,s.to_dict('records')))
    frames,metadata=[],[];outer_fits=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(outer_job,t) for t in tasks]
        for i,future in enumerate(as_completed(pending),1):
            frame,meta,count=future.result();frames.append(frame);metadata.extend(meta);outer_fits+=count
            if i%12==0 or i==len(tasks):
                print('OUTER',i,'/',len(tasks),'elapsed',round(time.monotonic()-start),flush=True)
    raw=pd.concat(frames,ignore_index=True)
    group=['mode','definition','removed_group','procedure','seed']
    fresh=pd.concat([a.calibrate(g) for _,g in raw.groupby(group)],ignore_index=True)
    fresh['evaluation']=fresh.origin>=pd.Timestamp(DESIGN['evaluation_start'])
    assert (fresh.loc[fresh.evaluation,'calibration_n']>=30).all()
    old=read_full()
    for (mode,definition,removed,procedure,seed),got in fresh.groupby(group):
        got=got.sort_values('origin')
        ref=old[(old['mode']==mode)&(old.definition==definition)&(old.seed==seed)].sort_values('origin')
        assert list(got.origin)==list(ref.origin)
        assert list(got.target_time)==list(ref.target_time)
        np.testing.assert_allclose(got[['target','e_now']],ref[['target','e_now']],atol=1e-10,rtol=0)
    columns=['mode','definition','fold','removed_group','procedure','seed','kind','candidate',
        'origin','target_time','target','e_now','pred_raw','pred_calibrated','correction',
        'calibration_n','latest_calibration_label','evaluation']
    pred=pd.concat([fresh[columns],old[columns]],ignore_index=True).sort_values(group+['origin'])
    pred.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_new(OUT/'selected_models.json',metadata)
    pd.DataFrame([{k:v for k,v in m.items() if k not in ['residualizer','warnings']} for m in metadata]).to_csv(OUT/'selected_models.csv',index=False)
    write_new(OUT/'PREDICTIONS_SEALED.json',dict(**stamp,sealed_utc=now(),
        predictions_sha256=sha(OUT/'predictions.csv.gz'),inner_sha256=sha(OUT/'inner_scores.csv'),
        selections_sha256=sha(OUT/'selected_models.json'),inner_fits=len(scores),
        invalid_candidates=int((~scores.valid).sum()),outer_fits=outer_fits,
        warnings=sum(len(w) for w in scores.warnings)+sum(len(m['warnings']) for m in metadata),
        elapsed_seconds=round(time.monotonic()-start,2),independent_confirmation=False,
        full_predictions_refitted=False))
    print('SEALED all forecasts; no outer scores displayed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12)
    main(p.parse_args().workers)
