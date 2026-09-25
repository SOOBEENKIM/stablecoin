import argparse
import gzip
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from r12_core import (HERE,ROOT,OUT,PARENT,old,hs,DEFS,VARIANTS,INFOS,STRATEGIES,SEEDS,FAMILIES,
    case,calibrate,select_all,standardize,file_key,read,save,sha,now,write_new)
from r12_guard import verify

DATA={}
CACHE=ROOT/'.runs/residual_12h_robustness'


def one_fit(definition,variant,info,fold,spec,seed):
    tr,te,r=DATA[definition,fold]
    cutoff=pd.Timestamp(fold+'-01',tz='UTC')
    fit=old.legacy.a.window(tr,cutoff,spec['window'])
    _,pred=old.fit_predict(spec,fit,te,hs.columns(info,variant),seed)
    d=te[['target','target_time','e_now']].reset_index()
    values=dict(definition=definition,variant=variant,information=info,fold=fold,
                candidate=spec['candidate'],kind=spec['kind'],seed=seed,horizon=12)
    for k,v in values.items():d[k]=v
    d['pred_raw']=pred
    note=dict(**values,features=hs.columns(info,variant),spec=spec,train_n=len(fit),test_n=len(te),
              train_last_label=fit.target_time.max().isoformat(),residualizer=r.metadata())
    return d,note


def first_job(task):
    definition,variant,info,fold,kind=task
    frames=[];meta=[]
    for spec in [s for s in old.SPECS if s['kind']==kind]:
        d,n=one_fit(definition,variant,info,fold,spec,SEEDS[0]);frames.append(d);meta.append(n)
    return pd.concat(frames,ignore_index=True),meta


def extra_job(task):
    definition,variant,info,seed,cids=task
    frames=[];meta=[]
    for cid in cids:
        stream=[]
        for cutoff in old.MONTHS:
            d,n=one_fit(definition,variant,info,cutoff.strftime('%Y-%m'),old.SPECS[cid],seed)
            stream.append(d);meta.append(n)
        frames.append(calibrate(pd.concat(stream,ignore_index=True)))
    return pd.concat(frames,ignore_index=True),meta


def run_jobs(tasks,worker,stage,stamp,workers):
    metadata=[];todo=[]
    for task in tasks:
        name=stage+'__'+file_key(*task)
        note=CACHE/(name+'.json')
        if note.exists():
            saved=json.loads(note.read_text());assert saved['lock_sha256']==stamp['lock_sha256']
            assert sha(CACHE/(name+'.pkl.gz'))==saved['sha256'];metadata.extend(saved['metadata'])
        else:todo.append((task,name))
    count=len(tasks)-len(todo)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(worker,t):(t,n) for t,n in todo}
        for fut in as_completed(futures):
            task,name=futures[fut];d,meta=fut.result();path=CACHE/(name+'.pkl.gz')
            d.to_pickle(path)
            write_new(CACHE/(name+'.json'),dict(**stamp,sha256=sha(path),metadata=meta))
            metadata.extend(meta);count+=1
            if count%24==0 or count==len(tasks):print('FITTED',stage,count,'/',len(tasks),flush=True)
    return metadata


def calibrate_file(task):
    definition,variant,info,kind=task
    if definition=='EQ':
        d=standardize(read(PARENT/'results/candidates'/file_key(12,variant,info,kind+'.csv.gz')),'EQ')
    else:
        raw=pd.concat([pd.read_pickle(CACHE/('first__'+file_key(definition,variant,info,c.strftime('%Y-%m'),kind)+'.pkl.gz')) for c in old.MONTHS],ignore_index=True)
        d=pd.concat([calibrate(g) for _,g in raw.groupby('candidate',sort=True)],ignore_index=True)
    path=OUT/'candidates'/(file_key(*task)+'.csv.gz');save(d,path)
    return len(d)


def main(workers):
    stamp=verify()
    if (OUT/'PREDICTIONS_SEALED.json').exists():raise ValueError('Already sealed')
    if (OUT/'RUN_STARTED.json').exists():assert json.loads((OUT/'RUN_STARTED.json').read_text())['lock_sha256']==stamp['lock_sha256']
    else:write_new(OUT/'RUN_STARTED.json',dict(**stamp,started_utc=now(),workers=workers))
    CACHE.mkdir(parents=True,exist_ok=True)
    panel=old.legacy.a.panel('available_macro');targets=[];states=[]
    state_cols=['e_now','e_change1','e_rms72','btc_vol24','account_log','funding_bp','oi_ret1','oi_change24','local_volume_surprise','fx_ret1']
    for cutoff in old.MONTHS:
        indexes=None
        for definition in DEFS:
            tr,te,r=case(panel,definition,cutoff);fold=cutoff.strftime('%Y-%m')
            DATA[definition,fold]=(tr,te,r)
            if indexes is None:indexes=te.index
            else:pd.testing.assert_index_equal(te.index,indexes)
            d=te.reset_index();d['definition']=definition;d['fold']=fold;targets.append(d)
            for col in state_cols:
                lo,med,hi=np.quantile(tr[col],[.01,.5,.99])
                q=np.quantile(te[col],[.01,.5,.99])
                states.append(dict(definition=definition,fold=fold,feature=col,train_n=len(tr),test_n=len(te),
                    train_p01=lo,train_median=med,train_p99=hi,test_p01=q[0],test_median=q[1],test_p99=q[2],
                    below_train_p01=float((te[col]<lo).mean()),above_train_p99=float((te[col]>hi).mean()),
                    residualizer=r.metadata()))
    save(pd.concat(targets,ignore_index=True),OUT/'targets.csv.gz')
    # These diagnostics are generated now but not interpreted until verification.
    (OUT/'state_distributions.json').write_text(json.dumps(states,indent=2)+'\n')
    tasks=[(d,v,i,c.strftime('%Y-%m'),k) for d in ['CAP','PCA'] for v in VARIANTS for i in INFOS for c in old.MONTHS for k in FAMILIES]
    metadata=run_jobs(tasks,first_job,'first',stamp,workers)
    cal_tasks=[(d,v,i,k) for d in DEFS for v in VARIANTS for i in INFOS for k in FAMILIES]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for n,_ in enumerate(pool.map(calibrate_file,cal_tasks),1):
            if n%12==0:print('CALIBRATED files',n,'/',len(cal_tasks),flush=True)
    predictions=[];selections=[];all_scores=[];choices={}
    parent_s=pd.DataFrame(json.loads((PARENT/'results/selections.json').read_text()))
    for definition in DEFS:
        for variant in VARIANTS:
            for info in INFOS:
                s=pd.concat([read(OUT/'candidates'/(file_key(definition,variant,info,k)+'.csv.gz')) for k in FAMILIES],ignore_index=True)
                cids=set()
                for cutoff in old.OUTER_MONTHS:
                    fold=cutoff.strftime('%Y-%m');picked,scores=select_all(s,cutoff)
                    context=dict(definition=definition,variant=variant,information=info,fold=fold)
                    all_scores.extend(dict(**context,**x) for x in scores)
                    for strategy,spec in picked.items():
                        choices[definition,variant,info,fold,strategy]=spec['candidate']
                        selections.append(dict(**context,strategy=strategy,**spec))
                        z=s[(s.fold==fold)&(s.candidate==spec['candidate'])].copy();z['strategy']=strategy
                        predictions.append(z)
                        if strategy=='ml_selected':cids.add(spec['candidate'])
                        if definition=='EQ':
                            a=parent_s[(parent_s.horizon==12)&(parent_s.variant==variant)&(parent_s.information==info)&(parent_s.fold==fold)&(parent_s.strategy==strategy)].iloc[0]
                            assert int(a.candidate)==spec['candidate'] and a.inner_score==spec['inner_score']
                choices[definition,variant,info,'cids']=tuple(sorted(cids))
                print('SELECTED past only',definition,variant,info,flush=True)
    extra_tasks=[(d,v,i,seed,choices[d,v,i,'cids']) for d in DEFS for v in VARIANTS for i in INFOS
                 for seed in SEEDS[1:] if (d,i)!=('EQ','F')]
    metadata+=run_jobs(extra_tasks,extra_job,'extra',stamp,workers)
    eq_extra=standardize(read(PARENT/'results/seed_sensitivity/candidate_streams.csv.gz'),'EQ')
    for definition in DEFS:
        for variant in VARIANTS:
            for info in INFOS:
                for seed in SEEDS[1:]:
                    if (definition,info)==('EQ','F'):
                        stream=eq_extra[(eq_extra.variant==variant)&(eq_extra.seed==seed)].copy()
                        stream['kind']='qrf'
                    else:
                        task=(definition,variant,info,seed,choices[definition,variant,info,'cids'])
                        stream=pd.read_pickle(CACHE/('extra__'+file_key(*task)+'.pkl.gz'))
                    save(stream,OUT/'extra_candidates'/(file_key(definition,variant,info,seed)+'.csv.gz'))
                    for cutoff in old.OUTER_MONTHS:
                        fold=cutoff.strftime('%Y-%m');cid=choices[definition,variant,info,fold,'ml_selected']
                        z=stream[(stream.fold==fold)&(stream.candidate==cid)].copy()
                        assert len(z)>0;z['strategy']='ml_selected';predictions.append(z)
    save(pd.concat(predictions,ignore_index=True),OUT/'predictions.csv.gz')
    save(pd.DataFrame(all_scores),OUT/'inner_scores.csv')
    (OUT/'selections.json').write_text(json.dumps(selections,indent=2)+'\n')
    (OUT/'fit_metadata.json.gz').write_bytes(gzip.compress(json.dumps(metadata).encode(),mtime=0))
    files=[p for p in OUT.rglob('*') if p.is_file()]
    write_new(OUT/'PREDICTIONS_SEALED.json',dict(**stamp,sealed_utc=now(),new_fits=len(metadata),
        independent_confirmation=False,file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('SEALED forecasts; verification required.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16)
    main(p.parse_args().workers)
