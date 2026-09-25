"""Reuse audited raw forecasts; refit only missing selected seed/candidate pairs."""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from cc_core import *

DATA={}
CACHE=ROOT/'.runs/residual_conditional_calibration'


def fit_extra(task):
    definition,seed,cid=task
    frames=[];notes=[];spec=old.SPECS[cid]
    for cutoff in old.MONTHS:
        fold=cutoff.strftime('%Y-%m');tr,te,r=DATA[definition,fold]
        fit=old.legacy.a.window(tr,cutoff,spec['window'])
        _,pred=old.fit_predict(spec,fit,te,r12.hs.columns('F','original'),seed)
        d=te[['target','target_time','e_now']].reset_index()
        for k,v in dict(definition=definition,fold=fold,candidate=cid,kind=spec['kind'],seed=seed).items():d[k]=v
        d['pred_raw']=pred;frames.append(d)
        notes.append(dict(definition=definition,fold=fold,candidate=cid,seed=seed,spec=spec,
            train_n=len(fit),test_n=len(te),last_label=fit.target_time.max().isoformat(),residualizer=r.metadata()))
    return pd.concat(frames,ignore_index=True),notes


def main(workers):
    stamp=verify_lock()
    assert not (OUT/'PREDICTIONS_SEALED.json').exists()
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if not (OUT/'RUN_STARTED.json').exists():write_new(OUT/'RUN_STARTED.json',dict(**stamp,started_utc=now()))
    else:assert json.loads((OUT/'RUN_STARTED.json').read_text())['lock_sha256']==stamp['lock_sha256']
    choices=[];scores=[];first={};cids={};todo=[];extra={};fitnotes=[]
    parent=pd.DataFrame(json.loads((PARENT/'results/selections.json').read_text()))
    for definition in DEFS:
        ctx=context_for(definition)
        d=calibrate_streams(first_streams(definition),ctx)
        save(d,OUT/'streams'/f'{definition}__{SEEDS[0]}.csv.gz');first[definition]=d
        for cutoff in old.OUTER_MONTHS:
            fold=cutoff.strftime('%Y-%m');selected,sc=select(d,cutoff)
            for c in selected:
                choices.append(dict(definition=definition,fold=fold,**c))
                if c['policy']=='legacy' and c['family'] in ['linear','threshold','ml_selected']:
                    p=parent[(parent.definition==definition)&(parent.variant=='original')&(parent.information=='F')&
                        (parent.fold==fold)&(parent.strategy==c['family'])].iloc[0]
                    assert int(p.candidate)==c['candidate']
                    assert abs(p.inner_score-c['score'])<1e-12
            scores.extend(dict(definition=definition,fold=fold,**x) for x in sc)
        cids[definition]=sorted({x['candidate'] for x in choices if x['definition']==definition and x['family']=='ml_selected'})
        for seed in SEEDS[1:]:
            archived=read(PARENT/'results/extra_candidates'/f'{definition}__original__F__{seed}.csv.gz')
            extra[definition,seed]=archived[archived.candidate.isin(cids[definition])].copy()
            for cid in cids[definition]:
                if cid not in set(archived.candidate):todo.append((definition,seed,cid))
        print('SELECTED using prior months only',definition,'ML candidates',cids[definition],flush=True)
    panel=old.legacy.a.panel('available_macro')
    for definition in sorted({t[0] for t in todo}):
        for cutoff in old.MONTHS:DATA[definition,cutoff.strftime('%Y-%m')]=r12.case(panel,definition,cutoff)
    pending=[]
    def accept(task,d,notes):
        key=task[:2];extra[key]=pd.concat([extra[key],d],ignore_index=True);fitnotes.extend(notes)
    for task in todo:
        path=CACHE/('__'.join(map(str,task))+'.pkl.gz');note=path.with_suffix('.json')
        if note.exists():
            meta=json.loads(note.read_text());assert meta['lock_sha256']==stamp['lock_sha256'] and meta['sha256']==sha(path)
            accept(task,pd.read_pickle(path),meta['fits'])
        else:pending.append(task)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(fit_extra,t):t for t in pending}
        for n,fut in enumerate(as_completed(futures),1):
            task=futures[fut];d,notes=fut.result();path=CACHE/('__'.join(map(str,task))+'.pkl.gz')
            d.to_pickle(path);write_new(path.with_suffix('.json'),dict(**stamp,sha256=sha(path),fits=notes))
            accept(task,d,notes);print('REFIT extra seed streams',n,'/',len(pending),flush=True)
    predictions=[];fixed=[]
    for definition in DEFS:
        ctx=context_for(definition)
        streams={SEEDS[0]:first[definition]}
        for seed in SEEDS[1:]:
            streams[seed]=calibrate_streams(aligned(extra[definition,seed],ctx),ctx)
            save(streams[seed],OUT/'streams'/f'{definition}__{seed}.csv.gz')
        for c in [c for c in choices if c['definition']==definition]:
            for seed in (SEEDS if c['family']=='ml_selected' else SEEDS[:1]):
                d=streams[seed]
                g=d[(d.candidate==c['candidate'])&(d.fold==c['fold'])].copy()
                assert len(g)>0
                g['policy']=c['policy'];g['family']=c['family'];g['rule']=c['rule'];g['q']=g['q_'+c['rule']]
                predictions.append(g)
                if c['policy']=='legacy':
                    for rule in RULES:
                        f=g.copy();f['rule']=rule;f['q']=f['q_'+rule];fixed.append(f)
    save(pd.concat(predictions,ignore_index=True),OUT/'predictions.csv.gz')
    save(pd.concat(fixed,ignore_index=True),OUT/'fixed_rule_predictions.csv.gz')
    save(pd.DataFrame(scores),OUT/'inner_scores.csv')
    save(pd.DataFrame(choices),OUT/'choices.csv')
    write_new(OUT/'new_fit_metadata.json',fitnotes)
    files=[p for p in OUT.rglob('*') if p.is_file()]
    write_new(OUT/'PREDICTIONS_SEALED.json',dict(**stamp,sealed_utc=now(),new_fits=len(fitnotes),
        independent_confirmation=False,file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('SEALED; independent verification precedes performance tables.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12)
    main(p.parse_args().workers)
