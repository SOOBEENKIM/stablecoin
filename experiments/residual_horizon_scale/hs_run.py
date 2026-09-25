import argparse
import gzip
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from hs_core import (HERE, ROOT, OUT, OLD, old, HORIZONS, VARIANTS, INFOS, STRATEGIES,
    SEED, START, make_case, columns, calibrate, choose, read, save, sha, now, write_new)
from hs_guard import verify

DATA = {}
CACHE = ROOT/'.runs/residual_horizon_scale'


def key(task):
    return '__'.join(map(str, task))


def fit_job(task):
    h, variant, fold, info, kind = task
    train, test, r = DATA[h, fold]
    cutoff = pd.Timestamp(fold+'-01',tz='UTC')
    frames, metadata = [], []
    for spec in [s for s in old.SPECS if s['kind'] == kind]:
        fit = old.legacy.a.window(train, cutoff, spec['window'])
        _, pred = old.fit_predict(spec, fit, test, columns(info,variant), SEED)
        d = test[['target','target_time','e_now']].reset_index()
        values = dict(horizon=h,variant=variant,fold=fold,information=info,kind=kind,candidate=spec['candidate'],seed=SEED)
        for k,v in values.items():
            d[k] = v
        d['pred_raw'] = pred
        frames.append(d)
        metadata.append(dict(**values,train_n=len(fit),test_n=len(test),features=columns(info,variant),
            train_last_label=fit.target_time.max().isoformat(),residualizer=r.metadata(),spec=spec))
    return pd.concat(frames,ignore_index=True), metadata


def main(workers):
    stamp = verify()
    if (OUT/'PREDICTIONS_SEALED.json').exists():
        raise ValueError('Already sealed')
    if (OUT/'RUN_STARTED.json').exists():
        assert json.loads((OUT/'RUN_STARTED.json').read_text())['lock_sha256'] == stamp['lock_sha256']
    else:
        write_new(OUT/'RUN_STARTED.json',dict(**stamp,started_utc=now(),workers=workers))
    CACHE.mkdir(parents=True,exist_ok=True)
    panel = old.legacy.a.panel('available_macro')
    targets, preflight = [], []
    for cutoff in old.MONTHS:
        for h in HORIZONS:
            train, test, r = make_case(panel,cutoff,h)
            fold = cutoff.strftime('%Y-%m')
            DATA[h,fold] = train,test,r
            d = test.reset_index().copy()
            d['horizon'],d['fold'] = h,fold
            bins = {}
            for col in ['e_now','e_change1','e_rms72']:
                cuts = np.quantile(train[col],[1/3,2/3])
                d['bin_'+col] = np.searchsorted(cuts,d[col],side='right')
                bins[col] = cuts.tolist()
            targets.append(d)
            preflight.append(dict(horizon=h,fold=fold,train_n=len(train),test_n=len(test),train_bins=bins))
    save(pd.concat(targets,ignore_index=True),OUT/'targets.csv.gz')
    (OUT/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    tasks = [(h,v,c.strftime('%Y-%m'),i,k) for h in HORIZONS for v in VARIANTS
             if (h,v)!=(1,'original') for c in old.MONTHS for i in INFOS for k in old.DESIGN['families']]
    todo, metadata = [], []
    for task in tasks:
        note = CACHE/(key(task)+'.json')
        if note.exists():
            m=json.loads(note.read_text())
            assert m['lock_sha256']==stamp['lock_sha256'] and m['sha256']==sha(CACHE/(key(task)+'.pkl.gz'))
            metadata.extend(m['metadata'])
        else:
            todo.append(task)
    count=len(tasks)-len(todo)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(fit_job,t):t for t in todo}
        for f in as_completed(futures):
            task=futures[f]
            frame,meta=f.result()
            path=CACHE/(key(task)+'.pkl.gz')
            frame.to_pickle(path)
            write_new(CACHE/(key(task)+'.json'),dict(**stamp,sha256=sha(path),metadata=meta))
            metadata.extend(meta)
            count+=1
            if count%24==0 or count==len(tasks):
                print('FITTED jobs',count,'/',len(tasks),flush=True)
    selected, selections, all_scores = [], [], []
    for h in HORIZONS:
        for v in VARIANTS:
            for info in INFOS:
                streams=[]
                for kind in old.DESIGN['families']:
                    if (h,v)==(1,'original'):
                        path=OLD/'results/candidates'/('available_macro__EQ__'+info+'__'+kind+'.csv.gz')
                        s=read(path)
                        s=s[s.seed==SEED].drop(columns=['mode','definition'])
                        s['horizon'],s['variant']=h,v
                        raw=s.drop(columns=['correction','calibration_n','latest_calibration_label','pred_calibrated'])
                    else:
                        raw=pd.concat([pd.read_pickle(CACHE/(key((h,v,c.strftime('%Y-%m'),info,kind))+'.pkl.gz')) for c in old.MONTHS],ignore_index=True)
                    cal=pd.concat([calibrate(g,h) for _,g in raw.groupby('candidate',sort=True)],ignore_index=True)
                    if (h,v)==(1,'original'):
                        a=cal.sort_values(['candidate','origin']).reset_index(drop=True)
                        b=s.sort_values(['candidate','origin']).reset_index(drop=True)
                        np.testing.assert_array_equal(a.pred_calibrated,b.pred_calibrated)
                    save(cal,OUT/'candidates'/(key((h,v,info,kind))+'.csv.gz'))
                    streams.append(cal)
                stream=pd.concat(streams,ignore_index=True)
                for cutoff in old.OUTER_MONTHS:
                    fold=cutoff.strftime('%Y-%m')
                    for strategy in STRATEGIES:
                        spec,scores=choose(stream,cutoff,strategy,h)
                        selections.append(dict(horizon=h,variant=v,information=info,fold=fold,strategy=strategy,**spec))
                        if strategy=='ml_selected':
                            all_scores.extend(dict(horizon=h,variant=v,information=info,fold=fold,**x) for x in scores)
                        g=stream[(stream.fold==fold)&(stream.candidate==spec['candidate'])].copy()
                        g['strategy']=strategy
                        selected.append(g)
                print('SELECTED past-only',h,v,info,flush=True)
    save(pd.concat(selected,ignore_index=True),OUT/'predictions.csv.gz')
    save(pd.DataFrame(all_scores),OUT/'inner_scores.csv')
    (OUT/'fit_metadata.json.gz').write_bytes(gzip.compress(json.dumps(metadata).encode(),mtime=0))
    (OUT/'selections.json').write_text(json.dumps(selections,indent=2)+'\n')
    paths=[p for p in OUT.rglob('*') if p.is_file()]
    write_new(OUT/'PREDICTIONS_SEALED.json',dict(**stamp,sealed_utc=now(),new_fits=len(metadata),
        reused_one_hour_candidate_streams=180,independent_confirmation=False,
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print('SEALED. Verification required before scoring.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16)
    main(p.parse_args().workers)
