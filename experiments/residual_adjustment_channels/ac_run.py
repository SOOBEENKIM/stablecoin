import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from ac_core import *

CACHE=ROOT/'.runs/residual_adjustment_channels'


def fit(task):
    definition,fold,h,learner,seed=task
    case=pd.read_pickle(PREP/'cases'/f'{definition}__{fold}__{h}.pkl.gz')
    # Fit all nonzero gaps once; fixed 0/5/10-bp analysis subsets share nuisances.
    tr=case['train'].query('abs_e > 0');te=case['test'].query('abs_e > 0')
    cutoff=pd.Timestamp(fold+'-01',tz='UTC')
    assert tr.target_time.max()<cutoff and min(tr.premium.sum(),(tr.premium==0).sum())>=20
    d=model_fit(tr,te,learner,seed)
    for k,v in dict(definition=definition,fold=fold,h=h).items():d[k]=v
    note=dict(definition=definition,fold=fold,h=h,learner=learner,seed=seed,train_n=len(tr),test_n=len(te),
        last_train_label=tr.target_time.max().isoformat(),features=FEATURES)
    return d,note


def main(workers):
    stamp=verify_lock();assert not (OUT/'PREDICTIONS_SEALED.json').exists()
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'RUN_STARTED.json').exists():write_new(OUT/'RUN_STARTED.json',dict(**stamp,started_utc=now()))
    assert json.loads((OUT/'RUN_STARTED.json').read_text())['lock_sha256']==stamp['lock_sha256']
    frames=[];notes=[];todo=[]
    tasks=[(d,c.strftime('%Y-%m'),h,l,s) for d in DEFS for c in old.OUTER_MONTHS for h in HORIZONS for l in LEARNERS for s in (SEEDS[:1] if l=='linear' else SEEDS)]
    for task in tasks:
        name='__'.join(map(str,task));path=CACHE/(name+'.pkl.gz');note=CACHE/(name+'.json')
        if note.exists():
            n=json.loads(note.read_text());assert n['lock_sha256']==stamp['lock_sha256'] and n['sha256']==sha(path)
            frames.append(pd.read_pickle(path));notes.append(n['fit'])
        else:todo.append(task)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(fit,t):t for t in todo}
        for i,f in enumerate(as_completed(futures),1):
            task=futures[f];d,n=f.result();name='__'.join(map(str,task));path=CACHE/(name+'.pkl.gz')
            d.to_pickle(path);write_new(CACHE/(name+'.json'),dict(**stamp,sha256=sha(path),fit=n))
            frames.append(d);notes.append(n)
            if i%24==0 or i==len(todo):print('FITTED',i,'/',len(todo),flush=True)
    pred=pd.concat(frames,ignore_index=True).sort_values(['definition','fold','h','learner','seed','origin'])
    save(pred,OUT/'predictions.csv.gz');write_new(OUT/'fit_metadata.json',notes)
    files=[p for p in OUT.glob('*') if p.is_file()]
    write_new(OUT/'PREDICTIONS_SEALED.json',dict(**stamp,sealed_utc=now(),fits=len(notes),
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('SEALED predictions; no hypothesis results opened.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=12)
    main(p.parse_args().workers)
