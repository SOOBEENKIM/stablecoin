"""Transparent post-primary extension: directional reversal versus absolute closure."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from ac_core import *
from ac_evaluate import inference

DEST=OUT/'overshoot';XLOCK=HERE/'OVERSHOOT_LOCK.json'
QUANTITIES=['gap','excess','cross','overshot']


def verify_parent_results():
    stamp=verify_seal()
    e=json.loads((OUT/'EVALUATION_COMPLETE.json').read_text())
    for name,digest in e['file_sha256'].items():assert sha(ROOT/name)==digest,name
    return stamp


def actual(d):
    a=d.copy()
    a['gap']=abs(a.e_now)-abs(a.target)
    a['excess']=2*abs(a.target)*(a.e_now*a.target<0)
    a['cross']=(a.e_now*a.target<0).astype(float)
    a['overshot']=((a['cross']==1)&(abs(a.target)>abs(a.e_now))).astype(float)
    np.testing.assert_allclose(a.gap+a.excess,a.close_total,atol=1e-10,rtol=0)
    return a


def learn(tr,te,learner,seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    import warnings
    tr=actual(tr);y=tr[['gap','cross','overshot']].to_numpy()
    center=y.mean(axis=0);scale=y.std(axis=0,ddof=1);scale=np.where(scale>1e-8,scale,1.)
    if learner=='linear':model=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    else:model=RandomForestRegressor(n_estimators=300,min_samples_leaf=int(learner.replace('forest','')),
        max_depth=6,max_features=.7,random_state=seed,n_jobs=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');model.fit(tr[FEATURES],(y-center)/scale)
        g=model.predict(te[FEATURES])*scale+center
    if w:raise RuntimeError('; '.join(str(z.message) for z in w))
    assert np.isfinite(g).all()
    return g,center


def fit(task):
    definition,fold,learner,seed=task
    case=pd.read_pickle(PREP/'cases'/f'{definition}__{fold}__12.pkl.gz')
    tr=case['train'].query('abs_e>0');te=case['test'].query('abs_e>0')
    assert tr.target_time.max()<pd.Timestamp(fold+'-01',tz='UTC')
    g,center=learn(tr,te,learner,seed)
    d=te.reset_index()[['origin']].copy()
    for j,c in enumerate(['gap','cross','overshot']):d['g_'+c]=g[:,j];d['constant_'+c]=center[j]
    for k,v in dict(definition=definition,fold=fold,h=12,learner=learner,seed=seed).items():d[k]=v
    return d


def create_lock():
    verify_parent_results()
    files=[Path(__file__),HERE/'OVERSHOOT_PROTOCOL_KO.md',HERE/'test_ac_overshoot.py',HERE/'OVERSHOOT_PRE_RUN_TESTS.txt',
        LOCK,OUT/'EVALUATION_COMPLETE.json',OUT/'PREDICTIONS_SEALED.json']
    write_new(XLOCK,dict(created_utc=now(),primary_results_already_seen=True,independent_confirmation=False,
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))


def check():
    verify_parent_results();d=json.loads(XLOCK.read_text())
    for name,digest in d['file_sha256'].items():assert sha(ROOT/name)==digest,name
    rel=str(XLOCK.relative_to(ROOT));assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==XLOCK.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True)
    return dict(lock_commit=commit,lock_sha256=sha(XLOCK))


def run(workers):
    stamp=check();DEST.mkdir(parents=True,exist_ok=True)
    write_new(DEST/'STARTED.json',dict(**stamp,started_utc=now()))
    tasks=[(d,c.strftime('%Y-%m'),l,s) for d in DEFS for c in old.OUTER_MONTHS for l in LEARNERS for s in (SEEDS[:1] if l=='linear' else SEEDS)]
    frames=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(fit,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            frames.append(f.result())
            if i%20==0:print('OVERSHOOT FIT',i,'/',len(tasks),flush=True)
    extra=pd.concat(frames,ignore_index=True)
    pred=read(OUT/'predictions.csv.gz');pred=pred[pred.h==12]
    keys=['definition','fold','h','learner','seed','origin']
    d=actual(pred.merge(extra,on=keys,validate='one_to_one'))
    assert len(d)==len(pred)
    d['g_excess']=d.g_total-d.g_gap
    d['constant_excess']=d.g_constant_total-d.constant_gap
    save(d.sort_values(keys),DEST/'predictions.csv.gz')
    write_new(DEST/'SEALED.json',dict(**stamp,sealed_utc=now(),fits=len(tasks),predictions_sha256=sha(DEST/'predictions.csv.gz')))


def verify():
    stamp=check();seal=json.loads((DEST/'SEALED.json').read_text())
    assert seal['predictions_sha256']==sha(DEST/'predictions.csv.gz')
    d=read(DEST/'predictions.csv.gz');base=read(OUT/'predictions.csv.gz');base=base[base.h==12]
    keys=['definition','fold','h','learner','seed','origin'];cols=['m','g_total','close_total']
    d=d.sort_values(keys);base=base.sort_values(keys)
    np.testing.assert_array_equal(d[keys],base[keys]);np.testing.assert_array_equal(d[cols],base[cols])
    # Scalar piecewise identity, including crossing and overshooting cases.
    error=0.
    for row in d.itertuples():
        gap=abs(row.e_now)-abs(row.target)
        extra=0. if row.e_now*row.target>=0 else 2*abs(row.target)
        error=max(error,abs(row.gap-gap),abs(row.excess-extra),abs(row.gap+row.excess-row.close_total))
    assert error<1e-9
    refit_error=0.;n=0
    for definition in DEFS:
        case=pd.read_pickle(PREP/'cases'/f'{definition}__2026-01__12.pkl.gz')
        for learner in LEARNERS:
            g,_=learn(case['train'].query('abs_e>0'),case['test'].query('abs_e>0'),learner,SEEDS[0])
            p=d[(d.definition==definition)&(d.fold=='2026-01')&(d.learner==learner)&(d.seed==SEEDS[0])].sort_values('origin')
            refit_error=max(refit_error,float(np.max(abs(p[['g_gap','g_cross','g_overshot']].to_numpy()-g))));n+=1
    assert refit_error<1e-10
    write_new(DEST/'VERIFIED.json',dict(**stamp,verified_utc=now(),piecewise_identity_error=error,refits=n,
        refit_error=refit_error,predictions_sha256=sha(DEST/'predictions.csv.gz')))


def evaluate():
    stamp=check();v=json.loads((DEST/'VERIFIED.json').read_text())
    assert v['predictions_sha256']==sha(DEST/'predictions.csv.gz')
    write_new(DEST/'SCORES_OPENED.json',dict(**stamp,opened_utc=now()))
    pred=read(DEST/'predictions.csv.gz');rows=[];quality=[]
    for (definition,learner,seed),g in pred.groupby(['definition','learner','seed']):
        g=g.sort_values('origin').reset_index(drop=True)
        key=dict(definition=definition,learner=learner,seed=int(seed))
        base=g[g.abs_e>5].copy();samples=[('full',5,base,b) for b in [1,5,10]]
        samples += [('full',t,g[g.abs_e>t],5) for t in [0,10]]
        samples += [('nonoverlap',5,base[dv.nonoverlap(base)],5),('overlap',5,base[base.m.between(.1,.9)],5)]
        for fold in sorted(g.fold.unique()):samples += [(fold,5,base[base.fold==fold],5),('without_'+fold,5,base[base.fold!=fold],5)]
        for sample,threshold,d,block in samples:
            if len(d)<10 or d.premium.nunique()!=2:continue
            d=d.reset_index(drop=True);w=dv.day_weights(d[['origin','fold']],block)
            vv=(d.premium-d.m).to_numpy();den=w@(vv*vv);est={}
            for quantity in QUANTITIES:
                u=(d[quantity]-d['g_'+quantity]).to_numpy();num=vv*u
                point=float(num.sum()/sum(vv*vv));draws=np.divide(w@num,den,out=np.full(len(w),np.nan),where=den>1e-12)
                est[quantity]=point
                rows.append(dict(**key,sample=sample,minimum_gap_bp=threshold,block_days=block,n=len(d),
                    quantity=quantity,**inference(point,draws)))
            assert abs(est['gap']+est['excess']-orthogonal(d,'total'))<1e-9
        for quantity in QUANTITIES:
            quality.append(dict(**key,quantity=quantity,mse=float(np.mean((base[quantity]-base['g_'+quantity])**2)),
                constant_mse=float(np.mean((base[quantity]-base['constant_'+quantity])**2))))
    rows=pd.DataFrame(rows)
    main=rows[(rows.learner=='forest40')&(rows.seed==SEEDS[0])&(rows['sample']=='full')&(rows.minimum_gap_bp==5)&(rows.block_days==5)].copy()
    assert len(main)==12
    main['p_holm4']=np.nan
    for _,g in main.groupby('definition'):main.loc[g.index,'p_holm4']=dv.holm(g.p)
    main['p_holm12']=dv.holm(main.p)
    prior=pd.read_csv(OUT/'primary_tests.csv')
    main['p_holm30']=dv.holm(pd.concat([prior.p,main.p],ignore_index=True))[-len(main):]
    for name,df in [('contrasts',rows),('primary_tests',main),('nuisance_quality',pd.DataFrame(quality))]:save(df,DEST/(name+'.csv'))
    files=list(DEST.glob('*.csv'))+[DEST/'SCORES_OPENED.json']
    write_new(DEST/'COMPLETE.json',dict(**stamp,completed_utc=now(),primary_results_already_seen=True,
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print(main[['definition','quantity','n','estimate','lo','hi','p_holm4','p_holm30']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['lock','run','verify','evaluate']);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    if a.command=='lock':create_lock()
    elif a.command=='run':run(a.workers)
    elif a.command=='verify':verify()
    else:evaluate()
