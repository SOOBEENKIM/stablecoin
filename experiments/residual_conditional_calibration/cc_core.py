"""Past-only empirical conditional calibration; no conformal coverage guarantee."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
PREP=HERE/'prepared'
sys.path.insert(0,str(HERE.parent/'residual_decision_value'))
import dv_core as dv
from dv_alignment import verify_alignment
r12=dv.r12
old=r12.old
PARENT=dv.PARENT
DEFS,SEEDS=dv.DEFS,dv.SEEDS
read,save,sha,now,write_new=dv.read,dv.save,dv.sha,dv.now,dv.write_new
RULES=['global60','global180','scaled180','local180','local_scaled180',
       'local_shrink180','local_scaled_shrink180']
POLICIES=['legacy','fixed_model_selected_cal','joint_selected','global_window_selected']
FAMILIES=['linear','threshold','ml_selected','historical','state_hist']
BRANCH='research/icaif2026-manuscript-development'
LOCK=HERE/'LOCK.json'


def context_for(definition):
    return read(PREP/'context.csv.gz').query('definition == @definition').sort_values('origin').reset_index(drop=True)


def aligned(d,ctx):
    """Exact key/label join, including archived extra seeds without e_now."""
    keys=['definition','origin','fold']
    assert not ctx.duplicated(keys).any()
    lookup=ctx.set_index(keys).reindex(pd.MultiIndex.from_frame(d[keys]))
    assert not lookup.target.isna().any()
    np.testing.assert_allclose(d.target,lookup.target,atol=1e-11,rtol=0)
    assert list(d.target_time)==list(lookup.target_time)
    out=d.copy()
    for col in ['e_now','e_rms72','e_rank','vol_rank','state']:
        if col in out:
            mask=out[col].notna().to_numpy()
            np.testing.assert_allclose(out[col].to_numpy()[mask],lookup[col].to_numpy()[mask],atol=1e-11,rtol=0)
        out[col]=lookup[col].to_numpy()
    assert np.isfinite(out[['target','pred_raw','e_now','e_rms72','e_rank','vol_rank']]).all().all()
    return out


def first_streams(definition):
    ctx=context_for(definition)
    frames=[read(PARENT/'results/candidates'/f'{definition}__original__F__{k}.csv.gz') for k in ['linear','threshold','qrf','boosting']]
    baseline=read(dv.OUT/'baseline_streams.csv.gz')
    for cid,name in [(60,'historical'),(61,'state_hist')]:
        d=baseline[(baseline.definition==definition)&(baseline.strategy==name)].copy()
        d['candidate']=cid;d['kind']=name;frames.append(d)
    d=aligned(pd.concat(frames,ignore_index=True),ctx)
    assert set(d.candidate)==set(range(62)) and set(d.seed)=={SEEDS[0]}
    for _,g in d.groupby('candidate'):
        assert list(g.sort_values('origin').origin)==list(ctx.origin)
    return d


def plans(ctx):
    t=pd.DatetimeIndex(ctx.origin).asi8
    lab=pd.DatetimeIndex(ctx.target_time).asi8
    assert (np.diff(t)>0).all() and (np.diff(lab)>0).all()
    xy=ctx[['e_rank','vol_rank']].to_numpy()
    age=pd.Timedelta(days=90).value
    out=[]
    for i,origin in enumerate(t):
        lo=np.searchsorted(lab,origin-age,side='left')
        hi=np.searchsorted(lab,origin,side='left')
        pool=np.arange(max(lo,hi-180),hi,dtype=int)
        distance=np.sum((xy[pool]-xy[i])**2,axis=1)
        local=pool[np.lexsort((-pool,distance))[:60]]
        out.append((pool[-60:],pool,local))
    return out


def q10(a):
    return np.quantile(a,.1,axis=0,method='inverted_cdf')


def calibrate_matrix(ctx,raw,plan=None):
    raw=np.asarray(raw,float)
    if raw.ndim==1:raw=raw[:,None]
    assert len(raw)==len(ctx) and np.isfinite(raw).all()
    plan=plans(ctx) if plan is None else plan
    error=ctx.target.to_numpy()[:,None]-raw
    sigma=np.maximum(ctx.e_rms72.to_numpy(),1.)
    corrections=np.zeros((len(ctx),raw.shape[1],len(RULES)))
    for i,(short,pool,local) in enumerate(plan):
        if len(short)<30:continue
        a,b=q10(error[short]),q10(error[pool])
        c=sigma[i]*q10(error[pool]/sigma[pool,None])
        d=q10(error[local]);e=sigma[i]*q10(error[local]/sigma[local,None])
        corrections[i]=np.stack([a,b,c,d,e,.5*(b+d),.5*(c+e)],axis=1)
    return raw[:,:,None]+corrections


def calibrate_streams(d,ctx):
    ids=sorted(d.candidate.unique())
    raw=np.column_stack([d[d.candidate==cid].sort_values('origin').pred_raw for cid in ids])
    cal=calibrate_matrix(ctx,raw)
    frames=[]
    for j,cid in enumerate(ids):
        g=d[d.candidate==cid].sort_values('origin').reset_index(drop=True).copy()
        for k,rule in enumerate(RULES):g['q_'+rule]=cal[:,j,k]
        # Legacy global calibration must remain exactly equivalent.
        if 'pred_calibrated' in g and g.pred_calibrated.notna().all():
            np.testing.assert_allclose(g.q_global60,g.pred_calibrated,atol=1e-11,rtol=0)
        frames.append(g)
    return pd.concat(frames,ignore_index=True)


def select(d,cutoff):
    inner=r12.inner_rows(d,cutoff)
    assert set(inner.candidate)==set(range(62))
    assert inner.origin.min()>=cutoff-pd.offsets.MonthBegin(3)
    assert inner.target_time.max()<cutoff
    rows=[]
    for cid,g in inner.groupby('candidate',sort=True):
        g=g.sort_values('origin')
        for rank,rule in enumerate(RULES):
            rows.append(dict(candidate=int(cid),kind=g.kind.iloc[0],rule=rule,rule_rank=rank,
                score=float(old.loss(g.target,g['q_'+rule]).mean()),n=len(g)))
    choices=[]
    for family in FAMILIES:
        kinds=['qrf','boosting'] if family=='ml_selected' else [family]
        pool=[x for x in rows if x['kind'] in kinds]
        legacy=min([x for x in pool if x['rule']=='global60'],key=lambda x:(x['score'],x['candidate']))
        for policy in POLICIES:
            eligible=pool
            if policy=='legacy':eligible=[legacy]
            elif policy=='fixed_model_selected_cal':eligible=[x for x in pool if x['candidate']==legacy['candidate']]
            elif policy=='global_window_selected':eligible=[x for x in pool if x['rule'] in RULES[:2]]
            choice=min(eligible,key=lambda x:(x['score'],x['candidate'],x['rule_rank']))
            choices.append(dict(family=family,policy=policy,**choice))
    return choices,rows


def verify_inputs():
    dv.verify_inputs();verify_alignment()


def create_lock():
    verify_inputs()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'PRE_RUN_TESTS.txt']+list(PREP.glob('*'))
    files+=list((PARENT/'results/candidates').glob('*__original__F__*.csv.gz'))
    files+=list((PARENT/'results/extra_candidates').glob('*__original__F__*.csv.gz'))
    files+=[PARENT/'results/targets.csv.gz',PARENT/'results/selections.json',dv.OUT/'baseline_streams.csv.gz']
    write_new(LOCK,dict(created_utc=now(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        independent_confirmation=False,file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))


def verify_lock():
    verify_inputs()
    lock=json.loads(LOCK.read_text())
    for n,s in lock['file_sha256'].items():assert sha(ROOT/n)==s,n
    rel=str(LOCK.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==LOCK.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True)
    return dict(lock_sha256=sha(LOCK),lock_commit=commit)


def verify_seal():
    stamp=verify_lock()
    seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    assert seal['lock_sha256']==stamp['lock_sha256']
    for n,s in seal['file_sha256'].items():assert sha(ROOT/n)==s,n
    return stamp


if __name__=='__main__':create_lock() if sys.argv[1]=='lock' else print(verify_lock())
