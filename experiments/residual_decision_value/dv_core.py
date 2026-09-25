"""Fixed downstream decisions for archived, temporally validated forecasts."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'residual_12h_robustness'
OUT=HERE/'results'
sys.path.insert(0,str(PARENT))
import r12_core as r12
from r12_guard import verify_seal as verify_parent
from factorial_stats import day_weights,holm,bootstrap_mean

read,save,sha,now,write_new=r12.read,r12.save,r12.sha,r12.now,r12.write_new
DEFS=r12.DEFS
SEEDS=r12.SEEDS
ALPHA=.1
MONTHS=pd.date_range('2025-08-01','2026-03-01',freq='MS',tz='UTC')
CONFIGS=[('original','F'),('original','R'),('original','B'),('own_scale','F')]
REFERENCES=['linear','threshold','historical','state_hist']
PARTS=['delta_local_bp','delta_quote_bp','delta_fx_bp','delta_market_bp','delta_g_bp']
LOCK=HERE/'LOCK.json'


def fit_state(train,cutoff):
    tr=train[train.target_time<cutoff]
    if len(tr)<100:raise ValueError('Insufficient prior labels')
    cuts=np.quantile(tr.e_now,[1/3,2/3])
    v=float(np.median(tr.e_rms72))
    q=float(np.quantile(tr.target-tr.e_now,ALPHA,method='inverted_cdf'))
    model=dict(e_cuts=cuts.tolist(),vol_cut=v,historical_q=q,n=len(tr),
        last_label=tr.target_time.max().isoformat(),state_q={},counts={})
    states=assign_state(tr,model)
    for state in range(6):
        z=(tr.target-tr.e_now)[states==state]
        model['counts'][str(state)]=len(z)
        model['state_q'][str(state)]=float(np.quantile(z,ALPHA,method='inverted_cdf')) if len(z)>=25 else q
    return model


def assign_state(d,model):
    return np.searchsorted(model['e_cuts'],np.asarray(d.e_now),side='right')*2+(np.asarray(d.e_rms72)>=model['vol_cut']).astype(int)


def baseline_predict(d,model,strategy):
    if strategy=='historical':delta=np.full(len(d),model['historical_q'])
    elif strategy=='state_hist':delta=np.array([model['state_q'][str(s)] for s in assign_state(d,model)])
    else:raise ValueError(strategy)
    return np.asarray(d.e_now)+delta


def pinball(y,q):
    y,q=np.asarray(y),np.asarray(q)
    return ALPHA*np.maximum(y-q,0)+(1-ALPHA)*np.maximum(q-y,0)


def decisions(y,q,c):
    """A lower-tail alarm; miss cost .9 and false-alarm cost .1."""
    y,q=np.asarray(y),np.asarray(q)
    event=y<c;alarm=q<c
    tp=alarm&event;fp=alarm&~event;fn=~alarm&event
    return dict(event=event.astype(float),alarm=alarm.astype(float),tp=tp.astype(float),
        fp=fp.astype(float),fn=fn.astype(float),cost=ALPHA*fp+(1-ALPHA)*fn,
        missed_severity=np.maximum(c-y,0)*(~alarm))


def independent_elementary_integral(y,q):
    """Exact finite interval integral; independent scalar decision accounting."""
    lo,hi=sorted([float(y),float(q)])
    if lo==hi:return 0.
    c=(lo+hi)/2
    a,e=q<c,y<c
    value=(ALPHA if a and not e else (1-ALPHA) if e and not a else 0.)
    return (hi-lo)*value


def nonoverlap(d):
    if not pd.DatetimeIndex(d.origin).is_monotonic_increasing:raise ValueError('Unordered origins')
    out=[];last=None
    for row in d.itertuples():
        keep=last is None or row.origin>=last
        out.append(keep)
        if keep:last=row.target_time
    return np.asarray(out)


def weighted_interval(w,value,mask=None):
    mask=np.ones(len(value)) if mask is None else np.asarray(mask,float)
    value=np.asarray(value,float)
    mask=mask*np.isfinite(value)
    value=np.where(np.isfinite(value),value,0.)
    denom=w@mask
    valid=denom>0
    draws=(w@(value*mask))[valid]/denom[valid]
    if not valid.any():return np.nan,np.nan,0.
    lo,hi=np.quantile(draws,[.025,.975])
    return float(lo),float(hi),float(valid.mean())


def comparison(ref,new,w):
    ref,new=np.asarray(ref,float),np.asarray(new,float)
    d=ref-new;point=float(d.mean())
    draws=bootstrap_mean(w,d)
    dlo,dhi=np.quantile(draws,[.025,.975])
    rb=bootstrap_mean(w,ref)
    valid=rb>0
    if valid.any():glo,ghi=np.quantile(100*draws[valid]/rb[valid],[.025,.975])
    else:glo,ghi=np.nan,np.nan
    return dict(reference_cost=float(ref.mean()),ml_cost=float(new.mean()),difference=point,
        difference_lo=float(dlo),difference_hi=float(dhi),
        improvement_pct=100*point/ref.mean() if ref.mean()>0 else np.nan,
        improvement_lo=float(glo),improvement_hi=float(ghi),
        ratio_valid_fraction=float(valid.mean()),
        p_centered_boot=float((1+np.count_nonzero(np.abs(draws-point)>=abs(point)))/(len(draws)+1)))


def verified_source():
    verify_parent()
    manifest=pd.read_json(PARENT/'results/VERIFICATION.json',typ='series')
    assert manifest.predictions_sha256==sha(PARENT/'results/predictions.csv.gz')


def create_lock():
    import subprocess
    verified_source()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'PRE_RUN_TESTS.txt']
    files += [PARENT/'LOCK.json',PARENT/'results/PREDICTIONS_SEALED.json',
        PARENT/'results/predictions.csv.gz',PARENT/'results/targets.csv.gz',PARENT/'results/VERIFICATION.json']
    write_new(LOCK,dict(created_utc=now(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        existing_data_only=True,independent_confirmation=False,new_primary_comparisons=12,
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))


def verify_lock():
    import subprocess,json
    verified_source()
    lock=json.loads(LOCK.read_text())
    for name,value in lock['file_sha256'].items():assert sha(ROOT/name)==value,name
    rel=str(LOCK.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==LOCK.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/research/icaif2026-manuscript-development'],cwd=ROOT,check=True)
    return dict(lock_commit=commit,lock_sha256=sha(LOCK))


def verify_inputs():
    import json
    stamp=verify_lock()
    seal=json.loads((OUT/'INPUTS_SEALED.json').read_text())
    assert seal['lock_sha256']==stamp['lock_sha256']
    for name,value in seal['file_sha256'].items():assert sha(ROOT/name)==value,name
    return stamp


if __name__=='__main__':create_lock() if sys.argv[1]=='lock' else print(verify_lock())
