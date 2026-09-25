"""Signed adjustment channels and past-only orthogonal association analysis."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'results';PREP=HERE/'prepared';LOCK=HERE/'LOCK.json'
sys.path.insert(0,str(HERE.parent/'residual_conditional_calibration'))
import cc_core as cc
dv,r12,old=cc.dv,cc.r12,cc.old
source=sys.modules[old.legacy.a.Residualizer.__module__]
read,save,sha,now,write_new=cc.read,cc.save,cc.sha,cc.now,cc.write_new
DEFS=cc.DEFS;HORIZONS=[1,6,12];LEARNERS=['linear','forest20','forest40']
SEEDS=[20260926,20260927]
PARTS=['local','basket_price','basket_weights','numeraire','global_control']
FEATURES=['abs_e','e_rms72','market_now_bp','g_now_bp']+[c for c in r12.hs.columns('F','original') if c!='e_now']
BRANCH=cc.BRANCH


def attribution(l0,l1,r0,r1,a0,a1,z0,z1,g0,g1,beta,beta_g):
    """Exact symmetric product decomposition; no log/zero-gap division."""
    l0,l1,z0,z1=np.asarray(l0),np.asarray(l1),np.asarray(z0),np.asarray(z1)
    r0,r1,a0,a1=map(np.asarray,[r0,r1,a0,a1])
    zb=(z0+z1)/2;ab=(a0+a1)/2;rb=(r0+r1)/2
    b0=np.sum(a0*r0,axis=1);b1=np.sum(a1*r1,axis=1)
    hb=((l0-beta*b0)+(l1-beta*b1))/2
    return 1e4*np.column_stack([zb*(l1-l0),-beta*zb*np.sum(ab*(r1-r0),axis=1),
        -beta*zb*np.sum(rb*(a1-a0),axis=1),hb*(z1-z0),-beta_g*(np.asarray(g1)-np.asarray(g0))])


def raw_market():
    d=source.read_panel(source.INPUT_PATHS[0]).join(source.read_panel(source.INPUT_PATHS[1]))
    d.index+=pd.Timedelta(hours=1)
    return d


def extended_frame(panel,raw,residualizer,h):
    r=residualizer
    d=old.with_calendar(old.legacy.a.frame(panel,r,'available_macro',h))
    e,m=r.transform(panel);d=d.join(r12.hs.own_scale(e))
    d['abs_e']=d.e_now.abs();d['premium']=(d.e_now>0).astype(int)
    d['market_now_bp']=(m*r.beta_market*1e4).reindex(d.index)
    d['g_now_bp']=(panel.g*1e4).reindex(d.index)
    ratios=np.column_stack([(raw[c+'_UPBIT_CLOSE']/raw[c+'_BINANCE_CLOSE']).reindex(panel.index) for c in source.COINS])
    if r.definition=='EQ':weights=np.full_like(ratios,.2)
    elif r.definition=='CAP':
        caps=np.column_stack([(raw[c+'_BINANCE_CLOSE']*source.SUPPLY[j]).reindex(panel.index) for j,c in enumerate(source.COINS)])
        weights=caps/caps.sum(axis=1)[:,None]
    else:weights=np.tile(r.loading/r.k_std,(len(panel),1))
    z=(panel.q/panel.fx).to_numpy()
    # Verify independent raw-price reconstruction before using the decomposition.
    np.testing.assert_allclose(z[:,None]*ratios-1,panel[['kp_'+c for c in source.COINS]],atol=1e-12,rtol=0,equal_nan=True)
    idx=panel.index.get_indexer(d.index);nxt=panel.index.get_indexer(d.index+pd.Timedelta(hours=h))
    assert (idx>=0).all() and (nxt>=0).all()
    parts=attribution(panel.local_usdt.to_numpy()[idx],panel.local_usdt.to_numpy()[nxt],
        ratios[idx],ratios[nxt],weights[idx],weights[nxt],z[idx],z[nxt],
        panel.g.to_numpy()[idx],panel.g.to_numpy()[nxt],r.beta_market,r.beta_g)
    sign=np.where(d.premium==1,1.,-1.)
    for j,name in enumerate(PARTS):
        d['delta_'+name]=parts[:,j];d['close_'+name]=-sign*parts[:,j]
    d['close_total']=-sign*d.delta_e
    d['close_dominance']=d.close_local-d.close_basket_price
    # Directed movement can pass through zero and overshoot: report both concepts.
    d['absolute_gap_reduction']=d.abs_e-d.target.abs()
    d['crossed_zero']=(d.e_now*d.target<0).astype(int)
    d['overshot_farther']=((d.crossed_zero==1)&(d.target.abs()>d.abs_e)).astype(int)
    d['local_return_bp']=np.log(panel.local_usdt.to_numpy()[nxt]/panel.local_usdt.to_numpy()[idx])*1e4
    d['basket_implied_return_bp']=np.log(np.sum(weights[nxt]*ratios[nxt],axis=1)/np.sum(weights[idx]*ratios[idx],axis=1))*1e4
    d=d.dropna(subset=FEATURES)
    assert np.isfinite(d[FEATURES+['close_'+c for c in PARTS]].to_numpy()).all()
    np.testing.assert_allclose(d[['delta_'+c for c in PARTS]].sum(axis=1),d.delta_e,atol=1e-7,rtol=0)
    return d


def model_fit(tr,te,learner,seed):
    """No hyperparameter/seed selection on these already-seen outcome periods."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge,LogisticRegression
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    import warnings
    x=tr[FEATURES];xt=te[FEATURES]
    y=tr[['close_'+c for c in PARTS]].to_numpy();label=tr.premium.to_numpy()
    center=y.mean(axis=0);scale=y.std(axis=0,ddof=1);scale=np.where(scale>1e-8,scale,1.)
    if learner=='linear':
        outcome=make_pipeline(StandardScaler(),Ridge(alpha=10.))
        probability=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,random_state=seed))
    else:
        leaf=int(learner.replace('forest',''))
        params=dict(n_estimators=300,min_samples_leaf=leaf,max_depth=6,max_features=.7,random_state=seed,n_jobs=1)
        outcome=RandomForestRegressor(**params);probability=RandomForestClassifier(**params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        outcome.fit(x,(y-center)/scale);probability.fit(x,label)
        g=outcome.predict(xt)*scale+center;m=probability.predict_proba(xt)[:,1]
    if caught:raise RuntimeError('; '.join(str(w.message) for w in caught))
    assert np.isfinite(g).all() and np.isfinite(m).all() and ((m>=0)&(m<=1)).all()
    d=te.reset_index().copy()
    for j,name in enumerate(PARTS):d['g_'+name]=g[:,j]
    d['g_total']=g.sum(axis=1);d['g_dominance']=g[:,0]-g[:,1]
    d['m']=m;d['m_constant']=label.mean()
    for j,name in enumerate(PARTS):d['g_constant_'+name]=center[j]
    d['g_constant_total']=center.sum();d['g_constant_dominance']=center[0]-center[1]
    d['learner']=learner;d['seed']=seed
    return d


def orthogonal(d,outcome,weights=None):
    v=d.premium.to_numpy()-d.m.to_numpy()
    u=d['close_'+outcome].to_numpy()-d['g_'+outcome].to_numpy()
    numerator=v*u;denominator=v*v
    if weights is None:return float(numerator.sum()/denominator.sum()) if denominator.sum()>1e-12 else np.nan
    num,den=weights@numerator,weights@denominator
    return np.divide(num,den,out=np.full(len(weights),np.nan),where=den>1e-12)


def create_lock():
    cc.verify_seal()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'PRE_RUN_TESTS.txt']+list(PREP.rglob('*'))
    files+=[cc.OUT/'EVALUATION_COMPLETE.json',cc.LOCK]
    files+=source.INPUT_PATHS
    files=[p for p in files if p.is_file()]
    write_new(LOCK,dict(created_utc=now(),independent_confirmation=False,causal_identification=False,
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))


def verify_lock():
    cc.verify_seal()
    lock=json.loads(LOCK.read_text())
    for name,digest in lock['file_sha256'].items():assert sha(ROOT/name)==digest,name
    rel=str(LOCK.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==LOCK.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True)
    return dict(lock_sha256=sha(LOCK),lock_commit=commit)


def verify_seal():
    stamp=verify_lock();seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    assert stamp['lock_sha256']==seal['lock_sha256']
    for name,digest in seal['file_sha256'].items():assert sha(ROOT/name)==digest,name
    return stamp


if __name__=='__main__':create_lock() if sys.argv[1]=='lock' else print(verify_lock())
