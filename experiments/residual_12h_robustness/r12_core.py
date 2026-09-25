"""Frozen extension across residual definitions, with efficient past-only scores."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
PARENT=HERE.parent/'residual_horizon_scale'
sys.path.insert(0,str(PARENT))
import hs_core as hs
from selection_check import verify_selection
from hs_guard import verify as verify_parent
from hs_amendment_guard import verify_amendment

old=hs.old
DEFS=['EQ','CAP','PCA']
VARIANTS=hs.VARIANTS
INFOS=hs.INFOS
STRATEGIES=hs.STRATEGIES
SEEDS=[20260925,20260926,20260927,20260928]
sha,now,write_new,read,save=hs.sha,hs.now,hs.write_new,hs.read,hs.save
FAMILIES=old.DESIGN['families']


def case(panel,definition,cutoff):
    r=old.legacy.a.Residualizer(definition).fit(panel,cutoff)
    e,_=r.transform(panel)
    d=old.with_calendar(old.legacy.a.frame(panel,r,'available_macro',12)).join(hs.own_scale(e))
    tr=d[d.target_time<cutoff]
    te=d[(d.index>=cutoff)&(d.index<cutoff+pd.offsets.MonthBegin(1))&(d.target_time<hs.END)]
    for z in [tr,te]:
        assert len(z)>0 and np.isfinite(z[hs.columns('F','own_scale')+['target']].to_numpy()).all()
        assert ((z.target_time-z.index)==pd.Timedelta(hours=12)).all()
    assert tr.target_time.max()<cutoff and r.fit_end<cutoff
    if definition=='EQ':
        a,b,_=hs.make_case(panel,cutoff,12)
        pd.testing.assert_frame_equal(tr,a);pd.testing.assert_frame_equal(te,b)
    return tr,te,r


def calibrate(g):
    assert g.definition.nunique()==1
    return hs.calibrate(g,12)


def inner_rows(s,cutoff):
    origins,labels=pd.DatetimeIndex(s.origin),pd.DatetimeIndex(s.target_time)
    if not ((labels-origins)==pd.Timedelta(hours=12)).all():raise ValueError('Wrong horizon')
    same_month=(origins.year*12+origins.month)==(labels.year*12+labels.month)
    return s[(origins>=cutoff-pd.offsets.MonthBegin(3))&(origins<cutoff)&(labels<cutoff)&same_month].copy()


def select_all(s,cutoff):
    for c in ['definition','variant','information','seed']:
        if s[c].nunique()!=1:raise ValueError('Mixed stream '+c)
    assert int(s.seed.iloc[0])==SEEDS[0]
    d=inner_rows(s,cutoff)
    expected={x.strftime('%Y-%m') for x in old.inner_months(cutoff)}
    if set(d.origin.dt.strftime('%Y-%m'))!=expected or set(d.candidate)!=set(range(60)):
        raise ValueError('Missing month/candidate')
    if d.duplicated(['candidate','origin']).any():raise ValueError('Duplicate candidate origin')
    if (d.calibration_n<30).any() or not np.isfinite(d.pred_calibrated).all():raise ValueError('Invalid score input')
    pairs=None;scores=[]
    for cid,g in d.groupby('candidate',sort=True):
        z=g.sort_values('origin')[['origin','target_time','target']].reset_index(drop=True)
        if pairs is None:pairs=z
        else:pd.testing.assert_frame_equal(z,pairs)
        scores.append(dict(candidate=int(cid),kind=old.SPECS[int(cid)]['kind'],n=len(g),score=float(old.loss(g.target,g.pred_calibrated).mean())))
    selections={}
    for strategy in STRATEGIES:
        kinds=['qrf','boosting'] if strategy=='ml_selected' else [strategy]
        value,cid=min((x['score'],x['candidate']) for x in scores if x['kind'] in kinds)
        selections[strategy]=dict(**old.SPECS[cid],inner_score=value,inner_months=sorted(expected))
    return selections,scores


def independent_calibration(g):
    """Integer clock scan, independent from production searchsorted/quantile."""
    d=g.sort_values('origin')
    t=pd.DatetimeIndex(d.origin).asi8
    lab=pd.DatetimeIndex(d.target_time).asi8
    err=(d.target-d.pred_raw).to_numpy()
    age=pd.Timedelta(days=90).value
    out=[]
    for origin in t:
        ids=np.flatnonzero((lab<origin)&(lab>=origin-age))[-60:]
        out.append(np.sort(err[ids])[int(np.ceil(.1*len(ids)))-1] if len(ids)>=30 else 0.)
    np.testing.assert_allclose(d.correction,out,atol=1e-12,rtol=0)
    return float(np.max(np.abs(d.correction-np.asarray(out))))


def standardize(g,definition):
    d=g.copy();d['definition']=definition;d['horizon']=12
    return d


def file_key(*args):return '__'.join(map(str,args))
