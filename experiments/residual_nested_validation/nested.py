"""Selection accepts ONLY earlier inner-month scores; no outer outcome argument."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from guard import HERE

sys.path.insert(0,str(HERE.parent/'residual_economic_validation'))
from economic import a, ThresholdQuantile, threshold_candidates
import models as model_library

DESIGN=json.loads((HERE/'design.json').read_text())
CASES=[tuple(x) for x in DESIGN['cases']]
SEEDS=DESIGN['seeds']
FOLDS=pd.date_range(DESIGN['warmup_start'],pd.Timestamp(DESIGN['evaluation_end_exclusive'])-pd.offsets.MonthBegin(1),freq='MS')


def candidate_specs():
    specs=[]
    for kind in DESIGN['algorithms']:
        grid=threshold_candidates() if kind=='threshold' else a.candidates(kind)
        for window,form,params in grid:
            specs.append(dict(candidate=len(specs),kind=kind,window=str(window),form=form,params=params))
    assert len(specs)==84
    return specs


SPECS=candidate_specs()


def inner_months(cutoff):
    return [cutoff-pd.offsets.MonthBegin(i) for i in [3,2,1]]


def purge_inner(frame,cutoff):
    return frame[(frame.index>=cutoff)&(frame.target_time<cutoff+pd.offsets.MonthBegin(1))]


def fit_model(spec,train,columns,seed):
    a.SEED=seed;model_library.SEED=seed
    model=ThresholdQuantile(spec['form'],spec['params']) if spec['kind']=='threshold' else a.DynamicModel(spec['kind'],spec['form'],spec['params'])
    return model.fit(train,columns)


def choose(inner_scores,cutoff,family=None):
    """Reject future rows rather than trusting callers to hide outer scores."""
    expected=[d.strftime('%Y-%m') for d in inner_months(cutoff)]
    d=inner_scores.copy()
    if not set(d.fold.unique()).issubset(set(expected)):
        raise ValueError('Selection received a month outside the three past inner folds')
    if d.duplicated(['candidate','fold']).any():
        raise ValueError('Repeated inner candidate/month')
    if family is not None:d=d[d.kind==family]
    valid=[]
    for candidate,g in d.groupby('candidate',sort=True):
        if set(g.fold)!=set(expected) or not g.valid.all():continue
        if (g.n<=0).any() or not np.isfinite(g.loss_sum).all():continue
        valid.append((float(g.loss_sum.sum()/g.n.sum()),int(candidate)))
    if not valid:raise ValueError('No candidate valid in all three inner months: '+str(family))
    score,candidate=min(valid)
    return dict(**SPECS[candidate],inner_score=score,inner_months=expected)


def outer_sample(panel,mode,definition,cutoff):
    r=a.Residualizer(definition).fit(panel,cutoff)
    d=a.frame(panel,r,mode)
    train=d[d.target_time<cutoff]
    test=d[(d.index>=cutoff)&(d.index<cutoff+pd.offsets.MonthBegin(1))&
           (d.target_time<pd.Timestamp(DESIGN['evaluation_end_exclusive']))]
    if len(train)==0 or len(test)==0:raise ValueError('Empty chronological split')
    assert train.target_time.max()<test.index.min()
    assert r.fit_end<cutoff
    return train,test,r
