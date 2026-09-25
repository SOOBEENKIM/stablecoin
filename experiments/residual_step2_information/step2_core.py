"""Feature-group removal with train-only reselection or fixed historical specs."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_nested_validation'))
import nested as n
from models import ForecastModel
a=n.a
DESIGN=json.loads((HERE/'design.json').read_text())
OUT=HERE/'results'
LEGACY=HERE.parent/'residual_nested_validation'
CASES=[tuple(x) for x in DESIGN['cases']]
GROUPS=DESIGN['groups']
PROCEDURES=DESIGN['procedures']
PAIRS=[('downside24','btc_vol24'),('e_now','downside24')]+[
    ('downside24',c) for c in ['account_log','funding_bp','oi_ret1','oi_surprise']]+[
    ('e_now','account_log')]


def features(mode,group):
    full=a.columns(mode,DESIGN['legacy_info'])
    removed=GROUPS[group]
    cols=[c for c in full if c not in removed]
    assert all(c in cols for c in DESIGN['always_keep'])
    assert not set(cols).intersection(removed)
    assert len(cols)<len(full)
    return cols


def interaction_pairs(columns):
    return [(x,y) for x,y in PAIRS if x in columns and y in columns]


class AvailableInteractions(ForecastModel):
    """Exactly the old basis when complete; no products of removed features."""
    def prepare(self,x,fit=False):
        if fit:
            self.scaler=StandardScaler().fit(x)
            self.cols=list(x.columns)
            self.pairs=interaction_pairs(self.cols)
        z=self.scaler.transform(x)
        lookup={c:z[:,j] for j,c in enumerate(self.cols)}
        basis=np.column_stack([z]+[lookup[c]*lookup[d] for c,d in self.pairs])
        if fit:self.second_scaler=StandardScaler().fit(basis)
        return self.second_scaler.transform(basis)


class InteractionDynamics:
    def __init__(self,spec):self.spec=spec

    def fit(self,train,columns):
        self.columns=list(columns)
        target=train.target.to_numpy()-(train.e_now.to_numpy() if self.spec['form']=='change' else 0)
        self.model=AvailableInteractions('interaction',.1,self.spec['params']).fit(train[self.columns],target)
        return self

    def predict(self,features):
        raw=self.model.predict(features[self.columns])
        return raw+(features.e_now.to_numpy() if self.spec['form']=='change' else 0)


def fit_model(spec,train,columns,seed):
    if spec['kind']=='interaction':return InteractionDynamics(spec).fit(train,columns)
    return n.fit_model(spec,train,columns,seed)


def choose(scores,cutoff,group):
    if set(scores.removed_group)!={group}:
        raise ValueError('Mixed or incorrect removal groups in selection')
    return n.choose(scores,cutoff)


def full_spec(mode,definition,fold):
    rows=json.loads((LEGACY/'results/selected_models.json').read_text())
    rows=[r for r in rows if (r['mode'],r['definition'],r['fold'],r['info'],r['strategy'],r['seed'])==
          (mode,definition,fold,DESIGN['legacy_info'],DESIGN['legacy_strategy'],n.SEEDS[0])]
    assert len(rows)==1
    m=rows[0];spec=n.SPECS[m['candidate']]
    for k in ['kind','window','form','params']:assert m[k]==spec[k]
    return dict(**spec,inner_score=m['inner_score'],inner_months=m['inner_months'])


def read_full():
    d=pd.read_csv(LEGACY/'results/predictions.csv.gz')
    for c in ['origin','target_time','latest_calibration_label']:
        d[c]=pd.to_datetime(d[c],utc=True)
    d=d[(d['info']==DESIGN['legacy_info'])&(d.strategy==DESIGN['legacy_strategy'])].copy()
    d['removed_group']='none';d['procedure']='full'
    return d
