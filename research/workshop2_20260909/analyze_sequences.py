"""Compare all sequence learners and matched controls without seed selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from run_pilot import scores,intercept_calibration
from analyze_pilot import paired
HERE=Path(__file__).resolve().parent


def main():
    p=pd.read_csv(HERE/'sequence_predictions.csv.gz',index_col=0,parse_dates=True)
    c=pd.read_csv(HERE/'sequence_controls.csv.gz',index_col=0,parse_dates=True)
    for v in [p,c]:
        v.index=pd.to_datetime(v.index,utc=True);v['target_time']=pd.to_datetime(v.target_time,utc=True)
    assert p.index.equals(c.index)
    np.testing.assert_array_equal(p.target,c.target)
    f=c.copy()
    for col in p.columns:
        if col in ['target','target_time']:continue
        f[col+'_raw']=p[col]
        if 'ensemble' in col:
            f[col+'_cal']=intercept_calibration(f,p[col].to_numpy())[0]
    names=[v for v in f.columns if v not in ['target','target_time']]
    test=f.loc[f.index>=pd.Timestamp('2026-01-01',tz='UTC')]
    rec=[];monthly=[]
    for name in names:
        rec.append(dict(model=name,**scores(test.target.to_numpy(),test[name].to_numpy())))
        for month,v in f.groupby(f.index.strftime('%Y-%m')):
            monthly.append(dict(model=name,month=month,**scores(v.target.to_numpy(),v[name].to_numpy())))
    score=pd.DataFrame(rec);score.to_csv(HERE/'sequence_scores.csv',index=False)
    pd.DataFrame(monthly).to_csv(HERE/'sequence_monthly.csv',index=False)
    f.to_csv(HERE/'sequence_all_forecasts.csv.gz',compression='gzip')
    primary=[(a+'_ensemble_raw',b) for a in ['gru','tcn'] for b in ['tree_current_raw','logistic_flat_raw','recent28']]
    secondary=[(a+'_ensemble_raw',b) for a in ['gru','tcn'] for b in ['logistic_current_raw','mlp_ensemble_raw']]
    rows=[]
    for block in [5,1,10]:
        for cand,ref in primary+secondary:
            rows.append(dict(primary=(cand,ref) in primary,**paired(test,cand,ref,block)))
    inf=pd.DataFrame(rows);inf['p_holm_primary']=np.nan
    mainmask=inf.primary&inf.block_days.eq(5)
    inf.loc[mainmask,'p_holm_primary']=multipletests(inf.loc[mainmask,'p_centered'],method='holm')[1]
    inf.to_csv(HERE/'sequence_inference.csv',index=False)
    mo=pd.DataFrame(monthly).query("month >= '2026-01'").pivot(index='month',columns='model',values='brier')
    decisions={}
    for family in ['gru','tcn']:
        name=family+'_ensemble_raw';r=inf.loc[mainmask&inf.candidate.eq(name)]
        decisions[family]=bool((r.relative_improvement_pct>=3).all() and (r.low95>0).all()
            and (r.p_holm_primary<.05).all() and all((mo[name]<mo[ref]).sum()>=2 for ref in r.reference))
    (HERE/'SEQUENCE_DECISION.json').write_text(json.dumps(dict(gates_passed=decisions,
        n_test=len(test),days=int(test.index.normalize().nunique()),
        status='exploratory, same previously inspected market period; not a test of all possible AI models'),indent=2))
    print(score.round(5).to_string(index=False))
    print(inf.loc[inf.block_days.eq(5)].round(5).to_string(index=False))
    print(mo[['recent28','tree_current_raw','logistic_current_raw','logistic_flat_raw','mlp_ensemble_raw','gru_ensemble_raw','tcn_ensemble_raw']].round(5).to_string())
    print(decisions)


if __name__=='__main__':main()
