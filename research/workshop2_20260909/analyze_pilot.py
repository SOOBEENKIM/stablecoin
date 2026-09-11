"""Conditional paired calendar-block inference for fixed pilot comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parent


def paired(f,candidate,reference,block,seed=20260909):
    delta=(f.target-f[reference])**2-(f.target-f[candidate])**2
    daily=pd.DataFrame({'sum':delta,'n':1},index=f.index).resample('D').sum()
    sums=daily['sum'].to_numpy();ns=daily.n.to_numpy();n=len(daily)
    estimate=delta.mean();rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(1999):
        starts=rng.integers(n,size=int(np.ceil(n/block)))
        ix=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        boot.append(sums[ix].sum()/ns[ix].sum())
    low,high=np.quantile(boot,[.025,.975])
    p=(1+(np.abs(np.asarray(boot)-estimate)>=abs(estimate)).sum())/2000
    baseline=((f.target-f[reference])**2).mean()
    return dict(candidate=candidate,reference=reference,block_days=block,n=len(f),
                brier_improvement=float(estimate),relative_improvement_pct=100*estimate/baseline,
                low95=float(low),high95=float(high),p_centered=float(p))


def main():
    f=pd.read_csv(HERE/'onset_forecasts.csv.gz',index_col=0,parse_dates=True)
    f.index=pd.to_datetime(f.index,utc=True)
    f=f.loc[f.index>=pd.Timestamp('2026-01-01',tz='UTC')]
    primary=[('tree_reference_cal',r) for r in ['logistic_reference_cal','spline_reference_cal','recent28']]
    secondary=[('tree_reference_cal','tree_local_cal'),('tree_full_cal','tree_reference_cal'),
        ('tree_reference_raw','logistic_reference_raw'),('tree_reference_raw','tree_local_raw'),
        ('tree_full_raw','tree_reference_raw'),('logistic_reference_raw','logistic_local_raw'),
        ('logistic_reference_raw','recent28'),('logistic_reference_cal','logistic_local_cal'),
        ('tree_reference_cal','tree_reference_raw'),('logistic_reference_cal','logistic_reference_raw'),
        ('tree_reference_cal','recent28_state')]
    rows=[]
    for block in [5,1,10]:
        for cand,ref in primary+secondary:
            rows.append(dict(primary=(cand,ref) in primary,**paired(f,cand,ref,block)))
    out=pd.DataFrame(rows);out['p_holm_primary']=np.nan
    mask=out.primary&out.block_days.eq(5)
    out.loc[mask,'p_holm_primary']=multipletests(out.loc[mask,'p_centered'],method='holm')[1]
    out.to_csv(HERE/'pilot_paired_inference.csv',index=False)
    monthly=pd.read_csv(HERE/'pilot_monthly.csv')
    monthly=monthly.loc[monthly.month>='2026-01'].pivot(index='month',columns='model',values='brier')
    monthly.to_csv(HERE/'pilot_monthly_brier.csv')
    mainrows=out.loc[mask]
    gate=bool((mainrows.relative_improvement_pct>=3).all() and (mainrows.low95>0).all()
        and (mainrows.p_holm_primary<.05).all()
        and all((monthly[c]<monthly[r]).sum()>=2 for c,r in primary))
    (HERE/'PILOT_DECISION.json').write_text(json.dumps(dict(strong_ai_gate_passed=gate,
        status='exploratory pilot, no independently unseen evaluation period',
        primary_comparisons=mainrows.to_dict(orient='records'),
        next_stage='No automatic model/horizon search. Report failure of the fixed AI gate; design decisions must be distinguished from validated effects.'),indent=2))
    print(out.loc[out.block_days.eq(5)].round(5).to_string(index=False))
    print(monthly[['recent28','logistic_reference_raw','logistic_reference_cal','tree_reference_raw','tree_reference_cal']].round(5).to_string())
    print('Strong AI gate:',gate)


if __name__=='__main__':
    main()
