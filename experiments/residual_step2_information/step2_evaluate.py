"""Paired full-vs-removed forecast losses, with prespecified three-group testing."""
import argparse
import json
import numpy as np
import pandas as pd
from step2_guard import write_new,now
from step2_core import OUT,DESIGN,GROUPS,PROCEDURES,a
from step2_verify import verify_seal
from analyze import day_weights,boot_mean,interval,holm


def main(reproduce=False):
    stamp,seal=verify_seal()
    verification=json.loads((OUT/'VERIFICATION.json').read_text())
    assert verification['predictions_sha256']==seal['predictions_sha256']
    marker=OUT/'SCORES_OPENED.json'
    if marker.exists():
        if not reproduce:raise RuntimeError('Scores already opened; --reproduce is only for the same sealed forecasts')
        assert json.loads(marker.read_text())['predictions_sha256']==seal['predictions_sha256']
    else:write_new(marker,dict(**stamp,opened_utc=now(),predictions_sha256=seal['predictions_sha256'],independent_confirmation=False))
    pred=pd.read_csv(OUT/'predictions.csv.gz');pred.origin=pd.to_datetime(pred.origin,utc=True)
    pred=pred[pred.evaluation].copy()
    metrics,monthly,comparisons,seed_results,monthly_effects=[],[],[],[],[]
    for (mode,definition),case in pred.groupby(['mode','definition']):
        for version in ['raw','calibrated']:
            maps={};streams={}
            for (removed,procedure),g in case.groupby(['removed_group','procedure']):
                g=g.copy();g['loss']=a.pinball(g.target,g['pred_'+version],.1)
                g['below']=(g.target<g['pred_'+version]).astype(float)
                r=g.groupby('origin',as_index=False).agg(loss=('loss','mean'),below=('below','mean'),
                    target=('target','first'),fold=('fold','first'),seeds=('seed','nunique'))
                assert r.seeds.nunique()==1
                maps[removed,procedure]=r;streams[removed,procedure]=g
            full=maps['none','full'];weights={b:day_weights(full,b) for b in [1,5,10]}
            for (removed,procedure),r in maps.items():
                assert r.origin.equals(full.origin)
                np.testing.assert_allclose(r.target,full.target,atol=1e-10,rtol=0)
                key=dict(mode=mode,definition=definition,removed_group=removed,procedure=procedure,calibration=version)
                lo,hi=interval(boot_mean(weights[5],r.below))
                metrics.append(dict(**key,n=len(r),days=r.origin.dt.normalize().nunique(),seeds=int(r.seeds.iloc[0]),
                    loss_bp=r.loss.mean(),below_rate=r.below.mean(),below_lo=lo,below_hi=hi))
                for fold,f in r.groupby('fold'):
                    monthly.append(dict(**key,fold=fold,n=len(f),loss_bp=f.loss.mean(),below_rate=f.below.mean()))
            for removed in GROUPS:
                for procedure in PROCEDURES:
                    without=maps[removed,procedure]
                    delta=without.loss.to_numpy()-full.loss.to_numpy();point=float(delta.mean())
                    key=dict(mode=mode,definition=definition,removed_group=removed,procedure=procedure,calibration=version,n=len(full))
                    for b,w in weights.items():
                        draws=boot_mean(w,delta);increase=100*draws/boot_mean(w,full.loss)
                        lo,hi=interval(increase)
                        p=(1+np.count_nonzero(np.abs(draws-point)>=abs(point)))/(len(draws)+1)
                        comparisons.append(dict(**key,block_days=b,difference_bp=point,
                            removal_loss_increase_pct=100*(without.loss.mean()/full.loss.mean()-1),
                            increase_lo=lo,increase_hi=hi,inclusion_gain_pct=100*(1-full.loss.mean()/without.loss.mean()),
                            p_centered_boot=p,full_winning_months=int((full.groupby('fold').loss.mean()<without.groupby('fold').loss.mean()).sum())))
                    for fold,f in full.groupby('fold'):
                        drop=without[without.fold==fold]
                        monthly_effects.append(dict(**key,fold=fold,month_n=len(f),
                            removal_loss_increase_pct=100*(drop.loss.mean()/f.loss.mean()-1),difference_bp=drop.loss.mean()-f.loss.mean()))
                    for seed,g in streams['none','full'].groupby('seed'):
                        x=g.sort_values('origin');y=streams[removed,procedure]
                        y=y[y.seed==seed].sort_values('origin')
                        assert list(x.origin)==list(y.origin)
                        seed_results.append(dict(**key,seed=int(seed),removal_loss_increase_pct=100*(y.loss.mean()/x.loss.mean()-1)))
    comp=pd.DataFrame(comparisons)
    primary=comp[(comp['mode']==DESIGN['primary_case'][0])&(comp.definition==DESIGN['primary_case'][1])&
        (comp.procedure==DESIGN['primary_procedure'])&(comp.calibration=='calibrated')&(comp.block_days==5)].copy()
    assert len(primary)==3
    primary['p_holm']=holm(primary.p_centered_boot)
    primary['supported_increment']=(primary.removal_loss_increase_pct>0)&(primary.increase_lo>0)&(primary.p_holm<.05)
    for name,frame in [('metrics',pd.DataFrame(metrics)),('monthly_metrics',pd.DataFrame(monthly)),
        ('comparisons',comp),('primary_tests',primary),('seed_comparisons',pd.DataFrame(seed_results)),
        ('monthly_effects',pd.DataFrame(monthly_effects))]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    selections=pd.read_csv(OUT/'selected_models.csv')
    selections[(selections.procedure=='retuned')&(selections.seed==DESIGN['seeds'][0])].to_csv(OUT/'algorithm_choices.csv',index=False)
    if not reproduce:write_new(OUT/'EVALUATION_COMPLETE.json',dict(**stamp,completed_utc=now(),
        primary_tests=primary.to_dict('records'),independent_confirmation=False))
    print(primary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reproduce',action='store_true');main(p.parse_args().reproduce)
