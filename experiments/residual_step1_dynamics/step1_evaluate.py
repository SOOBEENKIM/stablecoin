"""Paired, seed-averaged losses; new results never alter the old ML procedure."""
import argparse
import json
import numpy as np
import pandas as pd
from step1_guard import sha, write_new, now
from step1_core import OUT, DESIGN, a
from step1_verify import verify_seal
from analyze import day_weights, boot_mean, interval, holm


def main(reproduce=False):
    stamp,seal=verify_seal()
    verification=json.loads((OUT/'VERIFICATION.json').read_text())
    assert verification['predictions_sha256']==seal['predictions_sha256']
    marker=OUT/'SCORES_OPENED.json'
    if marker.exists():
        if not reproduce:raise RuntimeError('Scores already opened; use --reproduce only for sealed forecasts')
        assert json.loads(marker.read_text())['predictions_sha256']==seal['predictions_sha256']
    else:
        write_new(marker,dict(**stamp,opened_utc=now(),predictions_sha256=seal['predictions_sha256'],independent_confirmation=False))
    pred=pd.read_csv(OUT/'predictions.csv.gz')
    pred.origin=pd.to_datetime(pred.origin,utc=True)
    pred=pred[pred.evaluation].copy()
    metrics,monthly,comparisons,seeds=[],[],[],[]
    for (mode,definition),case in pred.groupby(['mode','definition']):
        for version in ['raw','calibrated']:
            maps={};streams={}
            for strategy,g in case.groupby('strategy'):
                g=g.copy()
                g['loss']=a.pinball(g.target,g['pred_'+version],.1)
                g['below']=(g.target<g['pred_'+version]).astype(float)
                r=g.groupby('origin',as_index=False).agg(loss=('loss','mean'),below=('below','mean'),
                    target=('target','first'),fold=('fold','first'),seeds=('seed','nunique'))
                maps[strategy]=r;streams[strategy]=g
            ref=maps['selected'];weights={b:day_weights(ref,b) for b in [1,5,10]}
            for strategy,r in maps.items():
                assert r.origin.equals(ref.origin)
                np.testing.assert_allclose(r.target,ref.target,atol=1e-10,rtol=0)
                key=dict(mode=mode,definition=definition,strategy=strategy,calibration=version)
                lo,hi=interval(boot_mean(weights[5],r.below))
                metrics.append(dict(**key,n=len(r),days=r.origin.dt.normalize().nunique(),
                    seeds=int(r.seeds.iloc[0]),loss_bp=r.loss.mean(),below_rate=r.below.mean(),below_lo=lo,below_hi=hi))
                for fold,f in r.groupby('fold'):
                    monthly.append(dict(**key,fold=fold,n=len(f),loss_bp=f.loss.mean(),below_rate=f.below.mean()))
            for reference in DESIGN['primary_references']+['linear']:
                x,y=maps['selected'],maps[reference]
                delta=x.loss.to_numpy()-y.loss.to_numpy();point=float(delta.mean())
                key=dict(mode=mode,definition=definition,reference=reference,calibration=version,n=len(x))
                for b,w in weights.items():
                    draws=boot_mean(w,delta);gain=-100*draws/boot_mean(w,y.loss)
                    lo,hi=interval(gain)
                    p=(1+np.count_nonzero(np.abs(draws-point)>=abs(point)))/(len(draws)+1)
                    comparisons.append(dict(**key,block_days=b,difference_bp=point,
                        improvement_pct=100*(1-x.loss.mean()/y.loss.mean()),improvement_lo=lo,improvement_hi=hi,
                        p_centered_boot=p,winning_months=int((x.groupby('fold').loss.mean()<y.groupby('fold').loss.mean()).sum())))
                for seed,g in streams['selected'].groupby('seed'):
                    xx=g.sort_values('origin')
                    yy=streams[reference]
                    # Legacy benchmarks are deterministic repeated seed streams;
                    # fresh deterministic benchmarks have a single seed=0.
                    yy=yy[yy.seed==(seed if seed in set(yy.seed) else 0)].sort_values('origin')
                    assert list(xx.origin)==list(yy.origin)
                    seeds.append(dict(**key,seed=int(seed),improvement_pct=100*(1-xx.loss.mean()/yy.loss.mean())))
    comp=pd.DataFrame(comparisons)
    primary=comp[(comp['mode']==DESIGN['primary_case'][0])&(comp.definition==DESIGN['primary_case'][1])&
        (comp.calibration=='calibrated')&(comp.block_days==5)&comp.reference.isin(DESIGN['primary_references'])].copy()
    assert len(primary)==4
    primary['p_holm']=holm(primary.p_centered_boot)
    primary['supported_conditional']=(primary.improvement_pct>0)&(primary.improvement_lo>0)&(primary.p_holm<.05)
    for name,frame in [('metrics',pd.DataFrame(metrics)),('monthly_metrics',pd.DataFrame(monthly)),
                        ('comparisons',comp),('primary_tests',primary),('seed_comparisons',pd.DataFrame(seeds))]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    if not reproduce:
        write_new(OUT/'EVALUATION_COMPLETE.json',dict(**stamp,completed_utc=now(),
            primary_tests=primary.to_dict('records'),independent_confirmation=False))
    print(primary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reproduce',action='store_true')
    main(p.parse_args().reproduce)
