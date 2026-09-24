"""Open the complete sealed evaluation once; never feed these scores to selection."""
import argparse
import json
import numpy as np
import pandas as pd
from guard import HERE,verify_lock,sha,now,write_new
from nested import a,DESIGN
from analyze import day_weights,boot_mean,interval,holm

OUT=HERE/'results'


def main(reproduce=False):
    stamp=verify_lock()
    seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    if seal['lock_sha256']!=stamp['lock_sha256'] or seal['prediction_sha256']!=sha(OUT/'predictions.csv.gz'):
        raise RuntimeError('Forecast seal mismatch')
    marker=OUT/'SCORES_OPENED.json'
    if marker.exists():
        if not reproduce:raise RuntimeError('Scores already opened; use --reproduce only to reproduce identical sealed forecasts')
        if json.loads(marker.read_text())['prediction_sha256']!=seal['prediction_sha256']:
            raise RuntimeError('Cannot replace already opened predictions')
    else:
        write_new(marker,dict(**stamp,opened_utc=now(),prediction_sha256=seal['prediction_sha256'],independent_confirmation=False))
    pred=pd.read_csv(OUT/'predictions.csv.gz')
    pred.origin=pd.to_datetime(pred.origin,utc=True)
    pred=pred[pred.evaluation].copy()
    metrics,monthly,comparisons,by_seed=[],[],[],[]
    pairs=[('selected','history','linear','history','selected_vs_linear'),
           ('selected','history','threshold','history','selected_vs_threshold'),
           ('selected','history','selected','base','position_increment'),
           ('selected','base','linear','base','selected_base_vs_linear_base'),
           ('qrf','history','linear','history','qrf_vs_linear'),
           ('qrf','history','threshold','history','qrf_vs_threshold'),
           ('qrf','history','qrf','base','qrf_position_increment')]
    for (mode,definition),case in pred.groupby(['mode','definition']):
        for version in ['raw','calibrated']:
            maps={};streams={}
            for (strategy,info),g in case.groupby(['strategy','info']):
                s=g.copy();s['loss']=a.pinball(s.target,s['pred_'+version],.1)
                s['below']=(s.target<s['pred_'+version]).astype(float)
                r=s.groupby('origin',as_index=False).agg(loss=('loss','mean'),below=('below','mean'),
                    target=('target','first'),fold=('fold','first'),seeds=('seed','nunique'))
                assert r.seeds.nunique()==1
                key=dict(mode=mode,definition=definition,strategy=strategy,info=info,calibration=version)
                metrics.append(dict(**key,n=len(r),days=r.origin.dt.normalize().nunique(),
                    seeds=int(r.seeds.iloc[0]),loss_bp=r.loss.mean(),below_rate=r.below.mean()))
                for fold,f in r.groupby('fold'):
                    monthly.append(dict(**key,fold=fold,n=len(f),loss_bp=f.loss.mean(),below_rate=f.below.mean()))
                maps[strategy,info]=r;streams[strategy,info]=s
            ref=maps['selected','history'];weights={b:day_weights(ref,b) for b in [1,5,10]}
            for sx,ix,sy,iy,label in pairs:
                x,y=maps[sx,ix],maps[sy,iy]
                assert x.origin.equals(y.origin)
                np.testing.assert_allclose(x.target,y.target,atol=1e-10,rtol=0)
                delta=x.loss.to_numpy()-y.loss.to_numpy();point=delta.mean()
                key=dict(mode=mode,definition=definition,comparison=label,calibration=version,n=len(x))
                for b,w in weights.items():
                    draws=boot_mean(w,delta);gain=-100*draws/boot_mean(w,y.loss)
                    lo,hi=interval(gain)
                    p=(1+np.count_nonzero(np.abs(draws-point)>=abs(point)))/(len(draws)+1)
                    comparisons.append(dict(**key,block_days=b,difference_bp=point,
                        improvement_pct=100*(1-x.loss.mean()/y.loss.mean()),improvement_lo=lo,improvement_hi=hi,
                        p_centered_boot=p,winning_months=int((x.groupby('fold').loss.mean()<y.groupby('fold').loss.mean()).sum())))
                a1,b1=streams[sx,ix],streams[sy,iy]
                for seed in sorted(a1.seed.unique()):
                    xx=a1[a1.seed==seed].sort_values('origin');yy=b1[b1.seed==seed].sort_values('origin')
                    assert list(xx.origin)==list(yy.origin)
                    by_seed.append(dict(**key,seed=int(seed),improvement_pct=100*(1-xx.loss.mean()/yy.loss.mean())))
    comp=pd.DataFrame(comparisons)
    primary=comp[(comp['mode']==DESIGN['primary_case'][0])&(comp.definition==DESIGN['primary_case'][1])&
        (comp.calibration=='calibrated')&(comp.block_days==5)&comp.comparison.isin(DESIGN['primary_comparisons'])].copy()
    assert len(primary)==3
    primary['p_holm']=holm(primary.p_centered_boot)
    primary['supported_conditional']=(primary.improvement_pct>0)&(primary.improvement_lo>0)&(primary.p_holm<.05)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    comp.to_csv(OUT/'comparisons.csv',index=False)
    primary.to_csv(OUT/'primary_tests.csv',index=False)
    pd.DataFrame(by_seed).to_csv(OUT/'seed_comparisons.csv',index=False)
    selections=pd.read_csv(OUT/'selected_models.csv')
    choices=selections[(selections.strategy=='selected')&(selections.seed==DESIGN['seeds'][0])]
    choices.to_csv(OUT/'algorithm_choices.csv',index=False)
    write_new(OUT/'EVALUATION_COMPLETE.json',dict(**stamp,completed_utc=now(),
        primary_tests=primary.to_dict('records'),independent_confirmation=False)) if not reproduce else None
    print(primary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reproduce',action='store_true');args=p.parse_args();main(args.reproduce)
