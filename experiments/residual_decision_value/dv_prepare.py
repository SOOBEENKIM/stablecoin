"""Construct past-only simple benchmarks and outcome paths without scores."""
import numpy as np
import pandas as pd
from dv_core import *


def main():
    stamp=verify_lock()
    if (OUT/'INPUTS_SEALED.json').exists():raise FileExistsError('Preserve completed inputs')
    panel=r12.old.legacy.a.panel('available_macro')
    contexts=[];baselines=[];paths=[];fits=[]
    archived=read(PARENT/'results/predictions.csv.gz')
    archived=archived[archived.strategy.isin(['ml_selected','linear','threshold'])].copy()
    for definition in DEFS:
        streams={'historical':[],'state_hist':[]}
        for cutoff in MONTHS:
            tr,te,r=r12.case(panel,definition,cutoff)
            model=fit_state(tr,cutoff)
            fits.append(dict(definition=definition,fold=cutoff.strftime('%Y-%m'),**model))
            if cutoff>=r12.hs.START:
                ctx=te.copy();ctx['state']=assign_state(ctx,model)
                ctx['definition']=definition;ctx['fold']=cutoff.strftime('%Y-%m')
                contexts.append(ctx.reset_index())
                for h in [1,6,12]:
                    path=r12.old.legacy.a.frame(panel,r,'available_macro',h)
                    path=path.loc[path.index.intersection(te.index)].copy()
                    path['local_logreturn_bp']=(np.log(r12.old.legacy.a.lead(panel.local_usdt,h)/panel.local_usdt)*1e4).reindex(path.index)
                    path['local_change_krw']=(r12.old.legacy.a.lead(panel.local_usdt,h)-panel.local_usdt).reindex(path.index)
                    path['definition']=definition;path['fold']=cutoff.strftime('%Y-%m');path['h']=h
                    paths.append(path.reset_index())
            for strategy in streams:
                out=te[['target','target_time','e_now']].reset_index()
                out['pred_raw']=baseline_predict(te,model,strategy)
                out['definition']=definition;out['fold']=cutoff.strftime('%Y-%m')
                out['strategy']=strategy;out['seed']=SEEDS[0];out['horizon']=12
                streams[strategy].append(out)
        for strategy,parts in streams.items():
            baselines.append(r12.calibrate(pd.concat(parts,ignore_index=True)))
    ctx=pd.concat(contexts,ignore_index=True)
    for definition,d in ctx.groupby('definition'):
        assert len(d)==509 and not d.origin.duplicated().any()
        frozen=read(PARENT/'results/targets.csv.gz').query('definition == @definition')
        pd.testing.assert_frame_equal(d[frozen.columns].reset_index(drop=True),frozen.reset_index(drop=True),check_exact=False,rtol=0,atol=1e-11)
    base=pd.concat(baselines,ignore_index=True)
    for name,d in [('context',ctx),('baseline_streams',base),('archived_forecasts',archived),('paths',pd.concat(paths,ignore_index=True))]:save(d,OUT/(name+'.csv.gz'))
    write_new(OUT/'baseline_fits.json',fits)
    files=[OUT/(n+'.csv.gz') for n in ['context','baseline_streams','archived_forecasts','paths']]+[OUT/'baseline_fits.json']
    write_new(OUT/'INPUTS_SEALED.json',dict(**stamp,sealed_utc=now(),file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        unique_evaluation_origins=509,first_stage_months=24,new_ml_fits=0,new_empirical_baseline_fits=48))
    print('Sealed aligned outcomes and past-only empirical baselines; no evaluation scores opened.')


if __name__=='__main__':main()
