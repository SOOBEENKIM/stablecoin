import json
import numpy as np
import pandas as pd
from r12_core import (OUT,ROOT,PARENT,old,hs,DEFS,VARIANTS,INFOS,FAMILIES,STRATEGIES,SEEDS,
    read,case,file_key,inner_rows,independent_calibration,verify_selection,sha,now,write_new)
from r12_guard import verify_seal


def main():
    stamp,_=verify_seal()
    pred=read(OUT/'predictions.csv.gz');targets=read(OUT/'targets.csv.gz')
    choices=pd.DataFrame(json.loads((OUT/'selections.json').read_text()))
    parent_pred=read(PARENT/'results/predictions.csv.gz')
    parent_pred=parent_pred[(parent_pred.horizon==12)&(parent_pred.seed==SEEDS[0])]
    parent_extra=read(PARENT/'results/seed_sensitivity/predictions.csv.gz')
    errors=[];selection_audit=[];checked=0;poison_checks=0
    for definition in DEFS:
        for variant in VARIANTS:
            for info in INFOS:
                first=pd.concat([read(OUT/'candidates'/(file_key(definition,variant,info,k)+'.csv.gz')) for k in FAMILIES],ignore_index=True)
                first['checkloss']=np.maximum(.1*(first.target-first.pred_calibrated),-.9*(first.target-first.pred_calibrated))
                for cid,g in first.groupby('candidate',sort=True):errors.append(independent_calibration(g))
                for cutoff in old.OUTER_MONTHS:
                    inner=inner_rows(first,cutoff)
                    fold=cutoff.strftime('%Y-%m')
                    for st in STRATEGIES:
                        note=choices[(choices.definition==definition)&(choices.variant==variant)&(choices.information==info)&(choices.fold==fold)&(choices.strategy==st)].iloc[0]
                        kinds=['qrf','boosting'] if st=='ml_selected' else [st]
                        ids=[s['candidate'] for s in old.SPECS if s['kind'] in kinds]
                        audit=verify_selection(inner,ids,int(note.candidate))
                        selection_audit.append(dict(definition=definition,variant=variant,information=info,fold=fold,strategy=st,**audit))
                for seed in SEEDS:
                    stream=first if seed==SEEDS[0] else read(OUT/'extra_candidates'/(file_key(definition,variant,info,seed)+'.csv.gz'))
                    if seed!=SEEDS[0]:
                        for cid,g in stream.groupby('candidate',sort=True):errors.append(independent_calibration(g))
                    for cutoff in old.OUTER_MONTHS:
                        fold=cutoff.strftime('%Y-%m')
                        for st in (STRATEGIES if seed==SEEDS[0] else ['ml_selected']):
                            note=choices[(choices.definition==definition)&(choices.variant==variant)&(choices.information==info)&(choices.fold==fold)&(choices.strategy==st)].iloc[0]
                            z=stream[(stream.fold==fold)&(stream.candidate==note.candidate)].sort_values('origin')
                            g=pred[(pred.definition==definition)&(pred.variant==variant)&(pred.information==info)&(pred.fold==fold)&(pred.strategy==st)&(pred.seed==seed)].sort_values('origin')
                            assert list(g.origin)==list(z.origin) and len(g)>0
                            np.testing.assert_array_equal(g.pred_calibrated,z.pred_calibrated)
                            np.testing.assert_array_equal(g.target,z.target);checked+=len(g)
                            if definition=='EQ' and (seed==SEEDS[0] or info=='F'):
                                archive=parent_pred if seed==SEEDS[0] else parent_extra
                                a=archive[(archive.variant==variant)&(archive.information==info)&(archive.seed==seed)&(archive.fold==fold)]
                                if seed==SEEDS[0]:a=a[a.strategy==st]
                                a=a.sort_values('origin');np.testing.assert_array_equal(a.pred_calibrated,g.pred_calibrated)
                    # Actual calibration future poison for each info/definition/variant/seed.
                    g=stream[stream.candidate==stream.candidate.min()].copy()
                    cut=pd.Timestamp('2026-03-01',tz='UTC');d=g.copy();d.loc[d.target_time>=cut,'target']=1e8
                    a=hs.calibrate(g,12);b=hs.calibrate(d,12)
                    np.testing.assert_array_equal(a[a.origin<=cut].pred_calibrated,b[b.origin<=cut].pred_calibrated)
                    poison_checks+=1
                print('AUDITED',definition,variant,info,flush=True)
    refits=0;refit_error=0.
    panel=old.legacy.a.panel('available_macro');cutoff=pd.Timestamp('2026-01-01',tz='UTC')
    for definition in DEFS:
        tr,te,_=case(panel,definition,cutoff)
        for variant in VARIANTS:
            for info in INFOS:
                for kind in FAMILIES:
                    n=choices[(choices.definition==definition)&(choices.variant==variant)&(choices.information==info)&(choices.fold=='2026-01')&(choices.strategy==kind)].iloc[0]
                    spec=old.SPECS[int(n.candidate)]
                    model=old.legacy.fit_model(spec,old.legacy.a.window(tr,cutoff,spec['window']),hs.columns(info,variant),SEEDS[0])
                    actual=model.predict(te[hs.columns(info,variant)])
                    g=pred[(pred.definition==definition)&(pred.variant==variant)&(pred.information==info)&(pred.fold=='2026-01')&(pred.strategy==kind)&(pred.seed==SEEDS[0])].sort_values('origin')
                    err=float(np.max(np.abs(actual-g.pred_raw.to_numpy())));assert err<1e-9
                    refit_error=max(refit_error,err);refits+=1
    # Same observation support across definitions; same targets within definition.
    origins=None
    for definition in DEFS:
        t=targets[(targets.definition==definition)&(targets.origin>=hs.START)].sort_values('origin')
        assert len(t)==509
        if origins is None:origins=list(t.origin)
        else:assert origins==list(t.origin)
        for _,g in pred[pred.definition==definition].groupby(['variant','information','strategy','seed']):
            g=g.sort_values('origin');assert list(g.origin)==origins
            np.testing.assert_array_equal(g.target,t.target)
    write_new(OUT/'SELECTION_AUDIT.json',selection_audit)
    write_new(OUT/'VERIFICATION.json',dict(**stamp,verified_utc=now(),candidate_streams=len(errors),
        calibration_max_error=max(errors),selected_rows_checked=checked,selection_checks=len(selection_audit),
        future_poison_checks=poison_checks,actual_refits=refits,actual_refit_max_error=refit_error,
        archived_EQ_predictions_identical=True,predictions_sha256=sha(OUT/'predictions.csv.gz')))
    print('VERIFIED',len(errors),'streams;',refits,'refits;',checked,'selected rows',flush=True)


if __name__=='__main__':main()
