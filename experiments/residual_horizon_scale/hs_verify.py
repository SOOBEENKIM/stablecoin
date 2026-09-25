import json
import numpy as np
import pandas as pd
from hs_core import OUT, ROOT, OLD, old, HORIZONS, VARIANTS, INFOS, STRATEGIES, SEED, read, make_case, columns, sha, now, write_new
from hs_guard import verify


def main():
    stamp=verify()
    seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    assert seal['lock_sha256']==stamp['lock_sha256']
    for f,s in seal['file_sha256'].items(): assert sha(ROOT/f)==s,f
    selections=pd.DataFrame(json.loads((OUT/'selections.json').read_text()))
    preds=read(OUT/'predictions.csv.gz')
    targets=read(OUT/'targets.csv.gz')
    oldpred=read(OLD/'results/predictions.csv.gz')
    oldpred=oldpred[(oldpred['mode']=='available_macro')&(oldpred.definition=='EQ')&(oldpred.seed==SEED)]
    streams=0; max_error=0.; selected_checked=0
    for h in HORIZONS:
        for v in VARIANTS:
            for info in INFOS:
                s=pd.concat([read(OUT/'candidates'/f'{h}__{v}__{info}__{kind}.csv.gz') for kind in old.DESIGN['families']],ignore_index=True)
                for cid,g in s.groupby('candidate'):
                    g=g.sort_values('origin').reset_index(drop=True)
                    errs=(g.target-g.pred_raw).to_numpy()
                    lab=pd.DatetimeIndex(g.target_time)
                    expected=[]
                    for t in g.origin:
                        ids=np.flatnonzero((lab<t)&(lab>=t-pd.Timedelta(days=90)))[-60:]
                        expected.append(np.sort(errs[ids])[int(np.ceil(len(ids)*.1))-1] if len(ids)>=30 else 0.)
                    error=float(np.max(np.abs(g.correction-np.asarray(expected))))
                    assert error<1e-12
                    max_error=max(max_error,error);streams+=1
                for fold in sorted(preds.fold.unique()):
                    cutoff=pd.Timestamp(fold+'-01',tz='UTC')
                    ends=s.origin.dt.normalize().map(lambda t:t.replace(day=1)+pd.offsets.MonthBegin(1))
                    inner=s[(s.origin>=cutoff-pd.offsets.MonthBegin(3))&(s.origin<cutoff)&(s.target_time<cutoff)&(s.target_time<ends)].copy()
                    error=inner.target-inner.pred_calibrated
                    inner['checkloss']=np.maximum(.1*error,-.9*error)
                    scores=inner.groupby('candidate').checkloss.mean()
                    for st in STRATEGIES:
                        ids=[i for i in scores.index if old.SPECS[i]['kind'] in (['qrf','boosting'] if st=='ml_selected' else [st])]
                        chosen=min(ids,key=lambda i:(scores.loc[i],i))
                        note=selections[(selections.horizon==h)&(selections.variant==v)&(selections.information==info)&(selections.fold==fold)&(selections.strategy==st)].iloc[0]
                        assert note.candidate==chosen
                        g=preds[(preds.horizon==h)&(preds.variant==v)&(preds.information==info)&(preds.fold==fold)&(preds.strategy==st)].sort_values('origin')
                        z=s[(s.fold==fold)&(s.candidate==chosen)].sort_values('origin')
                        assert list(g.origin)==list(z.origin)
                        np.testing.assert_array_equal(g.pred_calibrated,z.pred_calibrated)
                        if (h,v)==(1,'original'):
                            b=oldpred[(oldpred.information==info)&(oldpred.fold==fold)&(oldpred.strategy==st)].sort_values('origin')
                            np.testing.assert_array_equal(g.pred_calibrated,b.pred_calibrated)
                        selected_checked+=len(g)
    # Independently re-fit each Jan family at each h/variant/info, including reused h=1.
    panel=old.legacy.a.panel('available_macro');cutoff=pd.Timestamp('2026-01-01',tz='UTC')
    refits=0;refit_error=0.
    for h in HORIZONS:
        train,test,_=make_case(panel,cutoff,h)
        for v in VARIANTS:
            for info in INFOS:
                for kind in old.DESIGN['families']:
                    note=selections[(selections.horizon==h)&(selections.variant==v)&(selections.information==info)&(selections.fold=='2026-01')&(selections.strategy==kind)].iloc[0]
                    spec=old.SPECS[int(note.candidate)]
                    model=old.legacy.fit_model(spec,old.legacy.a.window(train,cutoff,spec['window']),columns(info,v),SEED)
                    new=model.predict(test[columns(info,v)])
                    g=preds[(preds.horizon==h)&(preds.variant==v)&(preds.information==info)&(preds.fold=='2026-01')&(preds.strategy==kind)].sort_values('origin')
                    error=float(np.max(np.abs(new-g.pred_raw.to_numpy())))
                    assert error<1e-9
                    refit_error=max(refit_error,error);refits+=1
    # Every cell has the same target and origin within horizon.
    for h in HORIZONS:
        t=targets[(targets.horizon==h)&(targets.origin>=pd.Timestamp('2025-12-01',tz='UTC'))].sort_values('origin')
        for _,g in preds[preds.horizon==h].groupby(['variant','information','strategy']):
            g=g.sort_values('origin')
            assert list(g.origin)==list(t.origin)
            np.testing.assert_array_equal(g.target,t.target)
    write_new(OUT/'VERIFICATION.json',dict(**stamp,verified_utc=now(),candidate_streams=streams,
        calibration_max_error=max_error,selected_rows_checked=selected_checked,
        actual_refits=refits,actual_refit_max_error=refit_error,
        one_hour_archived_predictions_and_selections_identical=True,
        predictions_sha256=sha(OUT/'predictions.csv.gz')))
    print('VERIFIED',streams,'streams;',refits,'actual refits; max errors',max_error,refit_error,flush=True)


if __name__=='__main__':main()
