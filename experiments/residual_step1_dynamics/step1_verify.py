"""Verify forecasts and selection without publishing model performance."""
import json
import numpy as np
import pandas as pd
from step1_guard import verify_lock, sha, write_new, now
from step1_core import OUT, DESIGN, CASES, Benchmark, choose, read_legacy, a, n


def verify_seal():
    stamp=verify_lock()
    seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    assert seal['lock_sha256']==stamp['lock_sha256']
    for name,key in [('predictions.csv.gz','predictions_sha256'),
                     ('selected_models.json','selections_sha256'),('inner_scores.csv','inner_sha256')]:
        assert sha(OUT/name)==seal[key]
    return stamp,seal


def main():
    stamp,seal=verify_seal()
    scores=pd.read_csv(OUT/'inner_scores.csv')
    metadata=json.loads((OUT/'selected_models.json').read_text())
    pred=pd.read_csv(OUT/'predictions.csv.gz')
    for c in ['origin','target_time','latest_calibration_label']:
        pred[c]=pd.to_datetime(pred[c],utc=True)
    assert not pred.duplicated(['mode','definition','strategy','seed','origin']).any()
    assert ((pred.target_time-pred.origin)==pd.Timedelta(hours=1)).all()
    for _,row in scores.iterrows():
        start=pd.Timestamp(row.fold+'-01',tz='UTC')
        assert pd.Timestamp(row.train_last_label)<start
        assert pd.Timestamp(row.residualizer_fit_end)<start
        assert pd.Timestamp(row.first_origin)>=start
        assert pd.Timestamp(row.last_label)<start+pd.offsets.MonthBegin(1)
    for m in metadata:
        cutoff=pd.Timestamp(m['fold']+'-01',tz='UTC')
        months=[x.strftime('%Y-%m') for x in n.inner_months(cutoff)]
        s=scores[(scores['mode']==m['mode'])&(scores.definition==m['definition'])&scores.fold.isin(months)]
        chosen=choose(s,cutoff,m['strategy'])
        assert chosen['candidate']==m['candidate']
        np.testing.assert_allclose(chosen['inner_score'],m['inner_score'],atol=1e-12,rtol=0)
        assert pd.Timestamp(m['train_last_label'])<cutoff
    old=read_legacy()
    max_old=0.
    for (mode,definition,strategy,seed),ref in old.groupby(['mode','definition','strategy','seed']):
        got=pred[(pred['mode']==mode)&(pred.definition==definition)&(pred.strategy==strategy)&(pred.seed==seed)].sort_values('origin')
        ref=ref.sort_values('origin')
        assert list(got.origin)==list(ref.origin)
        cols=['target','e_now','pred_raw','pred_calibrated','correction']
        error=float(np.max(np.abs(got[cols].to_numpy()-ref[cols].to_numpy())))
        max_old=max(max_old,error)
        np.testing.assert_allclose(got[cols],ref[cols],atol=1e-10,rtol=0)
    replay_error=0.;cal_error=0.;replays=0;streams=0
    for mode,definition in CASES:
        cutoff=pd.Timestamp('2026-01-01',tz='UTC')
        train,test,r=n.outer_sample(a.panel(mode),mode,definition,cutoff)
        for m in [x for x in metadata if (x['mode'],x['definition'],x['fold'])==(mode,definition,'2026-01')]:
            model=Benchmark(m).fit(a.window(train,cutoff,m['window']))
            values=model.predict(test[model.columns])
            coeff=model.coefficients()
            physical=coeff['intercept']+sum(test[c].to_numpy()*coeff[c] for c in model.columns)
            np.testing.assert_allclose(values,physical,atol=1e-10,rtol=0)
            got=pred[(pred['mode']==mode)&(pred.definition==definition)&(pred.strategy==m['strategy'])&
                (pred.fold=='2026-01')].sort_values('origin')
            assert list(got.origin)==list(test.index)
            replay_error=max(replay_error,float(np.max(np.abs(values-got.pred_raw.to_numpy()))))
            np.testing.assert_allclose(values,got.pred_raw,atol=1e-10,rtol=0)
            fake=test.copy();fake['target']+=1e9
            np.testing.assert_array_equal(values,model.predict(fake[model.columns]))
            replays+=1
    for (mode,definition,strategy,seed),g in pred.groupby(['mode','definition','strategy','seed']):
        g=g.sort_values('origin').reset_index(drop=True)
        replay=a.calibrate(g)
        cal_error=max(cal_error,float(np.max(np.abs(replay.pred_calibrated-g.pred_calibrated))))
        np.testing.assert_allclose(replay.pred_calibrated,g.pred_calibrated,atol=1e-10,rtol=0)
        middle=len(g)//2;fake=g.copy();fake.loc[middle:,'target']+=1e9
        before=fake.origin.iloc[middle]
        changed=a.calibrate(fake)
        np.testing.assert_allclose(changed.loc[changed.origin<=before,'pred_calibrated'],
            replay.loc[replay.origin<=before,'pred_calibrated'],atol=1e-10,rtol=0)
        ref=old[(old['mode']==mode)&(old.definition==definition)&(old.strategy=='selected')&
            (old.seed==n.SEEDS[0])].sort_values('origin')
        assert list(g.origin)==list(ref.origin)
        np.testing.assert_allclose(g[['target','e_now']],ref[['target','e_now']],atol=1e-10,rtol=0)
        streams+=1
    translation=0.
    for version in ['raw','calibrated']:
        level=a.pinball(pred.target,pred['pred_'+version],.1)
        change=a.pinball(pred.target_change,pred['pred_change_'+version],.1)
        translation=max(translation,float(np.max(np.abs(level-change))))
        np.testing.assert_allclose(level,change,atol=1e-10,rtol=0)
    counts={}
    for mode,definition in CASES:
        d=pred[(pred['mode']==mode)&(pred.definition==definition)&(pred.strategy=='persistence')&pred.evaluation]
        counts[mode+'/'+definition]=len(d)
    assert counts['available_macro/EQ']==1088 and counts['strict/EQ']==467
    write_new(OUT/'VERIFICATION.json',dict(**stamp, verified_utc=now(),
        predictions_sha256=seal['predictions_sha256'], selections_replayed=len(metadata),
        january_models_refitted=replays, calibration_streams_replayed=streams,
        max_replay_error_bp=replay_error,max_calibration_error_bp=cal_error,
        max_legacy_prediction_difference_bp=max_old,max_translation_loss_difference_bp=translation,
        counts=counts,future_outcome_perturbation_passed=True,independent_confirmation=False))
    print('VERIFIED: selections, inherited predictions, January refits, clocks, calibration, translation identity.')


if __name__=='__main__':main()
