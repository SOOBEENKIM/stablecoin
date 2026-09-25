"""Audit group removal, nested selection, refits and chronological calibration."""
import ast
import json
import numpy as np
import pandas as pd
from step2_guard import verify_lock,sha,write_new,now
from step2_core import (OUT,DESIGN,CASES,GROUPS,features,interaction_pairs,fit_model,
                        choose,full_spec,read_full,a,n)


def verify_seal():
    stamp=verify_lock();seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    assert stamp['lock_sha256']==seal['lock_sha256']
    for file,key in [('predictions.csv.gz','predictions_sha256'),('inner_scores.csv','inner_sha256'),
                     ('selected_models.json','selections_sha256')]:
        assert sha(OUT/file)==seal[key]
    return stamp,seal


def main():
    stamp,seal=verify_seal();scores=pd.read_csv(OUT/'inner_scores.csv')
    metadata=json.loads((OUT/'selected_models.json').read_text())
    pred=pd.read_csv(OUT/'predictions.csv.gz')
    for c in ['origin','target_time','latest_calibration_label']:pred[c]=pd.to_datetime(pred[c],utc=True)
    group=['mode','definition','removed_group','procedure','seed']
    assert not pred.duplicated(group+['origin']).any()
    assert ((pred.target_time-pred.origin)==pd.Timedelta(hours=1)).all()
    assert len(scores)==7056 and len(metadata)==210
    for _,row in scores.iterrows():
        start=pd.Timestamp(row.fold+'-01',tz='UTC')
        assert pd.Timestamp(row.train_last_label)<start
        assert pd.Timestamp(row.residualizer_fit_end)<start
        assert pd.Timestamp(row.first_origin)>=start
        assert pd.Timestamp(row.last_label)<start+pd.offsets.MonthBegin(1)
        assert ast.literal_eval(row.features)==features(row['mode'],row.removed_group)
    for m in metadata:
        cutoff=pd.Timestamp(m['fold']+'-01',tz='UTC')
        if m['procedure']=='retuned':
            months=[x.strftime('%Y-%m') for x in n.inner_months(cutoff)]
            s=scores[(scores['mode']==m['mode'])&(scores.definition==m['definition'])&
                (scores.removed_group==m['removed_group'])&scores.fold.isin(months)]
            selected=choose(s,cutoff,m['removed_group'])
        else:selected=full_spec(m['mode'],m['definition'],m['fold'])
        assert selected['candidate']==m['candidate']
        np.testing.assert_allclose(selected['inner_score'],m['inner_score'],atol=1e-12,rtol=0)
        assert m['features']==features(m['mode'],m['removed_group'])
        assert all(c in m['features'] for c in DESIGN['always_keep'])
        assert not set(m['features'])&set(GROUPS[m['removed_group']])
        if m['kind']=='interaction':
            assert m['interaction_pairs']==[list(pair) for pair in interaction_pairs(m['features'])]
        assert pd.Timestamp(m['train_last_label'])<cutoff
        assert pd.Timestamp(m['residualizer']['fit_end'])<cutoff
    old=read_full();max_old=0.;streams=0;cal_error=0.;counts={}
    for key,g in pred.groupby(group):
        mode,definition,removed,procedure,seed=key
        g=g.sort_values('origin').reset_index(drop=True)
        ref=old[(old['mode']==mode)&(old.definition==definition)&(old.seed==seed)].sort_values('origin')
        assert list(g.origin)==list(ref.origin)
        assert list(g.target_time)==list(ref.target_time)
        np.testing.assert_allclose(g[['target','e_now']],ref[['target','e_now']],atol=1e-10,rtol=0)
        counts[mode+'/'+definition]=int(g.evaluation.sum())
        if procedure=='full':
            cols=['target','e_now','pred_raw','pred_calibrated','correction']
            max_old=max(max_old,float(np.max(np.abs(g[cols].to_numpy()-ref[cols].to_numpy()))))
            np.testing.assert_allclose(g[cols],ref[cols],atol=1e-10,rtol=0)
        replay=a.calibrate(g)
        cal_error=max(cal_error,float(np.max(np.abs(replay.pred_calibrated-g.pred_calibrated))))
        np.testing.assert_allclose(replay.pred_calibrated,g.pred_calibrated,atol=1e-10,rtol=0)
        split=len(g)//2;fake=g.copy();fake.loc[split:,'target']+=1e8
        changed=a.calibrate(fake)
        mask=g.origin<=g.origin.iloc[split]
        np.testing.assert_allclose(changed.loc[mask,'pred_calibrated'],replay.loc[mask,'pred_calibrated'],atol=1e-10,rtol=0)
        assert (g.loc[g.calibration_n>=30,'latest_calibration_label']<g.loc[g.calibration_n>=30,'origin']).all()
        streams+=1
    assert counts=={'available_macro/CAP':1088,'available_macro/EQ':1088,'available_macro/PCA':1088,'strict/EQ':467}
    replay_error=0.;full_error=0.;replays=0;full_replays=0
    for mode,definition in CASES:
        cutoff=pd.Timestamp('2026-01-01',tz='UTC')
        train,test,r=n.outer_sample(a.panel(mode),mode,definition,cutoff)
        for m in [x for x in metadata if (x['mode'],x['definition'],x['fold'])==(mode,definition,'2026-01')]:
            cols=features(mode,m['removed_group'])
            model=fit_model(m,a.window(train,cutoff,m['window']),cols,m['seed'])
            values=model.predict(test[cols])
            g=pred[(pred['mode']==mode)&(pred.definition==definition)&(pred.fold=='2026-01')&
                (pred.removed_group==m['removed_group'])&(pred.procedure==m['procedure'])&(pred.seed==m['seed'])].sort_values('origin')
            assert list(g.origin)==list(test.index)
            replay_error=max(replay_error,float(np.max(np.abs(values-g.pred_raw.to_numpy()))))
            np.testing.assert_allclose(values,g.pred_raw,atol=1e-10,rtol=0)
            fake=test.copy();fake['target']+=1e8
            for col in GROUPS[m['removed_group']]:
                if col in fake:fake[col]+=1e8
            np.testing.assert_array_equal(values,model.predict(fake[cols]))
            replays+=1
        # Full-input January fits verify compatibility with the original pipeline.
        spec=full_spec(mode,definition,'2026-01');cols=a.columns(mode,'history')
        seeds=n.SEEDS if (mode,definition)==tuple(DESIGN['primary_case']) else n.SEEDS[:1]
        for seed in seeds:
            model=fit_model(spec,a.window(train,cutoff,spec['window']),cols,seed)
            values=model.predict(test[cols])
            ref=old[(old['mode']==mode)&(old.definition==definition)&(old.fold=='2026-01')&(old.seed==seed)].sort_values('origin')
            full_error=max(full_error,float(np.max(np.abs(values-ref.pred_raw.to_numpy()))))
            np.testing.assert_allclose(values,ref.pred_raw,atol=1e-10,rtol=0)
            full_replays+=1
        print('REFIT VERIFIED',mode,definition,flush=True)
    translation=0.
    for version in ['raw','calibrated']:
        original=a.pinball(pred.target,pred['pred_'+version],.1)
        shifted=a.pinball(pred.target-pred.e_now,pred['pred_'+version]-pred.e_now,.1)
        translation=max(translation,float(np.max(np.abs(original-shifted))))
        np.testing.assert_allclose(original,shifted,atol=1e-10,rtol=0)
    write_new(OUT/'VERIFICATION.json',dict(**stamp,verified_utc=now(),predictions_sha256=seal['predictions_sha256'],
        selections_replayed=len(metadata),new_january_refits=replays,full_january_refits=full_replays,
        calibration_streams_replayed=streams,max_replay_error_bp=replay_error,max_full_refit_error_bp=full_error,
        max_legacy_prediction_difference_bp=max_old,max_calibration_error_bp=cal_error,
        max_translation_loss_difference_bp=translation,counts=counts,group_removal_and_derived_terms_verified=True,
        future_outcome_perturbation_passed=True,independent_confirmation=False))
    print('VERIFIED all selections, feature removal, labels, refits and calibration.',flush=True)


if __name__=='__main__':main()
