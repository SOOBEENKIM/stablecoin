"""Replay nested choices, timing, calibration, and one month of actual fitted models."""
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import pandas as pd
from guard import HERE,sha,verify_lock,write_new,now
from nested import a,DESIGN,CASES,SEEDS,choose,inner_months
from run import outer_job

OUT=HERE/'results'


def replay(task):
    got,_,_=outer_job(task)
    saved=pd.read_csv(OUT/'predictions.csv.gz')
    saved.origin=pd.to_datetime(saved.origin,utc=True)
    mode,definition,fold,info,_=task
    saved=saved[(saved['mode']==mode)&(saved.definition==definition)&(saved.fold==fold)&(saved['info']==info)]
    keys=['strategy','seed','origin']
    got=got.sort_values(keys).set_index(keys);saved=saved.sort_values(keys).set_index(keys)
    assert got.index.equals(saved.index)
    np.testing.assert_allclose(got.pred_raw,saved.pred_raw,atol=1e-8,rtol=0)
    return dict(mode=mode,definition=definition,fold=fold,info=info,rows=len(got),
                max_error_bp=float(np.max(np.abs(got.pred_raw-saved.pred_raw))))


def main():
    stamp=verify_lock();seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    for key,name in [('prediction_sha256','predictions.csv.gz'),('inner_scores_sha256','inner_scores.csv'),('selections_sha256','selected_models.json')]:
        assert seal[key]==sha(OUT/name)
    scores=pd.read_csv(OUT/'inner_scores.csv')
    valid=scores[scores.valid]
    assert (pd.to_datetime(valid.train_last_target,utc=True)<pd.to_datetime(valid.first_validation_origin,utc=True)).all()
    assert (pd.to_datetime(valid.fit_end,utc=True)<pd.to_datetime(valid.first_validation_origin,utc=True)).all()
    for fold,g in valid.groupby('fold'):
        assert (pd.to_datetime(g.last_validation_target,utc=True)<pd.Timestamp(fold+'-01',tz='UTC')+pd.offsets.MonthBegin(1)).all()
    selected=json.loads((OUT/'selected_models.json').read_text());nchoices=0
    for s in selected:
        cutoff=pd.Timestamp(s['fold']+'-01',tz='UTC')
        wanted=[x.strftime('%Y-%m') for x in inner_months(cutoff)]
        inner=scores[(scores['mode']==s['mode'])&(scores.definition==s['definition'])&(scores['info']==s['info'])&scores.fold.isin(wanted)]
        got=choose(inner,cutoff,None if s['strategy']=='selected' else s['strategy'])
        assert got['candidate']==s['candidate'] and got['inner_months']==s['inner_months']
        assert pd.Timestamp(s['train_last_target'])<pd.Timestamp(s['first_test_origin'])
        assert pd.Timestamp(s['residualizer']['fit_end'])<cutoff
        nchoices+=1
    pred=pd.read_csv(OUT/'predictions.csv.gz')
    for c in ['origin','target_time','latest_calibration_label']:pred[c]=pd.to_datetime(pred[c],utc=True)
    assert (pred.target_time-pred.origin==pd.Timedelta(hours=1)).all()
    errors=[];groups=['mode','definition','info','strategy','seed']
    for _,g in pred.groupby(groups):
        g=g.sort_values('origin').reset_index(drop=True)
        check=a.calibrate(g)
        np.testing.assert_allclose(check.pred_calibrated,g.pred_calibrated,atol=1e-8,rtol=0)
        errors.append(float(np.max(np.abs(check.pred_calibrated-g.pred_calibrated))))
        ix=len(g)//2;changed=g.copy();changed.loc[ix:,'target']+=1e8
        future=a.calibrate(changed)
        np.testing.assert_allclose(future.pred_calibrated[:ix+1],check.pred_calibrated[:ix+1],atol=1e-8,rtol=0)
    tasks=[];cutoff=pd.Timestamp('2026-01-01',tz='UTC');months=[x.strftime('%Y-%m') for x in inner_months(cutoff)]
    for mode,definition in CASES:
        for info in DESIGN['information_sets']:
            s=scores[(scores['mode']==mode)&(scores.definition==definition)&(scores['info']==info)&scores.fold.isin(months)]
            tasks.append((mode,definition,'2026-01',info,s.to_dict('records')))
    with ProcessPoolExecutor(max_workers=8) as pool:replays=list(pool.map(replay,tasks))
    result=dict(**stamp,verified_utc=now(),all_choices_replayed=nchoices,calibration_streams=len(errors),
        max_calibration_error_bp=max(errors),model_replays=replays,actual_clock_labels=True,
        past_only_inner_selection=True,future_label_perturbation_passed=True,
        same_origins_and_targets_as_previous=seal['same_origins_and_targets_as_previous'],independent_confirmation=False)
    write_new(OUT/'VERIFICATION.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
