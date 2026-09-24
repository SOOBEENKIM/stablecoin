"""Replay model fits, calibration streams and outcome-blind matching."""
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import pandas as pd
from economic import HERE, ADAPTIVE, a, CASES, GROUPS, features, make_model
from state_design import match_month

OUT = HERE/'results'


def replay(task):
    mode, definition, model, info = task
    fold = '2026-02'
    j = json.loads((OUT/'checkpoints'/('__'.join([mode,definition,fold,model,info])+'.json')).read_text())
    _,_,train,test,_,_ = a.splits(a.panel(mode),mode,definition,pd.Timestamp(fold+'-01',tz='UTC'))
    stored = pd.read_csv(OUT/'checkpoints'/('__'.join([mode,definition,fold,model,info])+'.csv.gz'))
    stored.origin = pd.to_datetime(stored.origin,utc=True)
    rows=[]
    for spec in j['fits']:
        fit = make_model(model,spec['form'],spec['params'],spec['seed']).fit(
            a.window(train,pd.Timestamp(fold+'-01',tz='UTC'),spec['window']),features(mode,info))
        forecast=fit.predict(test)
        expected=stored[(stored.strategy==spec['strategy'])&(stored.seed==spec['seed'])].sort_values('origin')
        assert list(expected.origin)==list(test.index)
        np.testing.assert_allclose(forecast,expected.pred_raw,atol=1e-9,rtol=0)
        rows.append(dict(mode=mode,definition=definition,model=model,info=info,strategy=spec['strategy'],seed=spec['seed'],
            n=len(test),max_error_bp=float(np.max(np.abs(forecast-expected.pred_raw.to_numpy())))))
    return rows


def main():
    m=json.loads((OUT/'manifest.json').read_text())
    assert m['completed']
    for k in ['inherited_source_sha256','input_sha256','previous_forecast_sha256']:
        for name,digest in m[k].items():
            assert a.sha256(a.ROOT/name)==digest,name
    for name,digest in m['source_sha256'].items():
        assert a.sha256(HERE/name)==digest,name
    assert a.sha256(HERE/'PROTOCOL_KO.md')==m['protocol_sha256']
    p=pd.read_csv(OUT/'predictions.csv.gz')
    for c in ['origin','target_time','latest_calibration_label']:
        p[c]=pd.to_datetime(p[c],utc=True)
    assert (p.target_time-p.origin==pd.Timedelta(hours=1)).all()
    assert (p.loc[p.evaluation,'latest_calibration_label']<p.loc[p.evaluation,'origin']).all()
    keys=['mode','definition','model','info','strategy','seed','origin']
    assert not p.duplicated(keys).any()
    corrections=[]
    for key,g in p.groupby(keys[:-1]):
        g=g.sort_values('origin').reset_index(drop=True)
        again=a.calibrate(g)
        np.testing.assert_allclose(g.pred_calibrated,again.pred_calibrated,atol=1e-9,rtol=0)
        i=len(g)//2
        changed=g.copy()
        changed.loc[changed.target_time>=g.origin.iloc[i],'target']+=1e6
        altered=a.calibrate(changed)
        np.testing.assert_allclose(altered.loc[:i,'pred_calibrated'],again.loc[:i,'pred_calibrated'],atol=1e-9,rtol=0)
        corrections.append(float(np.max(np.abs(g.pred_calibrated-again.pred_calibrated))))
    for s in json.loads((OUT/'split_audit.json').read_text()):
        assert pd.Timestamp(s['inner_max_label'])<pd.Timestamp(s['validation_start'])
        assert pd.Timestamp(s['outer_max_label'])<pd.Timestamp(s['first_test_origin'])
        assert pd.Timestamp(s['inner']['fit_end'])<pd.Timestamp(s['validation_start'])
        assert pd.Timestamp(s['outer']['fit_end'])<pd.Timestamp(s['cutoff'])
    tasks=[(mode,definition,kind,info) for mode,definition in CASES
        for kind,info in [('qrf','without_'+g) for g in GROUPS]+[('threshold','base'),('threshold','history')]]
    with ProcessPoolExecutor(max_workers=8) as pool:
        fits=[row for result in pool.map(replay,tasks) for row in result]
    paths=pd.read_csv(OUT/'eligible_paths.csv.gz')
    paths.origin=pd.to_datetime(paths.origin,utc=True)
    pairs=pd.read_csv(OUT/'matched_pairs.csv')
    for c in ['high_origin','low_origin']:
        pairs[c]=pd.to_datetime(pairs[c],utc=True)
    _,_,train,_,_,_=a.splits(a.panel('available_macro'),'available_macro','EQ',pd.Timestamp('2026-02-01',tz='UTC'))
    matching_replays=0
    for cohort in ['one_hour','common_path']:
        for exposure in GROUPS:
            sample=paths[(paths.definition=='EQ')&(paths.fold=='2026-02')&(paths.cohort==cohort)&
                (paths.exposure==exposure)&(paths.h==1)].set_index('origin')
            got,_,_,_=match_month(train,sample,exposure)
            saved=pairs[(pairs.definition=='EQ')&(pairs.fold=='2026-02')&(pairs.cohort==cohort)&(pairs.exposure==exposure)]
            assert len(got)==len(saved)
            if len(got):
                assert list(zip(got.high_origin,got.low_origin))==list(zip(saved.high_origin,saved.low_origin))
            changed=sample.copy()
            for col in ['target','delta_e','delta_local_bp','predicted_q10_h1']:
                changed[col]=np.arange(len(changed))*1e7
            check,_,_,_=match_month(train,changed,exposure)
            pd.testing.assert_frame_equal(got,check)
            matching_replays+=1
    matched=pd.read_csv(OUT/'matched_observations.csv.gz')
    matched.origin=pd.to_datetime(matched.origin,utc=True)
    matched.target_time=pd.to_datetime(matched.target_time,utc=True)
    assert (matched.target_time-matched.origin==pd.to_timedelta(matched.h,unit='h')).all()
    for _,g in matched.groupby(['definition','cohort','exposure','h']):
        assert not g.origin.duplicated().any()
    for _,g in matched[matched.cohort=='common_path'].groupby(['definition','exposure','pair_id','arm']):
        assert set(g.h)=={1,6,12} and g.origin.nunique()==1
    result=dict(input_original_code_and_forecast_hashes_match=True,protocol_unchanged=True,
        actual_clock_targets=True,purged_training_labels=True,future_outcome_perturbation_passed=True,
        calibration_streams=len(corrections),max_calibration_replay_error_bp=max(corrections),
        refitted_forecast_streams=len(fits),max_model_replay_error_bp=max(r['max_error_bp'] for r in fits),
        matching_replays=matching_replays,outcome_blind_matching_verified=True,same_origin_paths_verified=True,
        fit_replays=fits,source_sha256=a.sha256(HERE/'verify_results.py'))
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='fit_replays'},indent=2))


if __name__=='__main__':
    main()
