"""Post-result seed sensitivity of the already selected 12-hour full QRF."""
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from hs_core import HERE,ROOT,OUT,old,read,make_case,columns,calibrate,save,sha,now,write_new,SEED
from hs_guard import verify,BRANCH
from hs_amendment_guard import verify_amendment
from factorial_stats import day_weights,effect

LOCK=HERE/'SEED_LOCK.json'
DEST=OUT/'seed_sensitivity'


def freeze():
    verify();verify_amendment()
    assert (OUT/'EVALUATION_COMPLETE.json').exists()
    paths=[Path for Path in [HERE/'hs_seed_check.py',HERE/'SEED_SENSITIVITY_KO.md',OUT/'predictions.csv.gz',OUT/'selections.json']]
    write_new(LOCK,dict(created_utc=now(),post_result=True,independent_confirmation=False,
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))


def run():
    verify();verify_amendment()
    lock=json.loads(LOCK.read_text())
    for p,s in lock['file_sha256'].items():assert sha(ROOT/p)==s
    rel=str(LOCK.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==LOCK.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True)
    write_new(DEST/'RUN_STARTED.json',dict(created_utc=now(),lock_commit=commit,lock_sha256=sha(LOCK)))
    sels=pd.DataFrame(json.loads((OUT/'selections.json').read_text()))
    sel=sels[(sels.horizon==12)&(sels.information=='F')&(sels.strategy=='ml_selected')]
    assert len(sel)==8 and (sel.candidate==16).all()
    panel=old.legacy.a.panel('available_macro')
    data={c:make_case(panel,c,12) for c in old.MONTHS}
    frames=[]
    for v in ['original','own_scale']:
        for seed in [20260926,20260927,20260928]:
            raw=[]
            for cutoff in old.MONTHS:
                tr,te,_=data[cutoff]
                model,pred=old.fit_predict(old.SPECS[16],tr,te,columns('F',v),seed)
                d=te[['target','target_time']].reset_index()
                d['horizon'],d['variant'],d['information'],d['seed'],d['candidate']=12,v,'F',seed,16
                d['fold']=cutoff.strftime('%Y-%m');d['pred_raw']=pred
                raw.append(d)
            d=calibrate(pd.concat(raw,ignore_index=True),12)
            # Verify maturity and independent order-statistic arithmetic.
            for row in d.itertuples():
                eligible=d[(d.target_time<row.origin)&(d.target_time>=row.origin-pd.Timedelta(days=90))].tail(60)
                expected=sorted((eligible.target-eligible.pred_raw).tolist())[int(np.ceil(len(eligible)*.1))-1] if len(eligible)>=30 else 0.
                assert row.correction==expected
            frames.append(d)
            print('VERIFIED extra seed',v,seed,flush=True)
    extra=pd.concat(frames,ignore_index=True)
    save(extra,DEST/'candidate_streams.csv.gz')
    base=read(OUT/'predictions.csv.gz')
    base=base[(base.horizon==12)&(base.information=='F')]
    forecasts=pd.concat([base[base.strategy=='ml_selected'],extra[extra.origin>=pd.Timestamp('2025-12-01',tz='UTC')]],ignore_index=True)
    save(forecasts,DEST/'predictions.csv.gz')
    rows,monthly,averages=[],[],[]
    for v in ['original','own_scale']:
        for version in ['raw','calibrated']:
            d=forecasts[forecasts.variant==v].copy()
            d['loss']=old.loss(d.target,d['pred_'+version])
            means=d.groupby('origin',as_index=False).agg(loss=('loss','mean'),seeds=('seed','nunique'),fold=('fold','first'),target=('target','first'))
            assert len(means)==509 and (means.seeds==4).all()
            weights=day_weights(means[['origin','fold']],5)
            for refname in ['linear','threshold']:
                ref=base[(base.variant==v)&(base.strategy==refname)].sort_values('origin').copy()
                np.testing.assert_array_equal(ref.target,means.target)
                ref['loss']=old.loss(ref.target,ref['pred_'+version])
                stats=effect(ref.loss,means.loss,weights)
                stats.pop('p_centered_boot')  # Sensitivity, not a new pass/fail test.
                averages.append(dict(variant=v,calibration=version,reference=refname,n=509,**stats))
                for seed,g in d.groupby('seed'):
                    g=g.sort_values('origin');assert list(g.origin)==list(ref.origin)
                    np.testing.assert_array_equal(g.target,ref.target)
                    rows.append(dict(variant=v,calibration=version,reference=refname,seed=int(seed),n=509,
                        loss_bp=g.loss.mean(),improvement_pct=100*(1-g.loss.mean()/ref.loss.mean())))
                    a,b=ref.groupby('fold').loss.mean(),g.groupby('fold').loss.mean()
                    for fold in a.index:
                        monthly.append(dict(variant=v,calibration=version,reference=refname,seed=int(seed),fold=fold,improvement_pct=100*(1-b[fold]/a[fold])))
    for name,rows_ in [('per_seed',rows),('monthly',monthly),('four_seed_mean',averages)]:save(pd.DataFrame(rows_),DEST/(name+'.csv'))
    write_new(DEST/'COMPLETE.json',dict(completed_utc=now(),lock_sha256=sha(LOCK),additional_fits=48,
        label_and_independent_calibration_checks=True,independent_confirmation=False,
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in DEST.iterdir() if p.is_file()}))
    print(pd.DataFrame(rows).query("calibration == 'calibrated'").to_string(index=False))
    print(pd.DataFrame(averages).query("calibration == 'calibrated'").to_string(index=False))


if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
