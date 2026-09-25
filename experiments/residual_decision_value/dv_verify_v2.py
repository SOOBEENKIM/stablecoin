"""Independent scalar interpolation preserves exact probability-scale ties."""
import json
import subprocess
import numpy as np
import pandas as pd
from dv_core import *
from dv_prepare_v2 import verify_amendment


def linear_quantile(values,p):
    values=sorted(float(x) for x in values)
    position=(len(values)-1)*p
    left=int(np.floor(position));right=min(left+1,len(values)-1)
    fraction=position-left;a,b=values[left],values[right]
    return b-(b-a)*(1-fraction) if fraction>=.5 else a+(b-a)*fraction


def verify_verifier_amendment():
    previous=verify_amendment()
    path=HERE/'VERIFIER_AMENDMENT.json';lock=json.loads(path.read_text())
    for name,value in lock['file_sha256'].items():assert sha(ROOT/name)==value,name
    rel=str(path.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==path.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/research/icaif2026-manuscript-development'],cwd=ROOT,check=True)
    return dict(**previous,verifier_amendment_commit=commit,verifier_amendment_sha256=sha(path))


def main():
    stamp=dict(**verify_inputs(),**verify_verifier_amendment())
    applied=json.loads((OUT/'PREPARATION_AMENDMENT_APPLIED.json').read_text())
    assert applied['inputs_seal_sha256']==sha(OUT/'INPUTS_SEALED.json')
    ctx=read(OUT/'context.csv.gz');base=read(OUT/'baseline_streams.csv.gz')
    archived=read(OUT/'archived_forecasts.csv.gz')
    original=read(PARENT/'results/predictions.csv.gz')
    original=original[original.strategy.isin(['ml_selected','linear','threshold'])]
    pd.testing.assert_frame_equal(archived.reset_index(drop=True),original.reset_index(drop=True),check_exact=True)
    recorded={(r['definition'],r['fold']):r for r in json.loads((OUT/'baseline_fits.json').read_text())}
    panel=r12.old.legacy.a.panel('available_macro')
    checks=0;calerr=0.;integralerr=0.
    for (definition,strategy),g in base.groupby(['definition','strategy']):
        calerr=max(calerr,r12.independent_calibration(g))
        for fold,s in g.groupby('fold'):
            cutoff=pd.Timestamp(fold+'-01',tz='UTC')
            tr,te,r=r12.case(panel,definition,cutoff)
            e1,e2=[linear_quantile(tr.e_now,x) for x in [1/3,2/3]]
            v=linear_quantile(tr.e_rms72,.5)
            model=recorded[definition,fold]
            np.testing.assert_array_equal([e1,e2],model['e_cuts']);assert v==model['vol_cut']
            train_states=2*((tr.e_now>=e1).astype(int)+(tr.e_now>=e2).astype(int))+(tr.e_rms72>=v).astype(int)
            values=(tr.target-tr.e_now).to_numpy()
            def inverse_quantile(a):return sorted(a)[int(np.ceil(.1*len(a)))-1]
            whole=inverse_quantile(values)
            preds=[]
            for row in te.itertuples():
                state=2*(int(row.e_now>=e1)+int(row.e_now>=e2))+int(row.e_rms72>=v)
                y=values[np.asarray(train_states)==state]
                preds.append(row.e_now+(inverse_quantile(y) if strategy=='state_hist' and len(y)>=25 else whole))
            assert list(s.origin)==list(te.index)
            np.testing.assert_allclose(s.pred_raw,preds,atol=1e-12,rtol=0)
            checks+=1
    for definition,g in ctx.groupby('definition'):
        assert len(g)==509
        for strategy,s in archived[(archived.definition==definition)&(archived.variant=='original')&(archived.information=='F')].groupby(['strategy','seed']):
            assert list(s.origin)==list(g.origin)
            np.testing.assert_allclose(s.target,g.target,atol=0,rtol=0)
            np.testing.assert_allclose(s.e_now,g.e_now,atol=1e-11,rtol=0)
            expected=pinball(s.target,s.pred_calibrated)
            integral=np.array([independent_elementary_integral(y-e,q-e) for y,q,e in zip(s.target,s.pred_calibrated,s.e_now)])
            integralerr=max(integralerr,float(np.max(np.abs(integral-expected))))
    path=read(OUT/'paths.csv.gz')
    err=float(np.max(np.abs(path[PARTS].sum(axis=1)-path.delta_e)))
    assert err<1e-8 and integralerr<1e-11
    assert ((path.target_time-path.origin).dt.total_seconds()/3600==path.h).all()
    for definition,g in ctx.groupby('definition'):
        path12=path[(path.definition==definition)&(path.h==12)]
        assert list(path12.origin)==list(g.origin)
        np.testing.assert_allclose(path12.target,g.target,atol=1e-11,rtol=0)
    write_new(OUT/'VERIFICATION.json',dict(**stamp,verified_utc=now(),archived_predictions_exact=True,
        independent_baseline_month_checks=checks,calibration_max_error=calerr,
        elementary_integral_max_error=integralerr,price_identity_max_error_bp=err,
        exact_state_boundaries=True,exact_horizons=True,independent_confirmation=False))
    print('Independent verification passed: 48 benchmark-month checks; archived forecasts unchanged; exact state boundaries, calibration, elementary integral and price identity.')


if __name__=='__main__':main()
