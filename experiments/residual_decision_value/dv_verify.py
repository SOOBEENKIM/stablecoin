"""Independent numerical checks before reporting downstream performance."""
import json
import numpy as np
import pandas as pd
from dv_core import *


def main():
    stamp=verify_inputs()
    ctx=read(OUT/'context.csv.gz');base=read(OUT/'baseline_streams.csv.gz')
    archived=read(OUT/'archived_forecasts.csv.gz')
    original=read(PARENT/'results/predictions.csv.gz')
    original=original[original.strategy.isin(['ml_selected','linear','threshold'])]
    pd.testing.assert_frame_equal(archived.reset_index(drop=True),original.reset_index(drop=True),check_exact=True)
    panel=r12.old.legacy.a.panel('available_macro')
    checks=0;calerr=0.;integralerr=0.
    for (definition,strategy),g in base.groupby(['definition','strategy']):
        calerr=max(calerr,r12.independent_calibration(g))
        for fold,s in g.groupby('fold'):
            cutoff=pd.Timestamp(fold+'-01',tz='UTC')
            tr,te,r=r12.case(panel,definition,cutoff)
            e1,e2=np.percentile(tr.e_now,[100/3,200/3]);v=np.percentile(tr.e_rms72,50)
            train_states=(tr.e_now>=e1).astype(int)+(tr.e_now>=e2).astype(int)
            train_states=2*train_states+(tr.e_rms72>=v).astype(int)
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
        exact_horizons=True,independent_confirmation=False))
    print('Verified archived predictions, 48 baseline-month reconstructions, causal calibration, elementary-score integral and exact price paths.')


if __name__=='__main__':main()
