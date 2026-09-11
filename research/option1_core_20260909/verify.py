from common import *
import statsmodels.api as sm
from scipy.sparse import hstack,eye,csr_matrix


def main():
    checks={};rng=np.random.default_rng(777)
    # Independently formulate primal quantile LP, including unequal weights.
    x=np.column_stack([np.ones(57),rng.normal(size=(57,3))])
    y=x@np.array([2.,-.4,1.7,.2])+rng.normal(size=57)
    w=rng.integers(0,4,size=57).astype(float)
    objective_gaps=[]
    for q in [.1,.5]:
        beta,gap=qr(x,y,q,w)
        primal=linprog(np.r_[np.zeros(x.shape[1]),q*w,(1-q)*w],
            A_eq=hstack([csr_matrix(x),eye(len(y)),-eye(len(y))]),b_eq=y,
            bounds=[(None,None)]*x.shape[1]+[(0,None)]*(2*len(y)),method='highs')
        assert primal.success
        e=y-x@beta
        obj=np.sum(w*np.maximum(q*e,(q-1)*e))
        assert abs(obj-primal.fun)<1e-6
        objective_gaps.append(abs(obj-primal.fun))
    expected=np.array([2.,-.4,1.7,.2])
    np.testing.assert_allclose(qr(x,x@expected,.1)[0],expected,atol=1e-8)
    checks['independent_primal_QR_objective_gap_max']=max(objective_gaps)
    checks['QR_exact_linear_model_recovery']=True
    dates=pd.date_range('2020-01-01',periods=len(y),tz='UTC')
    beta,se,p,cov=calendar_hac(x,y,dates)
    other=sm.OLS(y,x).fit(cov_type='HAC',cov_kwds={'maxlags':3,'use_correction':True})
    np.testing.assert_allclose(cov,other.cov_params(),atol=1e-10,rtol=1e-8)
    # Insert calendar gaps without compressing lag distances; independent meat.
    keep=np.ones(len(y),dtype=bool);keep[20:23]=False
    xx=x[keep];yy=y[keep];dd=dates[keep]
    b,s,p,c=calendar_hac(xx,yy,dd)
    scores=np.zeros_like(x);scores[keep]=xx*(yy-xx@b)[:,None]
    meat=scores.T@scores
    for lag in [1,2,3]:
        v=scores[lag:].T@scores[:-lag];meat+=(1-lag/4)*(v+v.T)
    bread=np.linalg.inv(xx.T@xx)
    np.testing.assert_allclose(c,bread@meat@bread*len(yy)/(len(yy)-xx.shape[1]),rtol=1e-8,atol=1e-10)
    checks['calendar_HAC_matches_statsmodels_and_explicit_missing_days']=True
    synthetic_dates=dates.repeat(2)
    weights=calendar_weights(synthetic_dates,5,99,rng)
    assert np.array_equal(weights[:,::2],weights[:,1::2])
    assert (weights.sum(axis=1)==len(synthetic_dates)).all()
    checks['calendar_weights_identical_within_days_and_preserve_draw_length']=True
    data=np.load(HERE/'economic_data.npz')
    basis=read_saved('basis_hourly.csv.gz')
    names=list(data['feature_names'])
    for h in [1,6,12]:
        origin=pd.to_datetime(data[f'origin{h}'],utc=True);target=pd.to_datetime(data[f'target{h}'],utc=True)
        assert ((target-origin)==pd.Timedelta(hours=h)).all()
        np.testing.assert_allclose(data[f'y{h}'],basis.reindex(target)[REFS],atol=1e-7)
        xx=data[f'x{h}'];yy=data[f'y{h}']
        assert len(xx)==len(yy) and yy.shape[1]==5
        assert np.isfinite(xx).all() and np.isfinite(yy).all()
        assert np.linalg.matrix_rank(xx)==xx.shape[1]
        assert (xx[:,names.index('down_pct')]>=0).all()
        np.testing.assert_allclose(xx[:,names.index('down_x_ls')],xx[:,names.index('down_pct')]*xx[:,names.index('ls_z')])
        b=ols(xx,yy);br=ols(xx,yy[:,0]-yy[:,2])
        np.testing.assert_allclose(b[:,0]-b[:,2],br,atol=1e-8)
    checks['exact_horizons_full_rank_same_samples_and_OLS_decomposition']=True
    manifest=json.loads((HERE/'RUN_MANIFEST.json').read_text())
    for name,digest in manifest['inputs_sha256'].items():assert sha(ROOT/name)==digest
    assert snapshot()==json.loads((HERE/'PRESERVED_OPTIONS_SHA256.json').read_text())
    assert manifest['economic_data_sha256']==sha(HERE/'economic_data.npz')
    for filename in ['RUN_MANIFEST.json','ECONOMIC_RUN_SETTINGS.json','POLICY_COMPLETION.json']:
        assert json.loads((HERE/filename).read_text())['protocol_sha256']==sha(HERE/'PROTOCOL_KO.md')
    assert json.loads((HERE/'ECONOMIC_RUN_SETTINGS.json').read_text())['code_sha256']==sha(HERE/'economics.py')
    assert json.loads((HERE/'POLICY_COMPLETION.json').read_text())['code_sha256']==sha(HERE/'policy.py')
    checks['preserved_options_files_unchanged']=len(snapshot())
    checks['input_data_and_preresult_protocol_unchanged']=True
    policy=pd.read_csv(HERE/'policy_regressions.csv')
    assert policy.primary.sum()==1
    raw=policy.loc[policy.spec=='unadjusted']
    np.testing.assert_allclose(raw.estimate,raw.mean_after-raw.mean_before,atol=1e-8)
    econ=pd.read_csv(HERE/'economic_contrasts.csv')
    assert econ.primary.sum()==24 and econ.loc[econ.primary,'p_holm_24'].notna().all()
    boot=np.load(HERE/'economic_bootstrap.npz')
    failures={k:int((~np.isfinite(boot[k]).all(axis=(1,2))).sum()) for k in boot.files}
    assert all(v<.05*len(boot[k]) for k,v in failures.items())
    checks['primary_policy_count']=1;checks['primary_economic_Holm_family_count']=24
    checks['economic_failed_replicates']=failures
    checks['data_reconstruction_checks']=manifest['checks']
    checks['economic_solver_and_identity']=json.loads((HERE/'ECONOMIC_COMPLETION.json').read_text())
    supplement=json.loads((HERE/'SUPPLEMENT_COMPLETION.json').read_text())
    assert supplement['scope_sha256']==sha(HERE/'SUPPLEMENT_SCOPE_KO.md')
    assert supplement['code_sha256']==sha(HERE/'supplement.py')
    for filename in ['policy_regressions.csv','pretrends.csv','policy_price_sensitivity.csv']:
        a=pd.read_csv(HERE/filename)
        assert (a.n_boot>=.95*1999).all()
        assert (a.condition_number<100).all()
    exposure=pd.read_csv(HERE/'tick_exposure.csv')
    assert (exposure.n==672).all()
    assert (exposure.price_on_new_order_grid_fraction==1).all()
    checks['policy_supplement_all_designs_full_rank_and_valid_bootstrap_rates']=True
    checks['policy_order_grid_exposure_matches_observed_price_bands']=True
    (HERE/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
