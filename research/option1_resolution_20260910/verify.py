from shared import *
from statsmodels.stats.multitest import multipletests

def main():
    checks={};panel=np.load(HERE/'measurement_panel.npz')
    np.testing.assert_allclose(panel['b'],aggregate(1e4*np.log(panel['u'][:,None]*panel['p']/panel['k'])),atol=1e-8)
    checks['independent_raw_price_reconstruction']=True
    for i in range(5):
        for sign_idx,sign in enumerate([-1,1]):
            kk=panel['k'].copy();kk[:,i]=price_step(kk[:,i],sign)
            direct=aggregate(1e4*np.log(panel['u'][:,None]*panel['p']/kk))
            np.testing.assert_allclose(direct,panel['single'][i,sign_idx],atol=1e-7)
            np.testing.assert_allclose(panel['equal'][i,sign_idx,:,0]-panel['b'][:,0],-sign/5,atol=1e-9)
    checks['all_single_quote_scenarios_and_equal_relative_shock_identity']=True
    rng=np.random.default_rng(77)
    for fraction in [.5,1.]:
        for _ in range(20):
            moved=panel['k']+rng.uniform(-fraction,fraction,size=panel['k'].shape)*TICKS
            bb=aggregate(1e4*np.log(panel['u'][:,None]*panel['p']/moved))
            assert (bb>=panel['joint_'+str(fraction)+'_low']-1e-8).all()
            assert (bb<=panel['joint_'+str(fraction)+'_high']+1e-8).all()
    checks['joint_intervals_contain_independent_multivariate_price_perturbations']=True
    np.testing.assert_allclose(price_step(np.array([100.,100000.,500000.,1000000.]),-1),[99.9,99950.,499900.,999500.])
    np.testing.assert_allclose(price_step(np.array([99.9,99950.,499900.,999500.]),1),[100.,100000.,500000.,1000000.])
    checks['integer_price_steps_cross_tick_boundaries_correctly']=True
    a=pd.read_csv(HERE/'measurement_sensitivity.csv')
    assert (a.risk_count==a.fragile_risk+a.stable_risk).all()
    np.testing.assert_allclose(a.flippable_all_share,(a.fragile_risk+a.possible_new_risk)/a.n)
    np.testing.assert_allclose(a.fragile_risk_share,a.fragile_risk/a.risk_count,equal_nan=True)
    assert (a.fragile_risk<=a.risk_count).all() and (a.possible_new_risk<=a.n-a.risk_count).all()
    old=pd.read_csv(ROOT/'option1_development_20260909/one_tick_scenarios.csv')
    for spec in ['q05','q10','q90','q95','minus5','minus10','minus20']:
        for coin in COINS+['USDT']:
            x=a.query('spec==@spec and reference=="mean5" and mode==@coin and period=="all"').iloc[0]
            y=old.query('threshold==@spec and coin==@coin and period=="all"').iloc[0]
            assert x.risk_count==y.tail_points and x.fragile_risk==y.scenario_exits
    checks['prior_one_tick_results_exactly_reproduced']=True
    minimum=np.load(HERE/'minimum_tick_arrays.npz')
    for key in minimum.files:
        spec,coin=key.rsplit('_',1)
        for j,ref in enumerate(REFS):
            x=a.query('spec==@spec and reference==@ref and mode==@coin and period=="all"').iloc[0]
            assert (minimum[key][:,j]==1).sum()==x.fragile_risk
    checks['minimum_steps_one_tick_matches_independent_scenarios']=True
    c=pd.read_csv(HERE/'primary_measurement_contrasts.csv')
    assert len(c)==4
    np.testing.assert_allclose(c.p_holm_4,multipletests(c.p_centered,method='holm')[1])
    influence=pd.read_csv(HERE/'policy_influence.csv')
    core=pd.read_csv(CORE/'policy_regressions.csv').query('primary').iloc[0]
    current=influence.query('pair=="DOGE_ETH" and variant=="all"').iloc[0]
    np.testing.assert_allclose([current.estimate,current.low95,current.high95],[core.estimate,core.low95,core.high95],atol=1e-7)
    assert influence.variant.str.startswith('leave_day').sum()==56
    checks['core_policy_result_and_block_CI_exactly_reproduced']=True
    rows=pd.read_csv(HERE/'prediction_evaluation_rows.csv.gz')
    orig=pd.to_datetime(rows.origin,utc=True);future=pd.to_datetime(rows.target_time,utc=True)
    assert ((future-orig)==pd.Timedelta(hours=6)).all()
    idx=pd.to_datetime(panel['dates'],utc=True);ix=idx.get_indexer(orig);assert (ix>=0).all()
    cuts=pd.read_csv(HERE/'thresholds.csv').query('spec=="q10"').set_index('reference').cut
    for j,ref in enumerate(REFS):
        keep=rows.reference.eq(ref).to_numpy()
        if not keep.any():continue
        expected=(panel['any_single_low'][ix[keep],j]<cuts[ref])&(panel['any_single_high'][ix[keep],j]>=cuts[ref])
        assert np.array_equal(expected,rows.current_fragile.to_numpy()[keep])
    checks['prediction_states_use_only_current_prices_and_past_thresholds']=True
    e=pd.read_csv(HERE/'prediction_difference_envelopes.csv');g=pd.read_csv(HERE/'prediction_scenario_gaps.csv')
    for ref in ['mean5','median5','BTC']:
        for metric in ['q10','tails']:
            limits=e.query('reference==@ref and metric==@metric').set_index('bound').mean_bound
            values=g.query('reference==@ref and metric==@metric').tree_minus_linear
            assert values.min()>=limits['lower']-1e-8 and values.max()<=limits['upper']+1e-8
            if ref=='BTC':np.testing.assert_allclose(limits['lower'],limits['upper'],atol=1e-9)
    checks['common_target_loss_bounds_and_DOGE_independent_BTC_invariance']=True
    common=np.load(HERE/'policy_common_controls.npz')
    assert common['x'].shape==(56,16)
    np.testing.assert_allclose(ols(common['x'],common['y']),common['coefficients'],atol=1e-9)
    for j in [1,2]:
        np.testing.assert_allclose(ols(common['x'],common['y'][:,0]-common['y'][:,j]),common['coefficients'][:,0]-common['coefficients'][:,j],atol=1e-8)
    assert np.isfinite(common['bootstrap_post']).all(axis=1).mean()>=.95
    smeta=json.loads((HERE/'SUPPLEMENT_COMPLETION.json').read_text())
    assert smeta['code_sha256']==sha(HERE/'supplement.py') and smeta['scope_sha256']==sha(HERE/'SUPPLEMENT_SCOPE_KO.md')
    multi=pd.read_csv(HERE/'prediction_secondary_multiplicity.csv');assert len(multi)==24
    np.testing.assert_allclose(multi.p_holm_24_secondary,multipletests(multi.p_centered,method='holm')[1])
    checks['common_policy_controls_and_secondary_multiplicity']=True
    meta=json.loads((HERE/'MEASUREMENT_COMPLETION.json').read_text())
    for p,v in meta['inputs_sha256'].items():assert sha(ROOT/p)==v
    for filename,source in [('MEASUREMENT_COMPLETION.json','measurement.py'),('POLICY_COMPLETION.json','policy_robustness.py'),('PREDICTION_COMPLETION.json','prediction_evaluation.py')]:
        m=json.loads((HERE/filename).read_text());assert m['code_sha256']==sha(HERE/source)
        assert m['protocol_sha256']==sha(HERE/'PROTOCOL_KO.md')
    pred=json.loads((HERE/'PREDICTION_COMPLETION.json').read_text())
    for path,v in pred['input_forecasts_sha256'].items():assert sha(ROOT/path)==v
    checks['unchanged_historical_files']=check_preserved()
    checks['all_data_code_protocol_and_frozen_prediction_hashes_verified']=True
    (HERE/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
