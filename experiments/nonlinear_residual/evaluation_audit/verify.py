"""Independent arithmetic, chronology, replay and bootstrap artifact audit."""
from common import *
from run_inference import BOOT, B, UNIQUE, CORE, GRID, ridge_refit
from verify_replay import replay


def main():
    metrics=json.loads((OUT/'METRICS_MANIFEST.json').read_text())
    for name,digest in metrics['input_sha256'].items():assert study.pilot.sha(HERE/name)==digest,name
    for name,digest in metrics['output_sha256'].items():assert study.pilot.sha(OUT/name)==digest,name
    source=json.loads((SOURCE/'RUN_MANIFEST.json').read_text())
    for name,digest in source['output_sha256'].items():assert study.pilot.sha(SOURCE/name)==digest,name
    verified=dict(equal_target_groups=0,chronological_fits=0,forecast_replays=0,quantile_replays=0,
                  original_model_replays=0,max_forecast_delta=0.,max_model_delta=0.)
    for scenario in study.pilot.SCENARIOS:
        panel,pred,pairs=load(scenario)
        p=pd.read_csv(OUT/scenario/'forecast_predictions.csv.gz',parse_dates=['time','target_time'])
        assert ((p.target_time-p.time).dt.total_seconds()==6*3600).all()
        observed=p.loc[p.task=='observed_change']
        assert observed.groupby('time').target.nunique().eq(1).all()
        np.testing.assert_allclose(observed.target,panel.y.reindex(observed.target_time).to_numpy()-panel.y.reindex(observed.time).to_numpy(),atol=1e-9)
        verified['equal_target_groups']+=int(observed.time.nunique())
        fits=json.loads((OUT/scenario/'forecast_fits.json').read_text())
        for f in fits:
            assert pd.Timestamp(f['train_target_last'])<pd.Timestamp(f['cutoff'])
        verified['chronological_fits']+=len(fits)
        np.testing.assert_allclose(p.loss,quantile_loss(p.target,p.prediction,p.q),atol=1e-10)
        np.testing.assert_allclose(p.baseline_loss,quantile_loss(p.target,p.baseline,p.q),atol=1e-10)
        d=pairs.loc[pairs.h==6].set_index('time').join(panel[['y']+study.KCOLS]).reset_index()
        d['downside_x_ls']=d.downside*d.account_ls
        d=d.dropna(subset=['y']+study.KCOLS+['g','downside','btc_vol','btc_ret','account_ls','downside_x_ls','target','y_change'])
        for f in fits:
            if f['month']!='2026-03' or f['method'] not in ['baseline','ae_selected'] or f['info']!='with_positioning':continue
            a=d.loc[d.method==('sequential_ols' if f['method']=='baseline' else f['method'])].copy()
            start=pd.Timestamp(f['cutoff']);end=start+pd.offsets.MonthBegin(1)
            train=a.loc[(a.time<start)&(a.target_time<start)]
            test=a.loc[(a.time>=start)&(a.time<end)]
            col='y_change' if f['task']=='observed_change' else 'target'
            m=estimator(f['algorithm'],f['q']).fit(train[f['columns']].to_numpy(),train[col].to_numpy())
            expected=m.predict(test[f['columns']].to_numpy())
            saved=p.loc[(p.task==f['task'])&(p.method==f['method'])&(p['info']==f['info'])&
                        (p.algorithm==f['algorithm'])&(p.q==f['q'])&(p.month==f['month'])]
            np.testing.assert_allclose(expected,saved.prediction,atol=1e-8)
            verified['max_forecast_delta']=max(verified['max_forecast_delta'],float(np.max(np.abs(expected-saved.prediction))))
            verified['forecast_replays']+=1
        probes=pd.read_csv(OUT/scenario/'probe_predictions.csv.gz',parse_dates=['time'])
        assert (pd.to_datetime(probes.train_last,utc=True)<pd.to_datetime(probes.cutoff,utc=True)).all()
        np.testing.assert_allclose(probes.loss,(probes.target-probes.prediction)**2,atol=1e-7)
        for method in METHODS:
            a=probes.loc[probes.method==method]; e=pred.loc[pred.model==method].set_index('time').residual_bp
            np.testing.assert_allclose(a.target,e.reindex(a.time),atol=1e-9)
        if scenario in UNIQUE:
            points=pd.read_csv(BOOT/scenario/'point_estimates.csv')
            for method in METHODS:
                f=pairs.loc[(pairs.method==method)&(pairs.h==6)&pairs.month.isin(MONTHS)].dropna(subset=CORE+['target'])
                x=f[CORE].to_numpy();scale=StandardScaler().fit(x);z=scale.transform(x)
                m=QuantileRegressor(quantile=.1,alpha=0.,solver='highs').fit(z,f.target)
                saved=points.loc[(points.method==method)&(points['sample']=='all')&(points.h==6)&(points.q==.1)].set_index('term')
                np.testing.assert_allclose(m.coef_/scale.scale_,saved.loc[CORE,'coefficient'],rtol=2e-5,atol=2e-5)
                verified['quantile_replays']+=1
        if scenario=='clock_UTC_quote':
            data=panel.dropna(subset=['y','m','g']+study.KCOLS);cutoff=pd.Timestamp('2026-01-01',tz='UTC')
            test=data.loc[(data.index>=cutoff)&(data.index<cutoff+pd.offsets.MonthBegin(1))]
            choices=json.loads((SOURCE/scenario/'selection.json').read_text())
            for method in PRIMARY:
                cid=next(c['candidate_id'] for c in choices if c['month']=='2026-01' and c['method']==method)
                fitted=ridge_refit(models.Model(GRID[cid]).fit(data,cutoff))
                state=json.loads((SOURCE/scenario/'checkpoints'/('2026-01_'+cid+'.json')).read_text())
                expected=replay(state,test);actual=fitted.predict(test)
                np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-10)
                verified['max_model_delta']=max(verified['max_model_delta'],float(np.max(np.abs(actual-expected))))
                verified['original_model_replays']+=1
    if (BOOT/'MANIFEST.json').exists():
        manifest=json.loads((BOOT/'MANIFEST.json').read_text())
        for name,digest in manifest['input_sha256'].items():assert study.pilot.sha(HERE/name)==digest,name
        for name,digest in manifest['output_sha256'].items():assert study.pilot.sha(BOOT/name)==digest,name
        assert json.loads((BOOT/'failures.json').read_text())==[]
        logs=[json.loads(p.read_text()) for s in UNIQUE for p in (BOOT/s/'refit_7d').glob('*.json')]
        assert len(logs)==len(UNIQUE)*B
        verified['bootstrap_refit_draws']=len(logs)
        verified['ae_month_refits']=len(logs)*3
        verified['ae_seed_refits']=len(logs)*9
        verified['ae_month_refits_with_warning']=sum(bool(f['warnings']) for l in logs for f in l['fits'])
        verified['ae_seed_refits_with_warning']=sum(bool(m['warnings']) for l in logs for f in l['fits'] for m in f['members'])
        for l in logs:
            for f in l['fits']:
                assert pd.Timestamp(f['train_last'])<pd.Timestamp(f['month']+'-01',tz='UTC')
        ci=pd.read_csv(BOOT/'coefficient_intervals.csv')
        assert ci.successful.eq(B).all()
        for scenario in UNIQUE:
            draws=pd.concat([pd.read_csv(p) for p in (BOOT/scenario/'refit_7d').glob('*.csv')])
            for method in METHODS:
                f=draws.loc[(draws.method==method)&(draws['sample']=='all')&(draws.h==6)&(draws.q==.1)&(draws.term=='downside')]
                row=ci.loc[(ci.scenario==scenario)&(ci.method==method)&(ci['sample']=='all')&(ci.h==6)&(ci.q==.1)&(ci.term=='downside')&ci.refit_first_stage].iloc[0]
                np.testing.assert_allclose(np.quantile(f.effect,[.025,.975]),[row.ci_low,row.ci_high],atol=1e-10)
    else:
        verified['bootstrap_complete']=False
    verified['archive_check']=study.pilot.audit_archive()
    (HERE/'VERIFICATION.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verified.items() if k!='archive_check'},indent=2))


if __name__=='__main__':main()
