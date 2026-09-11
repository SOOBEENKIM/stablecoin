from shared import *

MODELS=['update_linear_local_cal28','update_lgbm_local_cal28']
QS=np.array([.1,.5,.9])

def loss(y,p):
    e=np.asarray(y)[:,None]-p
    return np.maximum(QS*e,(QS-1)*e)

def main():
    panel=np.load(HERE/'measurement_panel.npz');dates=pd.to_datetime(panel['dates'],utc=True)
    cuts=pd.read_csv(HERE/'thresholds.csv').query("spec=='q10'").set_index('reference').cut
    frames={};inputs={}
    for ref,file in [('mean5','mean5'),('median5','median5'),('BTC','btc')]:
        path=ROOT/'extension_20260909'/(file+'_h6_forecasts.csv.gz');inputs[str(path.relative_to(ROOT))]=sha(path)
        f=pd.read_csv(path);f.origin=pd.to_datetime(f.origin,utc=True);f.target_time=pd.to_datetime(f.target_time,utc=True)
        f=f.loc[(f.origin>=TEST)&f.model.isin(MODELS)]
        for m in MODELS:
            x=f.loc[f.model==m].set_index('origin').sort_index();assert x.index.is_unique
            frames[(ref,m)]=x
    ix=frames[('mean5',MODELS[0])].index
    for f in frames.values():ix=ix.intersection(f.index)
    ix=ix.sort_values();rs=Resampler(ix,SEED+10);np.savez_compressed(HERE/'prediction_weights.npz',weights=rs.w.astype(np.int16),days=rs.days.asi8)
    origin_i=dates.get_indexer(ix);future_i=dates.get_indexer(ix+pd.Timedelta(hours=6));assert (origin_i>=0).all() and (future_i>=0).all()
    scores=[];gaps=[];conditional=[];envelopes=[];allrows=[];draws={};checks=[]
    for ref in ['mean5','median5','BTC']:
        j=REFS.index(ref);fs=[frames[(ref,m)].reindex(ix) for m in MODELS]
        for f in fs:
            assert ((f.target_time-f.index)==pd.Timedelta(hours=6)).all()
            np.testing.assert_allclose(f.target,panel['b'][future_i,j],atol=1e-7)
            np.testing.assert_allclose(f.basis,panel['b'][origin_i,j],atol=1e-7)
        ps=[f[['q10','q50','q90']].to_numpy() for f in fs]
        scenarios={'observed':panel['b'][future_i,j],
                   'DOGE_price_minus_one_tick':panel['single'][4,0,future_i,j],
                   'DOGE_price_plus_one_tick':panel['single'][4,1,future_i,j]}
        current_fragile=(panel['any_single_low'][origin_i,j]<cuts[ref])&(panel['any_single_high'][origin_i,j]>=cuts[ref])
        for scenario,y in scenarios.items():
            losses=[loss(y,p) for p in ps]
            for model,p,l in zip(MODELS,ps,losses):
                scores.append(dict(reference=ref,scenario=scenario,model=model,n=len(ix),q10_loss=l[:,0].mean(),tail_loss=l[:,[0,2]].mean(),
                    coverage80=((y>=p[:,0])&(y<=p[:,2])).mean(),width80=(p[:,2]-p[:,0]).mean()))
            for metric,col in [('q10',0),('tails',None)]:
                diff=losses[1][:,0]-losses[0][:,0] if col==0 else losses[1][:,[0,2]].mean(axis=1)-losses[0][:,[0,2]].mean(axis=1)
                b=rs.ratio(diff,np.ones(len(ix)));key=ref+'_'+scenario+'_'+metric;draws[key]=b
                gaps.append(dict(reference=ref,scenario=scenario,metric=metric,n=len(ix),tree_minus_linear=diff.mean(),**interval(diff.mean(),b)))
                if scenario=='observed':
                    groupboot={};groupest={}
                    for group,keep in [('current_fragile',current_fragile),('current_stable',~current_fragile)]:
                        n=int(keep.sum());nd=int(ix[keep].normalize().nunique())
                        if n:
                            vv=rs.ratio(diff*keep,keep);est=diff[keep].mean();groupboot[group]=vv;groupest[group]=est
                            row=dict(reference=ref,group=group,metric=metric,n=n,days=nd,adequate=(n>=100 and nd>=20),tree_minus_linear=est,**interval(est,vv))
                            for short,pred,ll in zip(['linear','tree'],ps,losses):
                                row[short+'_q10_loss']=ll[keep,0].mean();row[short+'_tail_loss']=ll[keep][:,[0,2]].mean()
                                row[short+'_coverage80']=((y[keep]>=pred[keep,0])&(y[keep]<=pred[keep,2])).mean()
                            conditional.append(row)
                    if len(groupboot)==2:
                        est=groupest['current_fragile']-groupest['current_stable'];v=groupboot['current_fragile']-groupboot['current_stable']
                        conditional.append(dict(reference=ref,group='fragile_minus_stable',metric=metric,n=len(ix),days=ix.normalize().nunique(),tree_minus_linear=est,**interval(est,v)))
            for row,t in enumerate(ix):
                allrows.append(dict(reference=ref,scenario=scenario,origin=str(t),target_time=str(t+pd.Timedelta(hours=6)),target=y[row],
                    current_fragile=bool(current_fragile[row]),linear_q10=ps[0][row,0],tree_q10=ps[1][row,0],
                    linear_q90=ps[0][row,2],tree_q90=ps[1][row,2]))
        # Exact difference bounds: common y for both models. All breakpoints.
        low=np.minimum(scenarios['DOGE_price_minus_one_tick'],scenarios['DOGE_price_plus_one_tick'])
        high=np.maximum(scenarios['DOGE_price_minus_one_tick'],scenarios['DOGE_price_plus_one_tick'])
        for metric in ['q10','tails']:
            levels=[0] if metric=='q10' else [0,2]
            points=[low,high]+[np.clip(pred[:,col],low,high) for pred in ps for col in levels]
            values=[]
            for y in points:
                dl=loss(y,ps[1])-loss(y,ps[0]);values.append(dl[:,levels].mean(axis=1))
            values=np.stack(values);lo=values.min(axis=0);hi=values.max(axis=0)
            for label,val in [('lower',lo),('upper',hi)]:
                bb=rs.ratio(val,np.ones(len(ix)));draws[ref+'_envelope_'+metric+'_'+label]=bb
                envelopes.append(dict(reference=ref,metric=metric,bound=label,mean_bound=val.mean(),**interval(val.mean(),bb)))
            # Dense independent grid never falls outside exact analytic envelope.
            for fraction in np.linspace(0,1,101):
                dl=loss(low+(high-low)*fraction,ps[1])-loss(low+(high-low)*fraction,ps[0]);vv=dl[:,levels].mean(axis=1)
                assert (vv>=lo-1e-9).all() and (vv<=hi+1e-9).all()
        frozen=pd.read_csv(ROOT/'extension_20260909'/(('btc' if ref=='BTC' else ref)+'_h6_scores.csv'))
        for m in MODELS:
            fresh=[s for s in scores if s['reference']==ref and s['scenario']=='observed' and s['model']==m][0]
            old=frozen.loc[frozen.model==m].iloc[0]
            if fresh['n']==old.n:
                np.testing.assert_allclose([fresh[k] for k in ['q10_loss','tail_loss','coverage80']],old[['q10_loss','tail_loss','coverage80']].to_numpy(dtype=float),atol=1e-8)
                checks.append(ref+' '+m+' exact frozen scores')
    pd.DataFrame(scores).to_csv(HERE/'prediction_scenario_scores.csv',index=False)
    pd.DataFrame(gaps).to_csv(HERE/'prediction_scenario_gaps.csv',index=False)
    pd.DataFrame(conditional).to_csv(HERE/'prediction_current_states.csv',index=False)
    pd.DataFrame(envelopes).to_csv(HERE/'prediction_difference_envelopes.csv',index=False)
    pd.DataFrame(allrows).to_csv(HERE/'prediction_evaluation_rows.csv.gz',index=False,compression='gzip')
    np.savez_compressed(HERE/'prediction_bootstrap.npz',**draws)
    (HERE/'PREDICTION_COMPLETION.json').write_text(json.dumps(dict(status='complete',code_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),
        n_common=len(ix),days=len(rs.days),input_forecasts_sha256=inputs,checks=checks,
        preserved_files=check_preserved(),scope='Frozen predictions; target-evaluation stress scenarios, not newly fitted predictive improvements'),indent=2))
    print(pd.DataFrame(gaps).round(5).to_string(index=False),flush=True)
    print(pd.DataFrame(conditional).round(5).to_string(index=False),flush=True)
    print(pd.DataFrame(envelopes).round(5).to_string(index=False),flush=True)

if __name__=='__main__':main()
