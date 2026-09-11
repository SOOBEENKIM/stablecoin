from shared import *
from statsmodels.stats.multitest import multipletests


def main():
    check_preserved();d,k,p,u,b,paths=load_prices();allbasis=aggregate(b)
    pre=allbasis[d.index<TEST];mask=d.index>=TEST
    dates=d.index[mask];k=k[mask];p=p[mask];u=u[mask];b=b[mask];base=aggregate(b)
    assert np.allclose(tick_at(k),np.broadcast_to(TICKS,k.shape))
    assert np.allclose(tick_at(u),1.)
    rs=Resampler(dates);np.savez_compressed(HERE/'measurement_weights.npz',weights=rs.w.astype(np.int16),days=rs.days.asi8)
    single=[];equal=[]
    for i,coin in enumerate(COINS):
        pair=[];eqpair=[]
        for sign in [-1,1]:
            kk=k.copy();kk[:,i]=price_step(kk[:,i],sign)
            bb=b.copy();bb[:,i]-=1e4*np.log(kk[:,i]/k[:,i]);v=aggregate(bb)
            np.testing.assert_allclose(v,aggregate(1e4*np.log(u[:,None]*p/kk)),atol=1e-7)
            pair.append(v)
            z=b.copy();z[:,i]-=sign;eqpair.append(aggregate(z))
        single.append(np.stack(pair));equal.append(np.stack(eqpair))
    single=np.stack(single);equal=np.stack(equal)
    bounds={}
    for i,c in enumerate(COINS):bounds[c]=(single[i].min(axis=0),single[i].max(axis=0))
    bounds['any_single']=(np.minimum(single.min(axis=(0,1)),base),np.maximum(single.max(axis=(0,1)),base))
    for half in [.5,1.]:
        lo=aggregate(b-1e4*np.log1p(half*TICKS/k));hi=aggregate(b-1e4*np.log1p(-half*TICKS/k))
        bounds['joint_'+str(half)]=(lo,hi)
    bounds['USDT']=(base+1e4*np.log1p(-1/u[:,None]),base+1e4*np.log1p(1/u[:,None]))
    bounds['equal_1bp_any']=(equal.min(axis=(0,1)),equal.max(axis=(0,1)))
    specs=[('minus10','lower',np.repeat(-10.,5)),('minus5','lower',np.repeat(-5.,5)),('minus20','lower',np.repeat(-20.,5)),
           ('plus10','upper',np.repeat(10.,5))]
    for q in [.05,.1,.9,.95]:specs.append(('q'+str(round(q*100)).zfill(2),'lower' if q<.5 else 'upper',np.quantile(pre,q,axis=0)))
    threshold=pd.DataFrame([dict(spec=n,side=s,reference=r,cut=c[i]) for n,s,c in specs for i,r in enumerate(REFS)])
    threshold.to_csv(HERE/'thresholds.csv',index=False)
    outliers=pd.Series(abs(base[:,0]),index=dates).groupby(dates.normalize()).max().sort_values(ascending=False,kind='stable')
    (HERE/'influence_dates.json').write_text(json.dumps({str(n):[str(v) for v in outliers.index[:n]] for n in [1,5]},indent=2))
    periods={'all':np.ones(len(dates),dtype=bool)}
    for month in sorted(set(dates.strftime('%Y-%m'))):periods[month]=dates.strftime('%Y-%m')==month
    for n in [1,5]:periods['drop_top'+str(n)+'_days']=~dates.normalize().isin(outliers.index[:n])
    records=[];primaries={};risksets={};minrows=[];minarrays={};overlap=[]
    for spec,side,cut in specs:
        risky=base<cut if side=='lower' else base>cut;risksets[spec]=risky
        for j,ref in enumerate(REFS):
            for mode,(low,high) in bounds.items():
                fragile=risky[:,j]&(high[:,j]>=cut[j]) if side=='lower' else risky[:,j]&(low[:,j]<=cut[j])
                enter=(~risky[:,j])&(low[:,j]<cut[j]) if side=='lower' else (~risky[:,j])&(high[:,j]>cut[j])
                flip=fragile|enter
                if spec=='minus10' and mode=='any_single':primaries[ref]=(flip,rs.ratio(flip,np.ones(len(flip))))
                for period,keep in periods.items():
                    n=int(keep.sum());nr=int((risky[:,j]&keep).sum());nf=int((fragile&keep).sum());ne=int((enter&keep).sum())
                    # Common calendar blocks quantify temporal sample variation only.
                    ci=interval((nf+ne)/n,rs.ratio(flip&keep,keep))
                    ci_tail=interval(nf/nr,rs.ratio(fragile&keep,risky[:,j]&keep)) if nr else dict(low95=np.nan,high95=np.nan)
                    records.append(dict(spec=spec,side=side,reference=ref,mode=mode,period=period,n=n,cut=cut[j],risk_count=nr,
                        fragile_risk=nf,stable_risk=nr-nf,possible_new_risk=ne,flippable_all_share=(nf+ne)/n,
                        fragile_risk_share=nf/nr if nr else np.nan,flip_low95=ci['low95'],flip_high95=ci['high95'],
                        fragile_low95=ci_tail['low95'],fragile_high95=ci_tail['high95']))
        # Pairwise classifications, before perturbation, on identical observations.
        for j in range(5):
            for l in range(j+1,5):
                x,y=risky[:,j],risky[:,l];union=(x|y).sum()
                overlap.append(dict(spec=spec,reference_a=REFS[j],reference_b=REFS[l],n=len(dates),risk_a=x.sum(),risk_b=y.sum(),
                    intersection=(x&y).sum(),disagreement=(x!=y).sum(),jaccard=(x&y).sum()/union if union else np.nan))
        # Exact integer grid steps. Changes use the price band reached at each step.
        for i,coin in enumerate(COINS):
            moved=k[:,i].copy();step_count=np.full(base.shape,101,dtype=np.int16)
            sign=-1 if side=='lower' else 1
            for nsteps in range(1,101):
                moved=price_step(moved,sign)
                assert (moved>0).all()
                bb=b.copy();bb[:,i]-=1e4*np.log(moved/k[:,i]);new=aggregate(bb)
                escaped=new>=cut if side=='lower' else new<=cut
                hit=risky&escaped&(step_count==101);step_count[hit]=nsteps
            step_count[~risky]=0;minarrays[spec+'_'+coin]=step_count
            for j,ref in enumerate(REFS):
                v=step_count[risky[:,j],j]
                minrows.append(dict(spec=spec,coin=coin,reference=ref,risk_count=len(v),within1=int((v==1).sum()),within2=int(((v>=1)&(v<=2)).sum()),
                    within5=int(((v>=1)&(v<=5)).sum()),within10=int(((v>=1)&(v<=10)).sum()),over100_or_no_effect=int((v==101).sum()),
                    median_capped_steps=float(np.median(v)) if len(v) else np.nan))
        print('Measurement complete',spec,flush=True)
    contrasts=[];draws={}
    for ref in ['BTC','ETH','median5','without_DOGE']:
        est=primaries['mean5'][0].mean()-primaries[ref][0].mean();boot=primaries['mean5'][1]-primaries[ref][1]
        contrasts.append(dict(contrast='mean5_minus_'+ref,estimate=est,**interval(est,boot)));draws[ref]=boot
    contrasts=pd.DataFrame(contrasts);contrasts['p_holm_4']=multipletests(contrasts.p_centered,method='holm')[1]
    contrasts.to_csv(HERE/'primary_measurement_contrasts.csv',index=False)
    pd.DataFrame(records).to_csv(HERE/'measurement_sensitivity.csv',index=False)
    pd.DataFrame(minrows).to_csv(HERE/'minimum_ticks.csv',index=False)
    pd.DataFrame(overlap).to_csv(HERE/'reference_overlap.csv',index=False)
    np.savez_compressed(HERE/'minimum_tick_arrays.npz',**minarrays)
    np.savez_compressed(HERE/'measurement_primary_bootstrap.npz',**draws)
    np.savez_compressed(HERE/'measurement_panel.npz',dates=dates.asi8,b=base,individual=b,k=k,p=p,u=u,single=single,equal=equal,
                        **{mode+'_low':v[0] for mode,v in bounds.items()},**{mode+'_high':v[1] for mode,v in bounds.items()})
    meta=dict(status='complete',protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),code_sha256=sha(Path(__file__)),shared_sha256=sha(HERE/'shared.py'),
              n=len(dates),days=len(rs.days),first=str(dates.min()),last=str(dates.max()),n_pre=len(pre),
              inputs_sha256={str(v.relative_to(ROOT)):sha(v) for v in paths},preserved_files=check_preserved(),
              interpretation='Deterministic local-price perturbation sensitivity, not probabilities of mispricing or false alarms')
    (HERE/'MEASUREMENT_COMPLETION.json').write_text(json.dumps(meta,indent=2))
    print(contrasts.round(5).to_string(index=False))
    print(pd.DataFrame(records).query("spec=='minus10' and mode=='any_single' and period=='all'").round(4).to_string(index=False))

if __name__=='__main__':main()
