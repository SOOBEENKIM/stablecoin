"""Independent price path and partialling-out inputs verification."""
from ac_core import *


def main():
    stamp=verify_seal();ctx=read(PREP/'context.csv.gz');pred=read(OUT/'predictions.csv.gz')
    panel=old.legacy.a.panel('available_macro');raw=raw_market()
    max_error=0.;identity_error=0.;price_checks=0
    for note in json.loads((PREP/'metadata.json').read_text()):
        definition,fold,h=note['definition'],note['fold'],note['h']
        r=note['residualizer'];b=r['beta_market'];bg=r['beta_g']
        cutoff=pd.Timestamp(fold+'-01',tz='UTC')
        d=ctx[(ctx.definition==definition)&(ctx.fold==fold)&(ctx.h==h)]
        assert pd.Timestamp(note['last_train_label'])<cutoff and pd.Timestamp(r['fit_end'])<cutoff
        for row in d.itertuples():
            t,u=row.origin,row.target_time
            assert u==t+pd.Timedelta(hours=h)
            p0,p1=panel.loc[t],panel.loc[u]
            r0=np.array([raw.loc[t,c+'_UPBIT_CLOSE']/raw.loc[t,c+'_BINANCE_CLOSE'] for c in source.COINS])
            r1=np.array([raw.loc[u,c+'_UPBIT_CLOSE']/raw.loc[u,c+'_BINANCE_CLOSE'] for c in source.COINS])
            if definition=='EQ':a0=a1=np.array([.2]*5)
            elif definition=='CAP':
                a0=np.array([raw.loc[t,c+'_BINANCE_CLOSE']*source.SUPPLY[j] for j,c in enumerate(source.COINS)]);a0=a0/sum(a0)
                a1=np.array([raw.loc[u,c+'_BINANCE_CLOSE']*source.SUPPLY[j] for j,c in enumerate(source.COINS)]);a1=a1/sum(a1)
            else:a0=a1=np.asarray(r['pca_loading'])/np.asarray(r['pca_std'])
            z0,z1=p0.q/p0.fx,p1.q/p1.fx
            price=0.;weight=0.
            for j in range(5):
                price+=(a0[j]+a1[j])/2*(r1[j]-r0[j]);weight+=(r0[j]+r1[j])/2*(a1[j]-a0[j])
            h0=p0.local_usdt-b*sum(a0*r0);h1=p1.local_usdt-b*sum(a1*r1)
            expected=np.array([(z0+z1)/2*(p1.local_usdt-p0.local_usdt),-b*(z0+z1)/2*price,
                -b*(z0+z1)/2*weight,(h0+h1)/2*(z1-z0),-bg*(p1.g-p0.g)])*1e4
            actual=np.array([getattr(row,'delta_'+c) for c in PARTS])
            max_error=max(max_error,float(np.max(abs(actual-expected))))
            identity_error=max(identity_error,abs(sum(expected)-row.delta_e));price_checks+=1
            sign=1 if row.e_now>0 else -1
            np.testing.assert_allclose([getattr(row,'close_'+c) for c in PARTS],-sign*expected,atol=1e-8,rtol=0)
    assert max_error<1e-8 and identity_error<1e-7
    for (definition,fold,h,learner,seed),d in pred.groupby(['definition','fold','h','learner','seed']):
        d=d.sort_values('origin');c=ctx[(ctx.definition==definition)&(ctx.fold==fold)&(ctx.h==h)&(ctx.abs_e>0)].sort_values('origin')
        assert list(d.origin)==list(c.origin)
        np.testing.assert_allclose(d[['e_now','delta_e']+['close_'+c for c in PARTS]],c[['e_now','delta_e']+['close_'+c for c in PARTS]],atol=1e-11,rtol=0)
        assert np.isfinite(d[['m']+['g_'+c for c in PARTS]]).all().all()
        np.testing.assert_allclose(d.g_total,d[['g_'+c for c in PARTS]].sum(axis=1),atol=1e-9,rtol=0)
    # Bounded exact refit of each learner at January/12h for each definition.
    refit_error=0.;checks=0
    for definition in DEFS:
        case=pd.read_pickle(PREP/'cases'/f'{definition}__2026-01__12.pkl.gz')
        for learner in LEARNERS:
            d=model_fit(case['train'].query('abs_e>0'),case['test'].query('abs_e>0'),learner,SEEDS[0])
            p=pred[(pred.definition==definition)&(pred.fold=='2026-01')&(pred.h==12)&(pred.learner==learner)&(pred.seed==SEEDS[0])].sort_values('origin')
            cols=['m']+['g_'+c for c in PARTS]
            refit_error=max(refit_error,float(np.max(np.abs(d[cols].to_numpy()-p[cols].to_numpy()))));checks+=1
    assert refit_error<1e-10
    write_new(OUT/'VERIFICATION.json',dict(**stamp,verified_utc=now(),price_rows_checked=price_checks,
        price_component_max_error=max_error,residual_identity_max_error=identity_error,
        refit_checks=checks,refit_max_error=refit_error,predictions_sha256=sha(OUT/'predictions.csv.gz')))
    print('VERIFIED independent prices, clocks, identities and refits.',flush=True)


if __name__=='__main__':main()
