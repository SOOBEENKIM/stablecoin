"""Independent integer clocks, scalar neighborhoods, sorted order statistics."""
import math
from cc_core import *


def independent(ctx,raw):
    times=list(ctx.origin);labels=list(ctx.target_time)
    sigma=np.array([max(float(x),1.) for x in ctx.e_rms72])
    errors=ctx.target.to_numpy()[:,None]-raw
    out=np.zeros((len(ctx),raw.shape[1],7))
    coords=list(zip(ctx.e_rank,ctx.vol_rank))
    min_history=10000
    for i,t in enumerate(times):
        pool=[j for j,label in enumerate(labels) if t-pd.Timedelta(days=90)<=label<t][-180:]
        if t>=pd.Timestamp('2025-09-01',tz='UTC'):min_history=min(min_history,len(pool))
        if len(pool)<30:continue
        a,b=coords[i]
        nearest=sorted(pool,key=lambda j:((coords[j][0]-a)**2+(coords[j][1]-b)**2,-j))[:60]
        def q(ids,scale=False):
            v=errors[ids]/sigma[ids,None] if scale else errors[ids]
            ans=np.sort(v,axis=0)[math.ceil(len(ids)/10)-1]
            return ans*sigma[i] if scale else ans
        short,globalq,scaled,local,local_scaled=q(pool[-60:]),q(pool),q(pool,True),q(nearest),q(nearest,True)
        out[i]=np.column_stack([short,globalq,scaled,local,local_scaled,(globalq+local)/2,(scaled+local_scaled)/2])
    assert min_history>=30
    return raw[:,:,None]+out


def verify_choices(stream,choices):
    for fold,chosen in choices.groupby('fold'):
        cutoff=pd.Timestamp(fold+'-01',tz='UTC');begin=cutoff-pd.DateOffset(months=3)
        scores=[]
        for cid,g in stream.groupby('candidate'):
            z=g[(g.origin>=begin)&(g.origin<cutoff)&(g.target_time<cutoff)&
                (g.origin.dt.strftime('%Y-%m')==g.target_time.dt.strftime('%Y-%m'))].sort_values('origin')
            assert set(z.fold)==set(pd.date_range(begin,cutoff-pd.offsets.MonthBegin(1),freq='MS').strftime('%Y-%m'))
            for rank,rule in enumerate(RULES):
                error=z.target.to_numpy()-z['q_'+rule].to_numpy()
                score=float(np.where(error>=0,error*.1,error*(-.9)).mean())
                scores.append((score,int(cid),rank,g.kind.iloc[0]))
        for c in chosen.itertuples():
            kinds=['qrf','boosting'] if c.family=='ml_selected' else [c.family]
            pool=[x for x in scores if x[3] in kinds]
            legacy=min(x for x in pool if x[2]==0)
            if c.policy=='legacy':pool=[legacy]
            elif c.policy=='fixed_model_selected_cal':pool=[x for x in pool if x[1]==legacy[1]]
            elif c.policy=='global_window_selected':pool=[x for x in pool if x[2]<2]
            expected=min(pool)
            assert (c.candidate,RULES.index(c.rule))==expected[1:3],(c,expected)
            assert abs(c.score-expected[0])<1e-12


def main():
    stamp=verify_seal()
    ctxall=read(PREP/'context.csv.gz');choices=read(OUT/'choices.csv')
    panel=old.legacy.a.panel('available_macro');rank_error=0.;cal_error=0.;streams_checked=0
    for definition in DEFS:
        ctx=context_for(definition)
        for cutoff in old.MONTHS:
            tr,te,r=r12.case(panel,definition,cutoff)
            g=ctx[ctx.fold==cutoff.strftime('%Y-%m')]
            for col,rank in [('e_now','e_rank'),('e_rms72','vol_rank')]:
                values=tr[col].to_numpy()
                expected=np.array([np.count_nonzero(values<=v)/len(values) for v in g[col]])
                rank_error=max(rank_error,float(np.max(abs(g[rank]-expected))))
            assert tr.target_time.max()<cutoff
            np.testing.assert_array_equal(g.state,dv.assign_state(g,dv.fit_state(tr,cutoff)))
        for seed in SEEDS:
            stream=read(OUT/'streams'/f'{definition}__{seed}.csv.gz')
            ids=sorted(stream.candidate.unique())
            for cid,g in stream.groupby('candidate'):
                assert list(g.sort_values('origin').origin)==list(ctx.origin)
            raw=np.column_stack([stream[stream.candidate==cid].sort_values('origin').pred_raw for cid in ids])
            expected=independent(ctx,raw)
            for j,cid in enumerate(ids):
                g=stream[stream.candidate==cid].sort_values('origin')
                err=np.max(np.abs(g[['q_'+r for r in RULES]].to_numpy()-expected[:,j,:]))
                cal_error=max(cal_error,float(err));streams_checked+=1
            if seed==SEEDS[0]:verify_choices(stream,choices[choices.definition==definition])
            print('VERIFIED clocks/calibration',definition,seed,flush=True)
    assert rank_error<1e-12 and cal_error<1e-11
    pred=read(OUT/'predictions.csv.gz');fixed=read(OUT/'fixed_rule_predictions.csv.gz')
    for (definition,policy,family,seed),g in pred.groupby(['definition','policy','family','seed']):
        g=g.sort_values('origin');ctx=ctxall[(ctxall.definition==definition)&(ctxall.origin>=r12.hs.START)].sort_values('origin')
        assert len(g)==509 and list(g.origin)==list(ctx.origin)
        np.testing.assert_allclose(g[['e_now','target']],ctx[['e_now','target']],atol=1e-11,rtol=0)
        for fold,s in g.groupby('fold'):
            choice=choices[(choices.definition==definition)&(choices.policy==policy)&(choices.family==family)&(choices.fold==fold)].iloc[0]
            assert set(s.candidate)=={choice.candidate} and set(s.rule)=={choice.rule}
            stream=read(OUT/'streams'/f'{definition}__{seed}.csv.gz')
            source=stream[(stream.candidate==choice.candidate)&(stream.fold==fold)].sort_values('origin')
            np.testing.assert_array_equal(s.q,source['q_'+choice.rule])
    assert len(pred)==3*509*4*8
    assert len(fixed)==3*509*7*8
    # Confirm legacy public forecasts, including extra-seed e_now reconstruction.
    archived=read(PARENT/'results/predictions.csv.gz')
    for (definition,family,seed),g in pred[pred.policy=='legacy'].groupby(['definition','family','seed']):
        if family in ['historical','state_hist']:
            source=read(dv.OUT/'baseline_streams.csv.gz')
            source=source[(source.definition==definition)&(source.strategy==family)&(source.origin>=r12.hs.START)]
        else:
            source=archived[(archived.definition==definition)&(archived.variant=='original')&(archived.information=='F')&
                (archived.strategy==family)&(archived.seed==seed)]
        np.testing.assert_allclose(g.sort_values('origin').q,source.sort_values('origin').pred_calibrated,atol=1e-11,rtol=0)
    notes=json.loads((OUT/'new_fit_metadata.json').read_text())
    for n in notes:assert pd.Timestamp(n['last_label'])<pd.Timestamp(n['fold']+'-01',tz='UTC')
    # Refit a bounded first job for each definition/seed that needed a new candidate.
    checked=set();refit_error=0.
    for n in notes:
        key=n['definition'],n['seed']
        if key in checked:continue
        checked.add(key);cutoff=pd.Timestamp(n['fold']+'-01',tz='UTC')
        tr,te,r=r12.case(panel,n['definition'],cutoff);spec=old.SPECS[n['candidate']]
        fit=old.legacy.a.window(tr,cutoff,spec['window'])
        _,q=old.fit_predict(spec,fit,te,r12.hs.columns('F','original'),n['seed'])
        s=read(OUT/'streams'/f'{n["definition"]}__{n["seed"]}.csv.gz')
        s=s[(s.candidate==n['candidate'])&(s.fold==n['fold'])].sort_values('origin')
        refit_error=max(refit_error,float(np.max(abs(q-s.pred_raw))))
    assert refit_error<1e-11
    write_new(OUT/'VERIFICATION.json',dict(**stamp,verified_utc=now(),streams_checked=streams_checked,
        calibration_max_error=cal_error,rank_max_error=rank_error,refit_checks=len(checked),refit_max_error=refit_error,
        legacy_predictions_reproduced=True,independent_selection_verified=True,predictions_sha256=sha(OUT/'predictions.csv.gz')))
    print('VERIFICATION COMPLETE; new performance may now be opened.',flush=True)


if __name__=='__main__':main()
