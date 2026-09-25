"""Decision costs, state/error accounting and bounded sensitivity tables."""
import json
import numpy as np
import pandas as pd
from dv_core import *


def load_cells(ctx,archived,baselines,definition,variant,info,version):
    context=ctx[ctx.definition==definition].sort_values('origin').reset_index(drop=True)
    subset=archived[(archived.definition==definition)&(archived.variant==variant)&(archived.information==info)]
    cells={}
    for name in ['ml_selected','linear','threshold','historical','state_hist']:
        g=subset[subset.strategy==name] if name in ['ml_selected','linear','threshold'] else baselines[(baselines.definition==definition)&(baselines.strategy==name)&(baselines.origin>=r12.hs.START)]
        expected=4 if name=='ml_selected' else 1
        seedcells=[]
        for seed,s in g.groupby('seed'):
            s=s.sort_values('origin').reset_index(drop=True)
            assert list(s.origin)==list(context.origin)
            np.testing.assert_allclose(s.target,context.target,atol=1e-11,rtol=0)
            d=context[['origin','target_time','fold','state','e_now','e_rms72','target','delta_e']].copy()
            d['seed']=seed;d['q']=s['pred_'+version];d['q_change']=d.q-d.e_now
            d['loss']=pinball(d.target,d.q)
            d['too_high']=.9*np.maximum(d.q-d.target,0)
            d['too_low']=.1*np.maximum(d.target-d.q,0)
            d['below']=(d.target<d.q).astype(float)
            seedcells.append(d)
        assert len(seedcells)==expected
        cells[name]=pd.concat(seedcells,ignore_index=True)
    for name,q in [('always_alarm',-np.inf),('never_alarm',np.inf)]:
        d=context[['origin','target_time','fold','state','e_now','e_rms72','target','delta_e']].copy()
        d['seed']=SEEDS[0];d['q_change']=q
        cells[name]=d
    return context,cells


def origin_decisions(d,c):
    out=d[['origin','target_time','fold']].copy()
    for k,v in decisions(d.delta_e,d.q_change,c).items():out[k]=v
    out=out.groupby('origin',as_index=False).agg(dict(target_time='first',fold='first',**{k:'mean' for k in decisions([0],[0],c)}))
    return out


def metrics(g):
    event,alarm,tp,fp,fn=[float(g[k].sum()) for k in ['event','alarm','tp','fp','fn']]
    return dict(n=len(g),events=event,alarms=alarm,true_positives=tp,false_positives=fp,false_negatives=fn,
        alarm_rate=alarm/len(g),recall=tp/event if event else np.nan,precision=tp/alarm if alarm else np.nan,
        cost=float(g.cost.mean()),missed_severity_bp=float(g.missed_severity.mean()))


def state_analysis(context,cells,definition,variant,info):
    metrics_rows=[];contrasts=[]
    w=day_weights(context[['origin','fold']],5)
    aggregates={}
    for name in ['ml_selected','linear','threshold','historical','state_hist']:
        d=cells[name].groupby('origin',as_index=False).agg({k:'mean' for k in ['loss','too_high','too_low','below','q_change']})
        assert list(d.origin)==list(context.origin)
        aggregates[name]=d
        for state in range(6):
            mask=context.state.to_numpy()==state;n=int(mask.sum())
            row=dict(definition=definition,variant=variant,information=info,strategy=name,state=state,n=n,
                e_state=['low','middle','high'][state//2],scale_state=['low','high'][state%2])
            for col in ['loss','too_high','too_low','below','q_change']:
                row[col]=float(d.loc[mask,col].mean())
            row['actual_delta_mean']=float(context.loc[mask,'delta_e'].mean())
            row['actual_delta_q10']=float(np.quantile(context.loc[mask,'delta_e'],.1,method='inverted_cdf')) if n else np.nan
            metrics_rows.append(row)
    for ref in REFERENCES:
        refd,new=aggregates[ref],aggregates['ml_selected']
        for state in range(6):
            mask=context.state.to_numpy()==state
            delta=(refd.loss-new.loss).to_numpy()
            lo,hi,valid=weighted_interval(w,delta,mask)
            contrasts.append(dict(definition=definition,variant=variant,information=info,reference=ref,state=state,
                n=int(mask.sum()),share=float(mask.mean()),difference_bp=float(delta[mask].mean()),
                difference_lo=lo,difference_hi=hi,bootstrap_valid_fraction=valid,
                contribution_bp=float(np.sum(delta[mask])/len(context)),
                too_high_contribution_bp=float(np.sum((refd.too_high-new.too_high)[mask])/len(context)),
                too_low_contribution_bp=float(np.sum((refd.too_low-new.too_low)[mask])/len(context))))
        just=contrasts[-6:]
        assert abs(sum(x['contribution_bp'] for x in just)-(refd.loss-new.loss).mean())<1e-12
    return metrics_rows,contrasts


def main():
    stamp=verify_inputs()
    assert json.loads((OUT/'VERIFICATION.json').read_text())['archived_predictions_exact']
    write_new(OUT/'SCORES_OPENED.json',dict(**stamp,opened_utc=now(),independent_confirmation=False))
    ctx=read(OUT/'context.csv.gz');pred=read(OUT/'archived_forecasts.csv.gz');base=read(OUT/'baseline_streams.csv.gz')
    rows=[];monthly=[];cmp=[];states=[];statecmp=[];seeds=[];murphy=[];information=[];saved_decisions=[]
    for definition in DEFS:
        for variant,info in CONFIGS:
            for version in ['raw','calibrated']:
                context,cells=load_cells(ctx,pred,base,definition,variant,info,version)
                key=dict(definition=definition,variant=variant,information=info,calibration=version)
                if version=='calibrated':
                    st,sc=state_analysis(context,cells,definition,variant,info);states+=st;statecmp+=sc
                for c in [-5.,-10.,-20.]:
                    aggregates={name:origin_decisions(d,c) for name,d in cells.items()}
                    for sample in ['full','nonoverlap']:
                        keep=np.ones(len(context),dtype=bool) if sample=='full' else nonoverlap(context)
                        samplectx=context[keep].reset_index(drop=True)
                        bkey=dict(**key,threshold_bp=c,sample=sample)
                        weights={b:day_weights(samplectx[['origin','fold']],b) for b in ([1,5,10] if sample=='full' and version=='calibrated' and variant=='original' and info=='F' and c==-10 else [5])}
                        for name,d in aggregates.items():
                            s=d[keep].reset_index(drop=True)
                            rows.append(dict(**bkey,strategy=name,**metrics(s)))
                            for fold,m in s.groupby('fold'):monthly.append(dict(**bkey,strategy=name,fold=fold,**metrics(m)))
                        new=aggregates['ml_selected'][keep].reset_index(drop=True)
                        for ref in REFERENCES:
                            reference=aggregates[ref][keep].reset_index(drop=True)
                            for block,w in weights.items():
                                cmp.append(dict(**bkey,reference=ref,n=len(new),block_days=block,**comparison(reference.cost,new.cost,w)))
                    if variant=='original' and info=='F' and version=='calibrated' and c==-10:
                        for seed,s in cells['ml_selected'].groupby('seed'):
                            d=origin_decisions(s,c)
                            for ref in REFERENCES:
                                seeds.append(dict(definition=definition,seed=int(seed),reference=ref,
                                    improvement_pct=100*(1-d.cost.mean()/aggregates[ref].cost.mean()) if aggregates[ref].cost.mean()>0 else np.nan,
                                    **metrics(d)))
                        for name,d in aggregates.items():
                            d=d.copy();d['definition']=definition;d['strategy']=name;saved_decisions.append(d)
                if variant=='original' and info=='F' and version=='calibrated':
                    for c in np.linspace(-30,0,61):
                        for name,d in cells.items():
                            scores=decisions(d.delta_e,d.q_change,float(c))
                            murphy.append(dict(definition=definition,strategy=name,threshold_bp=float(c),cost=float(np.mean(scores['cost']))))
        context=ctx[ctx.definition==definition].sort_values('origin').reset_index(drop=True)
        w=day_weights(context[['origin','fold']],5)
        infocells={}
        for info in ['R','B','F']:
            _,cell=load_cells(ctx,pred,base,definition,'original',info,'calibrated')
            infocells[info]=origin_decisions(cell['ml_selected'],-10.)
        for ref,new in [('R','B'),('B','F'),('R','F')]:
            information.append(dict(definition=definition,comparison=new+'_vs_'+ref,**comparison(infocells[ref].cost,infocells[new].cost,w)))
    comparisons=pd.DataFrame(cmp)
    primary=comparisons[(comparisons.variant=='original')&(comparisons.information=='F')&(comparisons.calibration=='calibrated')&
        (comparisons.threshold_bp==-10)&(comparisons['sample']=='full')&(comparisons.block_days==5)].copy()
    assert len(primary)==12
    primary['p_holm_12']=holm(primary.p_centered_boot)
    primary['within_extension_pass']=(primary.difference_lo>0)&(primary.p_holm_12<.05)
    for name,data in [('decision_metrics',rows),('monthly_metrics',monthly),('comparisons',comparisons),('primary_tests',primary),
        ('state_metrics',states),('state_contributions',statecmp),('seed_sensitivity',seeds),('murphy_curve',murphy),('information_value',information)]:
        save(data if isinstance(data,pd.DataFrame) else pd.DataFrame(data),OUT/(name+'.csv'))
    save(pd.concat(saved_decisions,ignore_index=True),OUT/'primary_origin_decisions.csv.gz')
    # Independent confusion-count reconstruction of the public primary means.
    max_error=0.
    for row in primary.itertuples():
        context,cells=load_cells(ctx,pred,base,row.definition,'original','F','calibrated')
        means={}
        for name in ['ml_selected',row.reference]:
            d=cells[name]
            alarms=(d.q-d.e_now).to_numpy() < -10
            events=(d.target-d.e_now).to_numpy() < -10
            means[name]=(.1*int(np.sum(alarms&~events))+.9*int(np.sum(~alarms&events)))/len(d)
        max_error=max(max_error,abs(means['ml_selected']-row.ml_cost),abs(means[row.reference]-row.reference_cost))
    assert max_error<1e-12
    write_new(OUT/'EVALUATION_COMPLETE.json',dict(**stamp,completed_utc=now(),primary_tests=12,
        independent_confusion_cost_max_error=max_error,state_contributions_add_exactly=True,independent_confirmation=False))
    print(primary[['definition','reference','improvement_pct','improvement_lo','improvement_hi','p_holm_12']].to_string(index=False))


if __name__=='__main__':main()
