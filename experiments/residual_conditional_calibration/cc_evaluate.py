"""Frozen paired comparisons; all candidates, policies and months are disclosed."""
from cc_core import *


def origin_metrics(g):
    d=g[['origin','fold','target_time','state']].copy()
    d['pinball']=old.loss(g.target,g.q)
    d['below']=(g.target<g.q).astype(float)
    d['q_change']=g.q-g.e_now
    d['too_low']=.1*np.maximum(g.target-g.q,0)
    d['too_high']=.9*np.maximum(g.q-g.target,0)
    for threshold in [5,10,20]:
        scores=dv.decisions(g.target-g.e_now,g.q-g.e_now,-threshold)
        for k,v in scores.items():d[f'{k}_{threshold}']=v
    agg={c:'mean' for c in d.columns if c not in ['origin','fold','target_time','state']}
    return d.groupby('origin',as_index=False).agg(dict(fold='first',target_time='first',state='first',**agg))


def summarize(d):
    row=dict(n=len(d))
    for c in ['pinball','below','q_change','too_low','too_high']+[f'{m}_{t}' for t in [5,10,20] for m in ['cost','missed_severity']]:
        row[c]=float(d[c].mean())
    for t in [5,10,20]:
        for c in ['event','alarm','tp','fp','fn']:row[f'{c}_{t}']=float(d[f'{c}_{t}'].sum())
    row['coverage_abs_error']=abs(row['below']-.1)
    row['conditional_abs_error']=sum(len(s)*abs(s.below.mean()-.1) for _,s in d.groupby('state'))/len(d)
    return row


def edges():
    result=[('model_advantage',('joint_selected',ref),('joint_selected','ml_selected')) for ref in dv.REFERENCES]
    result += [('calibration',('legacy','ml_selected'),('fixed_model_selected_cal','ml_selected')),
               ('calibration',('legacy','ml_selected'),('joint_selected','ml_selected')),
               ('calibration',('global_window_selected','ml_selected'),('joint_selected','ml_selected'))]
    return result


def main():
    stamp=verify_seal()
    v=json.loads((OUT/'VERIFICATION.json').read_text())
    assert v['predictions_sha256']==sha(OUT/'predictions.csv.gz') and v['legacy_predictions_reproduced']
    write_new(OUT/'SCORES_OPENED.json',dict(**stamp,opened_utc=now(),independent_confirmation=False))
    pred=read(OUT/'predictions.csv.gz')
    metrics=[];monthly=[];state_rows=[];comparisons=[];seed_rows=[];origin_rows=[]
    for definition in DEFS:
        cells={}
        for (policy,family),g in pred[pred.definition==definition].groupby(['policy','family']):
            d=origin_metrics(g);cells[policy,family]=d
            assert len(d)==509
            key=dict(definition=definition,policy=policy,family=family)
            metrics.append(dict(**key,**summarize(d)))
            for fold,s in d.groupby('fold'):monthly.append(dict(**key,fold=fold,**summarize(s)))
            w=dv.day_weights(d[['origin','fold']],5)
            for state,s in d.groupby('state'):
                mask=(d.state==state).to_numpy();lo,hi,valid=dv.weighted_interval(w,d.below,mask)
                state_rows.append(dict(**key,state=int(state),n=len(s),below=float(s.below.mean()),below_lo=lo,below_hi=hi,
                    q_change=float(s.q_change.mean()),pinball=float(s.pinball.mean()),cost_10=float(s.cost_10.mean()),
                    sparse=len(s)<30,bootstrap_valid_fraction=valid))
            for seed,s in g.groupby('seed'):seed_rows.append(dict(**key,seed=int(seed),**summarize(origin_metrics(s))))
            d=d.copy()
            for k,x in key.items():d[k]=x
            origin_rows.append(d)
        context=cells['legacy','linear']
        samples=[('full',np.ones(len(context),bool),b) for b in [1,5,10]]
        samples += [('nonoverlap',dv.nonoverlap(context),5)]
        samples += [(fold,(context.fold==fold).to_numpy(),5) for fold in sorted(context.fold.unique())]
        for sample,mask,block in samples:
            samplectx=context[mask].reset_index(drop=True)
            w=dv.day_weights(samplectx[['origin','fold']],block)
            for category,ref,new in edges():
                for endpoint in ['pinball','cost_10']:
                    a=cells[ref].loc[mask,endpoint];b=cells[new].loc[mask,endpoint]
                    effect=dv.comparison(a,b,w)
                    effect['reference_mean']=effect.pop('reference_cost');effect['new_mean']=effect.pop('ml_cost')
                    comparisons.append(dict(definition=definition,category=category,reference_policy=ref[0],reference_family=ref[1],
                        new_policy=new[0],new_family=new[1],endpoint=endpoint,sample=sample,block_days=block,n=len(a),**effect))
    cmp=pd.DataFrame(comparisons)
    primary=cmp[(cmp['sample']=='full')&(cmp.block_days==5)].copy()
    assert len(primary)==42
    primary['p_holm_all42']=dv.holm(primary.p_centered_boot)
    primary['p_holm_within_definition_question']=np.nan
    for (definition,category),g in primary.groupby(['definition','category']):
        assert len(g)==(8 if category=='model_advantage' else 6)
        primary.loc[g.index,'p_holm_within_definition_question']=dv.holm(g.p_centered_boot)
    primary['primary_EQ']=primary.definition=='EQ'
    fixed=read(OUT/'fixed_rule_predictions.csv.gz');fixed_rows=[]
    for (definition,family,rule),g in fixed.groupby(['definition','family','rule']):
        d=origin_metrics(g)
        for sample,s in [('full',d)]+list(d.groupby('fold')):
            fixed_rows.append(dict(definition=definition,family=family,rule=rule,sample=sample,**summarize(s)))
    for name,data in [('metrics',metrics),('monthly_metrics',monthly),('state_metrics',state_rows),('comparisons',cmp),
        ('primary_tests',primary),('seed_metrics',seed_rows),('fixed_rule_metrics',fixed_rows)]:
        save(data if isinstance(data,pd.DataFrame) else pd.DataFrame(data),OUT/(name+'.csv'))
    save(pd.concat(origin_rows,ignore_index=True),OUT/'origin_metrics.csv.gz')
    # Independent scalar event-count and quantile-score identities.
    max_cost_error=0.;max_loss_error=0.
    for row in pd.DataFrame(metrics).itertuples():
        g=pred[(pred.definition==row.definition)&(pred.policy==row.policy)&(pred.family==row.family)]
        alarm=(g.q-g.e_now).to_numpy()<-10;event=(g.target-g.e_now).to_numpy()<-10
        cost=(int(np.sum(alarm&~event))*.1+int(np.sum(~alarm&event))*.9)/len(g)
        loss=sum(dv.independent_elementary_integral(y,q) for y,q in zip(g.target,g.q))/len(g)
        max_cost_error=max(max_cost_error,abs(cost-row.cost_10));max_loss_error=max(max_loss_error,abs(loss-row.pinball))
    assert max_cost_error<1e-12 and max_loss_error<1e-11
    files=[p for p in OUT.glob('*.csv')]+[OUT/'origin_metrics.csv.gz',OUT/'SCORES_OPENED.json']
    write_new(OUT/'EVALUATION_COMPLETE.json',dict(**stamp,completed_utc=now(),independent_confirmation=False,
        cost_verification_error=max_cost_error,loss_verification_error=max_loss_error,
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    cols=['definition','category','reference_policy','reference_family','new_policy','endpoint','improvement_pct',
          'improvement_lo','improvement_hi','p_holm_within_definition_question']
    print(primary[cols].to_string(index=False))


if __name__=='__main__':main()
