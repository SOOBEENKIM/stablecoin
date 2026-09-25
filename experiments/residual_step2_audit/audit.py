"""Independent arithmetic audit of saved results, not another forecast experiment.

No imports from the forecasting/calibration/selection/evaluation implementation.
All old inputs are read-only. This is post-result diagnostic evidence, not a
preregistered new benchmark or independent-data confirmation.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import ast
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.parent/'residual_step2_information'
OUT=HERE/'results'
STREAM=['mode','definition','removed_group','procedure','seed']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def loss(actual,forecast):
    actual=np.asarray(actual);forecast=np.asarray(forecast)
    return np.where(actual>=forecast,.1*(actual-forecast),.9*(forecast-actual))


def direct_calibration(frame):
    # Search the complete history separately at each origin instead of using
    # the original forward pointer. Compute the order statistic explicitly.
    frame=frame.sort_values('origin').reset_index(drop=True)
    ends=frame.target_time.to_numpy();starts=frame.origin.to_numpy()
    errors=(frame.target-frame.pred_raw).to_numpy()
    correction=[];sizes=[]
    for t in starts:
        eligible=np.flatnonzero((ends<t)&(ends>=t-pd.Timedelta(days=90)))
        eligible=eligible[-60:]
        sizes.append(len(eligible))
        correction.append(float(np.sort(errors[eligible])[int(np.ceil(.1*len(eligible)))-1]) if len(eligible)>=30 else 0.)
    return np.asarray(correction),np.asarray(sizes)


def daily_aggregate_draws(frame,delta,base,block_days=5,repetitions=1999):
    # Resample daily sums/counts directly, not the old observation-weight matrix.
    rng=np.random.default_rng(20260925+block_days)
    totals=np.zeros((repetitions,3))
    frame=frame.copy();frame['delta']=delta;frame['base']=base;frame['count']=1
    for month,g in frame.groupby('fold',sort=True):
        days=pd.date_range(g.origin.dt.normalize().min(),g.origin.dt.normalize().max(),freq='D')
        daily=g.groupby(g.origin.dt.normalize())[['delta','base','count']].sum().reindex(days,fill_value=0).to_numpy()
        starts=rng.integers(0,len(days),size=(repetitions,int(np.ceil(len(days)/block_days))))
        indices=((starts[:,:,None]+np.arange(block_days))%len(days)).reshape(repetitions,-1)[:,:len(days)]
        totals+=daily[indices].sum(axis=1)
    return totals[:,0]/totals[:,2],100*totals[:,0]/totals[:,1]


def main():
    input_files=[OLD/p for p in ['LOCK.json','design.json','results/predictions.csv.gz',
        'results/metrics.csv','results/primary_tests.csv','results/inner_scores.csv','results/selected_models.json']]
    input_files += [OLD/'step2_run.py',OLD/'step2_evaluate.py',HERE.parent/'residual_dynamics_adaptive/adaptive.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in input_files}
    write_new(OUT/'STARTED.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(__file__),inputs=hashes,post_result_diagnostic=True))
    d=pd.read_csv(OLD/'results/predictions.csv.gz')
    for col in ['origin','target_time','latest_calibration_label']:d[col]=pd.to_datetime(d[col],utc=True)
    assert not d.duplicated(STREAM+['origin']).any()
    assert ((d.target_time-d.origin)==pd.Timedelta(hours=1)).all()
    assert (d.evaluation==(d.origin>=pd.Timestamp('2025-12-01',tz='UTC'))).all()
    assert (d.target_time<pd.Timestamp('2026-03-20',tz='UTC')).all()
    np.testing.assert_array_equal(loss([-20.,-10.],[-10.,-20.]),[9.,1.])
    calibration_rows=[]
    for keys,g in d.groupby(STREAM):
        g=g.sort_values('origin').reset_index(drop=True)
        corrections,sizes=direct_calibration(g)
        np.testing.assert_array_equal(sizes,g.calibration_n)
        error=float(np.max(np.abs(corrections-g.correction)))
        np.testing.assert_allclose(corrections,g.correction,atol=1e-10,rtol=0)
        np.testing.assert_allclose(g.pred_raw+corrections,g.pred_calibrated,atol=1e-10,rtol=0)
        fake=g.copy();split=len(fake)//2;fake.loc[split:,'target']+=1e6
        changed,_=direct_calibration(fake)
        np.testing.assert_array_equal(changed[:split+1],corrections[:split+1])
        calibration_rows.append(dict(zip(STREAM,keys),max_correction_difference_bp=error,n=len(g)))
    pd.DataFrame(calibration_rows).to_csv(OUT/'calibration_audit.csv',index=False)
    expected=pd.read_csv(OLD/'results/metrics.csv');metric_rows=[];max_loss_error=0.
    for (mode,definition,removed,procedure),g in d[d.evaluation].groupby(STREAM[:-1]):
        for version in ['raw','calibrated']:
            frame=g.assign(value=loss(g.target,g['pred_'+version]),below=g.target<g['pred_'+version])
            per_origin=frame.groupby('origin')[['value','below']].mean()
            ref=expected[(expected['mode']==mode)&(expected.definition==definition)&
                (expected.removed_group==removed)&(expected.procedure==procedure)&(expected.calibration==version)]
            assert len(ref)==1 and len(per_origin)==ref.n.iloc[0]
            error=abs(per_origin.value.mean()-ref.loss_bp.iloc[0]);max_loss_error=max(max_loss_error,error)
            np.testing.assert_allclose([per_origin.value.mean(),per_origin.below.mean()],
                [ref.loss_bp.iloc[0],ref.below_rate.iloc[0]],atol=1e-12,rtol=0)
            metric_rows.append(dict(mode=mode,definition=definition,removed_group=removed,procedure=procedure,
                calibration=version,loss_bp=per_origin.value.mean(),n=len(per_origin)))
    pd.DataFrame(metric_rows).to_csv(OUT/'recomputed_metrics.csv',index=False)
    primary=d[(d['mode']=='available_macro')&(d.definition=='EQ')&d.evaluation].copy()
    primary['loss']=loss(primary.target,primary.pred_calibrated)
    base=primary[primary.procedure=='full'].groupby('origin').agg(loss=('loss','mean'),fold=('fold','first'))
    table=[]
    for group in ['global_btc','local_volume','fx_macro']:
        without=primary[(primary.procedure=='retuned')&(primary.removed_group==group)].groupby('origin').loss.mean()
        assert without.index.equals(base.index)
        delta=(without-base.loss).to_numpy();point=delta.mean()
        draws,increase=daily_aggregate_draws(base.reset_index(),delta,base.loss.to_numpy())
        p=(1+sum(abs(draws-point)>=abs(point)))/(1+len(draws))
        lo,hi=np.quantile(increase,[.025,.975])
        table.append(dict(removed_group=group,removal_loss_increase_pct=100*(without.mean()/base.loss.mean()-1),
            increase_lo=lo,increase_hi=hi,p_centered_boot=p))
    order=sorted(range(3),key=lambda i:table[i]['p_centered_boot']);running=0.
    for rank,i in enumerate(order):
        running=max(running,(3-rank)*table[i]['p_centered_boot']);table[i]['p_holm']=min(1.,running)
    expected_primary=pd.read_csv(OLD/'results/primary_tests.csv').set_index('removed_group')
    for row in table:
        for col in ['removal_loss_increase_pct','increase_lo','increase_hi','p_centered_boot','p_holm']:
            np.testing.assert_allclose(row[col],expected_primary.loc[row['removed_group'],col],atol=1e-10,rtol=0)
    pd.DataFrame(table).to_csv(OUT/'recomputed_primary_tests.csv',index=False)
    # Independently recompute all chronological candidate choices from saved scores.
    scores=pd.read_csv(OLD/'results/inner_scores.csv');metadata=json.loads((OLD/'results/selected_models.json').read_text())
    design=json.loads((OLD/'design.json').read_text());groups=design['groups']
    all_pairs=[('downside24','btc_vol24'),('e_now','downside24')]+[
        ('downside24',x) for x in ['account_log','funding_bp','oi_ret1','oi_surprise']]+[('e_now','account_log')]
    choices=0
    for m in metadata:
        assert set(m['features']).isdisjoint(groups[m['removed_group']])
        assert {'e_now','e_change1'}.issubset(m['features'])
        if m['kind']=='interaction':
            assert m['interaction_pairs']==[[x,y] for x,y in all_pairs if x in m['features'] and y in m['features']]
        cutoff=pd.Timestamp(m['fold']+'-01',tz='UTC')
        assert pd.Timestamp(m['train_last_label'])<cutoff
        assert pd.Timestamp(m['residualizer']['fit_end'])<cutoff
        if m['procedure']=='retuned':
            months={(cutoff-pd.offsets.MonthBegin(i)).strftime('%Y-%m') for i in [1,2,3]}
            rows=scores[(scores['mode']==m['mode'])&(scores.definition==m['definition'])&
                (scores.removed_group==m['removed_group'])&scores.fold.isin(months)]
            options=[]
            for candidate,g in rows.groupby('candidate'):
                if len(g)==3 and set(g.fold)==months and g.valid.all():
                    options.append((g.loss_sum.sum()/g.n.sum(),int(candidate)))
            best=min(options)
            assert best[1]==m['candidate'];choices+=1
    # Confirm that saved inner scoring uses raw forecasts, while final primary
    # evaluation uses calibrated forecasts. This follows the old protocol but
    # is an objective-alignment limitation, not evidence of future-label leakage.
    tree=ast.parse((OLD/'step2_run.py').read_text())
    inner=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='inner_job')
    assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='calibrate' for x in ast.walk(inner))
    score_calls=[x for x in ast.walk(inner) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='pinball']
    assert len(score_calls)==1 and isinstance(score_calls[0].args[1],ast.Name) and score_calls[0].args[1].id=='pred'
    # Warm-up carryover: diagnostic comparison between two stored procedures,
    # not a newly optimized benchmark or a changed primary result.
    vol=d[(d['mode']=='available_macro')&(d.definition=='EQ')&(d.removed_group=='local_volume')]
    x=vol[vol.procedure=='retuned'].sort_values(['seed','origin']).reset_index(drop=True)
    y=vol[vol.procedure=='fixed_full_spec'].sort_values(['seed','origin']).reset_index(drop=True)
    assert x[['seed','origin']].equals(y[['seed','origin']])
    carry=pd.DataFrame({'origin':x.origin,'fold':x.fold,'evaluation':x.evaluation,
        'raw_difference_bp':abs(x.pred_raw-y.pred_raw),'calibrated_difference_bp':abs(x.pred_calibrated-y.pred_calibrated)})
    carry.groupby('fold')[['raw_difference_bp','calibrated_difference_bp']].max().to_csv(OUT/'volume_warmup_carryover.csv')
    assert carry[carry.evaluation].raw_difference_bp.max()<1e-10
    assert carry[(carry.fold>='2026-01')].calibrated_difference_bp.max()<1e-10
    changed=carry[carry.evaluation&(carry.calibrated_difference_bp>1e-10)]
    # Exact toy counterexample: redundant variables can each have zero leave-one-
    # out importance although removing both destroys all predictive information.
    z=np.linspace(-1,1,1001);target=z.copy();q0=float(np.quantile(target,.1))
    toy=dict(full_loss=float(loss(target,z).mean()),without_a_loss=float(loss(target,z).mean()),
        without_b_loss=float(loss(target,z).mean()),without_both_loss=float(loss(target,np.repeat(q0,len(z))).mean()),
        note='Constructed A=B=Y example only; not evidence of redundancy in the stablecoin sample.')
    assert toy['full_loss']==toy['without_a_loss']==toy['without_b_loss']==0 and toy['without_both_loss']>0
    assert all(digest(ROOT/name)==value for name,value in hashes.items())
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),post_result_diagnostic=True,
        independent_data_validation=False,old_sources_and_results_unchanged=True,
        independent_metric_rows=len(metric_rows),max_metric_error_bp=max_loss_error,
        independent_calibration_streams=len(calibration_rows),
        max_calibration_error_bp=max(r['max_correction_difference_bp'] for r in calibration_rows),
        independent_retuned_choices=choices,feature_metadata_checked=len(metadata),
        independent_daily_sum_bootstrap_primary_comparisons=len(table),
        selection_metric='raw_pinball',reported_primary_metric='calibrated_pinball',
        volume_equal_evaluation_raw_predictions=True,
        volume_calibration_only_affected_origins=int(changed.origin.nunique()),
        volume_carryover_first=changed.origin.min().isoformat(),volume_carryover_last=changed.origin.max().isoformat(),
        redundant_feature_counterexample=toy,
        bugs_found_in_checked_score_selection_calibration_arithmetic=False,
        overall_pipeline_correctness_proved=False)
    write_new(OUT/'AUDIT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
