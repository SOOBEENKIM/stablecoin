import json
import numpy as np
import pandas as pd
from r12_core import OUT,PARENT,old,hs,DEFS,VARIANTS,SEEDS,read,save,sha,now,write_new
from r12_guard import verify_seal
from factorial_stats import day_weights,effect,holm


def aggregate(g,version):
    g=g.copy();g['loss']=old.loss(g.target,g['pred_'+version]);g['below']=(g.target<g['pred_'+version]).astype(float)
    d=g.groupby('origin',as_index=False).agg(loss=('loss','mean'),below=('below','mean'),
       target=('target','first'),target_time=('target_time','first'),fold=('fold','first'),seeds=('seed','nunique'))
    expected=4 if g.strategy.iloc[0]=='ml_selected' else 1
    assert (d.seeds==expected).all()
    return d


def main():
    stamp,_=verify_seal()
    audit=json.loads((OUT/'VERIFICATION.json').read_text());assert audit['predictions_sha256']==sha(OUT/'predictions.csv.gz')
    write_new(OUT/'SCORES_OPENED.json',dict(**stamp,opened_utc=now(),independent_confirmation=False))
    pred=read(OUT/'predictions.csv.gz')
    metrics=[];monthly=[];comparisons=[];month_cmp=[];seed_cmp=[]
    for sample in ['full','first19days']:
        ends=pd.to_datetime(pred.fold+'-01',utc=True)+pd.Timedelta(days=19)
        allrows=pred if sample=='full' else pred[pred.target_time<ends]
        for definition in DEFS:
            for variant in VARIANTS:
                case=allrows[(allrows.definition==definition)&(allrows.variant==variant)]
                origins=case[['origin','fold']].drop_duplicates().sort_values('origin').reset_index(drop=True)
                blocks=[1,5,10] if sample=='full' else [5]
                weights={b:day_weights(origins,b) for b in blocks}
                for version in ['raw','calibrated']:
                    cells={};raws={}
                    for (info,strategy),g in case.groupby(['information','strategy']):
                        d=aggregate(g,version)
                        assert d.origin.equals(origins.origin)
                        cells[info,strategy]=d;raws[info,strategy]=g
                        key=dict(sample=sample,definition=definition,variant=variant,information=info,strategy=strategy,calibration=version)
                        metrics.append(dict(**key,n=len(d),seeds=int(d.seeds.iloc[0]),loss_bp=d.loss.mean(),below_rate=d.below.mean()))
                        for fold,m in d.groupby('fold'):
                            monthly.append(dict(**key,fold=fold,n=len(m),loss_bp=m.loss.mean(),below_rate=m.below.mean()))
                    for spec in old.DESIGN['primary_comparisons']:
                        rkey,nkey=tuple(spec['reference']),tuple(spec['candidate'])
                        ref,new=cells[rkey],cells[nkey];np.testing.assert_array_equal(ref.target,new.target)
                        key=dict(sample=sample,definition=definition,variant=variant,calibration=version,comparison=spec['id'],n=len(ref))
                        a,b=ref.groupby('fold').loss.mean(),new.groupby('fold').loss.mean()
                        for block,w in weights.items():comparisons.append(dict(**key,block_days=block,**effect(ref.loss,new.loss,w),winning_months=int((b<a).sum())))
                        for fold in a.index:
                            r=ref[ref.fold==fold].reset_index(drop=True);n=new[new.fold==fold].reset_index(drop=True)
                            month_cmp.append(dict(**key,fold=fold,month_n=len(r),**effect(r.loss,n.loss,day_weights(r[['origin','fold']],5))))
                        if sample=='full':
                            for seed in SEEDS:
                                r=raws[rkey];n=raws[nkey]
                                r=r[r.seed==(seed if rkey[1]=='ml_selected' else SEEDS[0])].sort_values('origin')
                                n=n[n.seed==seed].sort_values('origin')
                                assert list(r.origin)==list(n.origin)
                                rl=old.loss(r.target,r['pred_'+version]);nl=old.loss(n.target,n['pred_'+version])
                                seed_cmp.append(dict(**key,seed=seed,improvement_pct=100*(1-nl.mean()/rl.mean())))
    cmp=pd.DataFrame(comparisons)
    primary=cmp[(cmp['sample']=='full')&(cmp.calibration=='calibrated')&(cmp.block_days==5)].copy()
    assert len(primary)==36
    primary['p_holm_36']=holm(primary.p_centered_boot)
    primary['meets_within_extension_threshold']=(primary.improvement_pct>0)&(primary.improvement_lo>0)&(primary.p_holm_36<.05)
    for name,d in [('metrics',pd.DataFrame(metrics)),('monthly_metrics',pd.DataFrame(monthly)),('comparisons',cmp),
        ('development_comparisons',primary),('monthly_comparisons',pd.DataFrame(month_cmp)),('seed_comparisons',pd.DataFrame(seed_cmp))]:save(d,OUT/(name+'.csv'))
    # Exact preservation of the previous four-seed full-information EQ comparison.
    archived=pd.read_csv(PARENT/'results/seed_sensitivity/four_seed_mean.csv')
    for row in archived.itertuples():
        name='nonlinear_F' if row.reference=='linear' else 'ml_vs_threshold_F'
        got=cmp[(cmp['sample']=='full')&(cmp.definition=='EQ')&(cmp.variant==row.variant)&(cmp.calibration==row.calibration)&(cmp.block_days==5)&(cmp.comparison==name)].iloc[0]
        for c in ['improvement_pct','improvement_lo','improvement_hi','difference_bp']:
            assert abs(float(got[c])-float(getattr(row,c)))<1e-11
    write_new(OUT/'EVALUATION_COMPLETE.json',dict(**stamp,completed_utc=now(),comparisons=36,
        previous_four_seed_EQ_summary_identical=True,independent_confirmation=False))
    print(primary[primary.comparison.isin(['nonlinear_F','ml_vs_threshold_F','position_F_vs_B'])][
        ['definition','variant','comparison','improvement_pct','improvement_lo','improvement_hi','p_holm_36','winning_months']].to_string(index=False))


if __name__=='__main__':main()
