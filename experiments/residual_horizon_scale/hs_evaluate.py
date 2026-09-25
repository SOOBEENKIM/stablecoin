import json
import numpy as np
import pandas as pd
from hs_core import OUT, ROOT, old, HORIZONS, VARIANTS, INFOS, STRATEGIES, read, save, sha, now, write_new
from hs_guard import verify
from factorial_stats import day_weights, effect, holm


def main():
    stamp=verify()
    audit=json.loads((OUT/'VERIFICATION.json').read_text())
    assert audit['predictions_sha256']==sha(OUT/'predictions.csv.gz')
    for f,s in json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())['file_sha256'].items():
        assert sha(ROOT/f)==s
    write_new(OUT/'SCORES_OPENED.json',dict(**stamp,opened_utc=now(),independent_confirmation=False))
    pred=read(OUT/'predictions.csv.gz');target=read(OUT/'targets.csv.gz')
    common=set.intersection(*[set(pred[pred.horizon==h].origin) for h in HORIZONS])
    assert len(common)==239
    metrics,monthly,comparisons,monthly_effects,conditional=[],[],[],[],[]
    for sample in ['native','common_origins']:
        d=pred if sample=='native' else pred[pred.origin.isin(common)]
        for h in HORIZONS:
            case=d[d.horizon==h]
            origins=case[['origin','fold']].drop_duplicates().sort_values('origin').reset_index(drop=True)
            weights={b:day_weights(origins,b) for b in ([1,5,10] if sample=='native' else [5])}
            for version in ['raw','calibrated']:
                cells={}
                for (v,info,st),g in case.groupby(['variant','information','strategy']):
                    g=g.sort_values('origin').reset_index(drop=True).copy()
                    g['loss']=old.loss(g.target,g['pred_'+version]);g['below']=(g.target<g['pred_'+version]).astype(float)
                    assert g.origin.equals(origins.origin)
                    cells[v,info,st]=g
                    key=dict(sample=sample,horizon=h,variant=v,information=info,strategy=st,calibration=version)
                    metrics.append(dict(**key,n=len(g),loss_bp=g.loss.mean(),below_rate=g.below.mean()))
                    for fold,m in g.groupby('fold'):
                        monthly.append(dict(**key,fold=fold,n=len(m),loss_bp=m.loss.mean(),below_rate=m.below.mean()))
                    if sample=='native' and info=='F':
                        joined=g.merge(target[target.horizon==h][['origin','bin_e_now','bin_e_change1','bin_e_rms72']],on='origin',validate='one_to_one')
                        for state in ['e_now','e_change1','e_rms72']:
                            for bin_id,z in joined.groupby('bin_'+state):
                                conditional.append(dict(**key,state=state,bin=int(bin_id),fold='all',n=len(z),loss_bp=z.loss.mean(),below_rate=z.below.mean()))
                                for fold,m in z.groupby('fold'):
                                    conditional.append(dict(**key,state=state,bin=int(bin_id),fold=fold,n=len(m),loss_bp=m.loss.mean(),below_rate=m.below.mean()))
                specs=[]
                for v in VARIANTS:
                    for s in old.DESIGN['primary_comparisons']:
                        specs.append((v+'__'+s['id'],(v,*s['reference']),(v,*s['candidate'])))
                for info in INFOS:
                    for st in STRATEGIES:
                        specs.append(('scale__'+info+'__'+st,('original',info,st),('own_scale',info,st)))
                assert len(specs)==27
                for name,rkey,nkey in specs:
                    ref,new=cells[rkey],cells[nkey]
                    np.testing.assert_array_equal(ref.target,new.target)
                    base=dict(sample=sample,horizon=h,calibration=version,comparison=name,n=len(ref))
                    a,b=ref.groupby('fold').loss.mean(),new.groupby('fold').loss.mean()
                    for block,w in weights.items():
                        comparisons.append(dict(**base,block_days=block,**effect(ref.loss,new.loss,w),winning_months=int((b<a).sum())))
                    for fold in a.index:
                        monthly_effects.append(dict(**base,fold=fold,improvement_pct=100*(1-b[fold]/a[fold])))
    c=pd.DataFrame(comparisons)
    p=c[(c['sample']=='native')&(c.calibration=='calibrated')&(c.block_days==5)].copy()
    assert len(p)==81
    p['p_holm_81']=holm(p.p_centered_boot)
    # All comparisons are exploratory even if this diagnostic threshold passes.
    p['meets_within_extension_threshold']=(p.improvement_pct>0)&(p.improvement_lo>0)&(p.p_holm_81<.05)
    for name,frame in [('metrics',pd.DataFrame(metrics)),('monthly',pd.DataFrame(monthly)),('comparisons',c),
                       ('development_comparisons',p),('monthly_effects',pd.DataFrame(monthly_effects)),('conditional',pd.DataFrame(conditional))]:
        save(frame,OUT/(name+'.csv'))
    write_new(OUT/'EVALUATION_COMPLETE.json',dict(**stamp,completed_utc=now(),comparisons=81,
        independent_confirmation=False,seed_count=1,common_origins=len(common)))
    print(p[p.comparison.isin(['original__nonlinear_F','own_scale__nonlinear_F','original__position_F_vs_B',
        'own_scale__position_F_vs_B','scale__F__ml_selected','own_scale__ml_vs_threshold_F'])][
        ['horizon','comparison','improvement_pct','improvement_lo','improvement_hi','p_holm_81','winning_months']].to_string(index=False))


if __name__=='__main__':main()
