"""Observable price components after fixed, forecast-defined alarm groups."""
import json
import numpy as np
import pandas as pd
from dv_core import *


def groups(ml,reference):
    return np.where(ml&reference,'both',np.where(ml,'ml_only',np.where(reference,'reference_only','neither')))


def group_contrast(d,w,value):
    hi=(d.group=='ml_only').to_numpy();lo=(d.group=='reference_only').to_numpy()
    dh,dl=w@hi,w@lo;valid=(dh>0)&(dl>0)
    y=d[value].to_numpy()
    draws=(w@(hi*y))[valid]/dh[valid]-(w@(lo*y))[valid]/dl[valid]
    ci=np.quantile(draws,[.025,.975]) if len(draws) else [np.nan,np.nan]
    return dict(ml_only_n=int(hi.sum()),reference_only_n=int(lo.sum()),
        difference=float(y[hi].mean()-y[lo].mean()) if hi.any() and lo.any() else np.nan,
        ci_low=float(ci[0]),ci_high=float(ci[1]),bootstrap_valid_fraction=float(valid.mean()))


def main():
    stamp=verify_inputs();assert (OUT/'VERIFICATION.json').exists()
    ctx=read(OUT/'context.csv.gz');pred=read(OUT/'archived_forecasts.csv.gz');path=read(OUT/'paths.csv.gz')
    pred=pred[(pred.variant=='original')&(pred.information=='F')]
    summaries=[];contrasts=[];availability=[];seedrows=[];group_rows=[]
    outcomes=['e_now','delta_e','target','local_logreturn_bp','local_change_krw','delta_log_volume']+PARTS+['fall10','local_down']
    for definition in DEFS:
        c=ctx[ctx.definition==definition].sort_values('origin').reset_index(drop=True)
        p=path[path.definition==definition]
        available={h:set(p[p.h==h].origin) for h in [1,6,12]}
        common=available[1]&available[6]&available[12]
        availability.append(dict(definition=definition,origin_n=len(c),h1_n=len(available[1]),h6_n=len(available[6]),h12_n=len(available[12]),common_n=len(common)))
        for ref in ['linear','threshold']:
            reference=pred[(pred.definition==definition)&(pred.strategy==ref)&(pred.seed==SEEDS[0])].sort_values('origin').reset_index(drop=True)
            assert list(reference.origin)==list(c.origin)
            for seed in SEEDS:
                ml=pred[(pred.definition==definition)&(pred.strategy=='ml_selected')&(pred.seed==seed)].sort_values('origin').reset_index(drop=True)
                assert list(ml.origin)==list(c.origin)
                group=groups((ml.pred_calibrated-ml.e_now).to_numpy()<-10,(reference.pred_calibrated-reference.e_now).to_numpy()<-10)
                membership=pd.DataFrame(dict(origin=c.origin,group=group))
                twelve=p[p.h==12].merge(membership,on='origin',validate='one_to_one')
                for g,d in twelve.groupby('group'):
                    seedrows.append(dict(definition=definition,reference=ref,seed=seed,group=g,n=len(d),
                        delta_e_mean=d.delta_e.mean(),local_return_mean=d.local_logreturn_bp.mean()))
                if seed!=SEEDS[0]:continue
                for cohort,origins,horizons in [('full12',available[12],[12]),('common_path',common,[1,6,12])]:
                    for h in horizons:
                        d=p[(p.h==h)&p.origin.isin(origins)].merge(membership,on='origin',validate='one_to_one').sort_values('origin').reset_index(drop=True)
                        d['fall10']=(d.delta_e<-10).astype(float);d['local_down']=(d.local_logreturn_bp<0).astype(float)
                        d['reference']=ref;d['cohort']=cohort;group_rows.append(d)
                        w=day_weights(d[['origin','fold']],5)
                        for groupname in ['both','ml_only','reference_only','neither']:
                            mask=(d.group==groupname).to_numpy();n=int(mask.sum())
                            days=int(d.loc[mask,'origin'].dt.normalize().nunique())
                            for outcome in outcomes:
                                lo,hi,valid=weighted_interval(w,d[outcome],mask)
                                summaries.append(dict(definition=definition,reference=ref,cohort=cohort,h=h,group=groupname,
                                    outcome=outcome,n=n,outcome_n=int(np.isfinite(d.loc[mask,outcome]).sum()),days=days,status='adequate' if n>=20 and days>=10 else 'limited',
                                    mean=float(d.loc[mask,outcome].mean()),ci_low=lo,ci_high=hi,bootstrap_valid_fraction=valid))
                        if h==12:
                            for outcome in ['delta_e','local_logreturn_bp','fall10']:
                                contrasts.append(dict(definition=definition,reference=ref,cohort=cohort,outcome=outcome,**group_contrast(d,w,outcome)))
                        assert np.max(np.abs(d[PARTS].sum(axis=1)-d.delta_e))<1e-8
    for name,data in [('path_summary',summaries),('disagreement_contrasts',contrasts),('path_availability',availability),('path_seed_sensitivity',seedrows)]:save(pd.DataFrame(data),OUT/(name+'.csv'))
    allrows=pd.concat(group_rows,ignore_index=True)
    save(allrows,OUT/'path_group_observations.csv.gz')
    for _,g in allrows[allrows.cohort=='common_path'].groupby(['definition','reference','origin']):
        assert set(g.h)=={1,6,12}
        assert g.e_now.max()-g.e_now.min()<1e-10
    write_new(OUT/'PATHS_COMPLETE.json',dict(**stamp,completed_utc=now(),same_origins_verified=True,
        price_identity_max_error_bp=float(np.max(np.abs(allrows[PARTS].sum(axis=1)-allrows.delta_e))),
        groups_defined_from_frozen_forecasts=True,causal_identification=False,independent_confirmation=False))
    print(pd.DataFrame(availability).to_string(index=False))


if __name__=='__main__':main()
