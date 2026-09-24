"""Post-result interpretation checks, not additional preregistered contenders.

The PC2 loading concentration motivated checking every leave-one-coin-out EQ
factor. No diagnostic model is selected as a new winner or used downstream.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from models import study

HERE=Path(__file__).resolve().parent
OUT=HERE/'diagnostics'


def main():
 OUT.mkdir(exist_ok=True)
 score=[];weights=[];fits=[];predictions=[]
 for scenario in study.pilot.SCENARIOS:
  panel,_=study.build_panel(scenario)
  data=panel.dropna(subset=['y','m','g']+study.KCOLS)
  for month in study.pilot.MONTHS:
   start=pd.Timestamp(month+'-01',tz='UTC');end=start+pd.offsets.MonthBegin(1)
   train=data.loc[data.index<start];test=data.loc[(data.index>=start)&(data.index<end)]
   for omit in study.pilot.COINS+[None]:
    cols=[c for c in study.KCOLS if c!='kp_'+str(omit)]
    if omit is None:
     a=train[cols].to_numpy();b=test[cols].to_numpy();method='individual_five_ols'
    else:
     a=train[cols].mean(axis=1).to_numpy()[:,None];b=test[cols].mean(axis=1).to_numpy()[:,None]
     method='eq4_exclude_'+omit
    x=np.column_stack([np.ones(len(train)),a]);xt=np.column_stack([np.ones(len(test)),b])
    first=np.linalg.lstsq(x,train.y,rcond=None)[0]
    g=np.column_stack([np.ones(len(train)),train.g]);gt=np.column_stack([np.ones(len(test)),test.g])
    second=np.linalg.lstsq(g,train.y-x@first,rcond=None)[0]
    fitted=xt@first+gt@second
    predictions.append(pd.DataFrame(dict(time=test.index,scenario=scenario,evaluation_month=month,model=method,
     observed_bp=test.y.to_numpy(),fitted_bp=fitted,residual_bp=test.y.to_numpy()-fitted)))
    fits.append(dict(scenario=scenario,month=month,model=method,train_last=str(train.index.max()),
     cutoff=str(start),columns=cols,first=first.tolist(),second=second.tolist()))
   state=json.loads((HERE/'results'/scenario/'checkpoints'/(month+'_f02.json')).read_text())
   load=np.array(state['loading']);coef=np.array(state['first']);sd=np.array(state['sd'])
   for j,coin in enumerate(study.pilot.COINS):
    weights.append(dict(scenario=scenario,month=month,coin=coin,pc1_loading=load[j,0],pc2_loading=load[j,1],
     pca2_effective_first_stage_bp_per_bp=float((load[j,:2]@coef[1:])/sd[j]),
     pc1_variance_share=state['log']['pca_variance_shares'][0],pc2_variance_share=state['log']['pca_variance_shares'][1]))
 pred=pd.concat(predictions,ignore_index=True)
 for keys,group in pred.groupby(['scenario','model'],sort=False):
  score.append(dict(zip(['scenario','model'],keys),period='Jan_Mar',**study.pilot.metrics(group)))
 for keys,group in pred.groupby(['scenario','model','evaluation_month'],sort=False):
  score.append(dict(zip(['scenario','model','period'],keys),**study.pilot.metrics(group)))
 old=pd.concat([pd.read_csv(HERE/'results'/s/'predictions.csv.gz',parse_dates=['time']) for s in study.pilot.SCENARIOS])
 old=old.loc[(old.model=='pca_2')&old.evaluation_month.isin(study.pilot.MONTHS)]
 for s in study.pilot.SCENARIOS:
  times=set(old.loc[old.scenario==s,'time'])
  for model in pred.model.unique():
   assert set(pred.loc[(pred.scenario==s)&(pred.model==model),'time'])==times
 pred.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 pd.DataFrame(score).to_csv(OUT/'scores.csv',index=False)
 pd.DataFrame(weights).to_csv(OUT/'pca2_loadings.csv',index=False)
 (OUT/'fits.json').write_text(json.dumps(fits,indent=2)+'\n')
 inputs={str(p.relative_to(study.ROOT)):study.pilot.sha(p) for p in [Path(__file__).resolve(),HERE/'models.py',HERE/'results/RUN_MANIFEST.json']}
 outputs={p.name:study.pilot.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
 (OUT/'MANIFEST.json').write_text(json.dumps(dict(status='post_result_diagnostic',
  motivation='PC2 loads strongly on DOGE versus other reference coins; compare all five omissions, not only DOGE',
  candidate_selection=False,new_confirmatory_test=False,matched_primary_observations=True,
  timestamp_utc=pd.Timestamp.now(tz='UTC').isoformat(),input_sha256=inputs,output_sha256=outputs),indent=2)+'\n')
 print(pd.DataFrame(score).query("period=='Jan_Mar'")[['scenario','model','rmse_bp']].to_string(index=False))

if __name__=='__main__':main()
