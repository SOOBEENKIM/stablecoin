"""Prespecified secondary calibration/information comparisons for the linear model."""
from pathlib import Path
import pandas as pd
from analyze_extension import infer,losses
HERE=Path(__file__).resolve().parent
f=pd.read_csv(HERE/'mean5_h6_forecasts.csv.gz',parse_dates=['origin','target_time'])
f=f[f.origin>=pd.Timestamp('2026-01-01',tz='UTC')]
g={k:v.set_index('origin') for k,v in f.groupby('model')}
pairs=[('update_linear_local_cal28','update_linear_local_raw','linear_calibration'),
       ('update_linear_full_cal28','update_linear_local_cal28','linear_offshore'),
       ('update_linear_local_cal28','hist28','linear_vs_history'),
       ('update_linear_local_cal28','update_linear_price_cal28','linear_local_information')]
out=[]
for a,b,n in pairs:
    la,lb=losses(g[a]),losses(g[b])
    out.append({'candidate':a,'reference':b,'comparison':n,**infer(lb-la,lb)})
pd.DataFrame(out).to_csv(HERE/'supplementary_inference.csv',index=False)
