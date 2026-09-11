"""Reproduce onset feasibility separately from the frozen initial diagnostic."""
import pandas as pd
from feasibility import panel,labels,episode_starts,TRAIN_END,TEST_START,HERE
from run_pilot import future_any

z=panel();ys,th=labels(z);rows=[]
for name in ['btc_eth','three_of_five','three_of_four_no_doge']:
    st=ys[name+'_state'];y=future_any(st)
    for period,mask in [('train',z.index<TRAIN_END),('validation',(z.index>=TRAIN_END)&(z.index<TEST_START)),('test',z.index>=TEST_START)]:
        eligible=mask&y.notna()&st.eq(0)
        alarm=episode_starts(z.mean5.lt(th.mean5).astype(float).where(z.mean5.notna()))
        sub=eligible&alarm
        rows.append({'definition':name,'period':period,'at_risk_n':int(eligible.sum()),
            'future_any_n':int(y.loc[eligible].sum()),
            'positive_days':int(y.loc[eligible&y.eq(1)].index.normalize().nunique()),
            'fresh_alarm_n':int(sub.sum()),'fresh_alarm_future_any':int(y.loc[sub].sum())})
pd.DataFrame(rows).to_csv(HERE/'onset_feasibility.csv',index=False)
