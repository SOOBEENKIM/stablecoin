"""Paired sensitivity checks using exact quantile LPs; original timestamps retained.
These diagnose archived specifications, not a fully corrected final research model.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from audit_numeric import df, paper, exact_qr, ols_resid, ROOT

B=999
bc=['RESIDUAL_FINAL_lag1','DXY_ret_lag1','USDKRW_ret_lag1','BTC_DOWNSIDE_24_lag1']
glob=['BTC_VOL_24_lag1','BTC_RET_lag1','VIX_ret_lag1']
xc=bc+glob+['ACCOUNT_LS_lag1']
qpair=df.USDT_BINANCE_CLOSE
corrected_y=ols_resid(ols_resid(df.USDT_UPBIT_CLOSE*qpair/df.USDKRW-1,
                              (df.MKT_KP_EQ+1)*qpair-1),df.DEPEG_GLOBAL)
variants={}
for name,y in [('archived',df.RESIDUAL_FINAL),('quote_corrected',corrected_y)]:
    d=df.copy();d['RESIDUAL_FINAL']=np.asarray(y);d['RESIDUAL_FINAL_lag1']=d.RESIDUAL_FINAL.shift(1)
    variants[name]=d

rows=[];draws=[]
for model,xx,target,block_days in [('M2',xc,'ACCOUNT_LS_lag1',1),
                                   ('M2',xc,'ACCOUNT_LS_lag1',5),
                                   ('main',paper.BX,'BTC_DOWNSIDE_24_lag1',1)]:
    arrays={};idx=None
    for name,d in variants.items():
        d=d[['RESIDUAL_FINAL']+xx].dropna();idx=d.index
        if model=='M2': d=(d-d.mean())/d.std()
        arrays[name]=(paper.add_const(d[xx].values),d.RESIDUAL_FINAL.values)
    days=pd.unique(idx.normalize()); groups=[np.flatnonzero(idx.normalize()==day) for day in days]
    rng=np.random.RandomState(42); bs={n:[] for n in arrays}; start=time.time()
    for k in range(B):
        if block_days==1: chosen=rng.randint(0,len(days),len(days))
        else:
            starts=rng.randint(0,len(days),int(np.ceil(len(days)/block_days)))
            chosen=np.concatenate([(s+np.arange(block_days))%len(days) for s in starts])[:len(days)]
        rr=np.concatenate([groups[j] for j in chosen])
        for name,(X,y) in arrays.items(): bs[name].append(exact_qr(X[rr],y[rr],.1)[xx.index(target)+1])
        if (k+1)%100==0: print(model,block_days,k+1,'seconds',round(time.time()-start,1),flush=True)
    for name,(X,y) in arrays.items():
        a=np.asarray(bs[name]); point=exact_qr(X,y,.1)[xx.index(target)+1]
        p=2*min(np.mean(a>0),np.mean(a<0))
        row=dict(model=model,variant=name,block_days=block_days,n=len(y),B=B,q=.1,target=target,
                 beta=point,p_sign=p,p_sign_plus_one=min(1.,2*(min((a>0).sum(),(a<0).sum())+1)/(B+1)),
                 ci95_low=np.quantile(a,.025),ci95_high=np.quantile(a,.975),
                 note='first-stage residuals fixed; original row lags; circular moving-day blocks if block_days=5')
        rows.append(row);print(row,flush=True)
        draws.extend(dict(model=model,variant=name,block_days=block_days,rep=k,beta=b) for k,b in enumerate(a))
    pd.DataFrame(rows).to_csv(ROOT/'exact_bootstrap_sensitivity.csv',index=False)
    pd.DataFrame(draws).to_csv(ROOT/'exact_bootstrap_draws.csv',index=False)
