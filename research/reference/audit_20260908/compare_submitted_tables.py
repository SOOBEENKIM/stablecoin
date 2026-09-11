"""Compare submitted numerical table cells with saved and rerun outputs."""
from pathlib import Path
import json,re
import pandas as pd

R=Path(__file__).resolve().parent
tables=json.loads((R/'submitted_tables.json').read_text())
rows=[]
sources={'saved_v4':R.parent/'stablecoin_v4/paper_outputs','rerun_B300':R/'reproduction/paper_outputs'}
def num(s):return float(re.sub(r'\s+','',s))
def pair(s):return [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?',re.sub(r'\s+','',s))]
def add(table,label,stat,reported,filename,selector,column,scale=1):
    for source,folder in sources.items():
        data=pd.read_csv(folder/filename);value=float(data.loc[selector(data),column].iloc[0])*scale
        raw=re.sub(r'\s+','',str(reported));v=float(raw)
        dec=len(raw.split('.')[1]) if '.' in raw else 0
        rows.append(dict(table=table,label=label,stat=stat,submitted=raw,source=source,value=value,
                         matches_printed_precision=abs(value-v)<=.500001*10**(-dec),file=filename,column=column))
for row,reg in zip(tables[1][1:],['Normal','Downside']):
    for stat,text,col,scale in zip(['N','mean','std','skew'],row[1:],['n','mean','std','skew'],[1,1e4,1e4,1]):
        add(2,reg,stat,text,'table1_residual_descriptive.csv',lambda d,r=reg:d.regime==r,col,scale)
for row in tables[2][1:]:
    q=num(row[0])
    for st,t,c in [('beta',row[1],'beta_downside'),('p',row[2],'p_boot')]:
        add(3,str(q),st,t,'table3_main_quantile_tail.csv',lambda d,q=q:d.q==q,c)
for row,model in zip(tables[3][1:],['M0_base+downside','M1_+global_state','M2_+ACCOUNT_LS']):
    for cell,c in zip(row[1:],['downside','btc_vol','account_ls']):
        vals=pair(cell)
        if not vals:continue
        for st,v,col in [('beta',vals[0],c),('p',vals[1],'p_'+c)]:
            raw=f'{v:.3f}' if st=='beta' else f'{v:.2f}'
            add(4,model,st,raw,'table7_decisive_robustness.csv',lambda d,m=model:(d.q==.1)&(d.model==m),col)
for row,definition in zip(tables[4][1:],['EQ','CAP','PCA']):
    for cell,q in zip(row[1:3],[.1,.5]):
        b,p=pair(cell)
        for st,v,col in [('beta',b,'beta_downside'),('p',p,'p')]:
            add(5,definition+f'_q{q}',st,f'{v:.2f}','robustness_kp_main_quantile.csv',lambda d,k=definition,q=q:(d.definition==k)&(d.q==q),col)
    add(5,definition+'_M2','p',row[3],'robustness_kp_mediation.csv',lambda d,k=definition:(d.definition==k)&(d.channel=='ACCOUNT_LS (decisive M2)'),'ch_p')
for row,sample in zip(tables[5][1:],['FULL','DROP_TOP1_DAY','DROP_TOP5_DAYS','FIRST_HALF','SECOND_HALF']):
    b,p=pair(row[2]);levp=pair(row[3])[-1]
    for st,v,col,dec in [('N',num(row[1]),'n',0),('beta',b,'q10_dn',2),('p',p,'q10_p',2),('leverage_p',levp,'M2_leverage_p',2)]:
        add(6,sample,st,f'{v:.{dec}f}','exp2_subsample_stability.csv',lambda d,s=sample:d['sample']==s,col)
out=pd.DataFrame(rows);out.to_csv(R/'submitted_tables_comparison.csv',index=False)
print(out.groupby(['table','source']).matches_printed_precision.agg(['sum','count']).to_string())
print(out[~out.matches_printed_precision].to_string(index=False))
