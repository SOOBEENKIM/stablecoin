"""Descriptive target/price-grid diagnostics, not causal identification."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_extension import INPUTS,read_panel,COINS,TEST

d=read_panel(INPUTS[0]).join(read_panel(INPUTS[1]))
d.index+=pd.Timedelta(hours=1)
imp=pd.DataFrame({c:np.log(d[c+'_UPBIT_CLOSE']/d[c+'_BINANCE_CLOSE']) for c in COINS})
z=pd.DataFrame({c:1e4*(np.log(d.USDT_UPBIT_CLOSE)-imp[c]) for c in COINS}).dropna()
z['mean5']=z[COINS].mean(axis=1)
z['median5']=z[COINS].median(axis=1)
z['reference_component']=z.mean5-z.median5
z['dispersion']=imp.std(axis=1)*1e4
for c in COINS:
    z['without_'+c]=z[[k for k in COINS if k!=c]].mean(axis=1)
z.to_csv(HERE/'basis_definitions.csv')
summaries=[]
for period,keep in [('all',np.ones(len(z),bool)),('pre2026',z.index<TEST),('test2026',z.index>=TEST)]:
    for name in ['mean5','median5']+COINS+['without_'+c for c in COINS]:
        v=z.loc[keep,name]
        rec={'period':period,'measure':name,'n':len(v),'mean':v.mean(),'std':v.std(),
             'min':v.min(),'max':v.max(),'mean_abs':v.abs().mean()}
        for q in [.01,.1,.5,.9,.99]:
            rec['q%02d'%round(q*100)]=v.quantile(q)
        summaries.append(rec)
pd.DataFrame(summaries).to_csv(HERE/'basis_distribution.csv',index=False)
monthly=[]
for month,v in d.groupby(d.index.strftime('%Y-%m')):
    for c in COINS+['USDT']:
        close=v[c+'_UPBIT_CLOSE'].dropna()
        monthly.append({'month':month,'coin':c,'n':len(close),
            'noninteger_fraction':float(((close-close.round()).abs()>1e-6).mean()),
            'unchanged_fraction':float(close.eq(close.shift()).mean()),
            'median_close':float(close.median())})
pd.DataFrame(monthly).to_csv(HERE/'price_grid_monthly.csv',index=False)
extreme=z.loc[z.mean5.abs().nlargest(30).index].copy()
extreme.to_csv(HERE/'extreme_reference_comparison.csv')
thresholds={c:float(z.loc[z.index<TEST,c].quantile(.1)) for c in ['mean5','median5','BTC']}
tail_records=[]
test=z.loc[z.index>=TEST]
for a in thresholds:
    for b in thresholds:
        aa=test[a]<thresholds[a];bb=test[b]<thresholds[b]
        tail_records.append({'a':a,'b':b,'n':len(test),'threshold_a':thresholds[a],'threshold_b':thresholds[b],
            'a_events':int(aa.sum()),'b_events':int(bb.sum()),'intersection':int((aa&bb).sum()),
            'a_only':int((aa&~bb).sum()),'b_only':int((~aa&bb).sum()),
            'jaccard':float((aa&bb).sum()/(aa|bb).sum())})
pd.DataFrame(tail_records).to_csv(HERE/'tail_definition_overlap.csv',index=False)
# Verify an algebraic decomposition without treating its components as identified latent risks.
resid=(z.mean5-z.BTC)+1e4*(imp.mean(axis=1)-imp.BTC).reindex(z.index)
assert np.nanmax(abs(resid))<1e-8
(HERE/'measurement_metadata.json').write_text(json.dumps({
    'common_n':len(z),'definition_identity_max_abs_error':float(abs(resid).max()),
    'note':'Target comparisons and calendar co-occurrences are descriptive, not evidence of latent fair value or causal tick effects.',
    'official_policy_sources':[
       'https://docs.upbit.com/kr/kr/changelog/usdtkrw_tick_unit_change',
       'https://docs.upbit.com/kr/changelog/krw_tick_unit_change_250731',
       'https://docs.upbit.com/kr/docs/krw-market-info']},indent=2))
print(pd.DataFrame(summaries).query("measure in ['mean5','median5','BTC']").round(3).to_string(index=False))
print(pd.DataFrame(tail_records).round(3).to_string(index=False))
