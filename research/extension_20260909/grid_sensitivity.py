"""Observed reference sensitivity and deterministic one-tick perturbations.

Post-policy August 2025 onward only. This is an arithmetic sensitivity calculation,
not an estimate of the causal effect of the exchange's July policy change.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
u=pd.read_csv(HERE.parent/'data/raw data/upbit_1h_2025-06-01_2026-03-19.csv',index_col=0,parse_dates=True)
u.index+=pd.Timedelta(hours=1)
b=pd.read_csv(HERE/'basis_definitions.csv',index_col=0,parse_dates=True)
d=b.join(u[['DOGE_UPBIT_CLOSE','USDT_UPBIT_CLOSE']])
d=d.loc[d.index>=pd.Timestamp('2025-08-01',tz='UTC')].copy()
# Both assets in the relevant post-reform KRW price bands have a one-won increment.
assert d.DOGE_UPBIT_CLOSE.between(100,1000,inclusive='left').all()
assert d.USDT_UPBIT_CLOSE.between(1000,5000,inclusive='left').all()
d['doge_one_tick_basis_shift_abs']=2000*np.log1p(1/d.DOGE_UPBIT_CLOSE)
d['usdt_one_tick_basis_shift_abs']=10000*np.log1p(1/d.USDT_UPBIT_CLOSE)
d['sensitivity_ratio']=d.doge_one_tick_basis_shift_abs/d.usdt_one_tick_basis_shift_abs
rows=[]
for month,v in d.groupby(d.index.strftime('%Y-%m')):
    rows.append({'month':month,'n':len(v),'doge_price_median':v.DOGE_UPBIT_CLOSE.median(),
        'usdt_price_median':v.USDT_UPBIT_CLOSE.median(),
        'doge_tick_basis_bp_median':v.doge_one_tick_basis_shift_abs.median(),
        'usdt_tick_basis_bp_median':v.usdt_one_tick_basis_shift_abs.median(),
        'sensitivity_ratio_median':v.sensitivity_ratio.median(),
        'mean5_std':v.mean5.std(),'median5_std':v.median5.std(),
        'without_doge_std':v.without_DOGE.std(),'reference_component_std':v.reference_component.std()})
out=pd.DataFrame(rows)
out.to_csv(HERE/'tick_sensitivity_monthly.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(9,3.2))
x=np.arange(len(out))
axs[0].plot(x,out.doge_tick_basis_bp_median,'o-',color='#326a94',label='One DOGE tick through 20% reference weight')
axs[0].plot(x,out.usdt_tick_basis_bp_median,'s-',color='#666666',label='One USDT tick directly')
axs[0].set_ylabel('Absolute change in mean-reference basis (bp)');axs[0].legend(fontsize=7,frameon=False)
axs[1].plot(x,out.mean5_std,'o-',label='Five-coin mean',color='#326a94')
axs[1].plot(x,out.without_doge_std,'s-',label='Four-coin mean without DOGE',color='#666666')
axs[1].set_ylabel('Observed basis standard deviation (bp)');axs[1].legend(fontsize=7,frameon=False)
for ax in axs:
    ax.set_xticks(x);ax.set_xticklabels(out.month.str[2:],rotation=40)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.tight_layout();fig.savefig(HERE/'tick_sensitivity.png',dpi=180);fig.savefig(HERE/'tick_sensitivity.pdf')
print(out.round(4).to_string(index=False))
