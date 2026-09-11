from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
s=pd.read_csv(HERE/'one_tick_scenarios.csv')
p=pd.read_csv(HERE/'policy_window_descriptive.csv')
r=pd.read_csv(HERE/'virtual_regridding.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.right':False,'axes.spines.top':False})
fig,axs=plt.subplots(1,2,figsize=(11,4.8))
coins=['BTC','ETH','XRP','SOL','DOGE','USDT']
v=s.query("threshold=='q10' and period=='all'").set_index('coin').reindex(coins)
bars=axs[0].bar(np.arange(6),100*v.scenario_exit_share,
               color=['#888888']*4+['#315d81','#333333'])
axs[0].bar_label(bars,labels=[f'{x:.1f}%' for x in 100*v.scenario_exit_share],
                 padding=3,fontsize=8)
axs[0].set_xticks(range(6),coins)
axs[0].set_ylim(0,111)
axs[0].set_ylabel('Share of 384 lower-tail points exiting the tail (%)')
axs[0].set_title('One-tick sensitivity of the mean-reference label',pad=12)
coins2=['ETH','XRP','SOL','DOGE']
v=p.query('window_days==28').pivot(index='coin',columns='period',values='mean_abs_bp').reindex(coins2)
x=np.arange(4)
for offset,col,color,label in [(-.18,'before','#aaaaaa','Observed before'),
                                (.18,'after','#315d81','Observed after')]:
    bars=axs[1].bar(x+offset,v[col],width=.33,color=color,label=label)
    axs[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
virtual=float(r.query("window_days==28 and coin=='DOGE' and rule=='nearest_half_up'").mean_abs_bp.iloc[0])
axs[1].scatter([3],[virtual],marker='D',color='#111111',s=35,zorder=4,
                label='Pre-period DOGE, rounded to 1 KRW')
axs[1].annotate(f'{virtual:.2f}',(3,virtual),xytext=(-34,1),textcoords='offset points',fontsize=8)
axs[1].set_xticks(x,coins2)
axs[1].set_ylim(0,22)
axs[1].set_ylabel('Mean absolute BTC-relative implied-price gap (bp)')
axs[1].set_title('Reference-price gaps without the USDT price term',pad=12)
axs[1].legend(frameon=False,fontsize=7,loc='upper left')
for ax in axs:
    ax.grid(axis='y',alpha=.13)
    ax.set_axisbelow(True)
fig.text(.07,.07,'Left: 2026 observations; definition fixed using 2025. One local quote changed; others fixed. Not identified false alarms.\n'
         'Right: 28 days before / after the reform, with a transition gap. The diamond is a deterministic price transformation.\n'
         'Neither the before/after comparison nor virtual regridding identifies a causal market response.',fontsize=8,color='#444444')
fig.subplots_adjust(left=.07,right=.985,bottom=.25,top=.86,wspace=.29)
for ext in ['png','pdf']:
    fig.savefig(HERE/f'measurement_screening.{ext}',dpi=170)
plt.close(fig)
