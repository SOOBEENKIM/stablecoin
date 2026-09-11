"""Standalone audit visualization, using saved numerical audit results only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
h=pd.read_csv(R/'horizon_elapsed_time.csv')
b=pd.read_csv(R/'exact_bootstrap_sensitivity.csv').query("model == 'M2'")
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(13,4.7),gridspec_kw={'width_ratios':[1,1.25]})
ax=axs[0]
x=np.arange(len(h));med=h.median_hours.to_numpy()
ax.errorbar(x,med,yerr=np.array([med-h.min_hours.to_numpy(),h.max_hours.to_numpy()-med]),
            fmt='o',capsize=4,color='#224D70',label='Actual time: median and range')
ax.plot(x,h.h,'s--',color='#AA5A20',label='Hours claimed by row label')
ax.set_xticks(x);ax.set_xticklabels(h.h.astype(int));ax.set_yscale('log')
ax.set_yticks([1,3,6,12,24,48,72,120,168]);ax.set_yticklabels([1,3,6,12,24,48,72,120,168])
ax.set_xlabel('Row-shift horizon h');ax.set_ylabel('Elapsed clock hours (log scale)')
ax.set_title('A. Row shifts do not represent clock hours',loc='left',fontweight='bold')
ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8,loc='lower right')
ax=axs[1]
labels=[]
for i,(_,r) in enumerate(b.iterrows()):
    color='#224D70' if r.variant=='archived' else '#B35C37'
    y=3-i
    ax.errorbar(r.beta,y,xerr=[[r.beta-r.ci95_low],[r.ci95_high-r.beta]],fmt='o',capsize=4,color=color)
    labels.append(('Original quotes' if r.variant=='archived' else 'Quotes converted to USD')+f'\n{int(r.block_days)}-day blocks, p={r.p_sign:.3f}')
ax.set_yticks([3,2,1,0]);ax.set_yticklabels(labels,fontsize=9)
ax.axvline(0,color='#666666',lw=1,ls='--');ax.set_ylim(-.6,3.6);ax.set_xlim(-.37,.23)
ax.set_xlabel('Standardized account long/short coefficient; 95% percentile interval')
ax.set_title('B. Core leverage association is sensitive to quotes',loc='left',fontweight='bold')
ax.grid(axis='x',alpha=.15)
fig.text(.02,.02,'Diagnostic sensitivity: q=0.10, n=714, exact quantile LP, B=999. Original row lags and controls retained; first-stage residuals fixed.',fontsize=8,color='#555555')
fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(R/'audit_overview.png',dpi=190);fig.savefig(R/'audit_overview.pdf');plt.close(fig)
