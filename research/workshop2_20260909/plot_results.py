"""Static scientific comparison and a registry of primary sources."""
from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
scores=pd.read_csv(HERE/'sequence_scores.csv').set_index('model')
order=['logistic_current_raw','tcn_ensemble_raw','mlp_ensemble_raw','gru_ensemble_raw',
       'tree_current_raw','recent28','logistic_flat_raw']
names=['Logistic: current features','TCN: 24-hour history','MLP: current features','GRU: 24-hour history',
       'LightGBM: current features','Recent 28-day frequency','Logistic: 24-hour lags']
table=scores.loc[order].copy();table['label']=names;table.to_csv(HERE/'figure_scores.csv')
inf=pd.read_csv(HERE/'sequence_inference.csv')
refs=['logistic_current_raw','mlp_ensemble_raw','tree_current_raw','recent28']
sub=inf.loc[inf.candidate.eq('tcn_ensemble_raw')&inf.block_days.eq(5)].set_index('reference').loc[refs]
sub.to_csv(HERE/'figure_tcn_comparisons.csv')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(12,4))
for j,v in enumerate(table.brier):
    color='#126c8a' if j==1 else '#444444'
    ax[0].scatter(v,j,color=color,s=40)
    ax[0].text(v+.0007,j,'%.4f'%v,va='center',fontsize=9,color=color)
ax[0].set_yticks(range(len(names)),names);ax[0].invert_yaxis();ax[0].set_xlim(.210,.247)
ax[0].set_xlabel('Brier score (lower is better)');ax[0].set_title('Same 1,633 origins / 77 days',fontsize=11)
for j,(_,v) in enumerate(sub.iterrows()):
    ax[1].plot([v.low95,v.high95],[j,j],color='#126c8a')
    ax[1].scatter(v.brier_improvement,j,color='#126c8a',s=35)
ax[1].axvline(0,color='#777777',ls='--',lw=1)
ax[1].set_yticks(range(4),['vs current logistic','vs current MLP','vs LightGBM','vs recent frequency'])
ax[1].invert_yaxis();ax[1].set_xlabel('Reference loss minus TCN loss\nPositive favors TCN')
ax[1].set_title('Paired 5-day block 95% intervals',fontsize=11)
fig.tight_layout(w_pad=2)
fig.savefig(HERE/'model_comparison.png',dpi=180,bbox_inches='tight')
fig.savefig(HERE/'model_comparison.pdf',bbox_inches='tight');plt.close(fig)
doc=(HERE/'WORKSHOP2_DESIGN_AND_EVIDENCE_KO.md').read_text();refs=[]
for n,body in re.findall(r'^\[\^(\d+)\]: (.+)$',doc,re.M):
    links=re.findall(r'\[([^\]]+)\]\((https?://[^\s]+)\)',body)
    refs.append(dict(id=n,citation_and_access_note=body,title=links[0][0] if links else '',
        primary_url=links[0][1] if links else '',review_date='2026-09-09'))
pd.DataFrame(refs).to_csv(HERE/'references.csv',index=False)
