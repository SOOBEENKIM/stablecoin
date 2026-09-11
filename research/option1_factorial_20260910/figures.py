from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    s=pd.read_csv(HERE/'all_scores.csv').set_index('model');c=pd.read_csv(HERE/'factorial_contrasts.csv');c=c[c.stage=='cal'].set_index('name')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,f in zip(axs,['tcn','gru']):
        for place,color,label in [('tick','#b45339','Financial tick placement'),('shuffled','#4b789c','Magnitude-matched shuffled')]:
            vals=[s.loc[f+'_'+place+'_'+agg+'_ensemble_cal','worst_tail_h1'] for agg in ['mean','max']]
            ax.plot([0,1],vals,'o-',color=color,label=label,linewidth=2)
            for i,v in enumerate(vals):
                offset=(4,-14) if i==0 and place=='tick' else (4,6)
                ax.annotate(f'{v:.4f}',(i,v),xytext=offset,textcoords='offset points',fontsize=9,color=color)
        ax.set_xticks([0,1]);ax.set_xticklabels(['Mean candidate loss','Max candidate loss']);ax.set_xlim(-.15,1.3)
        ax.set_title(f.upper());ax.set_ylabel('Worst one-tick tail loss (lower is better)');ax.grid(axis='y',alpha=.15)
    axs[0].legend(fontsize=8);fig.suptitle('Same price candidates and training budget: the 2 x 2 comparison',y=1.03)
    fig.tight_layout();fig.savefig(HERE/'factorial_interactions.png',dpi=180);fig.savefig(HERE/'factorial_interactions.pdf');plt.close(fig)
    terms=['training_main','financial_main','interaction','max_minus_mean_tick','max_minus_mean_shuffled','tick_minus_shuffled_mean','tick_minus_shuffled_max']
    labels=['Max vs mean: average effect','Tick vs shuffled: average effect','Interaction','Max vs mean: tick candidates','Max vs mean: shuffled candidates','Tick vs shuffled: mean learning','Tick vs shuffled: max learning']
    fig,axs=plt.subplots(1,2,figsize=(13,5.5))
    for ax,f in zip(axs,['tcn','gru']):
        for i,term in enumerate(terms):
            r=c.loc[f+'_'+term];color='#b45339' if r.p_holm_19<.05 else '#929aa4'
            ax.plot([r.low95,r.high95],[i,i],color=color,linewidth=2);ax.scatter([r.difference],[i],color=color,s=35,zorder=3)
        ax.axvline(0,color='#444',linestyle=':',linewidth=1);ax.set_yticks(range(len(terms)));ax.set_yticklabels(labels if ax is axs[0] else ['']*len(terms));ax.invert_yaxis();ax.set_title(f.upper());ax.set_xlabel('Difference in worst one-tick tail loss (bp)')
    fig.suptitle('Factorial contrasts: negative values favor the named change',y=1.02)
    fig.text(.5,-.02,'Individual 95% day-block intervals; red points pass Holm19 | 1,814 origins / 77 observed days',ha='center',fontsize=10)
    fig.tight_layout();fig.savefig(HERE/'factorial_effects.png',dpi=180);fig.savefig(HERE/'factorial_effects.pdf');plt.close(fig)
    print('Saved 2 scientific figure pairs',flush=True)

if __name__=='__main__':main()
