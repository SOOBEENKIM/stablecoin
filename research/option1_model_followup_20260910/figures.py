from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    d=pd.read_csv(HERE/'all_scores.csv').set_index('model')
    names=['exact_qr_cal','tcn_clean_ensemble_cal','tcn_tick_ensemble_cal','gru_clean_ensemble_cal','gru_tick_ensemble_cal','mlp_clean_ensemble_cal','mlp_tick_ensemble_cal','tcn_hard_shuffled_ensemble_cal','tcn_hard_tick_ensemble_cal']
    labels=['Linear QR','TCN: clean','TCN: tick augmentation','GRU: clean','GRU: tick augmentation','MLP: clean','MLP: tick augmentation','TCN: hard shuffled','TCN: hard tick']
    colors=['#555d6a','#97b6cf','#366b98','#a6c6b4','#327557','#c6b7d0','#8366a2','#d2b4a3','#b05237']
    fig,axs=plt.subplots(1,3,figsize=(15,6))
    for ax,metric,title in zip(axs,['clean_tail','worst_tail_h1','sensitivity_h1'],['Observed-input loss','Worst one-tick input loss','Prediction sensitivity (bp)']):
        vals=d.loc[names,metric].to_numpy();span=max(float(np.ptp(vals)),.01)
        ax.scatter(vals,np.arange(len(names)),c=colors,s=55,zorder=3);ax.set_xlim(max(0,vals.min()-.15*span),vals.max()+.3*span)
        for i,v in enumerate(vals):ax.text(v+.02*span,i,f'{v:.3f}',va='center',fontsize=9)
        ax.set_yticks(np.arange(len(names)));ax.set_yticklabels(labels if ax is axs[0] else ['']*len(names));ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_title(title);ax.set_xlabel('Lower is better')
    fig.suptitle('Architecture and hard-scenario learning: the same fixed future target',fontsize=14,y=1.01)
    fig.text(.5,-.025,'1,814 origins / 77 observed days | 3 seeds averaged | Jan-Mar 2026 | strictly past calibration',ha='center')
    fig.tight_layout();fig.savefig(HERE/'model_comparison.png',dpi=180);fig.savefig(HERE/'model_comparison.pdf');plt.close(fig)
    monthly=pd.read_csv(HERE/'monthly_scores.csv').set_index(['model','month']);months=['2026-01','2026-02','2026-03']
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,metric,title in zip(axs,['clean_tail','worst_tail_h1','sensitivity_h1'],['Observed-input loss','Worst one-tick input loss','Prediction sensitivity (bp)']):
        for i in [0,2,4,6,8]:ax.plot(months,[monthly.loc[(names[i],t),metric] for t in months],marker='o',label=labels[i],color=colors[i])
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axs[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'monthly_comparison.png',dpi=180);fig.savefig(HERE/'monthly_comparison.pdf');plt.close(fig)
    print('Saved two figure pairs',flush=True)

if __name__=='__main__':main()
