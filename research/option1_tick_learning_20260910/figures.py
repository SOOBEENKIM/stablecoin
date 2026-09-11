from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    d=pd.read_csv(HERE/'all_scores.csv').set_index('model')
    names=['exact_qr_cal','tcn_clean_ensemble_cal','tcn_shuffled_ensemble_cal','tcn_tick_ensemble_cal','linear_tick_ensemble_cal','lgbm_cal']
    labels=['Linear QR','TCN: clean','TCN: shuffled','TCN: tick-aware','Linear: tick-aware','LightGBM']
    colors=['#555d6a','#4383ac','#a59ab8','#bd5340','#aab8bd','#c4c7cd']
    fig,axs=plt.subplots(1,3,figsize=(15,4.6))
    for ax,metric,title,xlim in zip(axs,['clean_tail','worst_tail_h1','sensitivity_h1'],
        ['Observed-input loss','Worst one-tick input loss','Prediction sensitivity (bp)'],[(1.60,1.73),(1.68,2.04),(0,6.5)]):
        vals=d.loc[names,metric].to_numpy();ax.scatter(vals,np.arange(6),c=colors,s=65,zorder=3);ax.grid(axis='x',alpha=.18)
        ax.set_yticks(np.arange(6));ax.set_yticklabels(labels if ax is axs[0] else ['']*6);ax.invert_yaxis();ax.set_xlim(*xlim)
        for i,v in enumerate(vals):ax.text(v+(xlim[1]-xlim[0])*.012,i,f'{v:.3f}',va='center',fontsize=9)
        ax.set_title(title);ax.set_xlabel('Lower is better')
    fig.suptitle('Tick-aware training: lower sensitivity did not improve forecast loss',fontsize=14,y=1.04)
    fig.text(.5,-.03,'Fixed future mean5 target at t+6h | 1,814 origins / 77 observed days | Jan-Mar 2026 | past-only calibration',ha='center',fontsize=10)
    fig.tight_layout();fig.savefig(HERE/'model_comparison.png',dpi=180);fig.savefig(HERE/'model_comparison.pdf');plt.close(fig)
    m=pd.read_csv(HERE/'monthly_scores.csv').set_index(['model','month']);months=['2026-01','2026-02','2026-03']
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for ax,metric,title in zip(axs,['clean_tail','worst_tail_h1','sensitivity_h1'],['Observed-input loss','Worst one-tick input loss','Prediction sensitivity (bp)']):
        for key,label,color in zip(names[:4],labels[:4],colors[:4]):
            ax.plot(months,[m.loc[(key,t),metric] for t in months],marker='o',label=label,color=color)
        ax.set_title(title);ax.grid(axis='y',alpha=.15)
    axs[-1].legend(fontsize=8);fig.suptitle('Monthly results: tick-aware stability gain is concentrated in March',y=1.03)
    fig.tight_layout();fig.savefig(HERE/'monthly_comparison.png',dpi=180);fig.savefig(HERE/'monthly_comparison.pdf');plt.close(fig)
    print('Saved two PNG/PDF figure pairs',flush=True)

if __name__=='__main__':main()
