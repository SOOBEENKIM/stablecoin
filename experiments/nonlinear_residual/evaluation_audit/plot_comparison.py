"""Static research figure: explanation error and forecast utility are distinct."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    score=pd.read_csv(SOURCE/'scores.csv')
    forecasts=pd.read_csv(OUT/'forecast_scores.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    names=['EQ5 OLS','PCA 1','PCA 2','Selected AE']
    colors=['#637488','#637488','#296b9b','#a25832']
    for i,method in enumerate(PRIMARY):
        a=score.loc[(score.model==method)&(score.period=='Jan_Mar'),'rmse_bp']
        b=forecasts.loc[(forecasts.method==method)&(forecasts.task=='observed_change')&(forecasts.q==.1)&
            (forecasts.period=='all')&(forecasts.algorithm=='quantile_lightgbm')&(forecasts['info']=='with_positioning'),'skill']*100
        c=forecasts.loc[(forecasts.method==method)&(forecasts.task=='own_residual')&(forecasts.q==.1)&
            (forecasts.period=='all')&(forecasts.algorithm=='linear_qr')&(forecasts['info']=='without_positioning'),'skill']*100
        for ax,values in zip(axes,[a,b,c]):
            ax.scatter(np.full(len(values),i)+np.linspace(-.09,.09,len(values)),values,s=29,color=colors[i],alpha=.8)
            ax.plot([i,i],[values.min(),values.max()],color=colors[i],alpha=.5)
    for ax in axes:
        ax.set_xticks(range(4),names,rotation=25,ha='right');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('Contemporaneous premium fit');axes[0].set_ylabel('Residual RMS (bp): lower is better')
    axes[1].set_title('Same observed 6-hour change');axes[1].set_ylabel('q10 forecast skill (%): higher is better')
    axes[2].set_title('Each method\'s own future residual');axes[2].set_ylabel('q10 forecast skill (%): higher is better')
    for ax in axes[1:]:ax.axhline(0,color='black',lw=.8,linestyle='--')
    fig.suptitle('January–March 2026: smaller residuals do not establish greater forecast utility',fontsize=12)
    fig.savefig(HERE/'comparison.png',dpi=160)


if __name__=='__main__':main()
