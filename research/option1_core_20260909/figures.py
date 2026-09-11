from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def save(fig,name):
    fig.savefig(HERE/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(HERE/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    d=read_saved('policy_daily.csv')
    p=pd.read_csv(HERE/'policy_regressions.csv')
    fig,axes=plt.subplots(2,1,figsize=(11,7.4),gridspec_kw={'height_ratios':[1.2,1]})
    v=d.loc['2025-06-02':'2025-09-12']
    ax=axes[0]
    for coin,color in [('DOGE','#bd4826'),('ETH','#246e95'),('XRP','#438d6c')]:
        ax.plot(v.index,v['abs_a_'+coin],lw=1.4,label=coin,color=color,alpha=.85)
    ax.axvspan(pd.Timestamp('2025-07-30',tz='UTC'),pd.Timestamp('2025-08-02',tz='UTC'),color='grey',alpha=.2,label='Excluded transition')
    ax.set(ylabel='Daily mean absolute gap to BTC (bp)',title='Relative implied exchange rates around the July 2025 grid change')
    ax.legend(ncol=4,fontsize=9);ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax=axes[1]
    labels={'unadjusted':'Unadjusted','controlled':'Market controls','controlled_trend':'Controls + linear trend'}
    colors={'unadjusted':'#999999','controlled':'#246e95','controlled_trend':'#438d6c'}
    for i,ref in enumerate(['ETH','XRP','SOL']):
        for j,spec in enumerate(labels):
            r=p.loc[(p.window_days==28)&(p.reference==ref)&(p.spec==spec)].iloc[0];yy=i+(j-1)*.21
            ax.hlines(yy,r.low95,r.high95,color=colors[spec],lw=2)
            ax.plot(r.estimate,yy,'o',color=colors[spec],label=labels[spec] if i==0 else None)
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(3));ax.set_yticklabels(['DOGE minus ETH','DOGE minus XRP','DOGE minus SOL*'])
    ax.set(xlabel='Post coefficient (bp), 95% calendar block bootstrap interval',title='28 days per side; *SOL also had a grid increase')
    ax.set_ylim(3.05,-.5);ax.legend(loc='lower right',fontsize=9)
    fig.text(.02,.005,'Conditional event-window changes, not isolated causal policy effects. Source: reconstructed Upbit/Binance hourly data.',fontsize=9)
    fig.tight_layout(rect=[0,.025,1,1]);save(fig,'policy_evidence')

    e=pd.read_csv(HERE/'economic_contrasts.csv');e=e.loc[e.primary]
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    names={'down_pct':'BTC downside\nper 1% drop at LS z = 0','ls_z':'Positioning\nper 1 SD at zero downside','down_x_ls':'Downside x positioning\nper 1% drop x 1 SD'}
    refs=['BTC','ETH','median5','without_DOGE']
    for i,q in enumerate([0.,.1]):
        for j,term in enumerate(TERMS):
            ax=axes[i,j]
            for k,ref in enumerate(refs):
                r=e.loc[(e.q==q)&(e.term==term)&(e.reference==ref)].iloc[0]
                ax.hlines(k,r.low95,r.high95,color='#246e95',lw=2)
                ax.plot(r.estimate,k,'o',color='#246e95')
            ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(4));ax.set_yticklabels(refs if j==0 else [])
            ax.invert_yaxis();ax.set_title(('OLS\n' if q==0 else 'Lower q=0.10\n')+names[term],fontsize=10)
            ax.set_xlabel('mean5 minus alternative coefficient (bp)')
    fig.suptitle('Same 1,837 origins and controls; actual +6 hours\n0 of 24 primary contrasts significant after joint Holm correction',fontsize=13)
    fig.text(.02,.005,'Intervals are marginal 95% block-bootstrap intervals, not simultaneous intervals. Associations are not leverage effects.',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.90]);save(fig,'economic_coefficient_comparisons')
    fig,ax=plt.subplots(figsize=(9,3.6))
    pre=d.loc['2025-06-02':'2025-07-29']
    ax.scatter(pre.index,pre.contrast_ETH,color='#246e95',s=17,alpha=.5,label='Daily DOGE - ETH absolute-gap difference')
    # Do not join missing dates in the rolling calendar series.
    v=pre.contrast_ETH.reindex(pd.date_range(pre.index.min(),pre.index.max(),freq='D'))
    ax.plot(v.index,v.rolling(7,min_periods=5).mean(),color='#bd4826',label='Trailing 7-day mean')
    ax.axhline(0,color='black',lw=.7)
    ax.set(ylabel='Difference (bp)',title='Pre-period levels; controlled linear slope = -0.002 bp/week [ -0.297, 0.271 ]')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));ax.legend(fontsize=9)
    fig.tight_layout();save(fig,'pretrend_levels')


if __name__=='__main__':main()
