from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save(fig,name):
    for suffix in ['png','pdf']:fig.savefig(HERE/(name+'.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    a=pd.read_csv(HERE/'measurement_sensitivity.csv');order=['mean5','median5','without_DOGE','ETH','BTC']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,mode,title in zip(axes,['any_single','equal_1bp_any'],['Actual one-tick change in any one reference coin','Equal 1 bp change in any one reference coin']):
        d=a.query('spec=="minus10" and period=="all" and mode==@mode').set_index('reference').loc[order]
        colors=['#c7542e']+['#397d9b']*4
        ax.barh(order,100*d.flippable_all_share,color=colors,height=.65)
        ax.hlines(np.arange(5),100*d.flip_low95,100*d.flip_high95,color='black',lw=1.3)
        for i,v in enumerate(100*d.flippable_all_share):ax.text(max(v,100*d.flip_high95.iloc[i])+.8,i,f'{v:.1f}%',va='center',fontsize=9)
        ax.invert_yaxis();ax.set(title=title,xlabel='Observations whose -10 bp classification can flip (%)',xlim=(0,76))
    fig.suptitle('Unequal relative price grids dominate the equal-weight benchmark',fontsize=14)
    fig.text(.02,.005,'Same 1,865 observations; USDT and offshore prices held fixed. These are sensitivity scenarios, not false-alarm rates.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);save(fig,'risk_resolution')
    b=pd.read_csv(HERE/'policy_influence.csv')
    v=b.loc[~b.variant.str.startswith('leave_day')].copy()
    labels=[]
    for _,r in v.iterrows():
        labels.append(r.pair.replace('_',' - ') if r.variant=='all' else r.variant.replace('_',' '))
    fig,ax=plt.subplots(figsize=(9,5))
    for j,(_,r) in enumerate(v.iterrows()):
        color='#c7542e' if r.pair=='DOGE_ETH' else '#397d9b'
        ax.hlines(j,r.low95,r.high95,color=color,lw=2);ax.plot(r.estimate,j,'o',color=color)
    ax.set_yticks(range(len(v)));ax.set_yticklabels(labels);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8)
    ax.set(title='Policy-window associations survive influence checks',xlabel='Controlled post coefficient (bp), 95% calendar-block interval')
    fig.text(.02,.005,'Pair-specific market controls; comparisons are not independently identified causal effects.',fontsize=9)
    fig.tight_layout(rect=[0,.03,1,1]);save(fig,'policy_influence')
    g=pd.read_csv(HERE/'prediction_scenario_gaps.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharex=True)
    scenarios=['observed','DOGE_price_minus_one_tick','DOGE_price_plus_one_tick']
    for ax,ref in zip(axes,['mean5','median5','BTC']):
        z=g.query('reference==@ref and metric=="q10"').set_index('scenario').loc[scenarios]
        for i,(_,r) in enumerate(z.iterrows()):
            ax.hlines(i,r.low95,r.high95,color='#397d9b',lw=2);ax.plot(r.tree_minus_linear,i,'o',color='#397d9b')
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(3));ax.set_yticklabels(['Observed target','DOGE price -1 tick','DOGE price +1 tick'])
        ax.invert_yaxis();ax.set(title=ref,xlabel='Tree minus linear q10 loss (bp)')
    fig.suptitle('Frozen forecasts: changing target prices can remove a ranking advantage',fontsize=13)
    fig.text(.02,.005,'Positive = lower loss for linear. Same 1,837 +6-hour origins. Target stress scenarios; no model is retrained.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);save(fig,'prediction_evaluation')

if __name__=='__main__':main()
