"""Paired inference and figures for all prespecified cells; no model selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
TEST=pd.Timestamp('2026-01-01',tz='UTC')
SEED=20260909
LOCAL='update_lgbm_local_cal28'
PAIRS=[(LOCAL,'update_linear_local_cal28','primary_linear'),
       (LOCAL,'hist28','primary_history'),
       (LOCAL,'update_lgbm_price_cal28','primary_local_info'),
       ('update_lgbm_full_cal28',LOCAL,'offshore_info'),
       (LOCAL,'update_lgbm_local_raw','calibration'),
       (LOCAL,'update_threshold_local_cal28','threshold'),
       (LOCAL,'update_qar_qar_cal28','qar'),
       (LOCAL,'hist7','recent_history')]


def losses(g):
    e1=g.target-g.q10;e9=g.target-g.q90
    return (np.maximum(.1*e1,-.9*e1)+np.maximum(.9*e9,-.1*e9))/2


def infer(delta,reference,block=5,B=1999):
    days=pd.date_range(delta.index.min().normalize(),delta.index.max().normalize(),freq='D')
    day_sum=delta.groupby(delta.index.normalize()).sum().reindex(days,fill_value=0).to_numpy()
    day_n=delta.groupby(delta.index.normalize()).size().reindex(days,fill_value=0).to_numpy()
    rng=np.random.RandomState(SEED)
    starts=rng.randint(0,len(days),(B,int(np.ceil(len(days)/block))))
    indices=((starts[:,:,None]+np.arange(block)[None,None,:])%len(days)).reshape(B,-1)[:,:len(days)]
    draws=day_sum[indices].sum(axis=1)/day_n[indices].sum(axis=1)
    observed=delta.mean()
    p=(1+np.sum(np.abs(draws-observed)>=abs(observed)))/(B+1)
    return {'n':len(delta),'days':len(days),'delta':observed,'relative_pct':100*observed/reference.mean(),
            'ci_low':np.quantile(draws,.025),'ci_high':np.quantile(draws,.975),
            'p_centered_approx':p,'block_days':block,'B':B}


def main():
    score_files=sorted(HERE.glob('*_h*_scores.csv'))
    assert len(score_files)==5
    allscores=pd.concat([pd.read_csv(p) for p in score_files],ignore_index=True)
    allscores.to_csv(HERE/'all_scores.csv',index=False)
    comparisons=[];influence=[];phase=[];daily=[]
    for p in score_files:
        label=p.name.replace('_scores.csv','')
        df=pd.read_csv(HERE/(label+'_forecasts.csv.gz'),parse_dates=['origin','target_time'])
        df=df.loc[df.origin>=TEST]
        grouped={k:v.set_index('origin').sort_index() for k,v in df.groupby('model')}
        reference_index=next(iter(grouped.values())).index
        assert all(v.index.equals(reference_index) for v in grouped.values())
        assert all(np.allclose(v.target,next(iter(grouped.values())).target) for v in grouped.values())
        loss={k:losses(v) for k,v in grouped.items()}
        pairs=PAIRS.copy()
        if 'freeze_lgbm_local_cal28' in loss:
            pairs += [(LOCAL,'freeze_lgbm_local_cal28','refitting'),
                ('update_linear_local_cal28','freeze_linear_local_cal28','linear_refitting'),
                ('update_lgbm_local_cal14',LOCAL,'window14'),
                ('update_lgbm_local_cal56',LOCAL,'window56')]
        for candidate,reference,name in pairs:
            delta=loss[reference]-loss[candidate]
            blocks=[1,5,10] if label=='mean5_h6' and name.startswith('primary') else [5]
            for block in blocks:
                comparisons.append({'cell':label,'comparison':name,'candidate':candidate,'reference':reference,
                                    **infer(delta,loss[reference],block)})
            if label=='mean5_h6':
                ds=delta.groupby(delta.index.normalize()).sum().sort_values(ascending=False)
                for k in [0,1,5]:
                    mask=~delta.index.normalize().isin(ds.index[:k])
                    influence.append({'comparison':name,'excluded_best_days':k,'n':int(mask.sum()),
                         'delta':delta[mask].mean(),'relative_pct':100*delta[mask].mean()/loss[reference][mask].mean(),
                         'excluded_dates':','.join(map(str,ds.index[:k].date))})
                for offset in range(6):
                    keep=delta.index.hour%6==offset
                    phase.append({'comparison':name,'offset_utc_hour_mod6':offset,**infer(delta[keep],loss[reference][keep])})
        for name in ['hist28','hist7','update_linear_local_cal28',LOCAL,'update_lgbm_full_cal28','update_lgbm_local_raw']:
            g=grouped[name]
            v=pd.DataFrame({'loss':loss[name], 'coverage':((g.target>=g.q10)&(g.target<=g.q90)).astype(float),
                            'width':g.q90-g.q10}).groupby(g.index.normalize()).mean().reset_index()
            v['model']=name;v['cell']=label
            daily.append(v)
    comp=pd.DataFrame(comparisons)
    comp['holm_primary_p']=np.nan
    ix=comp.index[(comp.cell=='mean5_h6')&comp.comparison.str.startswith('primary')&(comp.block_days==5)]
    order=comp.loc[ix,'p_centered_approx'].sort_values().index
    adjusted=np.maximum.accumulate(comp.loc[order,'p_centered_approx'].to_numpy()*np.arange(len(order),0,-1))
    comp.loc[order,'holm_primary_p']=np.minimum(adjusted,1)
    comp.to_csv(HERE/'paired_inference.csv',index=False)
    pd.DataFrame(influence).to_csv(HERE/'event_influence.csv',index=False)
    pd.DataFrame(phase).to_csv(HERE/'nonoverlapping_origins.csv',index=False)
    pd.concat(daily,ignore_index=True).to_csv(HERE/'daily_score_figure_data.csv',index=False)
    # Static research figures; source CSVs accompany every figure.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    main=allscores[(allscores.target=='mean5')&(allscores.h==6)].set_index('model')
    names=['hist7','hist28','update_linear_local_raw','update_linear_local_cal28','update_lgbm_local_raw',LOCAL,'update_lgbm_full_cal28']
    labels=['History 7d','History 28d','Linear','Linear + calibration','Tree','Tree + calibration','Tree + offshore + cal.']
    for ax,metric,title in zip(axs,['tail_loss','coverage80'],['Six-hour tail loss (lower is better)','Central 80% interval coverage']):
        ax.barh(np.arange(len(names)),main.loc[names,metric],color=['#999999','#999999','#555555','#555555','#326a94','#326a94','#7098b8'])
        ax.set_yticks(np.arange(len(names)));ax.set_yticklabels(labels);ax.invert_yaxis();ax.set_title(title)
        if metric=='coverage80':ax.axvline(.8,color='black',ls='--',lw=1);ax.set_xlim(0,1)
    fig.tight_layout();fig.savefig(HERE/'forecast_comparison.png',dpi=180);fig.savefig(HERE/'forecast_comparison.pdf');plt.close(fig)
    main.loc[names].to_csv(HERE/'forecast_comparison_figure_data.csv')
    basis=pd.read_csv(HERE/'basis_definitions.csv',index_col=0,parse_dates=True)
    segment=basis.loc['2025-12-01':'2025-12-02',['mean5','median5','BTC']]
    segment.to_csv(HERE/'reference_event_figure_data.csv')
    fig,ax=plt.subplots(figsize=(8,3))
    for c,color in [('mean5','#326a94'),('median5','#777777'),('BTC','#111111')]:
        ax.plot(segment.index,segment[c],label=c,color=color,lw=1.5)
    ax.axhline(0,color='#bbbbbb',lw=.7);ax.set_ylabel('Relative USDT price (bp)');ax.set_xlabel('UTC, candle end')
    ax.set_title('The same USDT price, different reference-asset tails');ax.legend(ncol=3,frameon=False)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(HERE/'reference_event.png',dpi=180);fig.savefig(HERE/'reference_event.pdf');plt.close(fig)
    # Cross-target differences in the information contribution, not cross-target loss rankings.
    sel=comp[(comp.block_days==5)&comp.comparison.isin(['primary_local_info','offshore_info'])]
    sel.to_csv(HERE/'information_contribution_figure_data.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,3.5))
    cells=['mean5_h1','mean5_h6','mean5_h12','btc_h6','median5_h6']
    for shift,name,color in [(-.12,'primary_local_info','#326a94'),(.12,'offshore_info','#777777')]:
        v=sel[sel.comparison==name].set_index('cell').loc[cells]
        ax.errorbar(np.arange(len(cells))+shift,v.delta,yerr=[v.delta-v.ci_low,v.ci_high-v.delta],
                    fmt='o',capsize=3,color=color,label=name.replace('_',' '))
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(len(cells)));ax.set_xticklabels(cells)
    ax.set_ylabel('Paired tail-loss reduction (bp)');ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(HERE/'information_contribution.png',dpi=180);fig.savefig(HERE/'information_contribution.pdf');plt.close(fig)
    print(main.loc[names,['tail_loss','coverage80','width80']].round(4).to_string())
    print(comp[(comp.cell=='mean5_h6')&(comp.block_days==5)].round(5).to_string(index=False))
    print('All prescribed cells analyzed; common origin/target equality verified.')


if __name__=='__main__':main()
