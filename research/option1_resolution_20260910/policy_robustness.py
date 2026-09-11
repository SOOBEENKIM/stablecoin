from shared import *
from statsmodels.stats.multitest import multipletests

def design(d,a,b):
    z=pd.DataFrame({'const':1.,'post':(d.index>=pd.Timestamp('2025-08-02',tz='UTC')).astype(float)},index=d.index)
    for c in ['ret_'+a,'ret_'+b,'abs_ret_'+a,'abs_ret_'+b,'btc_rv','volume_'+a,'volume_'+b]:
        z[c]=(d[c]-d[c].mean())/d[c].std(ddof=1)
    z['weekend']=d.weekend
    return z

def main():
    d=pd.read_csv(CORE/'policy_daily.csv',index_col=0);d.index=pd.to_datetime(d.index,utc=True)
    mask=((d.index>=pd.Timestamp('2025-07-02',tz='UTC'))&(d.index<pd.Timestamp('2025-07-30',tz='UTC')))|((d.index>=pd.Timestamp('2025-08-02',tz='UTC'))&(d.index<pd.Timestamp('2025-08-30',tz='UTC')))
    d=d.loc[mask];assert len(d)==56
    weights=np.load(CORE/'policy_weights.npz')['policy_w28'];post=d.index>=pd.Timestamp('2025-08-02',tz='UTC')
    variants={'all':np.ones(len(d),dtype=bool)}
    for day in d.index:variants['leave_day_'+day.strftime('%Y-%m-%d')]=d.index!=day
    for j in range(4):
        week=d.index[post][7*j:7*j+7];variants['drop_post_week_'+str(j+1)]=~d.index.isin(week)
    extremes=d.contrast_ETH.abs().sort_values(ascending=False,kind='stable').index
    for n in [1,5]:variants['drop_top_'+str(n)]=~d.index.isin(extremes[:n])
    rows=[];boot={};errors=[];point={}
    for a,b in [('DOGE','ETH'),('ETH','XRP'),('SOL','ETH')]:
        pair=a+'_'+b
        for name,keep in variants.items():
            if a!='DOGE' and name!='all':continue
            v=d.loc[keep];z=design(v,a,b);x=z.to_numpy();y=(v['abs_a_'+a]-v['abs_a_'+b]).to_numpy()
            assert np.linalg.matrix_rank(x)==x.shape[1]
            beta,se,p,cov=calendar_hac(x,y,v.index);key=pair+'_'+name
            row=dict(pair=pair,variant=name,n=len(v),estimate=beta[1],hac_low95=beta[1]-1.95996398454*se[1],hac_high95=beta[1]+1.95996398454*se[1],
                     hac_p=p[1],removed_dates=';'.join(str(v) for v in d.index[~keep]))
            if not name.startswith('leave_day'):
                draws=np.full(len(weights),np.nan)
                for i,w in enumerate(weights[:,keep]):
                    try:draws[i]=ols(x,y,w)[1]
                    except Exception as e:errors.append(dict(config=key,rep=i,error=str(e)))
                assert np.isfinite(draws).mean()>=.95
                boot[key]=draws;point[key]=beta[1];row.update(interval(beta[1],draws))
            rows.append(row)
    contrasts=[]
    for other in ['ETH_XRP','SOL_ETH']:
        est=point['DOGE_ETH_all']-point[other+'_all'];draws=boot['DOGE_ETH_all']-boot[other+'_all']
        contrasts.append(dict(comparison='DOGE_ETH_minus_'+other,estimate=est,**interval(est,draws)))
    dif=pd.DataFrame(contrasts);dif['p_holm_2']=multipletests(dif.p_centered,method='holm')[1]
    dif.to_csv(HERE/'policy_pair_contrasts.csv',index=False)
    out=pd.DataFrame(rows);out.to_csv(HERE/'policy_influence.csv',index=False)
    np.savez_compressed(HERE/'policy_influence_bootstrap.npz',**boot)
    (HERE/'POLICY_COMPLETION.json').write_text(json.dumps(dict(status='complete',code_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),
         variants=len(rows),bootstrap_errors=errors,preserved_files=check_preserved(),
         comparison_scope='Pair-specific control sets of identical form; cross-pair differences are conditional associations, not a common-X causal dose response'),indent=2))
    print(out.loc[~out.variant.str.startswith('leave_day')].round(4).to_string(index=False),flush=True)
    print('Leave one day range',out.loc[out.variant.str.startswith('leave_day'),'estimate'].agg(['min','max']).to_dict(),flush=True)
    print(dif.round(5).to_string(index=False),flush=True)

if __name__=='__main__':main()
