from common import *
from policy import design,windows,fit_boot


def main():
    d=read_saved('policy_daily.csv');weights=np.load(HERE/'policy_weights.npz')
    rows=[];boot={};errors=[];designs=[]
    for width in [14,28,42]:
        before,after=windows(d,width);sample=pd.concat([before,after])
        for ref in ['ETH','XRP','SOL']:
            for spec in ['controlled','controlled_trend']:
                z=design(sample,ref,spec)
                for c in ['DOGE',ref]:
                    v=sample['log_price_'+c]
                    z['log_price_'+c]=(v-v.mean())/v.std(ddof=1)
                x=z.to_numpy();y=sample['contrast_'+ref].to_numpy();k=z.columns.get_loc('post')
                assert np.linalg.matrix_rank(x)==x.shape[1]
                b,se,p,cov=calendar_hac(x,y,sample.index)
                draws,fail=fit_boot(x,y,weights['policy_w'+str(width)],k)
                key='w'+str(width)+'_'+ref+'_'+spec
                boot[key]=draws;errors.extend([dict(config=key,**e) for e in fail])
                rows.append(dict(window_days=width,reference=ref,spec=spec+'_prices',n=len(y),p=x.shape[1],
                    estimate=b[k],hac_se=se[k],hac_p=p[k],condition_number=np.linalg.cond(x),**interval(b[k],draws)))
                zz=z.copy();zz['outcome']=y;zz['config']=key;designs.append(zz)
    pd.DataFrame(rows).to_csv(HERE/'policy_price_sensitivity.csv',index=False)
    pd.concat(designs).to_csv(HERE/'policy_price_designs.csv.gz',compression='gzip')
    np.savez_compressed(HERE/'policy_price_bootstrap.npz',**boot)
    # Exposure to the official grid at the observed price levels, using the
    # exact same valid hours that entered the primary daily comparisons.
    raw=read(ROOT/'data/raw data/upbit_1h_2025-06-01_2026-03-19.csv')
    raw.index+=pd.Timedelta(hours=1)
    hourly=read_saved('policy_hourly.csv.gz')
    pre,post=windows(d,28)
    old={'BTC':1000.,'ETH':1000.,'XRP':1.,'SOL':50.,'DOGE':.1}
    new={'BTC':1000.,'ETH':1000.,'XRP':1.,'SOL':100.,'DOGE':1.}
    bands={'BTC':(2e6,np.inf),'ETH':(2e6,np.inf),'XRP':(1000,5000),'SOL':(1e5,5e5),'DOGE':(100,1000)}
    exposure=[]
    for period,dates,ticks in [('before',pre.index,old),('after',post.index,new)]:
        ix=hourly.index[hourly.index.normalize().isin(dates)]
        for coin in COINS:
            price=raw.reindex(ix)[coin+'_UPBIT_CLOSE'];lo,hi=bands[coin]
            assert price.between(lo,hi,inclusive='left').all()
            unit=ticks[coin];relative=1e4*np.log1p(unit/price)
            exposure.append(dict(coin=coin,period=period,n=len(price),tick_krw=unit,
                min_price_krw=price.min(),median_price_krw=price.median(),max_price_krw=price.max(),
                median_log_tick_bp=relative.median(),median_mean5_weighted_tick_bp=relative.median()/5,
                price_on_new_order_grid_fraction=np.isclose(price/unit,np.round(price/unit),atol=1e-5,rtol=0).mean()))
    pd.DataFrame(exposure).to_csv(HERE/'tick_exposure.csv',index=False)
    moving=pd.read_csv(HERE/'moving_window_contrasts.csv');policy=pd.read_csv(HERE/'policy_regressions.csv')
    ranks=[]
    for width in [14,28]:
        for ref in ['ETH','XRP','SOL']:
            v=moving.loc[(moving.window_days==width)&(moving.reference==ref),'estimate']
            actual=policy.loc[(policy.window_days==width)&(policy.reference==ref)&(policy.spec=='unadjusted'),'estimate'].iloc[0]
            ranks.append(dict(window_days=width,reference=ref,actual_change_bp=actual,n_overlapping_windows=len(v),
                count_at_least_actual=int((v>=actual).sum()),count_absolute_at_least_actual=int((v.abs()>=abs(actual)).sum()),
                min_other_change_bp=v.min(),max_other_change_bp=v.max()))
    pd.DataFrame(ranks).to_csv(HERE/'moving_window_summary.csv',index=False)
    (HERE/'SUPPLEMENT_COMPLETION.json').write_text(json.dumps(dict(status='complete',
        scope_sha256=sha(HERE/'SUPPLEMENT_SCOPE_KO.md'),code_sha256=sha(Path(__file__)),
        bootstrap_failures=errors,price_control_specs=len(rows),exposure='Observed-price-band official order-grid exposure, not measured bid-ask spread'),indent=2))
    print(pd.DataFrame(rows).loc[lambda a:a.reference=='ETH'].round(4).to_string(index=False))
    print(pd.DataFrame(exposure).round(4).to_string(index=False))


if __name__=='__main__':main()
