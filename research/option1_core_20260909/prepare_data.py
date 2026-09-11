from common import *


def main():
    inputs=[ROOT/'data/raw data'/f'{v}_1h_2025-06-01_2026-03-19.csv' for v in ['binance','upbit']]
    inputs+=[ROOT/'corrected_outputs/binance_btcusdt_data_vision_metrics.csv']
    previous=snapshot()
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    d=read(inputs[0]).join(read(inputs[1]),how='outer')
    d.index+=pd.Timedelta(hours=1)
    grid=pd.date_range(d.index.min(),d.index.max(),freq='h',tz='UTC')
    d=d.reindex(grid)
    lr=pd.DataFrame({c:logpos(d[c+'_UPBIT_CLOSE']/d[c+'_BINANCE_CLOSE']) for c in COINS})
    b=lr.rsub(logpos(d.USDT_UPBIT_CLOSE),axis=0)*1e4
    b['mean5']=b[COINS].mean(axis=1,skipna=False)
    b['median5']=b[COINS].median(axis=1,skipna=False)
    b['without_DOGE']=b[COINS[:-1]].mean(axis=1,skipna=False)
    old=pd.read_csv(ROOT/'extension_20260909/basis_definitions.csv',index_col=0)
    old.index=pd.to_datetime(old.index,utc=True)
    np.testing.assert_allclose(b.reindex(old.index)[REFS],old[REFS],atol=1e-7,rtol=1e-8,equal_nan=True)
    a=lr.sub(lr.BTC,axis=0)*1e4
    np.testing.assert_allclose(a,b[COINS].rsub(b.BTC,axis=0),atol=1e-7,rtol=1e-8,equal_nan=True)
    returns=pd.DataFrame({c:logpos(d[c+'_BINANCE_CLOSE']).diff()*100 for c in COINS})
    venue=pd.DataFrame({c:logpos(d[c+'_UPBIT_VOLUME']/d[c+'_BINANCE_VOLUME']) for c in COINS})
    hourly=pd.DataFrame(index=grid)
    for c in COINS[1:]:
        hourly['abs_a_'+c]=a[c].abs()
        hourly['ret_'+c]=returns[c]
        hourly['abs_ret_'+c]=returns[c].abs()
        hourly['volume_'+c]=venue[c]
        hourly['log_price_'+c]=logpos(d[c+'_UPBIT_CLOSE'])
    hourly['btc_r2']=returns.BTC**2
    valid=hourly.notna().all(axis=1)&np.isfinite(hourly).all(axis=1)
    hourly=hourly.loc[valid]
    daily=hourly.resample('D').mean()
    daily['n_hours']=hourly.resample('D').size()
    daily['btc_rv']=np.sqrt(hourly.btc_r2.resample('D').sum(min_count=1))
    daily=daily.loc[daily.n_hours>=20].copy()
    daily['weekend']=(daily.index.dayofweek>=5).astype(float)
    for c in ['ETH','XRP','SOL']:
        daily['contrast_'+c]=daily.abs_a_DOGE-daily['abs_a_'+c]
    daily.to_csv(HERE/'policy_daily.csv')
    b.to_csv(HERE/'basis_hourly.csv.gz',compression='gzip')
    hourly.to_csv(HERE/'policy_hourly.csv.gz',compression='gzip')

    metrics=read(inputs[2])
    ls=logpos(metrics.ACCOUNT_LS).dropna()
    available=ls.index+pd.Timedelta(minutes=5)
    matched=pd.merge_asof(pd.DataFrame({'t':grid}),
             pd.DataFrame({'available':available,'source':ls.index,'ls':ls.to_numpy()}),
             left_on='t',right_on='available',direction='backward',tolerance=pd.Timedelta(minutes=10))
    age=(matched.t-matched.available).dt.total_seconds()/60
    assert age.dropna().between(0,10).all()
    assert (matched.source.dropna()+pd.Timedelta(minutes=5)<=matched.t[matched.source.notna()]).all()
    f=pd.DataFrame(index=grid)
    f['current_btc']=b.BTC
    f['current_refspread']=b.mean5-b.BTC
    f['down_pct']=(-returns.BTC).clip(lower=0)
    f['up_pct']=returns.BTC.clip(lower=0)
    f['btc_vol24']=returns.BTC.rolling(24,min_periods=24).std()
    f['ls_z']=matched.ls.to_numpy()
    f['usdt_range']=logpos(d.USDT_UPBIT_HIGH/d.USDT_UPBIT_LOW)
    f['venue_volume']=venue.mean(axis=1,skipna=False)
    logv=logpos(d.USDT_UPBIT_VOLUME)
    f['usdt_volume_surprise']=logv-logv.shift().rolling(24,min_periods=24).mean()
    f['usdt_unchanged']=d.USDT_UPBIT_CLOSE.eq(d.USDT_UPBIT_CLOSE.shift()).astype(float)
    f['hour_sin']=np.sin(2*np.pi*grid.hour/24)
    f['hour_cos']=np.cos(2*np.pi*grid.hour/24)
    f['weekend']=(grid.dayofweek>=5).astype(float)
    standard=['current_btc','current_refspread','btc_vol24','ls_z','usdt_range','venue_volume','usdt_volume_surprise']
    pre=f.loc[(f.index<TEST)&np.isfinite(f).all(axis=1)]
    mean,sd=pre[standard].mean(),pre[standard].std(ddof=1)
    assert (sd>0).all()
    pd.DataFrame({'mean_pre2026':mean,'sd_pre2026':sd}).to_csv(HERE/'feature_scales.csv')
    f[standard]=(f[standard]-mean)/sd
    f['down_x_ls']=f.down_pct*f.ls_z
    f.insert(0,'const',1.)
    f['month_Feb']=(grid.month==2).astype(float)
    f['month_Mar']=(grid.month==3).astype(float)
    rows=[];datasets={}
    for h in [1,6,12]:
        target_time=grid+pd.Timedelta(hours=h)
        y=b[REFS].reindex(target_time).to_numpy()
        keep=(grid>=TEST)&np.isfinite(f).all(axis=1)&np.isfinite(y).all(axis=1)
        xx=f.loc[keep].to_numpy();yy=y[keep]
        assert np.linalg.matrix_rank(xx)==xx.shape[1]
        future_prices=d.reindex(target_time[keep])
        direct=np.column_stack([10000*np.log(future_prices.USDT_UPBIT_CLOSE*future_prices[c+'_BINANCE_CLOSE']/future_prices[c+'_UPBIT_CLOSE']) for c in COINS])
        direct_y=np.column_stack([direct.mean(axis=1),np.median(direct,axis=1),direct[:,0],direct[:,1],direct[:,:4].mean(axis=1)])
        np.testing.assert_allclose(yy,direct_y,atol=1e-7,rtol=1e-8)
        datasets.update({f'x{h}':xx,f'y{h}':yy,f'origin{h}':grid[keep].asi8,f'target{h}':target_time[keep].asi8})
        rows.append(dict(h=h,n=len(xx),days=int(grid[keep].normalize().nunique()),p=xx.shape[1],
                         condition_number=float(np.linalg.cond(xx)),first=str(grid[keep].min()),last=str(grid[keep].max())))
    np.savez_compressed(HERE/'economic_data.npz',feature_names=np.asarray(f.columns,dtype=str),reference_names=np.asarray(REFS),**datasets)
    f.to_csv(HERE/'economic_features.csv.gz',compression='gzip')
    pd.DataFrame(rows).to_csv(HERE/'sample_counts.csv',index=False)
    assert previous==snapshot()
    meta=dict(status='data and protocol frozen before new regression results',
        inputs_sha256=input_hashes,protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),
        source_prepare_sha256=sha(Path(__file__)),economic_data_sha256=sha(HERE/'economic_data.npz'),
        preserved_files=len(previous),n_policy_days=len(daily),n_scaling_pre2026=len(pre),
        features=list(f.columns),references=REFS,metric_age_max_min=float(age.max()),
        checks=['raw and frozen basis agreement','USDT cancels from policy contrasts',
                'exact raw-price h-hour targets','metric backward availability','full-rank common economic matrices',
                'historical files unchanged'])
    (HERE/'RUN_MANIFEST.json').write_text(json.dumps(meta,indent=2))
    (HERE/'PRESERVED_OPTIONS_SHA256.json').write_text(json.dumps(previous,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('Prepared policy days',len(daily),'past scaling rows',len(pre),flush=True)


if __name__=='__main__':main()
