from shared import *

def read(path):
    d=pd.read_csv(path);d.index=pd.to_datetime(d.pop('datetime_utc'),utc=True,format='mixed')+pd.Timedelta(hours=1)
    assert d.index.is_unique
    return d.sort_index()

def log(x):return np.log(x.where(x>0))

def main():
    assert_frozen();root=HERE.parent
    paths=[root/'data/raw data'/f'{v}_1h_2025-06-01_2026-03-19.csv' for v in ['binance','upbit']]
    d=read(paths[0]).join(read(paths[1]),how='outer');grid=pd.date_range(d.index.min(),d.index.max(),freq='h');d=d.reindex(grid)
    k=d[[c+'_UPBIT_CLOSE' for c in COINS]].to_numpy()
    b=pd.DataFrame({c:1e4*log(d.USDT_UPBIT_CLOSE*d[c+'_BINANCE_CLOSE']/d[c+'_UPBIT_CLOSE']) for c in COINS})
    mean=b.mean(axis=1,skipna=False);old=pd.read_csv(root/'extension_20260909/basis_definitions.csv',index_col=0);old.index=pd.to_datetime(old.index,utc=True)
    np.testing.assert_allclose(mean.reindex(old.index),old.mean5,atol=1e-7,equal_nan=True)
    f=b.rename(columns={c:'basis_'+c for c in COINS}).copy()
    f['mean5']=mean;f['median5']=b.median(axis=1,skipna=False);f['dispersion']=b.std(axis=1,skipna=False)
    returns=pd.DataFrame({c:100*log(d[c+'_BINANCE_CLOSE']).diff() for c in COINS})
    for c in COINS:f['offshore_return_'+c]=returns[c]
    for c in COINS:f['venue_volume_'+c]=log(d[c+'_UPBIT_VOLUME']/d[c+'_BINANCE_VOLUME'])
    f['usdt_range']=log(d.USDT_UPBIT_HIGH/d.USDT_UPBIT_LOW)
    lv=log(d.USDT_UPBIT_VOLUME);f['usdt_volume_surprise']=lv-lv.shift().rolling(24).mean()
    f['usdt_unchanged']=d.USDT_UPBIT_CLOSE.eq(d.USDT_UPBIT_CLOSE.shift()).astype(float)
    f['btc_vol24']=returns.BTC.rolling(24).std()
    f['hour_sin']=np.sin(2*np.pi*grid.hour/24);f['hour_cos']=np.cos(2*np.pi*grid.hour/24);f['weekend']=(grid.dayofweek>=5).astype(float)
    for j,c in enumerate(COINS):f['tick_bp_'+c]=1e4*np.log1p(tick(k[:,j])/k[:,j])
    assert len(f.columns)==30 and list(f.columns)[25]=='tick_bp_BTC'
    future=mean.reindex(grid+pd.Timedelta(hours=6)).to_numpy();start=pd.Timestamp('2025-08-02',tz='UTC')
    xs=[];ks=[];ix=[]
    for i in range(23,len(grid)):
        xx=f.iloc[i-23:i+1].to_numpy();kk=k[i-23:i+1]
        if grid[i-23]>=start and np.isfinite(xx).all() and np.isfinite(kk).all() and np.isfinite(future[i]):
            xs.append(xx);ks.append(kk);ix.append(i)
    ix=np.asarray(ix);X=np.stack(xs);K=np.stack(ks);origin=grid[ix];target=origin+pd.Timedelta(hours=6);y=future[ix]
    np.savez_compressed(HERE/'data.npz',X=X.astype(np.float32),K=K,y=y,origin_ns=origin.asi8,target_ns=target.asi8,
                        feature_names=np.asarray(f.columns,dtype=str))
    counts=[]
    for month in ['2025-12','2026-01','2026-02','2026-03']:
        cut=pd.Timestamp(month+'-01',tz='UTC');inner=cut-pd.Timedelta(days=28)
        tr=target<cut;it=target<inner;iv=(origin>=inner)&tr;te=origin.strftime('%Y-%m')==month
        counts.append(dict(month=month,n_train=int(tr.sum()),n_inner=int(it.sum()),n_valid=int(iv.sum()),n_forecast=int(te.sum()),
            days=int(origin[te].normalize().nunique()),last_training_target=str(target[tr].max()),first_forecast_origin=str(origin[te].min())))
        assert target[tr].max()<origin[te].min() and target[it].max()<origin[iv].min()
    pd.DataFrame(counts).to_csv(HERE/'sample_counts.csv',index=False)
    # Raw-level consistency and exact magnitude match for shuffled augmentations.
    take=np.linspace(0,len(X)-1,120,dtype=int);xx=X[take];kk=K[take];rng=np.random.default_rng(99)
    donor=rng.integers(0,5,len(xx));recipient=rng.integers(0,5,len(xx));sgn=rng.choice([-1,1],len(xx));span=rng.choice([1,6],len(xx))
    at=augment(xx,kk,donor,sgn,span);ag=augment(xx,kk,donor,sgn,span,recipient)
    np.testing.assert_allclose(abs(at[:,:,:5]-xx[:,:,:5]).sum(axis=2),abs(ag[:,:,:5]-xx[:,:,:5]).sum(axis=2),atol=1e-8)
    np.testing.assert_allclose(at[:,:,5]-xx[:,:,5],ag[:,:,5]-xx[:,:,5],atol=1e-8)
    for n in range(len(xx)):
        moved=kk[n].copy();j=donor[n];beg=24-span[n]
        anchor=moved[beg:,j] if sgn[n]>0 else moved[beg:,j]-np.maximum(1e-9,moved[beg:,j]*1e-12)
        moved[beg:,j]+=sgn[n]*tick(anchor)
        direct=xx[n,:,:5]-1e4*np.log(moved/kk[n])
        np.testing.assert_allclose(at[n,:,:5],direct,atol=1e-8)
    meta=dict(status='prepared',protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),code_sha256=sha(Path(__file__)),shared_sha256=sha(HERE/'shared.py'),
              input_hashes={str(p.relative_to(root)):sha(p) for p in paths},data_sha256=sha(HERE/'data.npz'),n=len(X),p=X.shape[-1],
              checks=['raw basis reproduction','exact 24h histories and 6h targets','chronological purges','raw scenario recomputation','equal log-price perturbation budget'])
    (HERE/'DATA_MANIFEST.json').write_text(json.dumps(meta,indent=2));assert_frozen();print(pd.DataFrame(counts).to_string(index=False),flush=True)

if __name__=='__main__':main()
