from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CORE=ROOT/'option1_core_20260909'
sys.path.insert(0,str(CORE))
from common import calendar_weights,interval,ols,calendar_hac,read
COINS=['BTC','ETH','XRP','SOL','DOGE']
REFS=['mean5','median5','BTC','ETH','without_DOGE']
TEST=pd.Timestamp('2026-01-01',tz='UTC')
SEED=20260910
TICKS=np.array([1000.,1000.,1.,100.,1.])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def aggregate(z):
    return np.column_stack([z.mean(axis=1),np.median(z,axis=1),z[:,0],z[:,1],z[:,:4].mean(axis=1)])

def tick_at(price):
    # Official post-2025-07-31 KRW bands, needed for multi-step diagnostics.
    floors=np.array([0.,.00001,.0001,.001,.01,.1,1.,10.,100.,1000.,5000.,10000.,50000.,100000.,500000.,1000000.,2000000.])
    units=np.array([1e-8,1e-7,1e-6,1e-5,1e-4,.001,.01,.1,1.,1.,5.,10.,50.,100.,500.,1000.,1000.])
    return units[np.searchsorted(floors,price,side='right')-1]

def price_step(price,sign):
    anchor=price if sign>0 else price-np.maximum(1e-9,price*1e-12)
    return price+sign*tick_at(anchor)

def load_prices():
    paths=[ROOT/'data/raw data'/f'{v}_1h_2025-06-01_2026-03-19.csv' for v in ['binance','upbit']]
    d=read(paths[0]).join(read(paths[1]),how='outer');d.index+=pd.Timedelta(hours=1)
    cols=[c+'_'+v+'_CLOSE' for c in COINS for v in ['BINANCE','UPBIT']]+['USDT_UPBIT_CLOSE']
    d=d.loc[np.isfinite(d[cols]).all(axis=1)&(d[cols]>0).all(axis=1)].copy()
    k=d[[c+'_UPBIT_CLOSE' for c in COINS]].to_numpy()
    p=d[[c+'_BINANCE_CLOSE' for c in COINS]].to_numpy()
    u=d.USDT_UPBIT_CLOSE.to_numpy()
    b=1e4*np.log(u[:,None]*p/k)
    old=pd.read_csv(ROOT/'extension_20260909/basis_definitions.csv',index_col=0);old.index=pd.to_datetime(old.index,utc=True)
    np.testing.assert_allclose(aggregate(b),old.reindex(d.index)[REFS],atol=1e-7,rtol=1e-8)
    return d,k,p,u,b,paths

class Resampler:
    def __init__(self,index,seed=SEED):
        self.index=pd.DatetimeIndex(index);self.days=self.index.normalize().unique().sort_values()
        self.ix=self.days.get_indexer(self.index.normalize())
        self.w=calendar_weights(self.days,5,1999,np.random.default_rng(seed)).astype(float)
    def sums(self,v):return np.bincount(self.ix,weights=np.asarray(v,dtype=float),minlength=len(self.days))
    def ratio(self,num,den):
        n=self.w@self.sums(num);d=self.w@self.sums(den)
        return np.divide(n,d,out=np.full_like(n,np.nan),where=d>0)

def check_preserved():
    saved=json.loads((HERE/'PRESERVED_OPTIONS_SHA256.json').read_text())
    for path,value in saved.items():assert sha(ROOT/path)==value,path
    assert sha(HERE/'PROTOCOL_KO.md')==json.loads((HERE/'PROTOCOL_LOCK.json').read_text())['protocol_sha256']
    return len(saved)
