from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
COINS=['BTC','ETH','XRP','SOL','DOGE']
SEEDS=[20260910,20260911,20260912]
QS=np.array([.1,.5,.9])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def tick(price):
    floors=np.array([0.,.00001,.0001,.001,.01,.1,1.,10.,100.,1000.,5000.,10000.,50000.,100000.,500000.,1000000.,2000000.])
    units=np.array([1e-8,1e-7,1e-6,1e-5,1e-4,.001,.01,.1,1.,1.,5.,10.,50.,100.,500.,1000.,1000.])
    return units[np.searchsorted(floors,price,side='right')-1]

def augment(x,k,donor,sign,span,recipient=None):
    n=len(x);row=np.arange(n)[:,None];hour=np.arange(24)[None,:]
    donor=np.broadcast_to(np.asarray(donor),(n,));sign=np.broadcast_to(np.asarray(sign),(n,));span=np.broadcast_to(np.asarray(span),(n,))
    recipient=donor if recipient is None else np.broadcast_to(np.asarray(recipient),(n,))
    kd=k[row,hour,donor[:,None]]
    anchor=np.where(sign[:,None]>0,kd,kd-np.maximum(1e-9,kd*1e-12))
    moved=kd+sign[:,None]*tick(anchor)
    delta=1e4*np.log(moved/kd)
    mask=hour>=24-span[:,None];delta*=mask
    out=x.copy();out[row,hour,recipient[:,None]]-=delta
    out[:,:,5]=out[:,:,:5].mean(axis=2)
    out[:,:,6]=np.median(out[:,:,:5],axis=2)
    out[:,:,7]=out[:,:,:5].std(axis=2,ddof=1)
    kr=k[row,hour,recipient[:,None]]*np.exp(delta/1e4)
    out[row,hour,25+recipient[:,None]]=1e4*np.log1p(tick(kr)/kr)
    return out

def tabular(x):return np.concatenate([x[:,-1],x[:,-6:].mean(axis=1),x.mean(axis=1)],axis=1)

def scenarios():return [('clean',0,0,0)]+[(f'h{s}_{c}_{sgn:+d}',i,sgn,s) for s in [1,6,24] for i,c in enumerate(COINS) for sgn in [-1,1]]

def pinball(y,p):
    e=y[...,None]-p
    return np.maximum(QS*e,(QS-1)*e)

def assert_frozen(root=None):
    root=HERE.parent if root is None else root
    for p,v in json.loads((HERE/'PRESERVED_OPTIONS_SHA256.json').read_text()).items():assert sha(root/p)==v,p
    assert sha(HERE/'PROTOCOL_KO.md')==json.loads((HERE/'PROTOCOL_LOCK.json').read_text())['protocol_sha256']
