from pathlib import Path
import hashlib,json,warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog,OptimizeWarning
from scipy.stats import norm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
COINS=['BTC','ETH','XRP','SOL','DOGE']
REFS=['mean5','median5','BTC','ETH','without_DOGE']
TERMS=['down_pct','ls_z','down_x_ls']
TEST=pd.Timestamp('2026-01-01',tz='UTC')
SEED=20260909
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot():
    return {str(p.relative_to(ROOT)):sha(p)
            for name in ['extension_20260909','workshop2_20260909','workshop3_20260909','option1_development_20260909']
            for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def read(path):
    d=pd.read_csv(path)
    d.index=pd.to_datetime(d.pop('datetime_utc'),utc=True,format='mixed')
    assert d.index.is_unique
    return d.sort_index()


def logpos(x):
    return np.log(x.where(x>0))


def ols(x,y,w=None):
    if w is None:
        return np.linalg.lstsq(x,y,rcond=None)[0]
    keep=w>0
    if np.linalg.matrix_rank(x[keep])<x.shape[1]:
        raise ValueError('rank deficient resample')
    sw=np.sqrt(w[keep])
    return np.linalg.lstsq(x[keep]*sw[:,None],y[keep]*sw[:,None] if y.ndim==2 else y[keep]*sw,rcond=None)[0]


def qr(x,y,q,w=None):
    if w is None:w=np.ones(len(y))
    keep=w>0
    xx=x[keep];yy=y[keep];ww=w[keep]
    if np.linalg.matrix_rank(xx)<xx.shape[1]:
        raise ValueError('rank deficient quantile resample')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',OptimizeWarning)
        fit=linprog(-yy,A_eq=xx.T,b_eq=np.zeros(xx.shape[1]),
             bounds=np.column_stack([(q-1)*ww,q*ww]),method='highs',
             options={'threads':1,'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    if not fit.success:raise RuntimeError(fit.message)
    beta=-fit.eqlin.marginals
    e=yy-xx@beta
    primal=float(np.sum(ww*np.maximum(q*e,(q-1)*e)))
    gap=abs(primal+fit.fun)
    if gap>max(1e-5,1e-7*max(1,primal)):
        raise RuntimeError('Quantile primal-dual objective gap '+str(gap))
    return beta,gap


def calendar_hac(x,y,dates,lags=3):
    beta=ols(x,y);err=y-x@beta
    days=pd.DatetimeIndex(dates).normalize()
    scores=pd.DataFrame(x*err[:,None],index=days).groupby(level=0).sum()
    scores=scores.reindex(pd.date_range(days.min(),days.max(),freq='D'),fill_value=0).to_numpy()
    meat=scores.T@scores
    for k in range(1,lags+1):
        cross=scores[k:].T@scores[:-k]
        meat+=(1-k/(lags+1))*(cross+cross.T)
    bread=np.linalg.pinv(x.T@x)
    cov=bread@meat@bread*len(x)/(len(x)-x.shape[1])
    se=np.sqrt(np.maximum(np.diag(cov),0))
    return beta,se,2*norm.sf(abs(beta/np.maximum(se,1e-15))),cov


def calendar_weights(dates,block,rep,rng):
    days=pd.DatetimeIndex(dates).normalize()
    grid=pd.date_range(days.min(),days.max(),freq='D')
    ix=grid.get_indexer(days);n=len(grid)
    counts=np.zeros((rep,n),dtype=np.int16)
    for i in range(rep):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        draw=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        counts[i]=np.bincount(draw,minlength=n)
    return counts[:,ix]


def interval(point,draws):
    v=np.asarray(draws);v=v[np.isfinite(v)]
    if len(v)==0:return dict(low95=np.nan,high95=np.nan,p_centered=np.nan,n_boot=0)
    low,high=np.quantile(v,[.025,.975])
    p=(1+np.sum(abs(v-point)>=abs(point)))/(len(v)+1)
    return dict(low95=float(low),high95=float(high),p_centered=float(p),n_boot=len(v))


def read_saved(name):
    d=pd.read_csv(HERE/name,index_col=0)
    d.index=pd.to_datetime(d.index,utc=True)
    return d
