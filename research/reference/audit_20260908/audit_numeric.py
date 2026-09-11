"""Read-only audit of the submitted paper against the archived data and code.
Outputs go to this audit directory. No original research files are modified.
Run: OPENBLAS_NUM_THREADS=1 python3 audit_numeric.py
"""
from pathlib import Path
import importlib.util, json, re, zipfile, hashlib, platform
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import hstack, eye, csr_matrix
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / 'stablecoin_v4'
spec = importlib.util.spec_from_file_location('paper', ROOT/'reproduction/stablecoin_paper_pipeline.py')
paper = importlib.util.module_from_spec(spec); spec.loader.exec_module(paper)
df, meta = paper.load_and_build()
RESULTS = {'rebuild': meta, 'n':len(df), 'python':platform.python_version()}

def exact_qr(X, y, q):
    """Solve the quantile check-loss LP after affine scaling; return original units."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    xm = X[:,1:].mean(0); xs = X[:,1:].std(0,ddof=1)
    ym = y.mean(); ys = y.std(ddof=1)
    Z = np.column_stack([np.ones(len(y)), (X[:,1:]-xm)/xs]); zy=(y-ym)/ys
    n,k=Z.shape
    opt = linprog(np.r_[np.zeros(k),np.full(n,q),np.full(n,1-q)],
                  A_eq=hstack([csr_matrix(Z),eye(n),-eye(n)],format='csr'),b_eq=zy,
                  bounds=[(None,None)]*k+[(0,None)]*(2*n),method='highs',
                  options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not opt.success: raise RuntimeError(opt.message)
    b=opt.x[:k]; slopes=b[1:]*ys/xs
    return np.r_[ym+ys*b[0]-xm@slopes,slopes]

def loss(X,y,b,q):
    r=y-X@b
    return float(np.where(r>=0,q*r,(q-1)*r).sum())

def design(d,ycol,xcols):
    return paper.add_const(d[xcols].to_numpy()),d[ycol].to_numpy()

def compare(name,d,ycol,xc,qs,targets,standardized=False):
    out=[]
    d=d[[ycol]+xc].dropna()
    if standardized: d=(d-d.mean())/d.std()
    X,y=design(d,ycol,xc)
    for q in qs:
        orig=paper.quantile_fit(X,y,q); exact=exact_qr(X,y,q)
        # Independent established solver, on scaled data to avoid tiny predictor units.
        z=(d-d.mean())/d.std(); zX,zy=design(z,ycol,xc)
        sr=sm.QuantReg(zy,zX).fit(q=q,max_iter=20000,p_tol=1e-10)
        sb=sr.params[1:]*d[ycol].std()/d[xc].std().to_numpy()
        for t in targets:
            j=xc.index(t)+1
            out.append(dict(model=name,q=q,n=len(d),target=t,archived_algorithm=orig[j],
                       exact_lp=exact[j],statsmodels_scaled=sb[j-1],
                       loss_original=loss(X,y,orig,q),loss_exact=loss(X,y,exact,q),
                       relative_loss_gap=(loss(X,y,orig,q)/loss(X,y,exact,q)-1),
                       standardized=standardized))
    return out

def ols_resid(y,x):
    X=np.column_stack([np.ones(len(y)),np.asarray(x)])
    return np.asarray(y)-X@np.linalg.lstsq(X,np.asarray(y),rcond=None)[0]

def main():
    # All 120 numerical cells in submitted Table 1.
    cols=['RESIDUAL_FINAL','USDT_KP','MKT_KP_EQ','BTC_DOWNSIDE_24','BTC_VOL_24',
          'ACCOUNT_LS','TOPTRADER_ACCOUNT_LS','OI','FUNDING','DXY_ret','VIX','USDKRW_ret']
    units=[1e4,100,100,1e4,100,1,1,.001,1e4,100,1,100]
    tab=json.loads((ROOT/'submitted_tables.json').read_text())[0]
    checks=[]
    for row,col,unit in zip(tab[1:],cols,units):
        s=df[col].dropna()*unit
        vals=[len(s),s.mean(),s.std(),s.min(),s.quantile(.25),s.median(),s.quantile(.75),s.max(),s.skew(),s.kurt()]
        for label,raw,calc in zip(['N','mean','std','min','q25','median','q75','max','skew','excess_kurtosis'],row[1:],vals):
            raw=re.sub(r'\s+','',raw); value=float(raw)
            decimals=len(raw.split('.')[1]) if '.' in raw else 0
            checks.append(dict(variable=col,stat=label,submitted=raw,recomputed=calc,
                               matches_rounding=abs(calc-value)<=.500001*10**(-decimals)))
    pd.DataFrame(checks).to_csv(ROOT/'table1_cell_audit.csv',index=False)
    RESULTS['table1_cells']=len(checks)
    RESULTS['table1_mismatches']=[x for x in checks if not x['matches_rounding']]
    # Actual elapsed time represented by row shifts.
    ts=df.index.to_series(); horizon=[]
    for h in [1,2,3,6,12]:
        elapsed=(ts.shift(-h)-ts).dt.total_seconds()/3600
        shock_elapsed=(ts.shift(-h)-ts.shift(1)).dt.total_seconds()/3600
        a=elapsed.dropna(); b=shock_elapsed.dropna()
        horizon.append(dict(h=h,n=len(a),min_hours=a.min(),median_hours=a.median(),max_hours=a.max(),
                            exact_h_hours=int((a==h).sum()),shock_to_outcome_median=b.median()))
    pd.DataFrame(horizon).to_csv(ROOT/'horizon_elapsed_time.csv',index=False)
    RESULTS['horizons']=horizon
    RESULTS['hours_utc']={str(k):int(v) for k,v in pd.Series(df.index.hour).value_counts().sort_index().items()}
    RESULTS['n_days']=df.index.normalize().nunique()
    RESULTS['daily_counts']=df.groupby(df.index.normalize()).size().value_counts().to_dict()
    RESULTS['lag_gap_counts']=(ts.diff().dt.total_seconds()/3600).value_counts().to_dict()
    RESULTS['return_rebuild_errors']={c:float((df[c+'_ret']-np.log(df[c]).diff()).abs().max()) for c in ['DXY','USDKRW','VIX']}
    RESULTS['lag_rebuild_errors']={c:float((df[c+'_lag1']-df[c].shift(1)).abs().max()) for c in ['BTC_DOWNSIDE_24','ACCOUNT_LS','TOPTRADER_ACCOUNT_LS']}
    # Two-stage projection and quote-currency checks.
    q=df['USDT_BINANCE_CLOSE']; u=df['USDT_UPBIT_CLOSE']; fx=df['USDKRW']
    corrected_kp=u*q/fx-1
    corrected_mkt=(df['MKT_KP_EQ']+1)*q-1
    currency_rf=ols_resid(ols_resid(corrected_kp,corrected_mkt),df['DEPEG_GLOBAL'])
    joint_rf=ols_resid(df['USDT_KP'],df[['MKT_KP_EQ','DEPEG_GLOBAL']])
    RESULTS['residual_checks']={
        'stored_usdt_kp_formula_error':float((df.USDT_KP-(u/(fx*q)-1)).abs().max()),
        'depeg_formula_error':float((df.DEPEG_GLOBAL-(q-1).abs()).abs().max()),
        'pair_price_min':q.min(),'pair_price_max':q.max(),
        'usdt_kp_quote_correction_max_bp':float((corrected_kp-df.USDT_KP).abs().max()*1e4),
        'currency_corrected_residual_std_bp':float(np.std(currency_rf,ddof=1)*1e4),
        'currency_corrected_residual_corr':float(np.corrcoef(currency_rf,df.RESIDUAL_FINAL)[0,1]),
        'currency_corrected_residual_difference_max_bp':float(np.max(np.abs(currency_rf-df.RESIDUAL_FINAL))*1e4),
        'sequential_residual_corr_market':float(df.RESIDUAL_FINAL.corr(df.MKT_KP_EQ)),
        'sequential_vs_joint_residual_max_bp':float(np.max(np.abs(joint_rf-df.RESIDUAL_FINAL))*1e4),
        'market_depeg_corr':float(df.MKT_KP_EQ.corr(df.DEPEG_GLOBAL)),
        'downside_vol_corr':float(df.BTC_DOWNSIDE_24.corr(df.BTC_VOL_24)),
    }
    comparisons=compare('main',df,'RESIDUAL_FINAL',paper.BX,[.1,.25,.5,.75,.9],['BTC_DOWNSIDE_24_lag1'])
    bc=['RESIDUAL_FINAL_lag1','DXY_ret_lag1','USDKRW_ret_lag1','BTC_DOWNSIDE_24_lag1']
    glob=['BTC_VOL_24_lag1','BTC_RET_lag1','VIX_ret_lag1']
    for name,xc in [('M0',bc),('M1',bc+glob),('M2',bc+glob+['ACCOUNT_LS_lag1'])]:
        targets=[x for x in ['BTC_DOWNSIDE_24_lag1','BTC_VOL_24_lag1','ACCOUNT_LS_lag1'] if x in xc]
        comparisons+=compare(name,df,'RESIDUAL_FINAL',xc,[.1,.25,.5],targets,True)
    for h in [1,2,3,6,12]:
        d=df.copy();d['YL']=d.RESIDUAL_FINAL.shift(-h)
        comparisons+=compare('LP_h'+str(h),d,'YL',paper.BX,[.1,.5,.9],['BTC_DOWNSIDE_24_lag1'])
    for name,res in [('currency_corrected',currency_rf),('joint_projection',joint_rf)]:
        d=df.copy();d['RESIDUAL_FINAL']=res;d['RESIDUAL_FINAL_lag1']=d.RESIDUAL_FINAL.shift(1)
        comparisons+=compare(name+'_main',d,'RESIDUAL_FINAL',paper.BX,[.1,.5],['BTC_DOWNSIDE_24_lag1'])
        comparisons+=compare(name+'_M2',d,'RESIDUAL_FINAL',bc+glob+['ACCOUNT_LS_lag1'],[.1],['ACCOUNT_LS_lag1'],True)
    pd.DataFrame(comparisons).to_csv(ROOT/'solver_comparison.csv',index=False)
    # Scale invariance and fixed iteration sensitivity on exactly the same sample.
    d=df[['RESIDUAL_FINAL']+paper.BX].dropna(); X,y=design(d,'RESIDUAL_FINAL',paper.BX)
    sensitivity=[]
    for q0 in [.1,.5]:
        for it in [30,40,60,70,80,300,3000]:
            b=paper.quantile_fit(X,y,q0,n_iter=it)
            sensitivity.append(dict(q=q0,iterations=it,downside=b[2],loss=loss(X,y,b,q0)))
    pd.DataFrame(sensitivity).to_csv(ROOT/'iteration_sensitivity.csv',index=False)
    # Unit-price lattice and claimed tick, without inferring official rules from it.
    lattice=[]
    for name,s in df.groupby(df.index.to_period('M')):
        p=s.USDT_UPBIT_CLOSE
        lattice.append(dict(month=str(name),n=len(p),non_integer_prices=int((p%1!=0).sum()),
                            min_positive_observed_price_change=p.diff().abs().replace(0,np.nan).min()))
    pd.DataFrame(lattice).to_csv(ROOT/'observed_price_lattice.csv',index=False)
    # Daily mean dependence (a one-day bootstrap does not preserve between-day dependence).
    daily=df.groupby(df.index.normalize())[['RESIDUAL_FINAL','BTC_DOWNSIDE_24','ACCOUNT_LS']].mean()
    RESULTS['daily_mean_lag1_autocorrelation']={c:daily[c].autocorr() for c in daily}
    (ROOT/'numeric_summary.json').write_text(json.dumps(RESULTS,ensure_ascii=False,indent=2,default=float))
    print(json.dumps(RESULTS,ensure_ascii=False,indent=2,default=float))
    print(pd.DataFrame(comparisons).query("model in ['main','M2','currency_corrected_main','currency_corrected_M2']").to_string(index=False))

if __name__=='__main__': main()
