from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,time
from common import *
from statsmodels.stats.multitest import multipletests

DATA=None
WEIGHTS=None


def initialize():
    global DATA,WEIGHTS
    DATA=np.load(HERE/'economic_data.npz')
    WEIGHTS=np.load(HERE/'economic_weights.npz')


def task(job):
    key,h,q,block,start,stop=job
    x=DATA[f'x{h}'];y=DATA[f'y{h}'];weights=WEIGHTS[key]
    out=np.full((stop-start,x.shape[1],len(REFS)),np.nan)
    errors=[];maxgap=0.;maxidentity=0.
    for row,rep in enumerate(range(start,stop)):
        try:
            w=weights[rep]
            if q==0:
                yy=np.column_stack([y,y[:,0]-y[:,2],y[:,0]-y[:,3]])
                beta=ols(x,yy,w)
                err=max(np.max(abs(beta[:,0]-beta[:,2]-beta[:,5])),
                        np.max(abs(beta[:,0]-beta[:,3]-beta[:,6])))
                if err>1e-7:raise RuntimeError('OLS identity failed')
                maxidentity=max(maxidentity,err)
                out[row]=beta[:,:5]
            else:
                for j in range(5):
                    beta,gap=qr(x,y[:,j],q,w)
                    out[row,:,j]=beta
                    maxgap=max(maxgap,gap)
        except Exception as e:
            errors.append(dict(rep=rep,error=str(e)));out[row]=np.nan
    return key,start,out,errors,maxgap,maxidentity


def main():
    begin=time.time()
    data=np.load(HERE/'economic_data.npz')
    names=list(data['feature_names'])
    point={};rows=[];checks=[]
    for h in [1,6,12]:
        x=data[f'x{h}'];y=data[f'y{h}']
        for q in [0,.1,.5]:
            if q==0:
                yy=np.column_stack([y,y[:,0]-y[:,2],y[:,0]-y[:,3]])
                bet=ols(x,yy)
                checks.append(float(np.max(abs(bet[:,0]-bet[:,2]-bet[:,5]))))
                checks.append(float(np.max(abs(bet[:,0]-bet[:,3]-bet[:,6]))))
                coeff=bet[:,:5]
            else:
                result=[qr(x,y[:,j],q) for j in range(5)]
                coeff=np.column_stack([v[0] for v in result])
            point[(h,q)]=coeff
            for j,ref in enumerate(REFS):
                for k,name in enumerate(names):
                    rows.append(dict(h=h,q=q,reference=ref,term=name,estimate=coeff[k,j],n=len(y),month_FE=True))
        if h==6:
            keep=[k for k,n in enumerate(names) if not n.startswith('month_')]
            for q in [0,.1,.5]:
                xx=x[:,keep]
                co=ols(xx,y) if q==0 else np.column_stack([qr(xx,y[:,j],q)[0] for j in range(5)])
                for j,ref in enumerate(REFS):
                    for k,name in enumerate(np.asarray(names)[keep]):
                        rows.append(dict(h=h,q=q,reference=ref,term=name,estimate=co[k,j],n=len(y),month_FE=False))
    assert max(checks)<1e-7
    pd.DataFrame(rows).to_csv(HERE/'economic_point_estimates.csv',index=False)
    configs=[(h,0,5,1999) for h in [1,6,12]]
    configs += [(6,.1,5,999),(6,.5,5,499),(1,.1,5,499),(12,.1,5,499),(6,.1,1,499),(6,.1,10,499)]
    weights={};draws={};lookup={};jobs=[]
    for i,(h,q,block,B) in enumerate(configs):
        key=f'h{h}_q{int(100*q):02d}_b{block}'
        dates=pd.to_datetime(data[f'origin{h}'],utc=True)
        w=calendar_weights(dates,block,B,np.random.default_rng(SEED+100+i))
        weights[key]=w;draws[key]=np.full((B,len(names),5),np.nan)
        lookup[key]=(h,q,block,B)
        for start in range(0,B,25):
            jobs.append((key,h,q,block,start,min(B,start+25)))
    np.savez_compressed(HERE/'economic_weights.npz',**weights)
    (HERE/'ECONOMIC_RUN_SETTINGS.json').write_text(json.dumps(dict(
        protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),code_sha256=sha(Path(__file__)),
        common_sha256=sha(HERE/'common.py'),configs=configs,seed=SEED,
        workers=2,solver='HiGHS LP dual with exact primal-dual gap check',status='running'),indent=2))
    # Jobs run in two single-thread processes; identical day weights for all references.
    errors=[];maxgap=0.;maxidentity=max(checks);completed=0;last_print=time.time()
    with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
        futures=[pool.submit(task,job) for job in jobs]
        for fut in as_completed(futures):
            key,start,result,err,gap,ident=fut.result()
            draws[key][start:start+len(result)]=result
            errors.extend([dict(config=key,**v) for v in err])
            maxgap=max(maxgap,gap);maxidentity=max(maxidentity,ident)
            completed+=len(result)
            if time.time()-last_print>15:
                print('Economic bootstrap completed',completed,'of',sum(v[3] for v in configs),
                      'draws;',round(time.time()-begin,1),'seconds',flush=True)
                last_print=time.time()
    np.savez_compressed(HERE/'economic_bootstrap.npz',**draws)
    (HERE/'economic_bootstrap_errors.json').write_text(json.dumps(errors,indent=2))
    coefrows=[];differences=[];marginals=[]
    for key,(h,q,block,B) in lookup.items():
        boot=draws[key];valid=np.isfinite(boot).all(axis=(1,2))
        if valid.sum()<.95*B:raise RuntimeError('Too many invalid replicates '+key)
        for j,ref in enumerate(REFS):
            for name in TERMS:
                k=names.index(name);estimate=float(point[h,q][k,j])
                coefrows.append(dict(h=h,q=q,block_days=block,reference=ref,term=name,
                    estimate=estimate,**interval(estimate,boot[:,k,j])))
            for d in [0,1]:
                k1=names.index('ls_z');k2=names.index('down_x_ls')
                est=float(point[h,q][k1,j]+d*point[h,q][k2,j])
                marginals.append(dict(h=h,q=q,block_days=block,reference=ref,
                    effect='ls_at_down_'+str(d)+'pct',estimate=est,
                    **interval(est,boot[:,k1,j]+d*boot[:,k2,j])))
        for ref in ['BTC','ETH','median5','without_DOGE']:
            j=REFS.index(ref)
            for name in TERMS:
                k=names.index(name);est=float(point[h,q][k,0]-point[h,q][k,j])
                differences.append(dict(h=h,q=q,block_days=block,reference=ref,term=name,
                    contrast='mean5_minus_'+ref,estimate=est,
                    primary=(h==6 and q in [0,.1] and block==5),
                    **interval(est,boot[:,k,0]-boot[:,k,j])))
    dif=pd.DataFrame(differences);dif['p_holm_24']=np.nan
    keep=dif.primary
    assert keep.sum()==24
    dif.loc[keep,'p_holm_24']=multipletests(dif.loc[keep,'p_centered'],method='holm')[1]
    pd.DataFrame(coefrows).to_csv(HERE/'economic_coefficients.csv',index=False)
    dif.to_csv(HERE/'economic_contrasts.csv',index=False)
    pd.DataFrame(marginals).to_csv(HERE/'economic_marginal_effects.csv',index=False)
    (HERE/'ECONOMIC_COMPLETION.json').write_text(json.dumps(dict(
        status='complete',seconds=time.time()-begin,bootstrap_draws=completed,
        failures=len(errors),max_quantile_primal_dual_gap=maxgap,
        max_OLS_decomposition_error=maxidentity,
        description='Same-period explanatory associations; not causal leverage effects or forecast gains'),indent=2))
    print('Economic comparisons complete',round(time.time()-begin,1),'seconds',flush=True)
    print(dif.loc[keep].round(5).to_string(index=False),flush=True)


if __name__=='__main__':main()
