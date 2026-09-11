from common import *
import time

PRE_END=pd.Timestamp('2025-07-30',tz='UTC')
POST_START=pd.Timestamp('2025-08-02',tz='UTC')
POLICY=pd.Timestamp('2025-07-31',tz='UTC')
B=1999


def design(d,ref,spec,pretrend=False):
    z=pd.DataFrame(index=d.index)
    z['const']=1.
    if not pretrend:z['post']=(d.index>=POST_START).astype(float)
    if pretrend or spec=='controlled_trend':
        z['time_weeks']=(d.index-POLICY).total_seconds()/(7*86400)
    if spec!='unadjusted':
        controls=['ret_DOGE','ret_'+ref,'abs_ret_DOGE','abs_ret_'+ref,
                  'btc_rv','volume_DOGE','volume_'+ref]
        for c in controls:
            z[c]=(d[c]-d[c].mean())/d[c].std(ddof=1)
        z['weekend']=d.weekend
    elif pretrend:z['weekend']=d.weekend
    return z


def fit_boot(x,y,w,term):
    result=np.full(len(w),np.nan);failures=[]
    for i,ww in enumerate(w):
        try:result[i]=ols(x,y,ww)[term]
        except Exception as e:failures.append(dict(rep=i,error=str(e)))
    if len(failures)>.05*len(w):raise RuntimeError('More than 5% failed policy resamples')
    return result,failures


def windows(d,width,date=POLICY):
    preend=date-pd.Timedelta(days=1)
    poststart=date+pd.Timedelta(days=2)
    return d.loc[(d.index>=preend-pd.Timedelta(days=width))&(d.index<preend)],d.loc[(d.index>=poststart)&(d.index<poststart+pd.Timedelta(days=width))]


def main():
    started=time.time();d=read_saved('policy_daily.csv')
    output=[];preoutput=[];allboot={};allweights={};allerrors=[];alldesign=[]
    pre=d.loc[(d.index>=pd.Timestamp('2025-06-02',tz='UTC'))&(d.index<PRE_END)]
    wpre=calendar_weights(pre.index,5,B,np.random.default_rng(SEED+200))
    allweights['pretrend']=wpre
    weekly=pre.copy()
    weekly['week_before']=np.floor((pre.index-PRE_END).days/7).astype(int)
    weekly.groupby('week_before').agg({**{c:['mean','count'] for c in ['contrast_ETH','contrast_XRP','contrast_SOL']},'n_hours':'sum'}).to_csv(HERE/'pretrend_weekly.csv')
    for ref in ['ETH','XRP','SOL']:
        for spec in ['unadjusted','controlled']:
            z=design(pre,ref,spec,pretrend=True);x=z.to_numpy();y=pre['contrast_'+ref].to_numpy()
            k=z.columns.get_loc('time_weeks');beta,se,p,cov=calendar_hac(x,y,pre.index)
            draws,errors=fit_boot(x,y,wpre,k);key='pre_'+ref+'_'+spec
            allboot[key]=draws;allerrors.extend([dict(config=key,**e) for e in errors])
            ci=interval(beta[k],draws)
            preoutput.append(dict(reference=ref,spec=spec,n=len(pre),p=x.shape[1],
                estimate_bp_per_week=beta[k],hac_se=se[k],hac_p=p[k],
                hac_low95=beta[k]-1.95996398454*se[k],hac_high95=beta[k]+1.95996398454*se[k],
                equivalent_within_1bp_week=(ci['low95']>-1 and ci['high95']<1),
                condition_number=np.linalg.cond(x),**ci))
            zz=z.copy();zz['outcome']=y;zz['config']=key;alldesign.append(zz)
    pd.DataFrame(preoutput).to_csv(HERE/'pretrends.csv',index=False)
    for width in [14,28,42]:
        before,after=windows(d,width);sample=pd.concat([before,after])
        rng=np.random.default_rng(SEED+300+width)
        w=np.column_stack([calendar_weights(before.index,5,B,rng),calendar_weights(after.index,5,B,rng)])
        allweights['policy_w'+str(width)]=w
        for ref in ['ETH','XRP','SOL']:
            for spec in ['unadjusted','controlled','controlled_trend']:
                z=design(sample,ref,spec);x=z.to_numpy();y=sample['contrast_'+ref].to_numpy()
                assert np.linalg.matrix_rank(x)==x.shape[1]
                k=z.columns.get_loc('post');beta,se,p,cov=calendar_hac(x,y,sample.index)
                key='w'+str(width)+'_'+ref+'_'+spec
                draws,errors=fit_boot(x,y,w,k)
                allboot[key]=draws;allerrors.extend([dict(config=key,**e) for e in errors])
                output.append(dict(window_days=width,reference=ref,spec=spec,
                    primary=(width==28 and ref=='ETH' and spec=='controlled'),
                    n_before=len(before),n_after=len(after),p=x.shape[1],
                    mean_before=before['contrast_'+ref].mean(),mean_after=after['contrast_'+ref].mean(),
                    estimate=beta[k],hac_se=se[k],hac_p=p[k],
                    hac_low95=beta[k]-1.95996398454*se[k],hac_high95=beta[k]+1.95996398454*se[k],
                    condition_number=np.linalg.cond(x),**interval(beta[k],draws)))
                zz=z.copy();zz['outcome']=y;zz['config']=key;alldesign.append(zz)
        print('Policy window',width,'complete',round(time.time()-started,1),'seconds',flush=True)
    moving=[]
    for width in [14,28]:
        for candidate in pd.date_range(d.index.min()+pd.Timedelta(days=width+1),d.index.max()-pd.Timedelta(days=width+1),freq='D'):
            start=candidate-pd.Timedelta(days=width+1);end=candidate+pd.Timedelta(days=width+2)
            if not (end<=PRE_END or start>=POST_START):continue
            before,after=windows(d,width,candidate)
            if min(len(before),len(after))<.9*width:continue
            for ref in ['ETH','XRP','SOL']:
                moving.append(dict(date=str(candidate),window_days=width,reference=ref,
                    n_before=len(before),n_after=len(after),
                    estimate=after['contrast_'+ref].mean()-before['contrast_'+ref].mean(),
                    regime='before' if end<=PRE_END else 'after'))
    pd.DataFrame(moving).to_csv(HERE/'moving_window_contrasts.csv',index=False)
    pd.DataFrame(output).to_csv(HERE/'policy_regressions.csv',index=False)
    pd.concat(alldesign).to_csv(HERE/'policy_designs.csv.gz',compression='gzip')
    np.savez_compressed(HERE/'policy_bootstrap.npz',**allboot)
    np.savez_compressed(HERE/'policy_weights.npz',**allweights)
    (HERE/'policy_bootstrap_errors.json').write_text(json.dumps(allerrors,indent=2))
    (HERE/'POLICY_COMPLETION.json').write_text(json.dumps(dict(status='complete',
        code_sha256=sha(Path(__file__)),common_sha256=sha(HERE/'common.py'),protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),
        seconds=time.time()-started,seed=SEED,B=B,bootstrap_failures=len(allerrors),
        primary='28-day ETH comparison, market controlled, no local trend',
        uncertainty='3-calendar-day HAC; 5-day circular pairs blocks separately within pre and post',
        causal_limit='One strongly exposed asset; contemporaneous controls and pretrend diagnostics cannot identify an isolated tick-policy causal effect'),indent=2))
    print(pd.DataFrame(preoutput).round(4).to_string(index=False),flush=True)
    print(pd.DataFrame(output).loc[lambda x:(x.reference=='ETH')].round(4).to_string(index=False),flush=True)


if __name__=='__main__':main()
