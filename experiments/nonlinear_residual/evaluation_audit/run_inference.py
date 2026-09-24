"""Refit generated residuals in calendar-block bootstrap; fixed prior choices."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import traceback
import time
from common import *

B=199
BOOT=HERE/'inference'
PANELS={}
PAIRS={}
CASES={}
CHOICES={}
WEIGHTS={}
GRID={c['candidate_id']:c for c in models.candidates()}
CORE=study.SPECS['core']['M2']
UNIQUE=[s for s in study.pilot.SCENARIOS if s!='clock_FXUTC_USNY_quote']


def ridge_refit(model):
    t=model.train
    if model.family.startswith('ae_'):
        members=[]
        for net,center,scale,_ in model.members:
            z=(models.latent(net,model.standardize(t))-center)/scale
            head=Ridge(alpha=1.).fit(z,t.y)
            members.append((net,center,scale,np.r_[head.intercept_,head.coef_]))
        model.members=members
    elif model.family.startswith('pca_'):
        z=model.standardize(t)@model.loading[:,:model.rank]
        center,scale=z.mean(axis=0),z.std(axis=0,ddof=1)
        head=Ridge(alpha=1.).fit((z-center)/scale,t.y)
        model.first=np.r_[head.intercept_-head.coef_@(center/scale),head.coef_/scale]
    model.second=np.linalg.lstsq(np.column_stack([np.ones(len(t)),t.g]),t.y-model.predict_first(t),rcond=None)[0]
    return model


def make_cases(frame):
    frame=frame.loc[frame.month.isin(MONTHS)].dropna(subset=CORE+['target']).copy()
    matched=set.intersection(*[set(frame.loc[frame.h==h,'time']) for h in [1,6,12]])
    result=[]
    for method in METHODS:
        for sample,h,q in [('all',6,.1),('all',6,.5)]+[('matched',h,.1) for h in [1,6,12]]:
            f=frame.loc[(frame.method==method)&(frame.h==h)].copy()
            if sample=='matched': f=f.loc[f.time.isin(matched)].copy()
            f=f.sort_values('time').reset_index(drop=True)
            assert study.enough(f,CORE)
            # Fixed original-sample scale allows paired coefficient contrasts.
            sd=f[CORE].std(ddof=1).to_numpy()
            result.append(dict(method=method,sample=sample,h=h,q=q,frame=f,sd=sd))
    return result


def estimates(cases, residuals=None, weights=None):
    rows=[]
    for c in cases:
        f=c['frame'].copy()
        if residuals is not None:
            for month in MONTHS:
                mask=f.month==month;e=residuals[(month,c['method'])]
                f.loc[mask,'e_origin']=e.reindex(f.loc[mask,'time']).to_numpy()
                f.loc[mask,'target']=e.reindex(f.loc[mask,'target_time']).to_numpy()
        w=None if weights is None else weights.reindex(f.time.dt.normalize()).to_numpy()
        assert f[CORE+['target']].notna().all().all()
        beta=study.fit_quantile(f,CORE,c['q'],w)
        for j,term in enumerate(CORE):
            rows.append(dict(method=c['method'],sample=c['sample'],h=c['h'],q=c['q'],term=term,n=len(f),
                effect=float(beta[j+1]*c['sd'][j]),coefficient=float(beta[j+1])))
    return pd.DataFrame(rows)


def replicate(scenario, rep, block, full):
    started=time.monotonic()
    calendar, draws=WEIGHTS[(scenario,block)]
    weights=pd.Series(draws[rep],index=calendar)
    residuals={};logs=[]
    if full:
        data=PANELS[scenario].dropna(subset=['y','m','g']+study.KCOLS)
        for month in MONTHS:
            cutoff=pd.Timestamp(month+'-01',tz='UTC');end=cutoff+pd.offsets.MonthBegin(1)+pd.Timedelta(hours=12)
            train=data.loc[data.index<cutoff]
            count=weights.reindex(train.index.normalize()).to_numpy()
            boot=train.iloc[np.repeat(np.arange(len(train)),count)]
            assert boot.index.max()<cutoff
            test=data.loc[(data.index>=cutoff)&(data.index<end)]
            for method in METHODS:
                if method=='joint_ols':
                    coef=np.linalg.lstsq(np.column_stack([np.ones(len(boot)),boot[['m','g']]]),boot.y,rcond=None)[0]
                    fitted=np.column_stack([np.ones(len(test)),test[['m','g']]])@coef
                else:
                    cid=CHOICES[scenario][(month,method)]
                    model=ridge_refit(models.Model(GRID[cid]).fit(boot,cutoff))
                    fitted=model.predict(test)
                    if method=='ae_selected':
                        logs.append(dict(month=month,candidate_id=cid,train_n=len(boot),train_last=str(boot.index.max()),
                            warnings=model.log['warnings'],members=model.log['members']))
                residuals[(month,method)]=pd.Series(test.y.to_numpy()-fitted,index=test.index)
    result=estimates(CASES[scenario],residuals if full else None,weights)
    result['replicate']=rep;result['block_days']=block;result['refit_first_stage']=full
    out=BOOT/scenario/('refit_7d' if full else 'fixed_3d')
    save(result,out/('%03d.csv'%rep))
    if full:
        (out/('%03d.json'%rep)).write_text(json.dumps(dict(scenario=scenario,replicate=rep,
            elapsed_seconds=time.monotonic()-started,fits=logs),indent=2)+'\n')
    return time.monotonic()-started


def contrasts(frame):
    rows=[]
    for (method,term),d in frame.groupby(['method','term']):
        def get(sample,h,q):return float(d.loc[(d['sample']==sample)&(d.h==h)&(d.q==q),'effect'].iloc[0])
        for name,value in [('q10_minus_q50_h6',get('all',6,.1)-get('all',6,.5)),
                           ('h6_minus_h1_q10',get('matched',6,.1)-get('matched',1,.1)),
                           ('h12_minus_h1_q10',get('matched',12,.1)-get('matched',1,.1))]:
            rows.append(dict(method=method,term=term,contrast=name,effect=value))
    return pd.DataFrame(rows)


def aggregate():
    allci,allcontrast=[] ,[]
    for scenario in UNIQUE:
        point=pd.read_csv(BOOT/scenario/'point_estimates.csv')
        cp=contrasts(point)
        for block,full,folder in [(7,True,'refit_7d'),(3,False,'fixed_3d')]:
            files=sorted((BOOT/scenario/folder).glob('*.csv'))
            if len(files)!=B: raise RuntimeError((scenario,folder,len(files)))
            draws=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
            cdraw=pd.concat([contrasts(g).assign(replicate=r) for r,g in draws.groupby('replicate')])
            keys=['method','sample','h','q','term']
            ci=draws.groupby(keys).effect.agg(ci_low=lambda x:x.quantile(.025),ci_high=lambda x:x.quantile(.975),successful='size').reset_index()
            ci=point.merge(ci,on=keys,validate='one_to_one')
            allci.append(ci.assign(scenario=scenario,block_days=block,refit_first_stage=full))
            keys=['method','term','contrast']
            ci=cdraw.groupby(keys).effect.agg(ci_low=lambda x:x.quantile(.025),ci_high=lambda x:x.quantile(.975),successful='size').reset_index()
            allcontrast.append(cp.merge(ci,on=keys,validate='one_to_one').assign(scenario=scenario,block_days=block,refit_first_stage=full))
    save(pd.concat(allci),BOOT/'coefficient_intervals.csv')
    save(pd.concat(allcontrast),BOOT/'contrast_intervals.csv')


def main():
    args=argparse.ArgumentParser();args.add_argument('--workers',type=int,default=10)
    args.add_argument('--smoke',action='store_true');args=args.parse_args()
    started=time.monotonic();BOOT.mkdir(exist_ok=True)
    for scenario in UNIQUE:
        panel,_,pairs=load(scenario)
        PANELS[scenario]=panel;PAIRS[scenario]=pairs;CASES[scenario]=make_cases(pairs)
        CHOICES[scenario]={(r['month'],r['method']):r['candidate_id'] for r in json.loads((SOURCE/scenario/'selection.json').read_text())}
        for block in [3,7]:
            WEIGHTS[(scenario,block)]=study.calendar_counts(panel.index,block,np.random.default_rng(SEED+block),B)
        save(estimates(CASES[scenario]),BOOT/scenario/'point_estimates.csv')
    if args.smoke:
        print('One complete nuisance-refit draw seconds:',replicate(UNIQUE[0],0,7,True),flush=True)
        return
    jobs=[]
    for s in UNIQUE:
        for block,full,folder in [(7,True,'refit_7d'),(3,False,'fixed_3d')]:
            for r in range(B):
                if not (BOOT/s/folder/('%03d.csv'%r)).exists():jobs.append((s,r,block,full))
    done=0;failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(replicate,*job):job for job in jobs}
        for f in as_completed(futures):
            try:f.result()
            except Exception:
                failures.append(dict(job=futures[f],traceback=traceback.format_exc()))
            done+=1
            if done%40==0:print('Bootstrap %d/%d; failures=%d; %.1fs'%(done,len(jobs),len(failures),time.monotonic()-started),flush=True)
    (BOOT/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    if failures: raise RuntimeError('Bootstrap failures retained in failures.json')
    aggregate()
    (BOOT/'MANIFEST.json').write_text(json.dumps(dict(repetitions=B,full_refit_block_days=7,
        conditional_block_days=3,unique_core_scenarios=UNIQUE,duplicate_core_scenario={'clock_FXUTC_USNY_quote':'clock_UTC_quote'},
        elapsed_seconds=time.monotonic()-started,source_manifest_sha256=study.pilot.sha(SOURCE/'RUN_MANIFEST.json'),
        input_sha256={p.name:study.pilot.sha(p) for p in [HERE/'PROTOCOL_KO.md',HERE/'common.py',Path(__file__).resolve()]},
        output_sha256={str(p.relative_to(BOOT)):study.pilot.sha(p) for p in BOOT.rglob('*') if p.is_file()}),indent=2)+'\n')
    print('A5 inference complete',flush=True)


if __name__=='__main__':main()
