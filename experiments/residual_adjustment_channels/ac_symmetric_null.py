"""Post-result falsification: asymmetric crossing under symmetric mean reversion."""
from ac_core import *
from ac_evaluate import inference
import ac_overshoot as x

NLOCK=HERE/'SYMMETRIC_NULL_LOCK.json';DEST=OUT/'symmetric_null'


def symmetric_probabilities(train,test,centered):
    tr=train.copy();te=test.copy()
    if centered:
        xx=(tr.e_now-tr.past_center72).to_numpy();yy=(tr.target-tr.past_center72).to_numpy()
        beta=float(np.dot(xx,yy)/np.dot(xx,xx));intercept=0.
        fitted=tr.past_center72+beta*xx;pred=te.past_center72+beta*(te.e_now-te.past_center72)
    else:
        coef=np.linalg.lstsq(np.column_stack([np.ones(len(tr)),tr.e_now]),tr.target,rcond=None)[0]
        intercept,beta=map(float,coef);fitted=intercept+beta*tr.e_now;pred=intercept+beta*te.e_now
    # Even empirical shocks: no skewness or sign-specific adjustment in the null.
    noise=((tr.target-fitted)/np.maximum(tr.e_rms72,1.)).to_numpy()
    even=np.sort(np.r_[noise,-noise]);scale=np.maximum(te.e_rms72.to_numpy(),1.)
    def prob(boundary):
        threshold=(boundary-np.asarray(pred))/scale
        below=np.searchsorted(even,threshold,side='left')/len(even)
        above=1-np.searchsorted(even,threshold,side='right')/len(even)
        return np.where(te.e_now.to_numpy()>0,below,above)
    return dict(cross=prob(np.zeros(len(te))),overshot=prob(-te.e_now.to_numpy())),dict(
        intercept=intercept,beta=beta,train_n=len(tr),noise_n=len(even),last_train_label=tr.target_time.max().isoformat())


def verify_parent():
    stamp=x.check()
    for name,digest in json.loads((x.DEST/'COMPLETE.json').read_text())['file_sha256'].items():assert sha(ROOT/name)==digest
    return stamp


def lock():
    verify_parent()
    files=[Path(__file__).resolve(),HERE/'SYMMETRIC_NULL_PROTOCOL_KO.md',HERE/'test_ac_null.py',HERE/'NULL_PRE_RUN_TESTS.txt',x.XLOCK,x.DEST/'COMPLETE.json']
    write_new(NLOCK,dict(created_utc=now(),prior_findings_seen=True,file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))


def check():
    verify_parent();d=json.loads(NLOCK.read_text())
    for name,digest in d['file_sha256'].items():assert sha(ROOT/name)==digest
    rel=str(NLOCK.relative_to(ROOT));assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==NLOCK.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True)
    return dict(lock_commit=commit,lock_sha256=sha(NLOCK))


def run():
    stamp=check();DEST.mkdir(parents=True,exist_ok=True);write_new(DEST/'STARTED.json',dict(**stamp,started_utc=now()))
    panel=old.legacy.a.panel('available_macro');frames=[];notes=[]
    for definition in DEFS:
        for cutoff in old.OUTER_MONTHS:
            fold=cutoff.strftime('%Y-%m');r=old.legacy.a.Residualizer(definition).fit(panel,cutoff)
            e,_=r.transform(panel);center=e.rolling('72h',min_periods=12,closed='left').mean().rename('past_center72')
            case=pd.read_pickle(PREP/'cases'/f'{definition}__{fold}__12.pkl.gz')
            tr=case['train'].query('abs_e>0').join(center).dropna(subset=['past_center72'])
            te=case['test'].query('abs_e>0').join(center)
            assert te.past_center72.notna().all() and tr.target_time.max()<cutoff
            # Check rolling center against a separate exact timestamp slice, every test origin.
            for t,value in te.past_center72.items():
                direct=e[(e.index>=t-pd.Timedelta(hours=72))&(e.index<t)].mean()
                assert abs(direct-value)<1e-10
            for centered in [False,True]:
                probabilities,note=symmetric_probabilities(tr,te,centered)
                name='local_center72' if centered else 'affine'
                d=te.reset_index()[['origin','past_center72']].copy()
                for k,v in probabilities.items():d['p_'+k]=v
                for k,v in dict(definition=definition,fold=fold,null=name).items():d[k]=v
                frames.append(d);notes.append(dict(definition=definition,fold=fold,null=name,**note))
    null=pd.concat(frames,ignore_index=True)
    orig=read(x.DEST/'predictions.csv.gz');orig=orig[(orig.learner=='forest40')&(orig.seed==SEEDS[0])]
    d=orig.merge(null,on=['definition','fold','origin'],validate='one_to_many')
    assert len(d)==2*len(orig) and np.isfinite(d[['p_cross','p_overshot']]).all().all()
    save(d,DEST/'predictions.csv.gz');write_new(DEST/'fit_metadata.json',notes)
    write_new(DEST/'SEALED.json',dict(**stamp,sealed_utc=now(),predictions_sha256=sha(DEST/'predictions.csv.gz')))
    print('Sealed fixed symmetric-null forecasts; rolling clocks independently checked.')


def evaluate():
    stamp=check();seal=json.loads((DEST/'SEALED.json').read_text());assert seal['predictions_sha256']==sha(DEST/'predictions.csv.gz')
    write_new(DEST/'SCORES_OPENED.json',dict(**stamp,opened_utc=now()))
    pred=read(DEST/'predictions.csv.gz');rows=[];groups=[];quality=[]
    for (definition,null),g in pred.groupby(['definition','null']):
        g=g.sort_values('origin').reset_index(drop=True);base=g[g.abs_e>5].copy()
        samples=[('full',base,b) for b in [1,5,10]]+[('nonoverlap',base[dv.nonoverlap(base)],5),('overlap',base[base.m.between(.1,.9)],5)]
        for fold in sorted(g.fold.unique()):samples.append((fold,base[base.fold==fold],5))
        for sample,d,block in samples:
            d=d.reset_index(drop=True);w=dv.day_weights(d[['origin','fold']],block)
            vv=(d.premium-d.m).to_numpy();den=w@(vv*vv)
            for quantity in ['cross','overshot']:
                error=(d[quantity]-d['p_'+quantity]).to_numpy();num=vv*error
                point=float(num.sum()/sum(vv*vv));draws=(w@num)/den
                rows.append(dict(definition=definition,null=null,sample=sample,block_days=block,n=len(d),quantity=quantity,**inference(point,draws)))
        for quantity in ['cross','overshot']:
            for sign in [0,1]:
                d=base[base.premium==sign]
                groups.append(dict(definition=definition,null=null,quantity=quantity,group='premium' if sign else 'discount',n=len(d),
                    observed=float(d[quantity].mean()),expected=float(d['p_'+quantity].mean()),past_center_mean_bp=float(d.past_center72.mean())))
            quality.append(dict(definition=definition,null=null,quantity=quantity,n=len(base),
                brier=float(np.mean((base[quantity]-base['p_'+quantity])**2)),constant_brier=float(np.mean((base[quantity]-base['constant_'+quantity])**2))))
    rows=pd.DataFrame(rows);main=rows[(rows['sample']=='full')&(rows.block_days==5)].copy();assert len(main)==12
    main['p_holm12']=dv.holm(main.p)
    for name,frame in [('contrasts',rows),('primary_tests',main),('observed_expected',pd.DataFrame(groups)),('quality',pd.DataFrame(quality))]:save(frame,DEST/(name+'.csv'))
    files=list(DEST.glob('*.csv'))+[DEST/'SCORES_OPENED.json']
    write_new(DEST/'COMPLETE.json',dict(**stamp,completed_utc=now(),post_result_falsification=True,
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print(main[['definition','null','quantity','estimate','lo','hi','p_holm12']].to_string(index=False))


if __name__=='__main__':
    {'lock':lock,'run':run,'evaluate':evaluate}[sys.argv[1]]()
