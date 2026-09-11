from shared import *
import importlib.util
FIRST=HERE.parent/'option1_tick_learning_20260910';PREVIOUS=HERE.parent/'option1_model_followup_20260910'
spec=importlib.util.spec_from_file_location('old_analysis',FIRST/'analyze.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
calibrate=old.calibrate;quantities=old.quantities;Resample=old.Resample;interval=old.interval
from statsmodels.stats.multitest import multipletests

def contrast_definitions():
    result=[];interactions={}
    for f in ['tcn','gru']:
        tm=f+'_tick_mean_ensemble_cal';tx=f+'_tick_max_ensemble_cal';sm=f+'_shuffled_mean_ensemble_cal';sx=f+'_shuffled_max_ensemble_cal'
        definitions=[('training_main',{tx:.5,tm:-.5,sx:.5,sm:-.5}),('financial_main',{tx:.5,sx:-.5,tm:.5,sm:-.5}),
            ('interaction',{tx:1,tm:-1,sx:-1,sm:1}),('max_minus_mean_tick',{tx:1,tm:-1}),('max_minus_mean_shuffled',{sx:1,sm:-1}),
            ('tick_minus_shuffled_mean',{tm:1,sm:-1}),('tick_minus_shuffled_max',{tx:1,sx:-1})]
        for term,w in definitions:result.append(dict(name=f+'_'+term,family=f,term=term,kind='factorial',weights=w))
        interactions[f]=dict(definitions)['interaction']
    result.append(dict(name='gru_minus_tcn_interaction',family='cross',term='interaction_difference',kind='factorial',weights={**interactions['gru'],**{k:-v for k,v in interactions['tcn'].items()}}))
    for f in ['tcn','gru']:
        a=f+'_tick_max_ensemble_cal'
        for label,b in [('clean',f+'_clean_ensemble_cal'),('qr','exact_qr_cal')]:result.append(dict(name=f+'_tick_max_vs_'+label,family=f,term=label,kind='utility',weights={a:1,b:-1}))
    assert len(result)==19 and all(abs(sum(r['weights'].values()))<1e-10 for r in result)
    return result

def main():
    assert json.loads((HERE/'TRAIN_COMPLETION.json').read_text())['status']=='complete';assert_frozen()
    new=np.load(HERE/'neural_predictions.npz');dates=pd.to_datetime(new['origin_ns'],utc=True);target=new['target'];use=dates>=pd.Timestamp('2026-01-01',tz='UTC');td=dates[use];y=target[use];rs=Resample(td)
    raw={};offset_archive={};newnames=[]
    for directory in [FIRST,PREVIOUS,HERE]:
        d=np.load(directory/'neural_predictions.npz');np.testing.assert_array_equal(d['origin_ns'],new['origin_ns']);np.testing.assert_array_equal(d['target'],target)
        offsets=None if directory==HERE else np.load(directory/'calibration_offsets.npz')
        prefixes=sorted({k.rsplit('_',1)[0] for k in d.files if k.endswith(tuple(str(s) for s in SEEDS))})
        for prefix in prefixes:
            names=[prefix+'_'+str(s) for s in SEEDS]
            for name in names:raw[name]=d[name].astype(float)
            raw[prefix+'_ensemble']=np.mean([raw[k] for k in names],axis=0)
            for name in names+[prefix+'_ensemble']:
                if offsets is None:newnames.append(name)
                else:offset_archive[name]=offsets[name]
    base=np.load(FIRST/'baseline_predictions.npz');offbase=np.load(FIRST/'calibration_offsets.npz')
    for name in ['exact_qr','lgbm']:raw[name]=base[name];offset_archive[name]=offbase[name]
    scores=[];monthly=[];metrics={};preds={};offsets={};calcheck=[]
    for name,p in raw.items():
        if name in newnames:
            cp,offset,ncal,last=calibrate(dates,target,p);offsets[name]=offset
            calcheck.append(dict(model=name,min_ncal=int(ncal[use].min()),strictly_past=bool((last[use]<td.asi8).all())))
        else:cp=np.sort(p+offset_archive[name][:,None,:],axis=-1)
        primary=name.endswith('_ensemble') or name in ['exact_qr','lgbm']
        for stage,pp in [('raw',p),('cal',cp)]:
            key=name+'_'+stage;v=pp[use];q=quantities(y,v);q['stress_excess']=q['worst_tail_h1']-q['clean_tail'];metrics[key]=q;preds[key]=v.astype(np.float32)
            scores.append(dict(model=key,primary_model=primary,new_model=name in newnames,n=len(y),days=len(rs.days),q10_prediction_sd=v[:,0,0].std(),q90_prediction_sd=v[:,0,2].std(),**{k:float(x.mean()) for k,x in q.items()}))
            for month in sorted(set(td.strftime('%Y-%m'))):
                ix=td.strftime('%Y-%m')==month;monthly.append(dict(model=key,month=month,n=int(ix.sum()),**{k:float(v[ix].mean()) for k,v in q.items()}))
    score=pd.DataFrame(scores);score.to_csv(HERE/'all_scores.csv',index=False);pd.DataFrame(monthly).to_csv(HERE/'monthly_scores.csv',index=False)
    np.savez_compressed(HERE/'evaluated_predictions.npz',origin_ns=td.asi8,target=y,scenario_names=new['scenario_names'],**preds)
    np.savez_compressed(HERE/'calibration_offsets.npz',origin_ns=dates.asi8,**offsets);pd.DataFrame(calcheck).to_csv(HERE/'calibration_checks.csv',index=False)
    prior=pd.read_csv(PREVIOUS/'all_scores.csv').set_index('model');now=score.set_index('model')
    for c in ['clean_tail','worst_tail_h1','sensitivity_h1','coverage80','width80']:np.testing.assert_allclose(now.loc[prior.index,c],prior[c],atol=1e-12)
    np.testing.assert_array_equal(rs.w,np.load(FIRST/'inference_weights.npz')['weights'])
    definitions=contrast_definitions();stored=json.loads((HERE/'contrasts.json').read_text());assert definitions==stored
    rows=[];draws={};aux=[];seedrows=[];monthrows=[]
    for r in definitions:
        for stage in ['cal','raw']:
            weights={k if stage=='cal' else k[:-3]+'raw':v for k,v in r['weights'].items()}
            delta=sum(w*metrics[k]['worst_tail_h1'] for k,w in weights.items());point=float(delta.mean());boot=rs.means(delta)
            row=dict(name=r['name'],family=r['family'],term=r['term'],kind=r['kind'],stage=stage,difference=point,**interval(point,boot));rows.append(row);draws[r['name']+'_'+stage]=boot
            for metric in ['clean_tail','stress_excess','sensitivity_h1']:
                v=sum(w*metrics[k][metric] for k,w in weights.items());bb=rs.means(v)
                aux.append(dict(name=r['name'],stage=stage,metric=metric,difference=float(v.mean()),low95=float(np.quantile(bb,.025)),high95=float(np.quantile(bb,.975))))
            for month in sorted(set(td.strftime('%Y-%m'))):
                ix=td.strftime('%Y-%m')==month;monthrows.append(dict(name=r['name'],stage=stage,month=month,difference=float(delta[ix].mean())))
            if all('_ensemble_' in k for k in weights):
                for seed in SEEDS:
                    dd=sum(w*metrics[k.replace('_ensemble_',f'_{seed}_')]['worst_tail_h1'] for k,w in weights.items())
                    seedrows.append(dict(name=r['name'],stage=stage,seed=seed,difference=float(dd.mean())))
    tests=pd.DataFrame(rows);cal=tests.stage=='cal';tests['p_holm_19']=np.nan;tests.loc[cal,'p_holm_19']=multipletests(tests.loc[cal,'p_centered'],method='holm')[1]
    oldp=np.r_[pd.read_csv(FIRST/'PRIMARY_DECISION.csv').p_centered,pd.read_csv(PREVIOUS/'PRIMARY_DECISION.csv').p_centered]
    tests['p_holm_35']=np.nan;tests.loc[cal,'p_holm_35']=multipletests(np.r_[oldp,tests.loc[cal,'p_centered']],method='holm')[1][16:]
    tests.to_csv(HERE/'factorial_contrasts.csv',index=False);pd.DataFrame(aux).to_csv(HERE/'secondary_contrasts.csv',index=False)
    pd.DataFrame(seedrows).to_csv(HERE/'seed_contrasts.csv',index=False);pd.DataFrame(monthrows).to_csv(HERE/'monthly_contrasts.csv',index=False)
    np.savez_compressed(HERE/'contrast_bootstrap.npz',**draws);np.savez_compressed(HERE/'inference_weights.npz',weights=rs.w.astype(np.int16),days=rs.days.asi8)
    utility=[]
    for r in definitions:
        if r['kind']!='utility':continue
        akey=next(k for k,w in r['weights'].items() if w==1);bkey=next(k for k,w in r['weights'].items() if w==-1);a=metrics[akey];b=metrics[bkey]
        row=dict(name=r['name'],proposal=akey,reference=bkey,clean_loss_ratio=a['clean_tail'].mean()/b['clean_tail'].mean(),clean_ratio_upper95=float(np.quantile(rs.means(a['clean_tail'])/rs.means(b['clean_tail']),.95)),
            worst_loss_ratio=a['worst_tail_h1'].mean()/b['worst_tail_h1'].mean(),sensitivity_ratio=a['sensitivity_h1'].mean()/b['sensitivity_h1'].mean(),width_ratio=a['width80'].mean()/b['width80'].mean(),coverage_error_increase=abs(a['coverage80'].mean()-.8)-abs(b['coverage80'].mean()-.8),p_holm_19=float(tests.loc[cal&(tests.name==r['name']),'p_holm_19'].iloc[0]))
        row['clean_noninferior_2pct']=row['clean_ratio_upper95']<=1.02;row['worst_reduction_10pct']=row['worst_loss_ratio']<=.9;row['sensitivity_reduction_10pct']=row['sensitivity_ratio']<=.9;row['width_increase_le5pct']=row['width_ratio']<=1.05;row['coverage_error_increase_le1pp']=row['coverage_error_increase']<=.01
        row['all_gates_pass']=all(row[k] for k in ['clean_noninferior_2pct','worst_reduction_10pct','sensitivity_reduction_10pct','width_increase_le5pct','coverage_error_increase_le1pp']) and row['p_holm_19']<.05
        utility.append(row)
    pd.DataFrame(utility).to_csv(HERE/'UTILITY_DECISION.csv',index=False)
    (HERE/'ANALYSIS_COMPLETION.json').write_text(json.dumps(dict(status='complete',n=len(y),days=len(rs.days),model_stages=len(scores),primary_contrasts=19,utility_joint_pass=sum(r['all_gates_pass'] for r in utility),previous_metrics_reproduced=True,code_sha256=sha(Path(__file__))),indent=2));assert_frozen()
    print(score.loc[score.primary_model&score.new_model&score.model.str.endswith('_cal')].round(6).to_string(index=False),flush=True)
    print(tests.loc[cal].round(6).to_string(index=False),flush=True);print(pd.DataFrame(utility).round(6).to_string(index=False),flush=True)

if __name__=='__main__':main()
