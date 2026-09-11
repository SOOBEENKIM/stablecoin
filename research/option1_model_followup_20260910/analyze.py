from shared import *
import importlib.util
OLD=HERE.parent/'option1_tick_learning_20260910'
spec=importlib.util.spec_from_file_location('previous_analysis',OLD/'analyze.py');previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
calibrate=previous.calibrate;quantities=previous.quantities;Resample=previous.Resample;interval=previous.interval
from statsmodels.stats.multitest import multipletests

def main():
    assert json.loads((HERE/'TRAIN_COMPLETION.json').read_text())['status']=='complete';assert_frozen()
    new=np.load(HERE/'neural_predictions.npz');old=np.load(OLD/'neural_predictions.npz');base=np.load(OLD/'baseline_predictions.npz');offold=np.load(OLD/'calibration_offsets.npz')
    np.testing.assert_array_equal(new['origin_ns'],old['origin_ns']);np.testing.assert_array_equal(new['target'],old['target'])
    dates=pd.to_datetime(new['origin_ns'],utc=True);target=new['target'];use=dates>=pd.Timestamp('2026-01-01',tz='UTC');td=dates[use];y=target[use]
    raw={};newnames=[]
    for dataset in [old,new]:
        prefixes=sorted({k.rsplit('_',1)[0] for k in dataset.files if k.endswith(tuple(str(s) for s in SEEDS))})
        for prefix in prefixes:
            names=[prefix+'_'+str(s) for s in SEEDS]
            for name in names:raw[name]=dataset[name].astype(float)
            raw[prefix+'_ensemble']=np.mean([raw[k] for k in names],axis=0)
            if dataset is new:newnames+=names+[prefix+'_ensemble']
    for name in ['exact_qr','lgbm']:raw[name]=base[name]
    scores=[];monthly=[];predictions={};metrics={};offsets={};metadata=[];rs=Resample(td)
    oldweights=np.load(OLD/'inference_weights.npz');np.testing.assert_array_equal(rs.w,oldweights['weights'])
    for name,p in raw.items():
        if name in newnames:
            cp,offset,ncal,last=calibrate(dates,target,p);offsets[name]=offset
            metadata.append(dict(model=name,min_ncal=int(ncal[use].min()),strictly_past=bool((last[use]<td.asi8).all())))
        else:cp=np.sort(p+offold[name][:,None,:],axis=-1)
        primary=name.endswith('_ensemble') or name in ['exact_qr','lgbm']
        for stage,pp in [('raw',p),('cal',cp)]:
            key=name+'_'+stage;v=pp[use];q=quantities(y,v);metrics[key]=q;predictions[key]=v.astype(np.float32)
            scores.append(dict(model=key,primary_model=primary,new_model=name in newnames,n=len(y),days=len(rs.days),
                q10_prediction_sd=v[:,0,0].std(),q90_prediction_sd=v[:,0,2].std(),**{k:float(x.mean()) for k,x in q.items()}))
            for month in sorted(set(td.strftime('%Y-%m'))):
                ix=td.strftime('%Y-%m')==month
                monthly.append(dict(model=key,month=month,n=int(ix.sum()),**{k:float(x[ix].mean()) for k,x in q.items()}))
    score=pd.DataFrame(scores);score.to_csv(HERE/'all_scores.csv',index=False);pd.DataFrame(monthly).to_csv(HERE/'monthly_scores.csv',index=False)
    np.savez_compressed(HERE/'evaluated_predictions.npz',origin_ns=td.asi8,target=y,scenario_names=new['scenario_names'],**predictions)
    np.savez_compressed(HERE/'calibration_offsets.npz',origin_ns=dates.asi8,**offsets);pd.DataFrame(metadata).to_csv(HERE/'calibration_checks.csv',index=False)
    origscore=pd.read_csv(OLD/'all_scores.csv').set_index('model');now=score.set_index('model')
    for col in ['clean_tail','worst_tail_h1','sensitivity_h1','coverage80','width80']:
        np.testing.assert_allclose(now.loc[origscore.index,col],origscore[col],atol=1e-12)
    pairs=[]
    for family in ['gru','mlp']:
        pairs += [(family+'_tick_ensemble_cal',r) for r in [family+'_clean_ensemble_cal',family+'_shuffled_ensemble_cal','tcn_tick_ensemble_cal','exact_qr_cal']]
    pairs += [('tcn_hard_tick_ensemble_cal',r) for r in ['tcn_clean_ensemble_cal','tcn_tick_ensemble_cal','tcn_hard_shuffled_ensemble_cal','exact_qr_cal']]
    rows=[];draws={}
    for j,(proposal,reference) in enumerate(pairs):
        a=metrics[proposal];b=metrics[reference];point=(a['worst_tail_h1']-b['worst_tail_h1']).mean();w=rs.means(a['worst_tail_h1']-b['worst_tail_h1'])
        clean=rs.means(a['clean_tail'])/rs.means(b['clean_tail']);sr=rs.means(a['sensitivity_h1'])/rs.means(b['sensitivity_h1'])
        row=dict(proposal=proposal,reference=reference,worst_loss_difference=point,**interval(point,w),
            clean_loss_ratio=a['clean_tail'].mean()/b['clean_tail'].mean(),clean_ratio_upper95=np.quantile(clean,.95),
            worst_loss_ratio=a['worst_tail_h1'].mean()/b['worst_tail_h1'].mean(),sensitivity_ratio=a['sensitivity_h1'].mean()/b['sensitivity_h1'].mean(),
            sensitivity_ratio_low95=np.quantile(sr,.025),sensitivity_ratio_high95=np.quantile(sr,.975),
            width_ratio=a['width80'].mean()/b['width80'].mean(),coverage_error_increase=abs(a['coverage80'].mean()-.8)-abs(b['coverage80'].mean()-.8))
        rows.append(row);draws['comparison_'+str(j)+'_worst_difference']=w;draws['comparison_'+str(j)+'_clean_ratio']=clean
    dec=pd.DataFrame(rows);dec['p_holm_12']=multipletests(dec.p_centered,method='holm')[1]
    oldp=pd.read_csv(OLD/'PRIMARY_DECISION.csv').p_centered.to_numpy();dec['p_holm_16']=multipletests(np.r_[oldp,dec.p_centered],method='holm')[1][4:]
    dec['clean_noninferior_2pct']=dec.clean_ratio_upper95<=1.02;dec['worst_reduction_10pct']=dec.worst_loss_ratio<=.9;dec['sensitivity_reduction_10pct']=dec.sensitivity_ratio<=.9
    dec['width_increase_le5pct']=dec.width_ratio<=1.05;dec['coverage_error_increase_le1pp']=dec.coverage_error_increase<=.01
    dec['all_gates_pass']=dec[['clean_noninferior_2pct','worst_reduction_10pct','sensitivity_reduction_10pct','width_increase_le5pct','coverage_error_increase_le1pp']].all(1)&(dec.p_holm_12<.05)&(dec.worst_loss_difference<0)
    dec.to_csv(HERE/'PRIMARY_DECISION.csv',index=False);np.savez_compressed(HERE/'primary_bootstrap.npz',**draws)
    np.savez_compressed(HERE/'inference_weights.npz',weights=rs.w.astype(np.int16),days=rs.days.asi8)
    seedrows=[]
    for a,b in pairs:
        if not b.endswith('_ensemble_cal'):continue
        for seed in SEEDS:
            an=a.replace('_ensemble_',f'_{seed}_');bn=b.replace('_ensemble_',f'_{seed}_');x=metrics[an];z=metrics[bn]
            seedrows.append(dict(proposal=a,reference=b,seed=seed,clean_ratio=x['clean_tail'].mean()/z['clean_tail'].mean(),worst_ratio=x['worst_tail_h1'].mean()/z['worst_tail_h1'].mean(),sensitivity_ratio=x['sensitivity_h1'].mean()/z['sensitivity_h1'].mean()))
    pd.DataFrame(seedrows).to_csv(HERE/'seed_diagnostics.csv',index=False)
    (HERE/'ANALYSIS_COMPLETION.json').write_text(json.dumps(dict(status='complete',n=len(y),days=len(rs.days),new_ensembles=8,model_stages=len(scores),primary_comparisons=12,all_gates_pass=int(dec.all_gates_pass.sum()),old_predictions_and_metrics_reproduced=True,code_sha256=sha(Path(__file__))),indent=2))
    assert_frozen();print(score.loc[score.primary_model&score.model.str.endswith('_cal')].round(5).to_string(index=False),flush=True)
    print(dec.round(5).to_string(index=False),flush=True)

if __name__=='__main__':main()
