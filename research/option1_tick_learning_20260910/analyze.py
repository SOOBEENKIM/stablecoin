from shared import *
import sys
sys.path.append(str(HERE.parent/'option1_core_20260909'))
from common import calendar_weights,interval
from statsmodels.stats.multitest import multipletests

def calibrate(dates,y,p):
    observed=(dates+pd.Timedelta(hours=6)).asi8
    err=y[:,None]-p[:,0,:];offset=np.zeros((len(y),3));ncal=[];last=[]
    for i,t in enumerate(dates.asi8):
        lo=np.searchsorted(observed,t-pd.Timedelta(days=28).value,side='left');hi=np.searchsorted(observed,t,side='left')
        if hi-lo>=120:
            offset[i]=[np.quantile(err[lo:hi,j],q) for j,q in enumerate(QS)]
            assert observed[hi-1]<t
        ncal.append(hi-lo);last.append(int(observed[hi-1]) if hi else 0)
    return np.sort(p+offset[:,None,:],axis=-1),offset,np.asarray(ncal),np.asarray(last,dtype=np.int64)

def quantities(y,p):
    loss=pinball(y[:,None],p);tail=loss[:,:,[0,2]].mean(2)
    out={'clean_tail':tail[:,0],'clean_q10':loss[:,0,0],
         'coverage80':((y>=p[:,0,0])&(y<=p[:,0,2])).astype(float),'width80':p[:,0,2]-p[:,0,0]}
    for h,start in [(1,1),(6,11),(24,21)]:
        jj=np.r_[0,np.arange(start,start+10)]
        out['worst_tail_h'+str(h)]=tail[:,jj].max(axis=1)
        out['sensitivity_h'+str(h)]=abs(p[:,start:start+10][:,:,[0,2]]-p[:,0][:,None,[0,2]]).max(axis=(1,2))
    return out

class Resample:
    def __init__(self,dates):
        self.days=dates.normalize().unique().sort_values();self.ix=self.days.get_indexer(dates.normalize())
        self.w=calendar_weights(self.days,5,1999,np.random.default_rng(20260910)).astype(float)
        self.count=np.bincount(self.ix,minlength=len(self.days));self.den=self.w@self.count
    def means(self,v):return (self.w@np.bincount(self.ix,weights=v,minlength=len(self.days)))/self.den

def main():
    assert json.loads((HERE/'TRAIN_COMPLETION.json').read_text())['status']=='complete'
    neural=np.load(HERE/'neural_predictions.npz');base=np.load(HERE/'baseline_predictions.npz')
    np.testing.assert_array_equal(neural['origin_ns'],base['origin_ns']);np.testing.assert_allclose(neural['target'],base['target'])
    dates=pd.to_datetime(neural['origin_ns'],utc=True);y=neural['target'];scenario=list(neural['scenario_names'])
    assert scenario==list(base['scenario_names']) and len(scenario)==31
    raw={}
    for family in ['tcn','linear']:
        for method in ['clean','shuffled','tick']:
            names=[f'{family}_{method}_{s}' for s in SEEDS]
            for name in names:raw[name]=neural[name].astype(float)
            raw[family+'_'+method+'_ensemble']=np.mean([raw[n] for n in names],axis=0)
    for name in ['exact_qr','lgbm']:raw[name]=base[name]
    select=dates>=pd.Timestamp('2026-01-01',tz='UTC');testdates=dates[select];testy=y[select];rs=Resample(testdates)
    scores=[];monthly=[];metrics={};predictions={};offsets={};calmetadata={};seedrows=[]
    for name,p in raw.items():
        assert np.isfinite(p).all()
        cp,offset,ncal,last=calibrate(dates,y,p);offsets[name]=offset
        calmetadata[name]=dict(min_eval_ncal=int(ncal[select].min()),max_used_time_before_origin=bool((last[select]<testdates.asi8).all()))
        primary=name.endswith('ensemble') or name in ['exact_qr','lgbm']
        for cal,v in [('raw',p),('cal',cp)]:
            key=name+'_'+cal;vv=v[select];qq=quantities(testy,vv);metrics[key]=qq
            predictions[key]=vv.astype(np.float32)
            row=dict(model=key,primary_model=primary,n=len(testy),days=len(rs.days),q10_prediction_sd=vv[:,0,0].std(),
                     q90_prediction_sd=vv[:,0,2].std(),**{k:float(x.mean()) for k,x in qq.items()})
            scores.append(row)
            for month in sorted(set(testdates.strftime('%Y-%m'))):
                keep=testdates.strftime('%Y-%m')==month
                monthly.append(dict(model=key,primary_model=primary,month=month,n=int(keep.sum()),**{k:float(x[keep].mean()) for k,x in qq.items()}))
    pd.DataFrame(scores).to_csv(HERE/'all_scores.csv',index=False);pd.DataFrame(monthly).to_csv(HERE/'monthly_scores.csv',index=False)
    np.savez_compressed(HERE/'evaluated_predictions.npz',origin_ns=testdates.asi8,target=testy,scenario_names=np.asarray(scenario),**predictions)
    np.savez_compressed(HERE/'calibration_offsets.npz',origin_ns=dates.asi8,**offsets)
    (HERE/'CALIBRATION_CHECKS.json').write_text(json.dumps(calmetadata,indent=2))
    np.savez_compressed(HERE/'inference_weights.npz',weights=rs.w.astype(np.int16),days=rs.days.asi8)
    proposal='tcn_tick_ensemble_cal';comparators=['tcn_clean_ensemble_cal','tcn_shuffled_ensemble_cal','linear_tick_ensemble_cal','exact_qr_cal']
    rows=[];draws={}
    for ref in comparators:
        a=metrics[proposal];b=metrics[ref]
        wc=rs.means(a['worst_tail_h1']-b['worst_tail_h1']);est=(a['worst_tail_h1']-b['worst_tail_h1']).mean()
        clean_ratio=rs.means(a['clean_tail'])/rs.means(b['clean_tail'])
        sensitivity=a['sensitivity_h1'].mean()/b['sensitivity_h1'].mean()
        width=a['width80'].mean()/b['width80'].mean();coverage=abs(a['coverage80'].mean()-.8)-abs(b['coverage80'].mean()-.8)
        row=dict(proposal=proposal,reference=ref,worst_loss_difference=est,**interval(est,wc),
            clean_loss_ratio=a['clean_tail'].mean()/b['clean_tail'].mean(),clean_ratio_upper95=float(np.quantile(clean_ratio,.95)),
            worst_loss_ratio=a['worst_tail_h1'].mean()/b['worst_tail_h1'].mean(),sensitivity_ratio=sensitivity,width_ratio=width,coverage_error_increase=coverage)
        row['clean_noninferior_2pct']=row['clean_ratio_upper95']<=1.02
        row['worst_reduction_10pct']=row['worst_loss_ratio']<=.9
        row['sensitivity_reduction_10pct']=sensitivity<=.9
        row['width_increase_le5pct']=width<=1.05
        row['coverage_error_increase_le1pp']=coverage<=.01
        rows.append(row);draws[ref+'_worst_difference']=wc;draws[ref+'_clean_ratio']=clean_ratio
    decision=pd.DataFrame(rows);decision['p_holm_4']=multipletests(decision.p_centered,method='holm')[1]
    decision['all_gates_pass']=decision[['clean_noninferior_2pct','worst_reduction_10pct','sensitivity_reduction_10pct','width_increase_le5pct','coverage_error_increase_le1pp']].all(axis=1)&(decision.p_holm_4<.05)&(decision.worst_loss_difference<0)
    decision.to_csv(HERE/'PRIMARY_DECISION.csv',index=False);np.savez_compressed(HERE/'primary_bootstrap.npz',**draws)
    # Matched seeds are diagnostics, never treated as independent time samples.
    for seed in SEEDS:
        a=metrics[f'tcn_tick_{seed}_cal']
        for ref in ['tcn_clean','tcn_shuffled','linear_tick']:
            b=metrics[f'{ref}_{seed}_cal']
            seedrows.append(dict(seed=seed,reference=ref,clean_ratio=a['clean_tail'].mean()/b['clean_tail'].mean(),
                worst_ratio=a['worst_tail_h1'].mean()/b['worst_tail_h1'].mean(),sensitivity_ratio=a['sensitivity_h1'].mean()/b['sensitivity_h1'].mean()))
    pd.DataFrame(seedrows).to_csv(HERE/'seed_diagnostics.csv',index=False)
    # Existing forecasts: common-origin observed-target checks only. No stored old input-stress predictions.
    old=pd.read_csv(HERE.parent/'extension_20260909/mean5_h6_forecasts.csv.gz');old.origin=pd.to_datetime(old.origin,utc=True)
    legacy=[]
    for name in ['update_linear_local_cal28','update_lgbm_local_cal28']:
        f=old.loc[old.model==name].set_index('origin');ix=testdates.intersection(f.index);pos=testdates.get_indexer(ix);f=f.reindex(ix)
        np.testing.assert_allclose(f.target,testy[pos],atol=1e-7)
        p=f[['q10','q50','q90']].to_numpy();ll=pinball(f.target.to_numpy(),p)
        legacy.append(dict(model=name,n=len(ix),clean_tail=ll[:,[0,2]].mean(),coverage80=((f.target>=p[:,0])&(f.target<=p[:,2])).mean(),width80=(p[:,2]-p[:,0]).mean()))
        for key in [proposal,'exact_qr_cal','lgbm_cal']:
            q=metrics[key];legacy.append(dict(model=key+'_matched_'+name,n=len(ix),clean_tail=q['clean_tail'][pos].mean(),coverage80=q['coverage80'][pos].mean(),width80=q['width80'][pos].mean()))
    pd.DataFrame(legacy).to_csv(HERE/'legacy_common_origin_comparison.csv',index=False)
    (HERE/'ANALYSIS_COMPLETION.json').write_text(json.dumps(dict(status='complete',code_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),
        n=len(testy),days=len(rs.days),primary_comparisons=4,all_gates_pass=int(decision.all_gates_pass.sum()),
        tick_specific_value=bool(decision.iloc[:2].all_gates_pass.all())),indent=2))
    assert_frozen()
    print(pd.DataFrame(scores).loc[lambda x:x.primary_model&x.model.str.endswith('_cal')].round(5).to_string(index=False),flush=True)
    print(decision.round(5).to_string(index=False),flush=True)
    print(pd.DataFrame(legacy).round(5).to_string(index=False),flush=True)

if __name__=='__main__':main()
