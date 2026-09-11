from shared import *
FIRST=HERE.parent/'option1_tick_learning_20260910';PREVIOUS=HERE.parent/'option1_model_followup_20260910'

def main():
    checks=[];assert_frozen();assert sha(HERE/'data.npz')==sha(FIRST/'data.npz');assert json.loads((FIRST/'VERIFICATION.json').read_text())['status']=='passed'
    checks.append('518 previous artifacts preserved; same validated raw-derived data and fixed target')
    d=np.load(HERE/'data.npz');n=np.load(HERE/'neural_predictions.npz');old=np.load(FIRST/'neural_predictions.npz');o=pd.to_datetime(d['origin_ns'],utc=True);target=pd.to_datetime(d['target_ns'],utc=True)
    for k in ['origin_ns','target','scenario_names']:np.testing.assert_array_equal(n[k],old[k])
    assert np.all(target.asi8-o.asi8==6*3600*10**9)
    f=pd.read_csv(HERE/'neural_training.csv');anchors=pd.read_csv(HERE/'anchor_epochs.csv');assert len(f)==96 and len(anchors)==24
    for _,r in anchors.iterrows():
        assert sha(HERE.parent/r.source)==r.source_sha256
        src=pd.read_csv(HERE.parent/r.source);match=src[(src.model==r.model)&(src.month==r.month)].iloc[0];assert match.epochs==r.epochs
    for _,g in f.groupby(['family','seed','month']):
        assert len(g)==4
        for c in ['epochs','optimizer_steps','initial_parameter_hash','batch_order_hash','final_torch_rng_hash']:assert g[c].nunique()==1
        r=g.iloc[0];a=anchors[(anchors.family==r.family)&(anchors.seed==r.seed)&(anchors.month==r.month)].iloc[0];assert r.epochs==a.epochs
    for _,g in f.groupby(['month','seed','placement']):assert g.candidate_hash.nunique()==1
    for _,r in f.iterrows():
        cut=pd.Timestamp(r.month+'-01',tz='UTC');tr=target<cut;te=o.strftime('%Y-%m')==r.month
        assert r.n_train==tr.sum() and target[tr].max()<o[te].min() and r.optimizer_steps==r.epochs*int(np.ceil(tr.sum()/128))
    checks.append('Fixed past-validation epoch anchors; identical four-cell initialization, candidate arrays, batches, RNG endpoint and optimizer steps')
    curves=pd.read_csv(HERE/'training_curves.csv')
    for _,r in f.iterrows():
        c=curves[(curves.model==r.model)&(curves.month==r.month)];assert len(c)==r.epochs and (c.candidate_max_loss_scaled>=c.candidate_mean_loss_scaled-1e-6).all()
        for counts in c.argmax_counts:assert sum(json.loads(counts))==r.n_train
        branch=c.candidate_mean_loss_scaled if r.aggregation=='mean' else c.candidate_max_loss_scaled
        np.testing.assert_allclose(c.train_loss_scaled,.5*c.clean_loss_scaled+.5*branch,atol=2e-6)
    settings=json.loads((HERE/'TRAIN_SETTINGS.json').read_text())
    for key,file in [('code_sha256','train.py'),('shared_sha256','shared.py'),('data_sha256','data.npz'),('anchor_sha256','anchor_epochs.csv'),('protocol_sha256','PROTOCOL_KO.md')]:assert settings[key]==sha(HERE/file)
    checks.append('Recorded train objectives decompose correctly; all candidates/rows accounted for; frozen code, data, anchors and protocol unchanged')
    p=np.load(HERE/'evaluated_predictions.npz');y=p['target'];score=pd.read_csv(HERE/'all_scores.csv').set_index('model');metrics={};maxerr=0.
    for name,r in score.iterrows():
        a=p[name].astype(float);e1=y[:,None]-a[:,:,0];e9=y[:,None]-a[:,:,2];tail=(np.where(e1>=0,.1*e1,-.9*e1)+np.where(e9>=0,.9*e9,-.1*e9))/2
        m=dict(clean_tail=tail[:,0],worst_tail_h1=tail[:,:11].max(1),stress_excess=tail[:,:11].max(1)-tail[:,0],
            sensitivity_h1=np.maximum(abs(a[:,1:11,0]-a[:,None,0,0]),abs(a[:,1:11,2]-a[:,None,0,2])).max(1),
            coverage80=((y>=a[:,0,0])&(y<=a[:,0,2])).astype(float),width80=a[:,0,2]-a[:,0,0]);metrics[name]=m
        for k,v in m.items():maxerr=max(maxerr,abs(v.mean()-r[k]));np.testing.assert_allclose(v.mean(),r[k],atol=4e-6,rtol=1e-6)
    checks.append('All 180 model-stage primary metrics independently reproduced from saved predictions')
    off=np.load(HERE/'calibration_offsets.npz');observed=n['origin_ns']+6*3600*10**9
    for name in off.files:
        if name=='origin_ns':continue
        pp=np.mean([n[name[:-9]+'_'+str(s)].astype(float) for s in SEEDS],axis=0) if name.endswith('_ensemble') else n[name].astype(float)
        ids=np.flatnonzero(n['origin_ns']>=pd.Timestamp('2026-01-01',tz='UTC').value)
        for i in ids[np.linspace(0,len(ids)-1,5,dtype=int)]:
            t=n['origin_ns'][i];past=(observed<t)&(observed>=t-28*24*3600*10**9);assert past.sum()>=120
            offset=np.array([np.quantile(n['target'][past]-pp[past,0,j],q) for j,q in enumerate(QS)]);np.testing.assert_allclose(offset,off[name][i],atol=1e-12)
            j=np.flatnonzero(p['origin_ns']==t)[0];np.testing.assert_allclose(p[name+'_cal'][j],np.sort(pp[i]+offset,axis=-1),atol=5e-6,rtol=1e-6)
    checks.append('All 32 new calibrators use strict-past observed labels; same offset across scenarios; ensemble first')
    contrasts=pd.read_csv(HERE/'factorial_contrasts.csv');definitions=json.loads((HERE/'contrasts.json').read_text())
    saved_boot=np.load(HERE/'contrast_bootstrap.npz');bw=np.load(HERE/'inference_weights.npz')
    days=pd.to_datetime(bw['days'],utc=True);evaldays=pd.to_datetime(p['origin_ns'],utc=True).normalize();ix=days.get_indexer(evaldays);assert (ix>=0).all()
    weights=bw['weights'].astype(float);den=weights@np.bincount(ix,minlength=len(days))
    for definition in definitions:
        for stage in ['cal','raw']:
            r=contrasts[(contrasts.name==definition['name'])&(contrasts.stage==stage)].iloc[0]
            delta=sum(w*metrics[k if stage=='cal' else k[:-3]+'raw']['worst_tail_h1'] for k,w in definition['weights'].items());np.testing.assert_allclose(delta.mean(),r.difference,atol=2e-6)
            independent_draws=(weights@np.bincount(ix,weights=delta,minlength=len(days)))/den
            draws=saved_boot[definition['name']+'_'+stage];np.testing.assert_allclose(independent_draws,draws,atol=2e-6)
            np.testing.assert_allclose(np.quantile(draws,[.025,.975]),[r.low95,r.high95],atol=1e-12)
            pp=(1+np.sum(abs(draws-r.difference)>=abs(r.difference)))/(len(draws)+1);np.testing.assert_allclose(pp,r.p_centered,atol=1e-12)
    # Independent 2x2 contrasts, including the sign of the interaction.
    for family in ['tcn','gru']:
        get=lambda place,agg:metrics[family+'_'+place+'_'+agg+'_ensemble_cal']['worst_tail_h1']
        a,b,c,e=get('tick','mean'),get('tick','max'),get('shuffled','mean'),get('shuffled','max')
        expected=dict(training_main=((b-a)+(e-c))/2,financial_main=((b-e)+(a-c))/2,interaction=(b-a)-(e-c))
        for term,v in expected.items():np.testing.assert_allclose(v.mean(),contrasts.loc[(contrasts.name==family+'_'+term)&(contrasts.stage=='cal'),'difference'].iloc[0],atol=2e-6)
    def holm(values):
        values=np.asarray(values);order=np.argsort(values);out=np.empty(len(values));out[order]=np.minimum(1,np.maximum.accumulate(values[order]*np.arange(len(values),0,-1)));return out
    c=contrasts[contrasts.stage=='cal'];assert len(c)==19
    np.testing.assert_allclose(holm(c.p_centered),c.p_holm_19,atol=1e-12)
    oldp=np.r_[pd.read_csv(FIRST/'PRIMARY_DECISION.csv').p_centered,pd.read_csv(PREVIOUS/'PRIMARY_DECISION.csv').p_centered]
    np.testing.assert_allclose(holm(np.r_[oldp,c.p_centered])[16:],c.p_holm_35,atol=1e-12)
    u=pd.read_csv(HERE/'UTILITY_DECISION.csv');g=(u.clean_ratio_upper95<=1.02)&(u.worst_loss_ratio<=.9)&(u.sensitivity_ratio<=.9)&(u.width_ratio<=1.05)&(u.coverage_error_increase<=.01)&(u.p_holm_19<.05);np.testing.assert_array_equal(g,u.all_gates_pass)
    checks.append('Factorial effects/interactions, Holm19/35 and four utility gates independently reconstructed')
    assert json.loads((HERE/'CHECKPOINT_VERIFICATION.json').read_text())['status']=='passed'
    checks.append('96 saved checkpoints replayed; temporal causality and trained common-dropout invariance pass')
    result=dict(status='passed',checks=checks,max_prediction_storage_metric_error=maxerr,code_sha256=sha(Path(__file__)))
    (HERE/'VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
