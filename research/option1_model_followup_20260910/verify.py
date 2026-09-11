from shared import *
OLD=HERE.parent/'option1_tick_learning_20260910'

def main():
    checks=[];assert_frozen();assert sha(HERE/'data.npz')==sha(OLD/'data.npz')
    assert json.loads((OLD/'VERIFICATION.json').read_text())['status']=='passed'
    checks.append('Same hash-verified raw-derived data and fixed targets; 381 previous artifacts unchanged')
    d=np.load(HERE/'data.npz');n=np.load(HERE/'neural_predictions.npz');o=pd.to_datetime(d['origin_ns'],utc=True);t=pd.to_datetime(d['target_ns'],utc=True)
    assert np.all(t.asi8-o.asi8==6*3600*10**9);oldn=np.load(OLD/'neural_predictions.npz')
    for key in ['origin_ns','target','scenario_names']:np.testing.assert_array_equal(n[key],oldn[key])
    f=pd.read_csv(HERE/'neural_training.csv');assert len(f)==96 and f.groupby('model').size().eq(4).all()
    assert (pd.to_datetime(f.last_training_target,utc=True)<pd.to_datetime(f.first_test_origin,utc=True)).all()
    for _,r in f.iterrows():
        cut=pd.Timestamp(r.month+'-01',tz='UTC');inner=cut-pd.Timedelta(days=28);tr=t<cut;it=t<inner;iv=(o>=inner)&tr
        assert r.n_train==tr.sum() and r.n_inner==it.sum() and r.n_valid==iv.sum() and t[it].max()<o[iv].min()
    checks.append('All 96 fits use strict chronological outer/inner splits and the same 1,814 evaluation origins')
    settings=json.loads((HERE/'TRAIN_SETTINGS.json').read_text())
    for key,file in [('code_sha256','train.py'),('shared_sha256','shared.py'),('protocol_sha256','PROTOCOL_KO.md'),('data_sha256','data.npz')]:assert settings[key]==sha(HERE/file)
    curves=pd.read_csv(HERE/'training_curves.csv');assert np.isfinite(curves.train_loss_scaled).all()
    for _,r in f.iterrows():
        c=curves[(curves.model==r.model)&(curves.month==r.month)];v=c[c.stage=='validation'];refit=c[c.stage=='refit']
        chosen=v.loc[v.epoch==r.epochs,'validation_tail_loss_bp'].iloc[0]
        assert chosen<=v.validation_tail_loss_bp.min()+1e-6 and len(refit)==r.epochs
    for _,r in curves[curves.model.str.contains('hard_')].iterrows():
        row=f[(f.model==r.model)&(f.month==r.month)].iloc[0];expected=row.n_inner if r.stage=='validation' else row.n_train
        assert sum(json.loads(r.selected_counts))==expected
    checks.append('Training code/protocol unchanged; hard-selection accounting covers each training example exactly once per epoch')
    e=np.load(HERE/'evaluated_predictions.npz');score=pd.read_csv(HERE/'all_scores.csv').set_index('model');y=e['target'];maxerr=0.
    for name,r in score.iterrows():
        p=e[name].astype(float);e1=y[:,None]-p[:,:,0];e9=y[:,None]-p[:,:,2]
        tail=(np.where(e1>=0,.1*e1,-.9*e1)+np.where(e9>=0,.9*e9,-.1*e9))/2
        metrics=dict(clean_tail=tail[:,0].mean(),worst_tail_h1=tail[:,:11].max(1).mean(),
            sensitivity_h1=np.maximum(abs(p[:,1:11,0]-p[:,None,0,0]),abs(p[:,1:11,2]-p[:,None,0,2])).max(1).mean(),
            coverage80=((y>=p[:,0,0])&(y<=p[:,0,2])).mean(),width80=(p[:,0,2]-p[:,0,0]).mean())
        for metric,v in metrics.items():maxerr=max(maxerr,abs(v-r[metric]));np.testing.assert_allclose(v,r[metric],atol=4e-6,rtol=1e-6)
    checks.append('All 116 model-stage primary metrics independently reproduced from predictions')
    offsets=np.load(HERE/'calibration_offsets.npz');observed=n['origin_ns']+6*3600*10**9
    for name in offsets.files:
        if name=='origin_ns':continue
        if name.endswith('_ensemble'):p=np.mean([n[name[:-9]+'_'+str(seed)].astype(float) for seed in SEEDS],axis=0)
        else:p=n[name].astype(float)
        ids=np.flatnonzero(n['origin_ns']>=pd.Timestamp('2026-01-01',tz='UTC').value)
        for i in ids[np.linspace(0,len(ids)-1,5,dtype=int)]:
            tt=n['origin_ns'][i];past=(observed<tt)&(observed>=tt-28*24*3600*10**9);assert past.sum()>=120
            off=np.array([np.quantile(n['target'][past]-p[past,0,j],q) for j,q in enumerate(QS)])
            np.testing.assert_allclose(off,offsets[name][i],atol=1e-12)
            j=np.flatnonzero(e['origin_ns']==tt)[0];np.testing.assert_allclose(e[name+'_cal'][j],np.sort(p[i]+off,axis=-1),atol=5e-6,rtol=1e-6)
    checks.append('All 32 new calibrators: strict past labels, ensemble before calibration, same offset for every input scenario')
    dec=pd.read_csv(HERE/'PRIMARY_DECISION.csv');assert len(dec)==12
    def holm(p):
        p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)));return out
    np.testing.assert_allclose(holm(dec.p_centered),dec.p_holm_12,atol=1e-12)
    np.testing.assert_allclose(holm(np.r_[pd.read_csv(OLD/'PRIMARY_DECISION.csv').p_centered,dec.p_centered])[4:],dec.p_holm_16,atol=1e-12)
    want=(dec.clean_ratio_upper95<=1.02)&(dec.worst_loss_ratio<=.9)&(dec.sensitivity_ratio<=.9)&(dec.width_ratio<=1.05)&(dec.coverage_error_increase<=.01)&(dec.p_holm_12<.05)&(dec.worst_loss_difference<0)
    np.testing.assert_array_equal(want,dec.all_gates_pass)
    checks.append('Holm12, combined Holm16 and all predeclared joint success gates recomputed')
    assert json.loads((HERE/'CHECKPOINT_VERIFICATION.json').read_text())['status']=='passed'
    checks.append('All 96 checkpoints replayed at 3 origins and 31 scenarios; GRU/TCN temporal causality verified')
    out=dict(status='passed',checks=checks,max_prediction_storage_metric_error=maxerr,code_sha256=sha(Path(__file__)))
    (HERE/'VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
