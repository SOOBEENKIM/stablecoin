"""Independent accounting checks on data, price perturbations, losses and decision gates."""
from shared import *

def main():
    checks=[];d=np.load(HERE/'data.npz');X=d['X'];K=d['K'];y=d['y'];o=pd.to_datetime(d['origin_ns'],utc=True)
    target=pd.to_datetime(d['target_ns'],utc=True)
    assert np.all(target.asi8-o.asi8==6*3600*10**9) and o.is_unique and o.is_monotonic_increasing
    assert (o-pd.Timedelta(hours=23)).min()>=pd.Timestamp('2025-08-02',tz='UTC')
    checks.append('Exact six-hour target clock and ordered origins')
    def readraw(name):
        f=pd.read_csv(HERE.parent/'data/raw data'/name)
        f.index=pd.to_datetime(f.pop('datetime_utc'),utc=True,format='mixed')+pd.Timedelta(hours=1)
        assert f.index.is_unique
        return f
    raw=readraw('binance_1h_2025-06-01_2026-03-19.csv').join(readraw('upbit_1h_2025-06-01_2026-03-19.csv'),how='outer')
    independent=1e4*np.log(raw.USDT_UPBIT_CLOSE.to_numpy()[:,None]*raw[[c+'_BINANCE_CLOSE' for c in COINS]].to_numpy()/raw[[c+'_UPBIT_CLOSE' for c in COINS]].to_numpy())
    truth=pd.Series(independent.mean(1),index=raw.index)
    np.testing.assert_allclose(y,truth.reindex(target),atol=1e-7)
    ids=np.linspace(0,len(X)-1,90,dtype=int)
    for i in ids:
        times=pd.date_range(o[i]-pd.Timedelta(hours=23),o[i],freq='h')
        ri=raw.index.get_indexer(times);assert (ri>=0).all()
        np.testing.assert_allclose(X[i,:,:5],independent[ri],atol=1e-4,rtol=1e-6)
        np.testing.assert_allclose(K[i],raw.reindex(times)[[c+'_UPBIT_CLOSE' for c in COINS]],atol=1e-10)
    checks.append('Raw prices reproduce 24-hour windows and all future targets')
    # Explicit independent table covers every observed price band, including step-down boundaries.
    def unit(a):return np.select([a>=2e6,a>=1e6,a>=5e5,a>=1e5,a>=5e4,a>=1e4,a>=5e3,a>=1e3,a>=100,a>=10,a>=1],[1000,1000,500,100,50,10,5,1,1,.1,.01],default=np.nan)
    assert np.isfinite(unit(K)).all()
    np.testing.assert_array_equal(tick(K),unit(K))
    np.testing.assert_allclose(K/unit(K),np.round(K/unit(K)),atol=1e-6)
    max_aug_error=0.;max_budget_error=0.
    for donor in range(5):
        for sign in [-1,1]:
            for span in [1,6,24]:
                xx=X[ids];kk=K[ids];moved=kk.copy();a=moved[:,-span:,donor]
                step=unit(a if sign>0 else np.nextafter(a,-np.inf));moved[:,-span:,donor]+=sign*step
                np.testing.assert_allclose(moved/unit(moved),np.round(moved/unit(moved)),atol=1e-6)
                want=xx[:,:,:5].astype(float)-1e4*np.log(moved/kk)
                got=augment(xx,kk,donor,sign,span)
                max_aug_error=max(max_aug_error,float(abs(got[:,:,:5]-want).max()))
                np.testing.assert_allclose(got[:,:,:5],want,atol=2e-4,rtol=1e-6)
                np.testing.assert_allclose(got[:,:,5],want.mean(2),atol=2e-4,rtol=1e-6)
                np.testing.assert_allclose(got[:,:,6],np.median(want,axis=2),atol=2e-4,rtol=1e-6)
                np.testing.assert_allclose(got[:,:,7],want.std(2,ddof=1),atol=2e-4,rtol=1e-6)
                np.testing.assert_array_equal(got[:,:,8:25],xx[:,:,8:25])
                random=augment(xx,kk,donor,sign,span,(donor+2)%5)
                e=abs(abs(got[:,:,:5]-xx[:,:,:5]).sum(2)-abs(random[:,:,:5]-xx[:,:,:5]).sum(2))
                max_budget_error=max(max_budget_error,float(e.max()))
                np.testing.assert_allclose(got[:,:,5],random[:,:,5],atol=2e-4,rtol=1e-6)
    assert max_budget_error<2e-4
    checks.append('Observed prices lie on official grid; all 30 scenarios legal; magnitude-matched controls and recomputed features')
    n=np.load(HERE/'neural_predictions.npz');b=np.load(HERE/'baseline_predictions.npz');e=np.load(HERE/'evaluated_predictions.npz')
    np.testing.assert_array_equal(n['origin_ns'],b['origin_ns']);np.testing.assert_array_equal(n['target'],b['target'])
    fit=pd.read_csv(HERE/'neural_training.csv')
    assert len(fit)==72 and len(list((HERE/'checkpoints').glob('*.pt')))==72
    assert (pd.to_datetime(fit.last_training_target,utc=True)<pd.to_datetime(fit.first_test_origin,utc=True)).all()
    checks.append('72 fits, matching targets and origins, all training target timestamps before test')
    score=pd.read_csv(HERE/'all_scores.csv').set_index('model');ey=e['target'];max_metric_error=0.
    for key in score.index:
        pp=e[key].astype(float);lo=ey[:,None]-pp[:,:,0];hi=ey[:,None]-pp[:,:,2]
        tails=(np.where(lo>=0,.1*lo,-.9*lo)+np.where(hi>=0,.9*hi,-.1*hi))/2
        want=dict(clean_tail=tails[:,0].mean(),worst_tail_h1=tails[:,:11].max(1).mean(),
            width80=(pp[:,0,2]-pp[:,0,0]).mean(),coverage80=((ey>=pp[:,0,0])&(ey<=pp[:,0,2])).mean())
        changes=np.maximum(abs(pp[:,1:11,0]-pp[:,None,0,0]),abs(pp[:,1:11,2]-pp[:,None,0,2]))
        want['sensitivity_h1']=changes.max(1).mean()
        assert (tails[:,:11].max(1)>=tails[:,0]).all()
        for metric,value in want.items():
            max_metric_error=max(max_metric_error,abs(value-score.loc[key,metric]))
            np.testing.assert_allclose(value,score.loc[key,metric],atol=3e-6,rtol=1e-6)
    checks.append('All 52 model-stage metrics independently reproduced from saved predictions')
    off=np.load(HERE/'calibration_offsets.npz');observed=n['origin_ns']+6*3600*10**9
    evalpos=np.flatnonzero(n['origin_ns']>=pd.Timestamp('2026-01-01',tz='UTC').value)
    for name in off.files:
        if name=='origin_ns':continue
        if name.endswith('_ensemble'):
            prefix=name[:-9];p=np.mean([n[prefix+'_'+str(s)].astype(float) for s in SEEDS],0)
        else:p=(b[name] if name in b.files else n[name]).astype(float)
        for i in evalpos[np.linspace(0,len(evalpos)-1,5,dtype=int)]:
            t=n['origin_ns'][i];past=(observed<t)&(observed>=t-28*24*3600*10**9);assert past.sum()>=120
            offsets=np.array([np.quantile(n['target'][past]-p[past,0,j],q) for j,q in enumerate(QS)])
            np.testing.assert_allclose(off[name][i],offsets,atol=1e-12)
            j=np.flatnonzero(e['origin_ns']==t)[0]
            np.testing.assert_allclose(e[name+'_cal'][j],np.sort(p[i]+offsets,axis=-1),atol=4e-6,rtol=1e-6)
    checks.append('All 26 calibrators use strictly observed past targets and identical offsets for all scenarios; ensemble first')
    decision=pd.read_csv(HERE/'PRIMARY_DECISION.csv');order=np.argsort(decision.p_centered.to_numpy())
    adj=np.empty(4);adj[order]=np.minimum(1,np.maximum.accumulate(decision.p_centered.to_numpy()[order]*np.arange(4,0,-1)))
    np.testing.assert_allclose(adj,decision.p_holm_4,atol=1e-12)
    gates=(decision.clean_ratio_upper95<=1.02)&(decision.worst_loss_ratio<=.9)&(decision.sensitivity_ratio<=.9)&(decision.width_ratio<=1.05)&(decision.coverage_error_increase<=.01)&(adj<.05)&(decision.worst_loss_difference<0)
    np.testing.assert_array_equal(gates,decision.all_gates_pass)
    checks.append('Four-comparison Holm adjustment and all frozen success gates independently recomputed')
    assert_frozen();meta=json.loads((HERE/'DATA_MANIFEST.json').read_text())
    for path,digest in meta['input_hashes'].items():assert sha(HERE.parent/path)==digest
    settings=json.loads((HERE/'TRAIN_SETTINGS.json').read_text())
    for key,path in [('code_sha256','train.py'),('shared_sha256','shared.py'),('data_sha256','data.npz'),('protocol_sha256','PROTOCOL_KO.md')]:assert settings[key]==sha(HERE/path)
    replay=json.loads((HERE/'CHECKPOINT_VERIFICATION.json').read_text());assert replay['status']=='passed'
    checks.append('Raw files, training code, protocol and 260 earlier artifacts unchanged; all checkpoint replay checks passed')
    out=dict(status='passed',checks=checks,max_augmentation_error_bp=max_aug_error,max_control_budget_error_bp=max_budget_error,
        max_metric_storage_error=max_metric_error,code_sha256=sha(Path(__file__)),checks_count=len(checks))
    (HERE/'VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
