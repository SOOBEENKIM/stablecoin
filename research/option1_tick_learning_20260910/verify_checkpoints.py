"""Replay every saved neural fit and check the temporal architecture in its runtime."""
from train import *

def main():
    d=np.load(HERE/'data.npz');p=np.load(HERE/'neural_predictions.npz')
    origins=pd.to_datetime(d['origin_ns'],utc=True);predorig=pd.to_datetime(p['origin_ns'],utc=True)
    target=pd.to_datetime(d['target_ns'],utc=True);results=[]
    for path in sorted((HERE/'checkpoints').glob('*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False)
        month=path.stem[-7:];name=path.stem[:-8];cut=pd.Timestamp(month+'-01',tz='UTC')
        tr=target<cut;xx=d['X'][tr]
        np.testing.assert_allclose(state['mu'],xx.reshape(-1,30).mean(0,dtype=np.float64),atol=1e-12)
        sd=xx.reshape(-1,30).std(0,dtype=np.float64);sd[sd<1e-6]=1
        np.testing.assert_allclose(state['sd'],sd,atol=1e-12)
        np.testing.assert_allclose(state['y_mean'],d['y'][tr].mean(),atol=1e-12)
        ids=np.flatnonzero(origins.strftime('%Y-%m')==month);ids=ids[np.linspace(0,len(ids)-1,3,dtype=int)]
        dest=predorig.get_indexer(origins[ids]);assert (dest>=0).all()
        m=Model(state['family'],30,[0,0,0]);m.load_state_dict(state['state_dict']);m.eval()
        maxerr=0.
        for j,(_,coin,sign,span) in enumerate(scenarios()):
            x=d['X'][ids] if not j else augment(d['X'][ids],d['K'][ids],coin,sign,span)
            actual=predict(m,x,state['mu'],state['sd'],state['y_mean'],state['y_sd'])
            expected=p[name][dest,j];maxerr=max(maxerr,float(abs(actual-expected).max()))
            np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=1e-6)
        if state['family']=='tcn':
            a=torch.from_numpy(((d['X'][ids]-state['mu'])/state['sd']).astype(np.float32)).transpose(1,2)
            b=a.clone();b[:,:,12:]+=30
            with torch.no_grad():early_a=m.enc(a)[:,:,:12];early_b=m.enc(b)[:,:,:12]
            torch.testing.assert_close(early_a,early_b,rtol=0,atol=0)
        results.append(dict(checkpoint=path.name,max_replay_error_bp=maxerr,causality_checked=state['family']=='tcn'))
    assert len(results)==72
    pd.DataFrame(results).to_csv(HERE/'checkpoint_replay.csv',index=False)
    (HERE/'CHECKPOINT_VERIFICATION.json').write_text(json.dumps(dict(status='passed',checkpoints=72,
        scenarios_per_checkpoint=31,origins_per_checkpoint=3,normalization='past training only',
        max_replay_error_bp=max(r['max_replay_error_bp'] for r in results),tcn_causality_checks=36,
        torch=torch.__version__,code_sha256=sha(Path(__file__))),indent=2))
    print((HERE/'CHECKPOINT_VERIFICATION.json').read_text(),flush=True)

if __name__=='__main__':main()
