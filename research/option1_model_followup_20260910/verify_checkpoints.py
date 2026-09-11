from train import *

def main():
    d=np.load(HERE/'data.npz');p=np.load(HERE/'neural_predictions.npz');o=pd.to_datetime(d['origin_ns'],utc=True);po=pd.to_datetime(p['origin_ns'],utc=True);t=pd.to_datetime(d['target_ns'],utc=True);rows=[]
    for path in sorted((HERE/'checkpoints').glob('*.pt')):
        s=torch.load(path,map_location='cpu',weights_only=False);month=path.stem[-7:];name=path.stem[:-8]
        tr=t<pd.Timestamp(month+'-01',tz='UTC');X=d['X'][tr];mu=X.reshape(-1,30).mean(0,dtype=np.float64);sd=X.reshape(-1,30).std(0,dtype=np.float64);sd[sd<1e-6]=1
        np.testing.assert_array_equal(mu,s['mu']);np.testing.assert_array_equal(sd,s['sd']);np.testing.assert_allclose(s['y_mean'],d['y'][tr].mean(),atol=1e-12)
        ids=np.flatnonzero(o.strftime('%Y-%m')==month);ids=ids[np.linspace(0,len(ids)-1,3,dtype=int)];dest=po.get_indexer(o[ids]);assert (dest>=0).all()
        m=Model(s['family'],30,[0,0,0]);m.load_state_dict(s['state_dict']);m.eval();err=0.
        for j,(_,coin,sign,span) in enumerate(scenarios()):
            x=d['X'][ids] if j==0 else augment(d['X'][ids],d['K'][ids],coin,sign,span)
            got=predict(m,x,mu,sd,s['y_mean'],s['y_sd']);want=p[name][dest,j]
            err=max(err,float(abs(got-want).max()));np.testing.assert_allclose(got,want,atol=3e-5,rtol=1e-6)
        if s['family'] in ['gru','tcn']:
            a=torch.from_numpy(((d['X'][ids]-mu)/sd).astype(np.float32));b=a.clone();b[:,12:]+=17
            with torch.no_grad():
                if s['family']=='gru':pa=m.enc(a)[0][:,:12];pb=m.enc(b)[0][:,:12]
                else:pa=m.enc(a.transpose(1,2))[:,:,:12];pb=m.enc(b.transpose(1,2))[:,:,:12]
            torch.testing.assert_close(pa,pb,rtol=0,atol=0)
        rows.append(dict(checkpoint=path.name,model=s['family'],max_error_bp=err,parameters=sum(v.numel() for v in m.parameters())))
    assert len(rows)==96;pd.DataFrame(rows).to_csv(HERE/'checkpoint_replay.csv',index=False)
    (HERE/'CHECKPOINT_VERIFICATION.json').write_text(json.dumps(dict(status='passed',checkpoints=96,scenarios_per_checkpoint=31,origins_per_checkpoint=3,causality_checks=60,max_error_bp=max(r['max_error_bp'] for r in rows),code_sha256=sha(Path(__file__))),indent=2))
    print((HERE/'CHECKPOINT_VERIFICATION.json').read_text(),flush=True)

if __name__=='__main__':main()
