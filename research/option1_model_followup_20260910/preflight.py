from train import *

def main():
    d=np.load(HERE/'data.npz');X=d['X'][:40];K=d['K'][:40];y=d['y'][:40];rows=[]
    for family in ['gru','mlp','tcn']:
        torch.manual_seed(20260910);m=Model(family,30,[-1,0,1]);m.eval()
        p=m(torch.from_numpy(X));assert p.shape==(40,3) and torch.isfinite(p).all()
        rows.append(dict(family=family,parameters=sum(v.numel() for v in m.parameters())))
    assert [r['parameters'] for r in rows]==[5673,5795,5923]
    mu=X.mean((0,1));sd=X.std((0,1));sd[sd<1e-6]=1
    a=candidates(X,K,mu,sd,'hard_tick',20260910);b=candidates(X,K,mu,sd,'hard_shuffled',20260910)
    # Undo normalization and compare total absolute individual-basis price change per candidate.
    au=a*sd+mu;bu=b*sd+mu
    np.testing.assert_allclose(abs(au[:,:,:,:5]-X[:,None,:,:5]).sum(-1),abs(bu[:,:,:,:5]-X[:,None,:,:5]).sum(-1),atol=3e-4,rtol=1e-4)
    assert np.isfinite(a).all() and a.shape==(40,11,24,30)
    # Nonconstant outputs exercise nontrivial per-example worst selection.
    with torch.no_grad():m.head.weight.normal_(std=.1)
    yy=torch.from_numpy(y.astype(np.float32));ii,values=choose(m,torch.from_numpy(a),yy)
    manual=[]
    for i in range(len(a)):
        with torch.no_grad():p=m(torch.from_numpy(a[i])).sort(dim=-1).values.numpy()
        e=y[i]-p;v=np.maximum(QS*e,(QS-1)*e).mean(1);manual.append(v)
    np.testing.assert_allclose(values.numpy(),np.asarray(manual),atol=1e-5,rtol=1e-5)
    for i,j in enumerate(ii):assert float(values[i,j])>=float(values[i].max())-1e-6
    for family,method in [('gru','tick'),('mlp','shuffled'),('tcn','hard_tick'),('tcn','hard_shuffled')]:
        fitted,*_=fit(family,method,X,K,y,20260910,epochs=1)
        assert all(torch.isfinite(v).all() for v in fitted.parameters()) and fitted.head.weight.abs().max()>0
    (HERE/'PREFLIGHT.json').write_text(json.dumps(dict(status='passed',models=rows,checks=['same 24x30 information','near-matched parameter count','matched candidate perturbation magnitude','independent worst-candidate loss and argmax','finite gradients for each new training branch'],code_sha256=sha(Path(__file__))),indent=2))
    print((HERE/'PREFLIGHT.json').read_text(),flush=True)

if __name__=='__main__':main()
