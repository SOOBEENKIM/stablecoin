from train import *

def main():
    d=np.load(HERE/'data.npz');X=d['X'][:24];K=d['K'][:24];y=d['y'][:24];checks=[]
    for family,nparams in [('tcn',5923),('gru',5673)]:
        torch.manual_seed(91);m=Model(family,30,[-1,0,1]);assert sum(v.numel() for v in m.parameters())==nparams
        with torch.no_grad():m.head.weight.normal_(std=.1)
        repeated=torch.randn(5,1,24,30).repeat(1,11,1,1);m.train();out=m(repeated)
        torch.testing.assert_close(out,out[:,0:1].expand_as(out),rtol=0,atol=2e-6)
        # With identical candidates, averaging and maximization give the same objective and gradient.
        z=out.detach().clone().requires_grad_(True);yy=torch.tensor([.137,.539,-.381,.725,1.113]);lm,_=objective(yy,z,'mean');lx,_=objective(yy,z,'max')
        torch.testing.assert_close(lm,lx,atol=1e-6,rtol=1e-6)
        # Independent analytic loss and gradient accounting for arbitrary candidates.
        z=torch.randn(5,11,3,requires_grad=True);a=z.detach().numpy();e=yy.numpy()[:,None,None]-a;manual=np.maximum(QS*e,(QS-1)*e).mean(2)
        for agg in ['mean','max']:
            loss,_=objective(yy,z,agg);expected=(.5*manual[:,0]+.5*(manual.mean(1) if agg=='mean' else manual.max(1))).mean()
            np.testing.assert_allclose(float(loss.detach()),expected,atol=1e-7)
            grad=torch.autograd.grad(loss,z,retain_graph=True)[0].numpy();w=np.zeros((5,11));w[:,0]=.5
            if agg=='mean':w+=.5/11
            else:w[np.arange(5),manual.argmax(1)]+=.5
            want=np.where(e>0,-QS,1-QS)*w[:,:,None]/(5*3)
            np.testing.assert_allclose(grad,want,atol=1e-8)
        checks.append(f'{family}: parameter count, common-mask duplicate invariance, independent mean/max objective and gradient')
    mu=X.mean((0,1));sd=X.std((0,1));sd[sd<1e-6]=1
    a=candidates(X,K,mu,sd,'tick',20260910);b=candidates(X,K,mu,sd,'shuffled',20260910)
    np.testing.assert_allclose(abs((a*sd+mu)[:,:,:,:5]-X[:,None,:,:5]).sum(-1),abs((b*sd+mu)[:,:,:,:5]-X[:,None,:,:5]).sum(-1),atol=3e-4,rtol=1e-4)
    rows=[]
    for placement in ['tick','shuffled']:
        for aggregation in ['mean','max']:
            m,*rest=fit('tcn',placement,aggregation,X,K,y,20260910,1);meta=rest[-1];rows.append(dict(placement=placement,aggregation=aggregation,**meta));assert m.head.weight.abs().max()>0
    r=pd.DataFrame(rows)
    for col in ['initial_parameter_hash','batch_order_hash','final_torch_rng_hash','optimizer_steps']:assert r[col].nunique()==1
    assert r.groupby('placement').candidate_hash.nunique().eq(1).all()
    checks.append('Four cells: identical initialization, batches, dropout RNG trajectory, steps and paired candidates; financial/control price budgets match')
    (HERE/'PREFLIGHT.json').write_text(json.dumps(dict(status='passed',checks=checks,code_sha256=sha(Path(__file__))),indent=2));print((HERE/'PREFLIGHT.json').read_text(),flush=True)

if __name__=='__main__':main()
