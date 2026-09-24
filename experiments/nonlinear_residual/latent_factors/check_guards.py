"""Research-critical behavioral checks, independent of the real evaluation data."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from models import Model, candidates, study, conditional_value_gradient, ReconstructionMLP


def main():
    rng=np.random.default_rng(784)
    u=rng.uniform(-2,2,500)
    k=np.column_stack([u,u+.5*u*u,u-.7*u*u,u+.25*u**3,u-.3*u**3])+rng.normal(0,.03,(500,5))
    data=pd.DataFrame(k,columns=study.KCOLS,index=pd.date_range("2025-01-01",periods=500,freq="h",tz="UTC"))
    data["m"]=data[study.KCOLS].mean(axis=1)
    data["g"]=rng.normal(size=500)
    data["y"]=3*u+.5*data.g+rng.normal(0,.05,500)
    cutoff=data.index[350];test=data.iloc[350:]
    altered=data.copy();altered.loc[altered.index>=cutoff]=1e8
    seen=set();checks=[]
    for c in candidates():
        if c["family"] in seen:continue
        seen.add(c["family"])
        a,b=Model(c).fit(data,cutoff),Model(c).fit(altered,cutoff)
        np.testing.assert_allclose(a.predict(test),b.predict(test),rtol=0,atol=1e-9)
        if c["family"].startswith("ae_"):
            changed=data.copy();changed["y"]=rng.normal(0,100,len(changed));changed["g"]=rng.normal(0,10,len(changed))
            other=Model(c).fit(changed,cutoff)
            for (na,*_),(nb,*_) in zip(a.members,other.members):
                for wa,wb in zip(na.coefs_,nb.coefs_):
                    np.testing.assert_allclose(wa,wb,rtol=0,atol=0)
            checks.append(c["family"]+": USDT/g excluded from encoder")
    old=study.PCASequential().fit(data,cutoff)
    pca=Model(next(c for c in candidates() if c["family"]=="pca_1")).fit(data,cutoff)
    np.testing.assert_allclose(old.predict(test),pca.predict(test),rtol=1e-10,atol=1e-10)
    z,c,y=rng.normal(size=(3,80));width=4;theta=rng.normal(0,.2,3+3*width)
    err=check_grad(lambda t:conditional_value_gradient(t,z,c,y,width,.01)[0],
                   lambda t:conditional_value_gradient(t,z,c,y,width,.01)[1],theta)
    assert err<1e-6,err
    # The nonlinear conditional model exactly nests the linear loading at v=0.
    base=np.array([.2,.8,-.3]);theta=np.r_[base,rng.normal(size=width),rng.normal(size=width),np.zeros(width)]
    loss,_=conditional_value_gradient(theta,z,c,y,width,0.)
    np.testing.assert_allclose(loss,np.mean((base[0]+z*(base[1]+base[2]*c)-y)**2),atol=1e-12)
    # Check the actual estimator objective/gradient, not a mirror implementation.
    from sklearn.neural_network._multilayer_perceptron import _pack
    import warnings
    x=rng.normal(size=(20,5))
    net=ReconstructionMLP(hidden_layer_sizes=(3,1,3),activation="tanh",alpha=.5,
        solver="lbfgs",max_iter=2,random_state=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore");net.fit(x,x)
    sizes=[5,3,1,3,5]
    args=(x,x,[x]+[np.empty((len(x),s)) for s in sizes[1:]],
        [np.empty((len(x),s)) for s in sizes[1:]],
        [np.empty_like(w) for w in net.coefs_],[np.empty_like(b) for b in net.intercepts_])
    packed=_pack(net.coefs_,net.intercepts_)
    ae_err=check_grad(lambda t:net._loss_grad_lbfgs(t,*args)[0],lambda t:net._loss_grad_lbfgs(t,*args)[1],packed)
    assert ae_err<1e-5,ae_err
    # A known one-dimensional curved manifold should be reconstructible more
    # accurately than a straight PCA line. This uses synthetic data only.
    ae=Model(next(c for c in candidates() if c["family"]=="ae_tanh_1")).fit(data,cutoff)
    ratio=float(ae.reconstruction_error(test).mean()/pca.reconstruction_error(test).mean())
    assert ratio<.8,ratio
    print(json.dumps(dict(status="passed",families_past_only=len(seen),gradient_error=float(err),
        autoencoder_gradient_error=float(ae_err),synthetic_curved_manifold_mse_ratio=ratio,
        checks=checks+["PCA1 reproduces existing implementation","conditional loading nests linear baseline",
                       "actual AE numerical gradient including multioutput loss/L2","nonlinear manifold recovery"])))


if __name__=="__main__":
    main()
