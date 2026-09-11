from shared import *
import time,copy
import torch
from torch import nn

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class Block(nn.Module):
    def __init__(self,p,q,d):
        super().__init__();self.pad=2*d
        self.a=nn.Conv1d(p,q,3,padding=self.pad,dilation=d);self.b=nn.Conv1d(q,q,3,padding=self.pad,dilation=d)
        self.skip=nn.Conv1d(p,q,1) if p!=q else nn.Identity();self.drop=nn.Dropout(.1)
    def forward(self,x):
        y=self.drop(torch.relu(self.a(x)[:,:,:-self.pad]));y=self.drop(torch.relu(self.b(y)[:,:,:-self.pad]))
        return torch.relu(self.skip(x)+y)

class Model(nn.Module):
    def __init__(self,family,p,bias):
        super().__init__();self.family=family
        if family=='tcn':self.enc=nn.Sequential(Block(p,16,1),Block(16,16,2),Block(16,16,4));self.head=nn.Linear(16,3)
        else:self.head=nn.Linear(3*p,3)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():self.head.bias.copy_(torch.as_tensor(bias,dtype=torch.float32))
    def forward(self,x):
        v=self.enc(x.transpose(1,2))[:,:,-1] if self.family=='tcn' else torch.cat([x[:,-1],x[:,-6:].mean(1),x.mean(1)],1)
        return self.head(v)

def criterion(y,p,tails=False):
    q=torch.as_tensor([.1,.5,.9]);e=y[:,None]-p;loss=torch.maximum(q*e,(q-1)*e)
    return loss[:,[0,2]].mean() if tails else loss.mean()

def predict(model,x,mu,sd,ym,ys):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(x),512):
            xx=torch.from_numpy(((x[start:start+512]-mu)/sd).astype(np.float32))
            out.append(model(xx).numpy()*ys+ym)
    return np.sort(np.concatenate(out),axis=1)

def fit(family,method,X,K,y,seed,epochs=60,validation=None):
    torch.manual_seed(seed)
    mu=X.reshape(-1,X.shape[-1]).mean(0,dtype=np.float64);sd=X.reshape(-1,X.shape[-1]).std(0,dtype=np.float64);sd[sd<1e-6]=1
    ym=float(y.mean());ys=max(float(y.std()),1.);yn=((y-ym)/ys).astype(np.float32)
    m=Model(family,X.shape[-1],np.quantile(yn,QS))
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001 if family=='tcn' else .005,weight_decay=.001)
    rng=np.random.default_rng(seed+31);shuffle=np.random.default_rng(seed+37)
    best=np.inf;best_epoch=1;stale=0;trace=[]
    for epoch in range(1,epochs+1):
        order=shuffle.permutation(len(X));m.train();total=0.
        for start in range(0,len(X),128):
            ii=order[start:start+128];xx=X[ii].copy();n=len(ii)
            donor=rng.integers(0,5,n);recipient=rng.integers(0,5,n);sign=rng.choice([-1,1],n);span=rng.choice([1,6],n);gate=rng.random(n)<.5
            if method!='clean' and gate.any():
                xx[gate]=augment(xx[gate],K[ii][gate],donor[gate],sign[gate],span[gate],recipient[gate] if method=='shuffled' else None)
            xb=torch.from_numpy(((xx-mu)/sd).astype(np.float32));yb=torch.from_numpy(yn[ii])
            optimizer.zero_grad();loss=criterion(yb,m(xb));loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step()
            total+=float(loss)*n
        val=np.nan
        if validation is not None:
            vx,vy=validation;pred=predict(m,vx,mu,sd,ym,ys)
            val=float(pinball(vy,pred)[:,[0,2]].mean())
            if val<best-1e-6:best=val;best_epoch=epoch;stale=0
            else:stale+=1
        trace.append(dict(epoch=epoch,train_loss_scaled=total/len(X),validation_tail_loss_bp=val))
        if validation is not None and stale>=8:break
    return m,mu,sd,ym,ys,best_epoch if validation is not None else epochs,trace

def main():
    started=time.time();d=np.load(HERE/'data.npz');X=d['X'];K=d['K'];y=d['y']
    origin=pd.to_datetime(d['origin_ns'],utc=True);target=pd.to_datetime(d['target_ns'],utc=True)
    valid=origin>=pd.Timestamp('2025-12-01',tz='UTC');dates=origin[valid];sc=scenarios()
    allpred={f'{f}_{m}_{s}':np.full((len(dates),len(sc),3),np.nan,dtype=np.float32) for f in ['tcn','linear'] for m in ['clean','shuffled','tick'] for s in SEEDS}
    fits=[];traces=[]
    settings=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device='cpu',seeds=SEEDS,
        code_sha256=sha(Path(__file__)),shared_sha256=sha(HERE/'shared.py'),data_sha256=sha(HERE/'data.npz'),protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),status='running')
    (HERE/'TRAIN_SETTINGS.json').write_text(json.dumps(settings,indent=2));(HERE/'checkpoints').mkdir(exist_ok=True)
    for month in sorted(set(dates.strftime('%Y-%m'))):
        cut=pd.Timestamp(month+'-01',tz='UTC');inner=cut-pd.Timedelta(days=28)
        tr=target<cut;it=target<inner;iv=(origin>=inner)&tr;te=origin.strftime('%Y-%m')==month;dest=dates.strftime('%Y-%m')==month
        assert target[tr].max()<origin[te].min() and target[it].max()<origin[iv].min()
        # Cache raw perturbations for the month; all models see exactly these inputs.
        sx=[X[te]]+[augment(X[te],K[te],coin,sign,span) for _,coin,sign,span in sc[1:]]
        for family in ['tcn','linear']:
            for method in ['clean','shuffled','tick']:
                for seed in SEEDS:
                    name=f'{family}_{method}_{seed}'
                    _,_,_,_,_,epochs,trace=fit(family,method,X[it],K[it],y[it],seed,validation=(X[iv],y[iv]))
                    m,mu,sd,ym,ys,_,_=fit(family,method,X[tr],K[tr],y[tr],seed,epochs=epochs)
                    for j,xx in enumerate(sx):allpred[name][dest,j]=predict(m,xx,mu,sd,ym,ys)
                    torch.save(dict(state_dict=m.state_dict(),mu=mu,sd=sd,y_mean=ym,y_sd=ys,family=family,method=method,seed=seed,epochs=epochs),HERE/'checkpoints'/(name+'_'+month+'.pt'))
                    fits.append(dict(model=name,month=month,epochs=epochs,n_train=int(tr.sum()),n_inner=int(it.sum()),n_valid=int(iv.sum()),n_forecast=int(te.sum()),
                        n_parameters=sum(p.numel() for p in m.parameters()),last_training_target=str(target[tr].max()),first_test_origin=str(origin[te].min())))
                    traces.extend([dict(model=name,month=month,**r) for r in trace])
                    np.savez_compressed(HERE/'neural_predictions.npz',origin_ns=dates.asi8,target=y[valid],scenario_names=np.asarray([s[0] for s in sc]),**allpred)
                    pd.DataFrame(fits).to_csv(HERE/'neural_training.csv',index=False);pd.DataFrame(traces).to_csv(HERE/'training_curves.csv',index=False)
                    print(name,month,'epochs',epochs,'fits',len(fits),'/72','seconds',round(time.time()-started,1),flush=True)
    assert all(np.isfinite(v).all() for v in allpred.values())
    (HERE/'TRAIN_COMPLETION.json').write_text(json.dumps(dict(status='complete',seconds=time.time()-started,fits=len(fits),
        code_unchanged=sha(Path(__file__))==settings['code_sha256'],data_unchanged=sha(HERE/'data.npz')==settings['data_sha256'],protocol_unchanged=sha(HERE/'PROTOCOL_KO.md')==settings['protocol_sha256']),indent=2))

if __name__=='__main__':main()
