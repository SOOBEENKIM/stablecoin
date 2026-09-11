from shared import *
import time
import torch
from torch import nn
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
FAMILIES=[('gru',m) for m in ['clean','shuffled','tick']]+[('mlp',m) for m in ['clean','shuffled','tick']]+[('tcn',m) for m in ['hard_shuffled','hard_tick']]

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
        super().__init__();self.family=family;self.drop=nn.Dropout(.1)
        if family=='gru':self.enc=nn.GRU(p,30,batch_first=True);self.head=nn.Linear(30,3)
        elif family=='mlp':self.enc=nn.Linear(24*p,8);self.head=nn.Linear(8,3)
        else:self.enc=nn.Sequential(Block(p,16,1),Block(16,16,2),Block(16,16,4));self.head=nn.Linear(16,3)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():self.head.bias.copy_(torch.as_tensor(bias,dtype=torch.float32))
    def forward(self,x):
        if self.family=='gru':v=self.drop(self.enc(x)[0][:,-1])
        elif self.family=='mlp':v=self.drop(torch.relu(self.enc(x.reshape(len(x),-1))))
        else:v=self.enc(x.transpose(1,2))[:,:,-1]
        return self.head(v)

def losses(y,p):
    q=torch.tensor([.1,.5,.9]);e=y[...,None]-p
    return torch.maximum(q*e,(q-1)*e).mean(-1)

def predict(m,x,mu,sd,ym,ys):
    m.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(x),512):out.append(m(torch.from_numpy(((x[i:i+512]-mu)/sd).astype(np.float32))).numpy()*ys+ym)
    return np.sort(np.concatenate(out),axis=-1)

def candidates(X,K,mu,sd,method,seed):
    rng=np.random.default_rng(seed+101);permutation=np.argsort(rng.random((len(X),5)),axis=1)
    xx=[X]
    for _,coin,sign,span in scenarios()[1:11]:
        xx.append(augment(X,K,coin,sign,span,permutation[:,coin] if method=='hard_shuffled' else None))
    return ((np.stack(xx,axis=1)-mu)/sd).astype(np.float32)

def choose(m,c,y):
    # Selection uses only training labels. Inference never calls this function.
    m.eval()
    with torch.no_grad():
        p=m(c.flatten(0,1)).reshape(len(c),11,3).sort(dim=-1).values
        values=losses(y[:,None],p);index=values.argmax(1)
    return index,values

def fit(family,method,X,K,y,seed,epochs=60,validation=None):
    torch.manual_seed(seed);mu=X.reshape(-1,30).mean(0,dtype=np.float64);sd=X.reshape(-1,30).std(0,dtype=np.float64);sd[sd<1e-6]=1
    ym=float(y.mean());ys=max(float(y.std()),1.);yn=((y-ym)/ys).astype(np.float32)
    m=Model(family,30,np.quantile(yn,QS));opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(seed+31);shuffle=np.random.default_rng(seed+37)
    hard=method.startswith('hard_');cache=candidates(X,K,mu,sd,method,seed) if hard else None
    best=np.inf;best_epoch=1;stale=0;trace=[]
    for epoch in range(1,epochs+1):
        order=shuffle.permutation(len(X));total=0.;selected_counts=np.zeros(11,dtype=int)
        for start in range(0,len(X),128):
            ii=order[start:start+128];xx=X[ii].copy();n=len(ii);yb=torch.from_numpy(yn[ii])
            donor=rng.integers(0,5,n);recipient=rng.integers(0,5,n);sign=rng.choice([-1,1],n);span=rng.choice([1,6],n);gate=rng.random(n)<.5
            if hard:
                c=torch.from_numpy(cache[ii]);index,_=choose(m,c,yb)
                selected_counts+=np.bincount(index.numpy(),minlength=11)
                xb=torch.cat([c[:,0],c[torch.arange(n),index]],dim=0)
                m.train();p=m(xb);loss=.5*losses(yb,p[:n]).mean()+.5*losses(yb,p[n:]).mean()
            else:
                if method!='clean' and gate.any():xx[gate]=augment(xx[gate],K[ii][gate],donor[gate],sign[gate],span[gate],recipient[gate] if method=='shuffled' else None)
                xb=torch.from_numpy(((xx-mu)/sd).astype(np.float32));m.train();loss=losses(yb,m(xb)).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();total+=float(loss.detach())*n
        val=np.nan
        if validation is not None:
            vx,vy=validation;vp=predict(m,vx,mu,sd,ym,ys);val=float(pinball(vy,vp)[:,[0,2]].mean())
            if val<best-1e-6:best=val;best_epoch=epoch;stale=0
            else:stale+=1
        trace.append(dict(epoch=epoch,train_loss_scaled=total/len(X),validation_tail_loss_bp=val,selected_counts=json.dumps(selected_counts.tolist())))
        if validation is not None and stale>=8:break
    return m,mu,sd,ym,ys,best_epoch if validation is not None else epochs,trace

def main():
    root=Path('/research') if Path('/research').exists() else HERE.parent;assert_frozen(root)
    start=time.time();d=np.load(HERE/'data.npz');X=d['X'];K=d['K'];y=d['y'];o=pd.to_datetime(d['origin_ns'],utc=True);t=pd.to_datetime(d['target_ns'],utc=True)
    use=o>=pd.Timestamp('2025-12-01',tz='UTC');dates=o[use];sc=scenarios()
    out={f'{f}_{m}_{s}':np.full((len(dates),31,3),np.nan,dtype=np.float32) for f,m in FAMILIES for s in SEEDS}
    settings=dict(status='started',torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,code_sha256=sha(Path(__file__)),shared_sha256=sha(HERE/'shared.py'),protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),data_sha256=sha(HERE/'data.npz'),families=FAMILIES,seeds=SEEDS)
    (HERE/'TRAIN_SETTINGS.json').write_text(json.dumps(settings,indent=2));(HERE/'checkpoints').mkdir(exist_ok=True);fits=[];curves=[]
    for month in sorted(set(dates.strftime('%Y-%m'))):
        cut=pd.Timestamp(month+'-01',tz='UTC');inner=cut-pd.Timedelta(days=28);tr=t<cut;it=t<inner;iv=(o>=inner)&tr;te=o.strftime('%Y-%m')==month;dest=dates.strftime('%Y-%m')==month
        assert t[tr].max()<o[te].min() and t[it].max()<o[iv].min()
        sx=[X[te]]+[augment(X[te],K[te],c,sign,span) for _,c,sign,span in sc[1:]]
        for family,method in FAMILIES:
            for seed in SEEDS:
                name=f'{family}_{method}_{seed}'
                _,_,_,_,_,epochs,trace=fit(family,method,X[it],K[it],y[it],seed,validation=(X[iv],y[iv]))
                m,mu,sd,ym,ys,_,refittrace=fit(family,method,X[tr],K[tr],y[tr],seed,epochs=epochs)
                for j,xx in enumerate(sx):out[name][dest,j]=predict(m,xx,mu,sd,ym,ys)
                torch.save(dict(state_dict=m.state_dict(),mu=mu,sd=sd,y_mean=ym,y_sd=ys,family=family,method=method,seed=seed,epochs=epochs),HERE/'checkpoints'/(name+'_'+month+'.pt'))
                fits.append(dict(model=name,month=month,epochs=epochs,n_train=int(tr.sum()),n_inner=int(it.sum()),n_valid=int(iv.sum()),n_forecast=int(te.sum()),n_parameters=sum(p.numel() for p in m.parameters()),last_training_target=str(t[tr].max()),first_test_origin=str(o[te].min())))
                for stage,tt in [('validation',trace),('refit',refittrace)]:curves.extend([dict(model=name,month=month,stage=stage,**v) for v in tt])
                np.savez_compressed(HERE/'neural_predictions.npz',origin_ns=dates.asi8,target=y[use],scenario_names=np.array([x[0] for x in sc]),**out)
                pd.DataFrame(fits).to_csv(HERE/'neural_training.csv',index=False);pd.DataFrame(curves).to_csv(HERE/'training_curves.csv',index=False)
                print(name,month,'epochs',epochs,'fits',len(fits),'/96','seconds',round(time.time()-start,1),flush=True)
    assert all(np.isfinite(x).all() for x in out.values());assert_frozen(root)
    assert sha(Path(__file__))==settings['code_sha256'] and sha(HERE/'data.npz')==settings['data_sha256']
    (HERE/'TRAIN_COMPLETION.json').write_text(json.dumps(dict(status='complete',fits=len(fits),seconds=time.time()-start,protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),training_code_unchanged=True,data_unchanged=True),indent=2))

if __name__=='__main__':main()
