from shared import *
import time
import torch
from torch import nn
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class SharedDropout(nn.Module):
    """One random mask per original example, broadcast to its price candidates."""
    def __init__(self,p=.1):super().__init__();self.p=p;self.groups=1
    def forward(self,x):
        if not self.training or self.p==0:return x
        assert len(x)%self.groups==0
        n=len(x)//self.groups;shape=(n,1)+tuple(x.shape[1:])
        mask=(torch.rand(shape,device=x.device)>=self.p).to(x.dtype)/(1-self.p)
        return (x.reshape((n,self.groups)+tuple(x.shape[1:]))*mask).reshape_as(x)

class Block(nn.Module):
    def __init__(self,p,q,d):
        super().__init__();self.pad=2*d
        self.a=nn.Conv1d(p,q,3,padding=self.pad,dilation=d);self.b=nn.Conv1d(q,q,3,padding=self.pad,dilation=d)
        self.skip=nn.Conv1d(p,q,1) if p!=q else nn.Identity();self.drop=SharedDropout(.1)
    def forward(self,x):
        y=self.drop(torch.relu(self.a(x)[:,:,:-self.pad]));y=self.drop(torch.relu(self.b(y)[:,:,:-self.pad]))
        return torch.relu(self.skip(x)+y)

class Model(nn.Module):
    def __init__(self,family,p,bias):
        super().__init__();self.family=family
        if family=='gru':self.enc=nn.GRU(p,30,batch_first=True);self.drop=SharedDropout(.1);self.head=nn.Linear(30,3)
        else:self.enc=nn.Sequential(Block(p,16,1),Block(16,16,2),Block(16,16,4));self.head=nn.Linear(16,3)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():self.head.bias.copy_(torch.as_tensor(bias,dtype=torch.float32))
    def forward(self,x):
        grouped=x.ndim==4;n=len(x);groups=x.shape[1] if grouped else 1
        for module in self.modules():
            if isinstance(module,SharedDropout):module.groups=groups
        xx=x.flatten(0,1) if grouped else x
        v=self.drop(self.enc(xx)[0][:,-1]) if self.family=='gru' else self.enc(xx.transpose(1,2))[:,:,-1]
        out=self.head(v)
        return out.reshape(n,groups,3) if grouped else out

def losses(y,p):
    q=torch.tensor([.1,.5,.9]);e=y[...,None]-p
    return torch.maximum(q*e,(q-1)*e).mean(-1)

def objective(y,p,aggregation):
    values=losses(y[:,None],p)
    branch=values.mean(1) if aggregation=='mean' else values.max(1).values
    return (.5*values[:,0]+.5*branch).mean(),values

def candidates(X,K,mu,sd,placement,seed):
    rng=np.random.default_rng(seed+101);permutation=np.argsort(rng.random((len(X),5)),axis=1);xs=[X]
    for _,coin,sign,span in scenarios()[1:11]:xs.append(augment(X,K,coin,sign,span,permutation[:,coin] if placement=='shuffled' else None))
    return ((np.stack(xs,axis=1)-mu)/sd).astype(np.float32)

def parameter_hash(m):
    h=hashlib.sha256()
    for name,t in sorted(m.state_dict().items()):h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def predict(m,X,mu,sd,ym,ys):
    m.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(X),512):out.append(m(torch.from_numpy(((X[start:start+512]-mu)/sd).astype(np.float32))).numpy()*ys+ym)
    return np.sort(np.concatenate(out),axis=-1)

def fit(family,placement,aggregation,X,K,y,seed,epochs):
    torch.manual_seed(seed);mu=X.reshape(-1,30).mean(0,dtype=np.float64);sd=X.reshape(-1,30).std(0,dtype=np.float64);sd[sd<1e-6]=1
    ym=float(y.mean());ys=max(float(y.std()),1.);yn=((y-ym)/ys).astype(np.float32)
    m=Model(family,30,np.quantile(yn,QS));initial=parameter_hash(m)
    cache=candidates(X,K,mu,sd,placement,seed);cache_hash=hashlib.sha256(cache.tobytes()).hexdigest()
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);shuffle=np.random.default_rng(seed+37)
    order_hash=hashlib.sha256();curves=[];steps=0
    for epoch in range(1,epochs+1):
        order=shuffle.permutation(len(X));order_hash.update(order.tobytes());total=np.zeros(4);counts=np.zeros(11,dtype=int)
        m.train()
        for start in range(0,len(X),128):
            ix=order[start:start+128];xb=torch.from_numpy(cache[ix]);yb=torch.from_numpy(yn[ix]);p=m(xb)
            loss,values=objective(yb,p,aggregation);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();steps+=1
            v=values.detach();counts+=np.bincount(v.argmax(1).numpy(),minlength=11)
            total+=len(ix)*np.array([float(loss.detach()),float(v[:,0].mean()),float(v.mean()),float(v.max(1).values.mean())])
        curves.append(dict(epoch=epoch,train_loss_scaled=total[0]/len(X),clean_loss_scaled=total[1]/len(X),candidate_mean_loss_scaled=total[2]/len(X),candidate_max_loss_scaled=total[3]/len(X),argmax_counts=json.dumps(counts.tolist())))
    meta=dict(initial_parameter_hash=initial,candidate_hash=cache_hash,batch_order_hash=order_hash.hexdigest(),
              final_torch_rng_hash=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),optimizer_steps=steps)
    return m,mu,sd,ym,ys,curves,meta

def main():
    start=time.time();root=Path('/research') if Path('/research').exists() else HERE.parent;assert_frozen(root)
    d=np.load(HERE/'data.npz');X=d['X'];K=d['K'];y=d['y'];o=pd.to_datetime(d['origin_ns'],utc=True);target=pd.to_datetime(d['target_ns'],utc=True)
    anchors=pd.read_csv(HERE/'anchor_epochs.csv');use=o>=pd.Timestamp('2025-12-01',tz='UTC');dates=o[use];sc=scenarios()
    out={f'{f}_{p}_{a}_{s}':np.full((len(dates),31,3),np.nan,dtype=np.float32) for f in ['tcn','gru'] for p in ['tick','shuffled'] for a in ['mean','max'] for s in SEEDS}
    settings=dict(status='started',code_sha256=sha(Path(__file__)),shared_sha256=sha(HERE/'shared.py'),protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),data_sha256=sha(HERE/'data.npz'),anchor_sha256=sha(HERE/'anchor_epochs.csv'),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device='cpu')
    (HERE/'TRAIN_SETTINGS.json').write_text(json.dumps(settings,indent=2));(HERE/'checkpoints').mkdir(exist_ok=True);fits=[];traces=[]
    for month in sorted(set(dates.strftime('%Y-%m'))):
        cut=pd.Timestamp(month+'-01',tz='UTC');tr=target<cut;te=o.strftime('%Y-%m')==month;dest=dates.strftime('%Y-%m')==month
        assert target[tr].max()<o[te].min()
        sx=[X[te]]+[augment(X[te],K[te],c,sign,span) for _,c,sign,span in sc[1:]]
        for family in ['tcn','gru']:
            for seed in SEEDS:
                anchor=anchors[(anchors.family==family)&(anchors.seed==seed)&(anchors.month==month)].iloc[0];epochs=int(anchor.epochs)
                for placement in ['tick','shuffled']:
                    for aggregation in ['mean','max']:
                        name=f'{family}_{placement}_{aggregation}_{seed}'
                        m,mu,sd,ym,ys,curve,meta=fit(family,placement,aggregation,X[tr],K[tr],y[tr],seed,epochs)
                        for j,xx in enumerate(sx):out[name][dest,j]=predict(m,xx,mu,sd,ym,ys)
                        torch.save(dict(state_dict=m.state_dict(),mu=mu,sd=sd,y_mean=ym,y_sd=ys,family=family,placement=placement,aggregation=aggregation,seed=seed,epochs=epochs),HERE/'checkpoints'/(name+'_'+month+'.pt'))
                        fits.append(dict(model=name,family=family,placement=placement,aggregation=aggregation,seed=seed,month=month,epochs=epochs,n_train=int(tr.sum()),n_forecast=int(te.sum()),n_parameters=sum(v.numel() for v in m.parameters()),last_training_target=str(target[tr].max()),first_test_origin=str(o[te].min()),**meta))
                        traces.extend([dict(model=name,month=month,**r) for r in curve]);pd.DataFrame(fits).to_csv(HERE/'neural_training.csv',index=False);pd.DataFrame(traces).to_csv(HERE/'training_curves.csv',index=False)
                        np.savez_compressed(HERE/'neural_predictions.npz',origin_ns=dates.asi8,target=y[use],scenario_names=np.array([s[0] for s in sc]),**out)
                        print(name,month,'epochs',epochs,'fits',len(fits),'/96','seconds',round(time.time()-start,1),flush=True)
    assert all(np.isfinite(v).all() for v in out.values());assert_frozen(root)
    f=pd.DataFrame(fits)
    for _,g in f.groupby(['family','seed','month']):
        for c in ['epochs','initial_parameter_hash','batch_order_hash','final_torch_rng_hash','optimizer_steps']:assert g[c].nunique()==1,c
        for _,pair in g.groupby('placement'):assert pair.candidate_hash.nunique()==1
    for key,file in [('code_sha256','train.py'),('data_sha256','data.npz'),('anchor_sha256','anchor_epochs.csv')]:assert settings[key]==sha(HERE/file)
    (HERE/'TRAIN_COMPLETION.json').write_text(json.dumps(dict(status='complete',fits=len(fits),seconds=time.time()-start,matched_candidate_and_rng_checks='passed',training_code_unchanged=True),indent=2))

if __name__=='__main__':main()
