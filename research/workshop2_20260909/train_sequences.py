"""CPU PyTorch sequence comparison, fixed architectures and three seeds."""
from pathlib import Path
import json,hashlib,time,copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
SEEDS=[20260909,20260910,20260911]


class CausalBlock(nn.Module):
    def __init__(self,inputs,outputs,dilation):
        super().__init__();self.pad=2*dilation
        self.c1=nn.Conv1d(inputs,outputs,3,dilation=dilation,padding=self.pad)
        self.c2=nn.Conv1d(outputs,outputs,3,dilation=dilation,padding=self.pad)
        self.skip=nn.Conv1d(inputs,outputs,1) if inputs!=outputs else nn.Identity()
        self.drop=nn.Dropout(.1)
    def forward(self,x):
        y=self.drop(torch.relu(self.c1(x)[:,:,:-self.pad]))
        y=self.drop(torch.relu(self.c2(y)[:,:,:-self.pad]))
        return torch.relu(y+self.skip(x))


class Predictor(nn.Module):
    def __init__(self,family,p):
        super().__init__();self.family=family
        if family=='gru':
            self.encoder=nn.GRU(p,16,batch_first=True)
        elif family=='tcn':
            self.encoder=nn.Sequential(CausalBlock(p,16,1),CausalBlock(16,16,2),CausalBlock(16,16,4))
        else:
            self.encoder=nn.Sequential(nn.Linear(p,16),nn.ReLU(),nn.Dropout(.1))
        self.head=nn.Linear(16,1)
    def forward(self,x):
        if self.family=='gru':
            _,h=self.encoder(x);h=h[-1]
        elif self.family=='tcn':
            h=self.encoder(x.transpose(1,2))[:,:,-1]
        else:
            h=self.encoder(x[:,-1,:])
        return self.head(h).squeeze(-1)


def fit(family,X,y,seed,epochs=50,validation=None):
    torch.manual_seed(seed);np.random.seed(seed)
    mean=X.reshape(-1,X.shape[-1]).mean(axis=0)
    sd=X.reshape(-1,X.shape[-1]).std(axis=0);sd[sd<1e-8]=1
    xt=torch.from_numpy(((X-mean)/sd).astype(np.float32));yt=torch.from_numpy(y)
    loader=DataLoader(TensorDataset(xt,yt),batch_size=128,shuffle=True,
        generator=torch.Generator().manual_seed(seed),num_workers=0)
    m=Predictor(family,X.shape[-1]);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
    loss=nn.BCEWithLogitsLoss();best=np.inf;best_epoch=1;stale=0;trace=[]
    if validation is not None:
        xv,yv=validation;vx=torch.from_numpy(((xv-mean)/sd).astype(np.float32));vy=torch.from_numpy(yv)
    for epoch in range(1,epochs+1):
        m.train();total=0.
        for xb,yb in loader:
            opt.zero_grad();l=loss(m(xb),yb);l.backward();nn.utils.clip_grad_norm_(m.parameters(),1)
            opt.step();total+=l.item()*len(xb)
        vl=np.nan
        if validation is not None:
            m.eval()
            with torch.no_grad():vl=float(loss(m(vx),vy))
            if vl<best-1e-6:
                best=vl;best_epoch=epoch;stale=0
            else:stale+=1
        trace.append({'epoch':epoch,'training_logloss':total/len(X),'validation_logloss':vl})
        if validation is not None and stale>=7:break
    if validation is None:best_epoch=epochs
    m.eval()
    return m,mean,sd,best_epoch,trace


def main():
    frozen=[HERE/'SEQUENCE_PROTOCOL_KO.md',HERE/'prepare_sequences.py',HERE/'train_sequences.py',HERE/'sequence_data.npz']
    manifest={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    (HERE/'SEQUENCE_RUN_MANIFEST.json').write_text(json.dumps({'torch':torch.__version__,'numpy':np.__version__,
        'hashes':manifest,'seeds':SEEDS,'device':'cpu'},indent=2))
    data=np.load(HERE/'sequence_data.npz');X=data['X'];y=data['y'];ix=pd.to_datetime(data['origin_ns'],utc=True)
    observed=ix+pd.Timedelta(hours=6);forecast=ix>=pd.Timestamp('2025-11-01',tz='UTC')
    origins=ix[forecast];out=pd.DataFrame({'target':y[forecast],'target_time':observed[forecast]},index=origins)
    allfits=[];alltraces=[];start=time.time()
    for family in ['mlp','gru','tcn']:
        for seed in SEEDS:
            p=np.zeros(len(origins));name=family+'_'+str(seed)
            for month in sorted(set(origins.strftime('%Y-%m'))):
                cut=pd.Timestamp(month+'-01',tz='UTC');inner=cut-pd.Timedelta(days=28)
                train=observed<cut;innertrain=observed<inner
                innerval=(ix>=inner)&train;test=ix.strftime('%Y-%m')==month
                dest=origins.strftime('%Y-%m')==month
                assert observed[innertrain].max()<ix[innerval].min()
                assert observed[train].max()<ix[test].min()
                _,_,_,epochs,trace=fit(family,X[innertrain],y[innertrain],seed,validation=(X[innerval],y[innerval]))
                m,mu,sd,_,_=fit(family,X[train],y[train],seed,epochs=epochs)
                with torch.no_grad():p[dest]=torch.sigmoid(m(torch.from_numpy(((X[test]-mu)/sd).astype(np.float32)))).numpy()
                allfits.append(dict(model=name,month=month,epochs=epochs,n_train=int(train.sum()),
                    inner_n_train=int(innertrain.sum()),inner_n_valid=int(innerval.sum()),
                    inner_positive=int(y[innertrain].sum()),valid_positive=int(y[innerval].sum()),
                    n_parameters=sum(v.numel() for v in m.parameters()),
                    last_training_target=str(observed[train].max()),first_test_origin=str(ix[test].min())))
                alltraces.extend([dict(model=name,month=month,**r) for r in trace])
                print(name,month,'epochs',epochs,'elapsed',round(time.time()-start,1),flush=True)
            out[name]=p
            out.to_csv(HERE/'sequence_predictions.csv.gz',compression='gzip')
            pd.DataFrame(allfits).to_csv(HERE/'sequence_training.csv',index=False)
            pd.DataFrame(alltraces).to_csv(HERE/'sequence_training_curves.csv',index=False)
        out[family+'_ensemble']=out[[family+'_'+str(seed) for seed in SEEDS]].mean(axis=1)
    out.to_csv(HERE/'sequence_predictions.csv.gz',compression='gzip')
    checks={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h for name,h in manifest.items()}
    (HERE/'SEQUENCE_TRAINING_COMPLETION.json').write_text(json.dumps(dict(complete=True,elapsed_sec=time.time()-start,
        fits=len(allfits),freeze_checks=checks),indent=2))


if __name__=='__main__':main()
