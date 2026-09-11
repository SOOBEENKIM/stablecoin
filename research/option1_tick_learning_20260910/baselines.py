from shared import *
import time
from sklearn.linear_model import QuantileRegressor
from lightgbm import LGBMRegressor

def main():
    start=time.time();d=np.load(HERE/'data.npz');X=d['X'];K=d['K'];y=d['y'];origin=pd.to_datetime(d['origin_ns'],utc=True);target=pd.to_datetime(d['target_ns'],utc=True)
    use=origin>=pd.Timestamp('2025-12-01',tz='UTC');dates=origin[use];sc=scenarios()
    out={m:np.full((len(dates),len(sc),3),np.nan) for m in ['exact_qr','lgbm']};records=[]
    for month in sorted(set(dates.strftime('%Y-%m'))):
        cut=pd.Timestamp(month+'-01',tz='UTC');tr=target<cut;te=origin.strftime('%Y-%m')==month;dest=dates.strftime('%Y-%m')==month
        assert target[tr].max()<origin[te].min()
        z=tabular(X[tr]);mu=z.mean(0);sd=z.std(0);sd[sd<1e-6]=1;z=(z-mu)/sd
        sx=[tabular(X[te])]+[tabular(augment(X[te],K[te],coin,sign,span)) for _,coin,sign,span in sc[1:]]
        for model in out:
            for j,q in enumerate(QS):
                if model=='exact_qr':m=QuantileRegressor(quantile=q,alpha=.01,solver='highs')
                else:m=LGBMRegressor(objective='quantile',alpha=q,num_leaves=7,n_estimators=150,learning_rate=.03,min_child_samples=100,reg_lambda=10,
                                     n_jobs=1,random_state=20260910,verbosity=-1,deterministic=True,force_col_wise=True)
                m.fit(z,y[tr])
                for s,xx in enumerate(sx):out[model][dest,s,j]=m.predict((xx-mu)/sd)
            out[model][dest]=np.sort(out[model][dest],axis=-1)
            records.append(dict(model=model,month=month,n_train=int(tr.sum()),n_forecast=int(te.sum()),last_training_target=str(target[tr].max()),first_test_origin=str(origin[te].min())))
            print('baseline',model,month,'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(HERE/'baseline_predictions.npz',origin_ns=dates.asi8,target=y[use],scenario_names=np.asarray([s[0] for s in sc]),**out)
    pd.DataFrame(records).to_csv(HERE/'baseline_training.csv',index=False)
    (HERE/'BASELINE_COMPLETION.json').write_text(json.dumps(dict(status='complete',code_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'PROTOCOL_KO.md'),
        seconds=time.time()-start,data_sha256=sha(HERE/'data.npz')),indent=2))

if __name__=='__main__':main()
