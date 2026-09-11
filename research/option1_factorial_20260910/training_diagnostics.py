"""Prespecified descriptive argmax accounting, not feature importance or market causality."""
from shared import *

def main():
    fits=pd.read_csv(HERE/'neural_training.csv');curves=pd.read_csv(HERE/'training_curves.csv');rows=[]
    assert len(fits)==96
    for _,r in fits.iterrows():
        c=curves[(curves.model==r.model)&(curves.month==r.month)];counts=np.sum([json.loads(x) for x in c.argmax_counts],axis=0);total=int(counts.sum());assert total==r.epochs*r.n_train
        row=dict(model=r.model,family=r.family,placement=r.placement,aggregation=r.aggregation,month=r.month,seed=int(r.seed),examples_over_epochs=total,clean_argmax_fraction=counts[0]/total)
        for j,coin in enumerate(COINS):row['donor_'+coin+'_argmax_fraction']=(counts[1+2*j]+counts[2+2*j])/total
        row['note']='Donor identity; shuffled candidates may be applied to a different recipient; mean learning does not select this candidate'
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/'training_candidate_summary.csv',index=False)

if __name__=='__main__':main()
