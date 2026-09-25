"""Feature clocks only; no new performance calculation."""
from cc_core import *


def main():
    verify_inputs()
    assert not (PREP/'context.csv.gz').exists()
    panel=old.legacy.a.panel('available_macro')
    canonical=read(PARENT/'results/targets.csv.gz')
    frames=[];notes=[]
    for definition in DEFS:
        for cutoff in old.MONTHS:
            fold=cutoff.strftime('%Y-%m')
            tr,te,r=r12.case(panel,definition,cutoff)
            d=te[['target','target_time','e_now','e_rms72']].reset_index()
            d['definition']=definition;d['fold']=fold
            for col,out in [('e_now','e_rank'),('e_rms72','vol_rank')]:
                prior=np.sort(tr[col].to_numpy())
                d[out]=np.searchsorted(prior,d[col].to_numpy(),side='right')/len(prior)
            model=dv.fit_state(tr,cutoff);d['state']=dv.assign_state(d,model)
            c=canonical[(canonical.definition==definition)&(canonical.fold==fold)].sort_values('origin')
            assert list(d.origin)==list(c.origin)
            np.testing.assert_allclose(d[['target','e_now','e_rms72']],c[['target','e_now','e_rms72']],atol=1e-11,rtol=0)
            frames.append(d)
            notes.append(dict(definition=definition,fold=fold,train_n=len(tr),last_train_label=tr.target_time.max().isoformat(),
                residualizer=r.metadata(),state_model=model))
        print('PREPARED feature clocks',definition,flush=True)
    save(pd.concat(frames,ignore_index=True),PREP/'context.csv.gz')
    write_new(PREP/'metadata.json',notes)


if __name__=='__main__':main()
