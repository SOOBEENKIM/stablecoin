"""Prepare exact price identities and eligibility counts; do not score hypotheses."""
from ac_core import *


def main():
    cc.verify_seal();assert not (PREP/'context.csv.gz').exists()
    panel=old.legacy.a.panel('available_macro');raw=raw_market()
    frames=[];metadata=[];counts=[]
    canonical=cc.read(cc.PARENT/'results/targets.csv.gz')
    for definition in DEFS:
        for cutoff in old.OUTER_MONTHS:
            fold=cutoff.strftime('%Y-%m');r=old.legacy.a.Residualizer(definition).fit(panel,cutoff)
            base=canonical[(canonical.definition==definition)&(canonical.fold==fold)]
            for h in HORIZONS:
                d=extended_frame(panel,raw,r,h)
                tr=d[d.target_time<cutoff]
                te=d.loc[d.index.intersection(pd.DatetimeIndex(base.origin))]
                te=te[te.target_time<r12.hs.END]
                if h==12:
                    assert len(te)==len(base)
                    np.testing.assert_allclose(te.target,base.target,atol=1e-11,rtol=0)
                path=PREP/'cases'/f'{definition}__{fold}__{h}.pkl.gz';path.parent.mkdir(parents=True,exist_ok=True)
                pd.to_pickle(dict(train=tr,test=te),path,compression='gzip')
                out=te.reset_index();out['definition']=definition;out['fold']=fold;out['h']=h;frames.append(out)
                metadata.append(dict(definition=definition,fold=fold,h=h,train_n=len(tr),test_n=len(te),
                    last_train_label=tr.target_time.max().isoformat(),residualizer=r.metadata(),case_sha256=sha(path)))
                for threshold in [0,5,10]:
                    z=te[te.abs_e>threshold]
                    counts.append(dict(definition=definition,fold=fold,h=h,minimum_gap_bp=threshold,n=len(z),
                        premium_n=int(z.premium.sum()),discount_n=int((z.premium==0).sum())))
        print('PREPARED exact channels and clocks',definition,flush=True)
    save(pd.concat(frames,ignore_index=True),PREP/'context.csv.gz')
    save(pd.DataFrame(counts),PREP/'eligibility_counts.csv')
    write_new(PREP/'metadata.json',metadata)


if __name__=='__main__':main()
