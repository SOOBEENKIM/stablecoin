"""Read-only independent final audit; no new hypotheses or model selection."""
from ac_core import *
import ac_symmetric_null as sn
import ac_overshoot as x


def main():
    stamp=sn.check()
    for path in [OUT/'EVALUATION_COMPLETE.json',x.DEST/'COMPLETE.json',sn.DEST/'COMPLETE.json']:
        for name,digest in json.loads(path.read_text())['file_sha256'].items():assert sha(ROOT/name)==digest,name
    for dest,sealname in [(x.DEST,'SEALED.json'),(sn.DEST,'SEALED.json')]:
        seal=json.loads((dest/sealname).read_text());assert seal['predictions_sha256']==sha(dest/'predictions.csv.gz')
    panel=old.legacy.a.panel('available_macro');pred=read(sn.DEST/'predictions.csv.gz')
    max_error=0.;n=0
    for definition in DEFS:
        for cutoff in old.OUTER_MONTHS:
            fold=cutoff.strftime('%Y-%m');r=old.legacy.a.Residualizer(definition).fit(panel,cutoff)
            e,_=r.transform(panel)
            center=e.rolling('72h',min_periods=12,closed='left').mean().rename('past_center72')
            case=pd.read_pickle(PREP/'cases'/f'{definition}__{fold}__12.pkl.gz')
            tr=case['train'].query('abs_e>0').join(center).dropna(subset=['past_center72'])
            for null in ['affine','local_center72']:
                xx=tr.e_now.to_numpy();yy=tr.target.to_numpy()
                if null=='affine':
                    beta=float(sum((xx-xx.mean())*(yy-yy.mean()))/sum((xx-xx.mean())**2))
                    alpha=float(yy.mean()-beta*xx.mean());fit=alpha+beta*xx
                else:
                    xx=xx-tr.past_center72.to_numpy();yy=yy-tr.past_center72.to_numpy()
                    beta=float(sum(xx*yy)/sum(xx*xx));alpha=0.;fit=tr.past_center72.to_numpy()+beta*xx
                shocks=((tr.target-fit)/np.maximum(tr.e_rms72,1.)).to_numpy()
                shocks=np.concatenate([shocks,-shocks])
                d=pred[(pred.definition==definition)&(pred.fold==fold)&(pred['null']==null)]
                for row in d.itertuples():
                    mu=alpha+beta*row.e_now if null=='affine' else row.past_center72+beta*(row.e_now-row.past_center72)
                    future=mu+max(row.e_rms72,1.)*shocks
                    cross=future<0 if row.e_now>0 else future>0
                    over=future<-row.e_now if row.e_now>0 else future>-row.e_now
                    max_error=max(max_error,abs(cross.mean()-row.p_cross),abs(over.mean()-row.p_overshot));n+=1
    assert max_error<1e-10
    primary=read(OUT/'primary_tests.csv');extra=read(x.DEST/'primary_tests.csv')
    assert len(primary)==18 and len(extra)==12
    # Numerical claims appearing in the short conclusion and root README.
    d=primary[(primary.definition=='EQ')&(primary.test=='adjusted_premium_minus_discount_total')].iloc[0]
    assert round(d.estimate,2)==3.70
    d=extra[(extra.definition=='EQ')&(extra.quantity=='cross')].iloc[0]
    assert round(d.estimate*100,2)==29.60
    mapping=read(OUT/'residual_vs_price.csv');m=mapping[(mapping.definition=='EQ')&(mapping.residual_fall_bp==10)].iloc[0]
    assert int(m.n)==104 and int(m.local_up_or_flat_n)==57
    write_new(HERE/'FINAL_AUDIT.json',dict(**stamp,audited_utc=now(),symmetric_null_rows_checked=n,
        independent_probability_max_error=max_error,primary_tests=18,overshoot_tests=12,symmetric_null_tests=12,
        archived_sources_unchanged=True,statistical_claims_are_exploratory=True))
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='FINAL_MANIFEST.json']
    write_new(HERE/'FINAL_MANIFEST.json',dict(created_utc=now(),file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('Final independent null probabilities and all result/source seals verified:',n,max_error)


if __name__=='__main__':main()
