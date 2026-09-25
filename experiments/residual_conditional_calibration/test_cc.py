import unittest
from cc_core import *


def artificial(n=220):
    origin=pd.date_range('2025-08-01',periods=n,freq='12h',tz='UTC')
    d=pd.DataFrame(dict(origin=origin,target_time=origin+pd.Timedelta(hours=12),
        target=np.sin(np.arange(n)/7)*8,e_rms72=2+np.arange(n)%5,
        e_rank=(np.arange(n)%11)/11,vol_rank=(np.arange(n)%7)/7))
    return d


class CalibrationTests(unittest.TestCase):
    def test_strict_clock_and_window(self):
        d=artificial();p=plans(d)
        for i,(a,b,c) in enumerate(p):
            self.assertLessEqual(len(a),60);self.assertLessEqual(len(b),180);self.assertLessEqual(len(c),60)
            self.assertTrue((d.target_time.iloc[b]<d.origin.iloc[i]).all())
            self.assertTrue((d.target_time.iloc[b]>=d.origin.iloc[i]-pd.Timedelta(days=90)).all())
            self.assertNotIn(i-1,b)  # label exactly at origin is excluded

    def test_global_and_warmup(self):
        d=artificial();raw=np.linspace(-4,4,len(d));cal=calibrate_matrix(d,raw)[:,0]
        for i,(a,b,c) in enumerate(plans(d)):
            expected=0 if len(a)<30 else sorted((d.target.to_numpy()-raw)[a])[int(np.ceil(.1*len(a)))-1]
            self.assertAlmostEqual(cal[i,0],raw[i]+expected)
            if len(a)<30:np.testing.assert_allclose(cal[i],raw[i])

    def test_future_poison(self):
        d=artificial();raw=np.zeros(len(d));original=calibrate_matrix(d,raw)
        z=d.copy();z.loc[150:,'target']=1e9
        poisoned=calibrate_matrix(z,raw)
        np.testing.assert_array_equal(original[:152],poisoned[:152])

    def test_neighbors_and_ties(self):
        d=artificial();d['e_rank']=0.;d['vol_rank']=0.
        for a,b,c in plans(d):np.testing.assert_array_equal(c,b[::-1][:60])

    def test_scaling_and_constant_scale(self):
        d=artificial();raw=np.full(len(d),-1.)
        a=calibrate_matrix(d,raw)
        z=d.copy();z['target']*=10;z['e_rms72']*=10
        np.testing.assert_allclose(calibrate_matrix(z,raw*10),a*10,atol=1e-12)
        d['e_rms72']=2.;a=calibrate_matrix(d,raw)
        np.testing.assert_allclose(a[:,:,1],a[:,:,2]);np.testing.assert_allclose(a[:,:,3],a[:,:,4])

    def test_selection_ignores_outer_labels(self):
        rows=[]
        for cid in range(62):
            kind=old.SPECS[cid]['kind'] if cid<60 else ['historical','state_hist'][cid-60]
            for origin in pd.date_range('2025-09-05','2026-03-05',freq='7D',tz='UTC'):
                row=dict(candidate=cid,kind=kind,origin=origin,target_time=origin+pd.Timedelta(hours=12),target=0.)
                row.update({'q_'+rule:float(1+cid+rank/10) for rank,rule in enumerate(RULES)})
                rows.append(row)
        d=pd.DataFrame(rows);cutoff=pd.Timestamp('2025-12-01',tz='UTC')
        chosen,scores=select(d,cutoff)
        self.assertEqual(len(chosen),20);self.assertEqual(len(scores),434)
        z=d.copy();mask=z.target_time>=cutoff
        z.loc[mask,'target']=1e12
        for rule in RULES:z.loc[mask,'q_'+rule]=-1e12
        self.assertEqual((chosen,scores),select(z,cutoff))
        for c in chosen:self.assertEqual(c['rule'],'global60')

    def test_real_archived_global60_and_metadata(self):
        for definition in DEFS:
            ctx=context_for(definition);d=first_streams(definition)
            p=plans(ctx)
            for _,g in d.groupby('candidate'):
                g=g.sort_values('origin').reset_index(drop=True)
                errors=(g.target-g.pred_raw).to_numpy()
                correction=np.array([np.sort(errors[a])[int(np.ceil(.1*len(a)))-1] if len(a)>=30 else 0. for a,_,_ in p])
                np.testing.assert_allclose(g.pred_raw+correction,g.pred_calibrated,atol=1e-11,rtol=0)
            for seed in SEEDS[1:]:
                e=read(PARENT/'results/extra_candidates'/f'{definition}__original__F__{seed}.csv.gz')
                fixed=aligned(e,ctx)
                self.assertTrue(np.isfinite(fixed.e_now).all())
            for i,(a,_,_) in enumerate(p):
                if ctx.origin.iloc[i]>=pd.Timestamp('2025-09-01',tz='UTC'):self.assertGreaterEqual(len(a),30)


if __name__=='__main__':unittest.main(verbosity=2)
