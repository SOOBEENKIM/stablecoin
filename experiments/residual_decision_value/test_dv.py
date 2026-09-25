import unittest
import numpy as np
import pandas as pd
from dv_core import *


class ScientificChecks(unittest.TestCase):
    def test_elementary_decision_orientation_and_integral(self):
        d=decisions(np.array([-15.,-15.,0.,0.]),np.array([-20.,0.,-20.,0.]),-10.)
        np.testing.assert_array_equal(d['cost'],[0.,.9,.1,0.])
        rng=np.random.default_rng(8)
        for y,q,e in rng.normal(size=(200,3))*30:
            self.assertAlmostEqual(float(pinball(y,q)),float(pinball(y-e,q-e)),places=12)
            self.assertAlmostEqual(float(pinball(y,q)),independent_elementary_integral(y,q),places=12)
        self.assertEqual(float(decisions([-10.],[-10.],-10.)['cost'][0]),0.)

    def test_future_poison_cannot_change_states_or_benchmarks(self):
        rng=np.random.default_rng(2)
        ix=pd.date_range('2025-01-01',periods=800,freq='h',tz='UTC')
        d=pd.DataFrame(dict(e_now=rng.normal(size=800),e_rms72=rng.uniform(.1,2,800),
            target=rng.normal(size=800),target_time=ix+pd.Timedelta(hours=12)),index=ix)
        cutoff=ix[600]
        a=fit_state(d,cutoff)
        corrupted=d.copy();corrupted.loc[corrupted.target_time>=cutoff,['target','e_now','e_rms72']]=1e9
        self.assertEqual(a,fit_state(corrupted,cutoff))
        evaluation=d.loc[d.index>=cutoff].copy();poison=evaluation.copy();poison['target']=-1e9
        for strategy in ['historical','state_hist']:
            np.testing.assert_array_equal(baseline_predict(evaluation,a,strategy),baseline_predict(poison,a,strategy))
        self.assertLess(pd.Timestamp(a['last_label']),cutoff)

    def test_exact_clock_nonoverlap(self):
        ix=pd.to_datetime(['2026-01-01T00:00Z','2026-01-01T01:00Z','2026-01-01T12:00Z','2026-01-03T00:00Z'])
        d=pd.DataFrame(dict(origin=ix,target_time=ix+pd.Timedelta(hours=12)))
        np.testing.assert_array_equal(nonoverlap(d),[True,False,True,True])

    def test_seed_cost_average_is_not_forecast_ensemble(self):
        from dv_evaluate import origin_decisions
        t=pd.Timestamp('2026-01-01',tz='UTC')
        d=pd.DataFrame(dict(origin=[t,t],target_time=[t+pd.Timedelta(hours=12)]*2,
            fold=['2026-01']*2,delta_e=[-15.,-15.],q_change=[0.,-30.]))
        score=origin_decisions(d,-10.)
        self.assertEqual(len(score),1)
        self.assertAlmostEqual(float(score.cost.iloc[0]),.45)
        self.assertAlmostEqual(float(decisions([-15.],[-15.],-10.)['cost'][0]),0.)

    def test_missing_outcome_does_not_pollute_other_groups(self):
        w=np.ones((3,3))
        lo,hi,valid=weighted_interval(w,[2.,4.,np.nan],[1,1,0])
        self.assertEqual((lo,hi,valid),(3.,3.,1.))

    def test_month_state_fits_and_price_accounting_on_existing_data(self):
        p=r12.old.legacy.a.panel('available_macro')
        cutoff=pd.Timestamp('2026-01-01',tz='UTC')
        for definition in DEFS:
            tr,te,r=r12.case(p,definition,cutoff)
            state=fit_state(tr,cutoff)
            poison=p.copy();future=poison.index>=cutoff
            poison.loc[future,'local_usdt']*=1.17
            poison.loc[future,'y']=poison.loc[future,'local_usdt']*poison.loc[future,'q']/poison.loc[future,'fx']-1
            ptr,pte,pr=r12.case(poison,definition,cutoff)
            self.assertEqual(state,fit_state(ptr,cutoff))
            np.testing.assert_allclose(te[PARTS].sum(axis=1),te.target-te.e_now,atol=1e-8,rtol=0)
            self.assertTrue(((te.target_time-te.index)==pd.Timedelta(hours=12)).all())


if __name__=='__main__':unittest.main(verbosity=2)
