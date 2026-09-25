import unittest
import numpy as np
import pandas as pd
from hs_core import old, own_scale, calibrate, make_case, choose, SEED, OLD, read


class MechanicsTests(unittest.TestCase):
    def test_exact_horizon_not_row_shift(self):
        ix=pd.DatetimeIndex(['2025-01-01T00:00Z','2025-01-01T01:00Z','2025-01-01T06:00Z','2025-01-01T12:00Z'])
        e=pd.Series([1.,2.,6.,12.],index=ix)
        self.assertEqual(old.legacy.a.lead(e,6).iloc[0],6.)
        self.assertTrue(np.isnan(old.legacy.a.lead(e,6).iloc[1]))
        self.assertEqual(old.legacy.a.lead(e,12).iloc[0],12.)

    def test_scale_gaps_and_no_future(self):
        ix=pd.date_range('2025-01-01',periods=200,freq='h',tz='UTC')
        e=pd.Series(np.arange(200,dtype=float),index=ix)
        e.iloc[40:80]=np.nan
        actual=own_scale(e)
        for t in ix:
            changes=[e.loc[s]-e.loc[s-pd.Timedelta(hours=1)] for s in ix
                     if t-pd.Timedelta(hours=72)<s<=t and s-pd.Timedelta(hours=1) in ix]
            changes=np.asarray(changes); changes=changes[np.isfinite(changes)]
            want=np.sqrt(np.mean(changes**2)) if len(changes)>=12 else np.nan
            np.testing.assert_allclose(actual.loc[t],want,equal_nan=True)
        poison=e.copy();poison.loc[ix[100]:]=1e8
        pd.testing.assert_series_equal(own_scale(poison).loc[:ix[99]],actual.loc[:ix[99]])

    def test_calibration_mature_labels_each_horizon(self):
        rng=np.random.default_rng(37)
        ix=pd.date_range('2025-01-01',periods=160,freq='h',tz='UTC')
        for h in [1,6,12]:
            d=pd.DataFrame(dict(origin=ix,target_time=ix+pd.Timedelta(hours=h),target=rng.normal(size=len(ix)),pred_raw=np.zeros(len(ix))))
            got=calibrate(d,h)
            for i,t in enumerate(ix):
                past=d[(d.target_time<t)&(d.target_time>=t-pd.Timedelta(days=90))].tail(60)
                expected=sorted(past.target)[int(np.ceil(len(past)*.1))-1] if len(past)>=30 else 0.
                self.assertEqual(got.correction.iloc[i],expected)
            bad=d.copy();bad.loc[bad.target_time>=ix[100],'target']=1e9
            np.testing.assert_array_equal(calibrate(bad,h).pred_calibrated.iloc[:101],got.pred_calibrated.iloc[:101])
            with self.assertRaises(ValueError):calibrate(d,h+1)

    def test_one_hour_parity_and_sample_support(self):
        panel=old.legacy.a.panel('available_macro')
        total={h:0 for h in [1,6,12]}; common=0
        for cutoff in old.MONTHS:
            indexes=[]
            for h in total:
                tr,te,_=make_case(panel,cutoff,h)
                self.assertTrue(np.isfinite(tr.e_rms72).all())
                self.assertTrue(np.isfinite(te.e_rms72).all())
                if cutoff>=pd.Timestamp('2025-12-01',tz='UTC'):
                    total[h]+=len(te); indexes.append(te.index)
            if indexes: common+=len(indexes[0].intersection(indexes[1]).intersection(indexes[2]))
        self.assertEqual(total,{1:1088,6:703,12:509})
        self.assertEqual(common,239)

    def test_selection_ignores_outer_outcomes_and_matches_old(self):
        stream=pd.concat([read(OLD/'results/candidates'/('available_macro__EQ__R__'+k+'.csv.gz')) for k in old.DESIGN['families']],ignore_index=True)
        stream=stream[stream.seed==SEED].copy();stream['horizon']=1;stream['variant']='original'
        cutoff=pd.Timestamp('2026-01-01',tz='UTC')
        got,_=choose(stream,cutoff,'ml_selected',1)
        expected=old.choose_past(old.purged_inner(stream,cutoff),cutoff,'ml_selected')
        self.assertEqual(got['candidate'],expected['candidate'])
        poison=stream.copy();poison.loc[poison.target_time>=cutoff,['target','pred_calibrated']]=1e9
        self.assertEqual(choose(poison,cutoff,'ml_selected',1)[0],got)


if __name__=='__main__':unittest.main(verbosity=2)
