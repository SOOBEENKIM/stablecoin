import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from step1_guard import write_new, sha, check_hashes
from step1_core import a, n, SPECS, Benchmark, choose


class Integrity(unittest.TestCase):
    def scores(self):
        return pd.DataFrame([dict(candidate=c, strategy='persistence', fold=f, valid=True,
            n=v, loss_sum=v*(2-c)) for c in [0,1] for f,v in
            [('2025-10',10),('2025-11',20),('2025-12',30)]])

    def test_selection_and_missing_month(self):
        d=self.scores();t=pd.Timestamp('2026-01-01',tz='UTC')
        self.assertEqual(choose(d,t,'persistence')['candidate'],1)
        self.assertEqual(len(SPECS),26)
        d=d[~((d.candidate==1)&(d.fold=='2025-10'))]
        self.assertEqual(choose(d,t,'persistence')['candidate'],0)
        with self.assertRaises(ValueError):choose(pd.concat([d,d.iloc[:1]]),t,'persistence')

    def test_future_score_rejected_and_boundary_purged(self):
        d=self.scores();d.loc[0,'fold']='2026-01'
        with self.assertRaises(ValueError):choose(d,pd.Timestamp('2026-01-01',tz='UTC'),'persistence')
        times=pd.date_range('2025-12-31 22:00',periods=3,freq='h',tz='UTC')
        d=pd.DataFrame({'target_time':times+pd.Timedelta(hours=1)},index=times)
        self.assertEqual(list(n.purge_inner(d,pd.Timestamp('2025-12-01',tz='UTC')).index),[times[0]])

    def test_persistence_is_quantile_and_shift_equivariant(self):
        train=pd.DataFrame({'e_now':np.arange(100.),'target':np.arange(100.)+np.linspace(-10,10,100)})
        m=Benchmark(SPECS[0]).fit(train)
        expected=np.quantile(train.target-train.e_now,.1)
        self.assertAlmostEqual(m.predict(pd.DataFrame({'e_now':[20.]}))[0],20+expected)
        self.assertAlmostEqual(m.predict(pd.DataFrame({'e_now':[30.]}))[0]-m.predict(pd.DataFrame({'e_now':[20.]}))[0],10.)

    def test_physical_coefficients_and_no_outcome_feature(self):
        rng=np.random.default_rng(1)
        d=pd.DataFrame({c:rng.normal(size=180) for c in ['e_now','e_change1','btc_vol24','downside24']})
        d['target']=2+.6*d.e_now+.2*d.e_change1+rng.normal(size=len(d))
        for strategy in ['ar_quantile','dynamics_vol']:
            for form in ['level','change']:
                spec=next(s for s in SPECS if s['strategy']==strategy and s['form']==form)
                m=Benchmark(spec).fit(d);prediction=m.predict(d[m.columns]);coeff=m.coefficients()
                reconstructed=coeff['intercept']+sum(d[c].to_numpy()*coeff[c] for c in m.columns)
                np.testing.assert_allclose(prediction,reconstructed,atol=1e-12)
                perturbed=d.copy();perturbed['target']+=1e8
                np.testing.assert_array_equal(prediction,m.predict(perturbed[m.columns]))

    def test_translation_loss_and_future_calibration(self):
        t=pd.date_range('2025-01-01',periods=100,freq='h',tz='UTC')
        d=pd.DataFrame(dict(origin=t,target_time=t+pd.Timedelta(hours=1),
            target=np.sin(np.arange(100)),pred_raw=np.zeros(100)))
        first=a.calibrate(d);d.loc[60:,'target']+=1e6;second=a.calibrate(d)
        np.testing.assert_array_equal(first.pred_calibrated[:61],second.pred_calibrated[:61])
        shift=np.linspace(-100,100,100)
        np.testing.assert_allclose(a.pinball(first.target,first.pred_calibrated,.1),
            a.pinball(first.target-shift,first.pred_calibrated-shift,.1),atol=1e-12)

    def test_changed_lock_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'a';f.write_text('before');hashes={'a':sha(f)}
            check_hashes(hashes,root);f.write_text('after')
            with self.assertRaises(RuntimeError):check_hashes(hashes,root)
            write_new(root/'opened',{})
            with self.assertRaises(FileExistsError):write_new(root/'opened',{})


if __name__=='__main__':unittest.main()
