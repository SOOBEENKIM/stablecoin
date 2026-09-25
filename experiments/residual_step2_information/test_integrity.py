import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from step2_guard import write_new,sha,check_hashes
from step2_core import (a,n,DESIGN,GROUPS,features,interaction_pairs,fit_model,choose,
                        AvailableInteractions,ForecastModel)


class Integrity(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(44)
        d=pd.DataFrame({c:rng.normal(size=250) for c in a.columns('available_macro','history')})
        d['target']=.6*d.e_now+.5*d.downside24*d.account_log+rng.normal(size=len(d))
        return d

    def scores(self):
        return pd.DataFrame([dict(candidate=c,kind='linear',removed_group='global_btc',
            fold=f,n=v,loss_sum=v*(2-c),valid=True) for c in [0,1] for f,v in
            [('2025-10',10),('2025-11',20),('2025-12',30)]])

    def test_groups_and_residual_inputs(self):
        for mode in ['available_macro','strict']:
            full=a.columns(mode,'history')
            for group,removed in GROUPS.items():
                cols=features(mode,group)
                self.assertFalse(set(cols)&set(removed))
                self.assertTrue(set(DESIGN['always_keep']).issubset(cols))
                self.assertEqual(cols,[c for c in full if c not in removed])
                self.assertFalse(any(x in removed or y in removed for x,y in interaction_pairs(cols)))
        self.assertEqual(interaction_pairs(features('available_macro','global_btc')),[('e_now','account_log')])
        self.assertNotIn('vix_known_age_hours',features('available_macro','fx_macro'))
        self.assertIn('hour_sin',features('available_macro','fx_macro'))

    def test_full_interactions_match_original(self):
        d=self.data();cols=a.columns('available_macro','history')
        for form in ['level','change']:
            spec=next(s for s in n.SPECS if s['kind']=='interaction' and s['form']==form)
            old=n.fit_model(spec,d,cols,n.SEEDS[0]);new=fit_model(spec,d,cols,n.SEEDS[0])
            np.testing.assert_array_equal(old.model.prepare(d[cols]),new.model.prepare(d[cols]))
            np.testing.assert_allclose(old.predict(d[cols]),new.predict(d[cols]),atol=1e-12,rtol=0)

    def test_removed_training_values_cannot_reenter_models(self):
        d=self.data()
        # Each algorithm and group is refitted after only deleted feature columns change.
        for group,removed in GROUPS.items():
            cols=features('available_macro',group);fake=d.copy()
            for c in removed:
                if c in fake:fake[c]+=1e7
            for kind in n.DESIGN['algorithms']:
                spec=next(s for s in n.SPECS if s['kind']==kind and s['form']=='change')
                m=fit_model(spec,d,cols,n.SEEDS[0]);other=fit_model(spec,fake,cols,n.SEEDS[0])
                np.testing.assert_allclose(m.predict(d[cols]),other.predict(d[cols]),atol=1e-12,rtol=0)
                fake_target=d.copy();fake_target['target']+=1e8
                np.testing.assert_array_equal(m.predict(d[cols]),m.predict(fake_target[cols]))

    def test_selection_uses_three_past_months_and_correct_group(self):
        t=pd.Timestamp('2026-01-01',tz='UTC');d=self.scores()
        self.assertEqual(choose(d,t,'global_btc')['candidate'],1)
        d=d[~((d.candidate==1)&(d.fold=='2025-10'))]
        self.assertEqual(choose(d,t,'global_btc')['candidate'],0)
        with self.assertRaises(ValueError):choose(d,t,'fx_macro')
        with self.assertRaises(ValueError):choose(pd.concat([d,d.iloc[:1]]),t,'global_btc')
        d.loc[d.index[0],'fold']='2026-01'
        with self.assertRaises(ValueError):choose(d,t,'global_btc')

    def test_month_boundary_labels(self):
        t=pd.date_range('2025-12-31 22:00',periods=3,freq='h',tz='UTC')
        d=pd.DataFrame({'target_time':t+pd.Timedelta(hours=1)},index=t)
        self.assertEqual(list(n.purge_inner(d,pd.Timestamp('2025-12-01',tz='UTC')).index),[t[0]])

    def test_future_calibration_and_pinball_translation(self):
        t=pd.date_range('2025-01-01',periods=100,freq='h',tz='UTC')
        d=pd.DataFrame(dict(origin=t,target_time=t+pd.Timedelta(hours=1),target=np.sin(np.arange(100)),pred_raw=np.zeros(100)))
        before=a.calibrate(d);d.loc[60:,'target']+=1e8;after=a.calibrate(d)
        np.testing.assert_array_equal(before.pred_calibrated[:61],after.pred_calibrated[:61])
        shift=np.linspace(-100,100,100)
        np.testing.assert_allclose(a.pinball(before.target,before.pred_calibrated,.1),
            a.pinball(before.target-shift,before.pred_calibrated-shift,.1),atol=1e-12)

    def test_effect_direction(self):
        full=np.array([1.,2.]);without=np.array([2.,4.])
        self.assertEqual(100*(without.mean()/full.mean()-1),100.)
        self.assertEqual(100*(1-full.mean()/without.mean()),50.)

    def test_changed_files_and_reopening_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'source';f.write_text('before');hashes={'source':sha(f)}
            check_hashes(hashes,root);f.write_text('after')
            with self.assertRaises(RuntimeError):check_hashes(hashes,root)
            write_new(root/'opened',{})
            with self.assertRaises(FileExistsError):write_new(root/'opened',{})


if __name__=='__main__':unittest.main()
