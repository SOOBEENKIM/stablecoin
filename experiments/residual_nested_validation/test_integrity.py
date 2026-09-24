"""Synthetic checks run BEFORE the frozen nested analysis."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from guard import sha,check_hashes,write_new
from nested import a,SPECS,choose,inner_months,purge_inner


class IntegrityTests(unittest.TestCase):
    def scores(self):
        rows=[]
        for candidate in [0,1]:
            for fold,n in [('2025-10',10),('2025-11',20),('2025-12',30)]:
                rows.append(dict(candidate=candidate,kind='linear',fold=fold,n=n,
                    loss_sum=n*(1 if candidate==1 else 2),valid=True))
        return pd.DataFrame(rows)

    def test_selection_uses_all_three_past_months_and_weighted_loss(self):
        cutoff=pd.Timestamp('2026-01-01',tz='UTC');d=self.scores()
        self.assertEqual(choose(d,cutoff)['candidate'],1)
        self.assertEqual([x.strftime('%Y-%m') for x in inner_months(cutoff)],['2025-10','2025-11','2025-12'])
        d.loc[(d.candidate==1)&(d.fold=='2025-12'),'valid']=False
        self.assertEqual(choose(d,cutoff)['candidate'],0)
        self.assertEqual(len(SPECS),84)

    def test_outer_score_injection_is_rejected(self):
        d=self.scores();d.loc[0,'fold']='2026-01'
        with self.assertRaises(ValueError):choose(d,pd.Timestamp('2026-01-01',tz='UTC'))

    def test_duplicate_month_and_missing_month_cannot_win(self):
        d=self.scores()
        with self.assertRaises(ValueError):choose(pd.concat([d,d.iloc[:1]]),pd.Timestamp('2026-01-01',tz='UTC'))
        d=d[~((d.candidate==1)&(d.fold=='2025-10'))]
        self.assertEqual(choose(d,pd.Timestamp('2026-01-01',tz='UTC'))['candidate'],0)

    def test_inner_label_on_outer_cutoff_is_purged(self):
        t=pd.date_range('2025-12-31 22:00',periods=3,freq='h',tz='UTC')
        d=pd.DataFrame(dict(target_time=t+pd.Timedelta(hours=1)),index=t)
        got=purge_inner(d,pd.Timestamp('2025-12-01',tz='UTC'))
        self.assertEqual(list(got.index),[t[0]])

    def test_future_labels_do_not_change_past_calibration(self):
        t=pd.date_range('2025-01-01',periods=100,freq='h',tz='UTC')
        d=pd.DataFrame(dict(origin=t,target_time=t+pd.Timedelta(hours=1),
            target=np.sin(np.arange(100)),pred_raw=np.zeros(100)))
        before=a.calibrate(d);d.loc[60:,'target']+=1e6;after=a.calibrate(d)
        np.testing.assert_array_equal(before.pred_calibrated[:61],after.pred_calibrated[:61])
        self.assertTrue((after.loc[after.calibration_n>=30,'latest_calibration_label']<after.loc[after.calibration_n>=30,'origin']).all())

    def test_changed_source_and_second_open_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'source.py';f.write_text('original')
            expected={'source.py':sha(f)};check_hashes(expected,root)
            f.write_text('changed')
            with self.assertRaises(RuntimeError):check_hashes(expected,root)
            write_new(root/'opened.json',{'hash':'first'})
            with self.assertRaises(FileExistsError):write_new(root/'opened.json',{'hash':'second'})


if __name__=='__main__':unittest.main()
