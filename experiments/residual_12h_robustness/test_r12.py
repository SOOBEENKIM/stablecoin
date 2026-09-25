import json
import unittest
import numpy as np
import pandas as pd
from r12_core import PARENT,old,hs,DEFS,read,case,select_all,inner_rows,independent_calibration,standardize


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream=pd.concat([standardize(read(PARENT/'results/candidates'/f'12__original__R__{k}.csv.gz'),'EQ') for k in old.DESIGN['families']],ignore_index=True)

    def test_fast_purge_and_tie_selection_match_archive(self):
        stored=pd.DataFrame(json.loads((PARENT/'results/selections.json').read_text()))
        for cut in old.OUTER_MONTHS:
            pd.testing.assert_frame_equal(inner_rows(self.stream,cut),old.purged_inner(self.stream,cut))
            choices,_=select_all(self.stream,cut)
            for st,spec in choices.items():
                want=stored[(stored.horizon==12)&(stored.variant=='original')&(stored.information=='R')&(stored.fold==cut.strftime('%Y-%m'))&(stored.strategy==st)].iloc[0]
                self.assertEqual(spec['candidate'],want.candidate)
                self.assertEqual(spec['inner_score'],want.inner_score)

    def test_future_poison_does_not_change_selection(self):
        cut=pd.Timestamp('2026-03-01',tz='UTC')
        a,_=select_all(self.stream,cut)
        d=self.stream.copy();d.loc[d.target_time>=cut,['target','pred_calibrated']]=1e8
        self.assertEqual(a,select_all(d,cut)[0])
        with self.assertRaises(ValueError):select_all(self.stream[self.stream.candidate!=0],cut)

    def test_independent_clock_scan_matches_production(self):
        g=self.stream[self.stream.candidate==16].copy()
        self.assertEqual(independent_calibration(g),0.)
        cut=pd.Timestamp('2026-03-01',tz='UTC')
        altered=g.copy();altered.loc[altered.target_time>=cut,'target']=1e7
        a=hs.calibrate(g,12);b=hs.calibrate(altered,12)
        np.testing.assert_array_equal(a[a.origin<=cut].pred_calibrated,b[b.origin<=cut].pred_calibrated)

    def test_definitions_same_clocks_and_targets_distinct(self):
        panel=old.legacy.a.panel('available_macro')
        totals={d:0 for d in DEFS}
        for cut in old.MONTHS:
            pairs={d:case(panel,d,cut) for d in DEFS}
            for d,(tr,te,r) in pairs.items():
                pd.testing.assert_index_equal(te.index,pairs['EQ'][1].index)
                self.assertTrue((tr.target_time<cut).all())
                self.assertTrue(((te.target_time-te.index)==pd.Timedelta(hours=12)).all())
                if cut>=hs.START:totals[d]+=len(te)
            self.assertGreater(np.max(np.abs(pairs['CAP'][1].target-pairs['EQ'][1].target)),0.)
        self.assertEqual(totals,dict(EQ=509,CAP=509,PCA=509))


if __name__=='__main__':unittest.main(verbosity=2)
