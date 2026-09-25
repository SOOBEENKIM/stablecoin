import unittest
import numpy as np
import pandas as pd
from hs_core import OUT,old,read
from selection_check import verify_selection


class SelectionCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s=read(OUT/'candidates/1__original__R__linear.csv.gz')
        cutoff=pd.Timestamp('2026-03-01',tz='UTC')
        ends=s.origin.dt.normalize().map(lambda t:t.replace(day=1)+pd.offsets.MonthBegin(1))
        d=s[(s.origin>=cutoff-pd.offsets.MonthBegin(3))&(s.origin<cutoff)&(s.target_time<cutoff)&(s.target_time<ends)].copy()
        e=d.target-d.pred_calibrated
        d['checkloss']=np.maximum(.1*e,-.9*e)
        cls.inner=d

    def test_reproduce_original_strict_checker_disagreement(self):
        scores=self.inner.groupby('candidate').checkloss.mean()
        self.assertEqual(min(scores.index,key=lambda i:(scores[i],i)),3)
        score0=self.inner[self.inner.candidate==0].checkloss.to_numpy().mean()
        score3=self.inner[self.inner.candidate==3].checkloss.to_numpy().mean()
        self.assertEqual(score0,score3)
        self.assertEqual(verify_selection(self.inner,list(scores.index),0)['saved_candidate'],0)

    def test_reject_different_frozen_winner(self):
        with self.assertRaises(AssertionError):verify_selection(self.inner,sorted(self.inner.candidate.unique()),3)

    def test_reject_substantive_score_difference(self):
        d=self.inner.copy();d.loc[d.candidate==0,'checkloss']+=.001
        with self.assertRaises(AssertionError):verify_selection(d,sorted(d.candidate.unique()),0)


if __name__=='__main__':unittest.main(verbosity=2)
