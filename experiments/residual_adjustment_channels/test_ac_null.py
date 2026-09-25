import unittest
from ac_symmetric_null import *

class NullTests(unittest.TestCase):
    def test_symmetric_null_reflection(self):
        rng=np.random.default_rng(29);n=600
        e=rng.normal(0,10,n);tr=pd.DataFrame(dict(e_now=e,target=-2+.3*e+rng.normal(0,3,n),e_rms72=np.ones(n)*3,past_center72=np.ones(n)*-2,
            target_time=pd.date_range('2025-01-01',periods=n,freq='h',tz='UTC')))
        te=tr.iloc[:4].copy();te.e_now=[10,10,-10,-10]
        for centered in [False,True]:
            a,_=symmetric_probabilities(tr,te,centered)
            neg=tr.copy();neg.e_now=-neg.e_now;neg.target=-neg.target;neg.past_center72=-neg.past_center72
            nt=te.copy();nt.e_now=-nt.e_now;nt.past_center72=-nt.past_center72
            b,_=symmetric_probabilities(neg,nt,centered)
            for c in ['cross','overshot']:np.testing.assert_allclose(a[c],b[c],atol=1e-12,rtol=0)
            self.assertTrue((a['overshot']<=a['cross']).all())
    def test_future_labels_irrelevant(self):
        case=pd.read_pickle(PREP/'cases'/'EQ__2026-01__12.pkl.gz')
        tr=case['train'].assign(past_center72=-2.);te=case['test'].assign(past_center72=-2.)
        bad=te.copy();bad.target=1e12
        for centered in [False,True]:
            a,_=symmetric_probabilities(tr,te,centered);b,_=symmetric_probabilities(tr,bad,centered)
            for c in a:np.testing.assert_array_equal(a[c],b[c])

if __name__=='__main__':unittest.main(verbosity=2)
