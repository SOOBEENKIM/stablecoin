import unittest
from ac_overshoot import *

class OvershootTests(unittest.TestCase):
    def test_piecewise(self):
        d=pd.DataFrame(dict(e_now=[10.,10.,10.,-10.,-10.,-10.],target=[5.,-5.,-15.,-5.,5.,15.]))
        d['close_total']=-np.sign(d.e_now)*(d.target-d.e_now)
        a=actual(d)
        np.testing.assert_array_equal(a.gap,[5,5,-5,5,5,-5])
        np.testing.assert_array_equal(a.excess,[0,10,30,0,10,30])
        np.testing.assert_array_equal(a.overshot,[0,0,1,0,0,1])
    def test_no_target_leakage(self):
        case=pd.read_pickle(PREP/'cases'/'EQ__2026-01__12.pkl.gz')
        tr=case['train'].query('abs_e>0');te=case['test'].query('abs_e>0').iloc[:5].copy()
        bad=te.copy();bad['target']=1e12;bad['close_total']=-np.sign(bad.e_now)*(bad.target-bad.e_now)
        for learner in LEARNERS:
            a,_=learn(tr,te,learner,SEEDS[0]);b,_=learn(tr,bad,learner,SEEDS[0])
            np.testing.assert_array_equal(a,b)
    def test_archived_identity(self):
        d=actual(read(PREP/'context.csv.gz'))
        np.testing.assert_allclose(d.gap,d.absolute_gap_reduction,atol=1e-10,rtol=0)
        np.testing.assert_array_equal(d['cross'],d.crossed_zero)
        np.testing.assert_array_equal(d.overshot,d.overshot_farther)

if __name__=='__main__':unittest.main(verbosity=2)
