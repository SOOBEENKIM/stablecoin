import unittest
from ac_core import *


class ChannelTests(unittest.TestCase):
    def test_exact_product_identity_with_moving_weights(self):
        rng=np.random.default_rng(19);n=120
        l0=rng.uniform(1200,1600,n);l1=l0+rng.normal(0,3,n)
        r0=rng.uniform(1200,1600,(n,5));r1=r0+rng.normal(0,3,(n,5))
        a0=rng.dirichlet(np.ones(5),n);a1=rng.dirichlet(np.ones(5),n)
        z0=rng.uniform(.0006,.0008,n);z1=z0+rng.normal(0,.000001,n)
        g0=rng.uniform(0,.001,n);g1=rng.uniform(0,.001,n);b=.98;bg=.2
        parts=attribution(l0,l1,r0,r1,a0,a1,z0,z1,g0,g1,b,bg)
        expected=[]
        for i in range(n):
            e0=z0[i]*l0[i]-b*z0[i]*sum(a0[i]*r0[i])-bg*g0[i]
            e1=z1[i]*l1[i]-b*z1[i]*sum(a1[i]*r1[i])-bg*g1[i]
            expected.append((e1-e0)*1e4)
        np.testing.assert_allclose(parts.sum(axis=1),expected,atol=1e-10,rtol=0)

    def test_shared_numeraire_and_balanced_prices(self):
        l=np.array([1400.,1410.]);r=np.full((2,5),1400.);a=np.full((2,5),.2)
        p=attribution(l,l,r,r,a,a,np.array([1/1400]*2),np.array([1/1500]*2),[0,0],[0,0],1.,0.)
        np.testing.assert_array_equal(p[:,:3],0)
        self.assertEqual(p[0].sum(),0.)
        self.assertAlmostEqual(p[1,3],10*(1/1500-1/1400)*1e4)

    def test_basket_weight_channel(self):
        r=np.array([[1400.,1500.]]);a=np.array([[.5,.5]]);b=np.array([[.2,.8]])
        p=attribution([1450.],[1450.],r,r,a,b,[1/1400],[1/1400],[0],[0],1.,0.)
        self.assertAlmostEqual(p[0,2],-30/1400*1e4)
        np.testing.assert_array_equal(p[0,[0,1,3,4]],0)

    def test_orthogonal_coefficient_and_additivity(self):
        rng=np.random.default_rng(3);n=120
        d=pd.DataFrame(dict(premium=rng.integers(0,2,n),m=rng.uniform(.2,.8,n)))
        for j,c in enumerate(PARTS):
            d['g_'+c]=rng.normal(size=n);d['close_'+c]=d['g_'+c]+(j+1)*(d.premium-d.m)
            self.assertAlmostEqual(orthogonal(d,c),j+1)
        d['g_total']=d[['g_'+c for c in PARTS]].sum(axis=1)
        d['close_total']=d[['close_'+c for c in PARTS]].sum(axis=1)
        self.assertAlmostEqual(orthogonal(d,'total'),15.)
        draws=orthogonal(d,'total',np.ones((10,n)))
        np.testing.assert_allclose(draws,15.)

    def test_future_outcomes_not_used_in_nuisance_predictions(self):
        rng=np.random.default_rng(8);n=240
        tr=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
        tr['premium']=rng.integers(0,2,n)
        for c in PARTS:tr['close_'+c]=rng.normal(size=n)
        tr.index=pd.date_range('2025-01-01',periods=n,freq='h',tz='UTC',name='origin')
        te=tr.iloc[:8].copy();bad=te.copy()
        for c in PARTS:bad['close_'+c]=1e12
        for learner in LEARNERS:
            a=model_fit(tr,te,learner,SEEDS[0]);b=model_fit(tr,bad,learner,SEEDS[0])
            cols=['m']+['g_'+c for c in PARTS]
            np.testing.assert_array_equal(a[cols],b[cols])

    def test_actual_frames_and_existing_targets(self):
        ctx=read(PREP/'context.csv.gz')
        for definition in DEFS:
            d=ctx[(ctx.definition==definition)&(ctx.h==12)]
            self.assertEqual(len(d),509)
        for note in json.loads((PREP/'metadata.json').read_text()):
            self.assertLess(pd.Timestamp(note['last_train_label']),pd.Timestamp(note['fold']+'-01',tz='UTC'))
        np.testing.assert_allclose(ctx[['delta_'+c for c in PARTS]].sum(axis=1),ctx.delta_e,atol=1e-7,rtol=0)
        np.testing.assert_allclose(ctx.absolute_gap_reduction,abs(ctx.e_now)-abs(ctx.target),atol=1e-11,rtol=0)
        # Zero-crossing overshoots cannot be reported as absolute gap reductions.
        crossed=ctx[ctx.overshot_farther==1]
        self.assertTrue((crossed.absolute_gap_reduction<0).all())
        self.assertTrue((crossed.close_total>0).all())


if __name__=='__main__':unittest.main(verbosity=2)
