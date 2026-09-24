import unittest
import numpy as np
import pandas as pd
from adaptive import (panel, frame, Residualizer, splits, QuantileForest, calibrate,
                      columns, BASE, POSITION, HISTORY)


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strict = panel('strict')
        cls.available = panel('available_macro')
        cls.cutoff = pd.Timestamp('2026-01-01', tz='UTC')

    def test_expansion_does_not_fill_fx_or_change_residual_targets(self):
        for col in ['fx', 'q', 'y', 'g', 'm_EQ', 'm_CAP']:
            np.testing.assert_array_equal(self.strict[col], self.available[col])
        for definition in ['EQ', 'CAP', 'PCA']:
            a = Residualizer(definition).fit(self.strict, self.cutoff)
            b = Residualizer(definition).fit(self.available, self.cutoff)
            self.assertEqual(a.metadata(), b.metadata())
            ea, _ = a.transform(self.strict)
            eb, _ = b.transform(self.available)
            np.testing.assert_array_equal(ea, eb)

    def test_available_macro_ages_are_nonnegative_and_bounded(self):
        p = self.available
        for c in ['vix_known_age_hours', 'dxy_known_age_hours']:
            s = p[c].dropna()
            self.assertTrue(((s >= 0) & (s <= 96)).all())
        ti, va, tr, te, _, _ = splits(p, 'available_macro', 'EQ', self.cutoff)
        self.assertLess(tr.target_time.max(), te.index.min())
        self.assertLess(ti.target_time.max(), va.index.min())
        self.assertTrue((te.target_time - te.index == pd.Timedelta(hours=1)).all())

    def test_forest_weights_and_actual_conditional_quantile(self):
        x = np.zeros((100, 2))
        y = np.arange(100, dtype=float)
        model = QuantileForest(q=.1, max_depth=2, min_samples_leaf=10,
                               max_features=1.).fit(x, y)
        w = model.weights(np.zeros((2, 2)))
        np.testing.assert_allclose(w, .01, atol=1e-12)
        np.testing.assert_allclose(model.predict(np.zeros((2, 2))), 9.)
        self.assertNotEqual(model.predict(np.zeros((1, 2)))[0], y.mean())

    def test_online_correction_excludes_current_and_unobserved_labels(self):
        t = pd.date_range('2026-01-01', periods=150, freq='h', tz='UTC')
        d = pd.DataFrame(dict(origin=t, target_time=t + pd.Timedelta(hours=1),
                              target=np.sin(np.arange(150)), pred_raw=np.zeros(150)))
        a = calibrate(d)
        b = d.copy()
        b.loc[80:, 'target'] = 1e9
        after = calibrate(b)
        np.testing.assert_array_equal(a.loc[:81, 'pred_calibrated'], after.loc[:81, 'pred_calibrated'])
        valid = a.calibration_n >= 30
        self.assertTrue((a.loc[valid, 'latest_calibration_label'] < a.loc[valid, 'origin']).all())
        self.assertTrue((a.calibration_n <= 60).all())


if __name__ == '__main__':
    unittest.main()
