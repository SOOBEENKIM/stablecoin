"""Research-integrity tests: quote direction, time gaps, future contamination."""
import unittest
import numpy as np
import pandas as pd
from data import (BASE, POSITION, Residualizer, build_panel, make_design,
                  past_observation, lead, feature_columns)
from models import ForecastModel
from run import prepare_splits


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = build_panel('NY_delay1')
        cls.cutoff = pd.Timestamp('2025-12-01', tz='UTC')

    def test_parity_conversion(self):
        q, fx = 1.01, 1400.
        u = fx / q
        self.assertAlmostEqual(u * q / fx - 1, 0.)
        self.assertNotAlmostEqual(u / (q * fx) - 1, 0.)
        p = self.p.dropna(subset=['y']).iloc[:10]
        np.testing.assert_allclose(p.y, p.local_usdt * p.q / p.fx - 1)

    def test_no_future_asof_and_no_row_leads(self):
        idx = pd.to_datetime(['2026-01-01T00:00Z', '2026-01-01T03:00Z'])
        s = pd.Series([1., 9.], index=idx)
        grid = pd.date_range(idx[0], idx[-1], freq='h')
        x, age = past_observation(s, grid, '59min59s')
        self.assertTrue(np.isnan(x.iloc[1]))
        self.assertTrue(np.isnan(x.iloc[2]))
        self.assertTrue(lead(s, 1).isna().all())
        self.assertEqual(lead(s, 3).iloc[0], 9.)

    def test_future_mutation_cannot_change_first_stage_or_origin_features(self):
        origin = pd.Timestamp('2025-12-02 14:00', tz='UTC')
        altered = self.p.copy()
        mask = altered.index > origin
        altered.loc[mask, 'y'] += 100
        altered.loc[mask, ['kp_' + c for c in ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']]] += 10
        altered.loc[mask, 'm_EQ'] += 10
        altered.loc[mask, 'm_CAP'] += 10
        for definition in ['EQ', 'CAP', 'PCA']:
            a = Residualizer(definition).fit(self.p, self.cutoff)
            b = Residualizer(definition).fit(altered, self.cutoff)
            self.assertEqual(a.metadata(), b.metadata())
            ea, _ = a.transform(self.p)
            eb, _ = b.transform(altered)
            np.testing.assert_allclose(ea.loc[:origin], eb.loc[:origin], equal_nan=True)

    def test_split_purge_and_price_identity(self):
        for h in [1, 6, 12]:
            ti, va, tr, te, audit = prepare_splits(self.p, 'EQ', 'sequential', h, self.cutoff)
            self.assertLess(tr.target_time.max(), te.index.min())
            self.assertLess(ti.target_time.max(), va.index.min())
            self.assertTrue(((te.target_time - te.index) == pd.Timedelta(hours=h)).all())
            self.assertTrue(set(feature_columns('full')).isdisjoint(
                {'target', 'delta_e', 'target_time', 'delta_local_bp'}))
            parts = ['delta_local_bp', 'delta_quote_bp', 'delta_fx_bp', 'delta_market_bp', 'delta_g_bp']
            np.testing.assert_allclose(te[parts].sum(axis=1), te.delta_e, atol=1e-8)
        for definition in ['CAP', 'PCA']:
            for projection in ['sequential', 'joint']:
                r = Residualizer(definition, projection).fit(self.p, self.cutoff)
                self.assertGreater(len(make_design(self.p, r, 1)), 100)

    def test_model_features_do_not_include_test_outcomes(self):
        ti, va, tr, te, _ = prepare_splits(self.p, 'EQ', 'sequential', 1, self.cutoff)
        cols = feature_columns('full')
        model = ForecastModel('linear', .1, {'alpha': .01}).fit(tr[cols], tr.target)
        before = model.predict(te[cols])
        changed = te.copy()
        changed['target'] = -999999
        changed['delta_e'] = 999999
        np.testing.assert_array_equal(before, model.predict(changed[cols]))


if __name__ == '__main__':
    unittest.main()
