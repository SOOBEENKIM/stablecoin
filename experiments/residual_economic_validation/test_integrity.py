import unittest
import numpy as np
import pandas as pd
from economic import features, GROUPS, ThresholdQuantile, a
from state_design import match_month


class IntegrityTests(unittest.TestCase):
    def test_ablation_removes_all_group_lags_and_preserves_other_inputs(self):
        full = features('available_macro', 'history')
        for group, cols in GROUPS.items():
            reduced = features('available_macro', 'without_' + group)
            self.assertEqual(set(full) - set(reduced), set(cols))
            self.assertEqual(len(reduced), len(set(reduced)))

    def test_threshold_recovers_known_regime_function_and_uses_frozen_boundaries(self):
        e = np.linspace(-3, 3, 600)
        regime = np.searchsorted(np.quantile(e, [.2,.8]), e, side='right')
        d = pd.DataFrame(dict(e_now=e, x=np.sin(np.arange(600))))
        d['target'] = np.choose(regime, [2 * e - 2, .1 * e, -e + 2]) + .3 * d.x
        model = ThresholdQuantile('change', dict(band=[.2,.8], alpha=0)).fit(d, ['e_now', 'x'])
        np.testing.assert_allclose(model.predict(d), d.target, atol=1e-5)
        before = model.cuts.copy()
        changed = d.copy(); changed['e_now'] += 100; changed['target'] += 1e6
        model.predict(changed)
        np.testing.assert_array_equal(model.cuts, before)
        without_target = d.drop(columns='target')
        np.testing.assert_allclose(model.predict(without_target), model.predict(d), atol=0)

    def test_matching_is_one_to_one_calipered_and_outcome_blind(self):
        t = pd.date_range('2025-01-01', periods=180, freq='h', tz='UTC')
        rng = np.random.default_rng(1)
        d = pd.DataFrame(dict(e_now=rng.normal(size=180), btc_vol24=rng.uniform(1,2,180),
            downside24=rng.uniform(1,2,180), e_change1=rng.normal(size=180),
            account_change24=rng.normal(size=180), target=rng.normal(size=180)), index=t)
        p, _, _, _ = match_month(d, d, 'account')
        self.assertGreater(len(p), 0)
        self.assertTrue(p.high_origin.is_unique and p.low_origin.is_unique)
        self.assertTrue((p[['difference_e_now','difference_log_vol','difference_log_downside']].abs() <= .5).all().all())
        changed = d.copy(); changed['target'] = rng.normal(size=180) * 1e8
        q, _, _, _ = match_month(changed, changed, 'account')
        pd.testing.assert_frame_equal(p, q)


if __name__ == '__main__':
    unittest.main()
