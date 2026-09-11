"""Regression checks for quote direction, clock time, and quantile estimation."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop_core import (COINS, at_hour, correct_quotes, prepare_data,
                           quantile_fit, residualize, tick_krw)


def panel(index):
    n = len(index)
    x = np.arange(n, dtype=float)
    q = 1 + .001 * np.sin(x / 3)
    fx = 1400 + np.cos(x / 5)
    d = pd.DataFrame({'USDT_BINANCE_CLOSE': q, 'USDKRW': fx,
                      'USDT_UPBIT_CLOSE': fx / q,
                      'DXY': 100 + .02 * x, 'VIX': 20 + .03 * x,
                      'ACCOUNT_LS': 1 + .01 * x}, index=index)
    for i, c in enumerate(COINS):
        usd = (i + 1) * (100 + x + np.sin(x))
        d[c + '_BINANCE_CLOSE'] = usd * q
        d[c + '_UPBIT_CLOSE'] = usd * fx
    return d


class WorkshopRegression(unittest.TestCase):
    def test_equal_usd_prices_have_zero_premium(self):
        d = panel(pd.date_range('2025-08-01', periods=40, freq='h', tz='UTC'))
        out = correct_quotes(d)
        np.testing.assert_allclose(out[['USDT_KP'] + [c+'_KP' for c in COINS]], 0, atol=1e-14)
        np.testing.assert_allclose(out.USDT_USD_CLOSE * out.USDT_BINANCE_CLOSE, 1)
        # The archived expression has a spurious premium in this same market.
        bad = d.USDT_UPBIT_CLOSE / (d.USDKRW*d.USDT_BINANCE_CLOSE)-1
        self.assertGreater(bad.abs().max(), .001)

    def test_clock_lags_never_cross_missing_hours(self):
        idx = pd.to_datetime(['2025-08-01T09:00Z', '2025-08-01T10:00Z',
                              '2025-08-04T09:00Z'])
        s = pd.Series([10., 20., 30.], index=idx)
        lag = at_hour(s, -1)
        self.assertTrue(pd.isna(lag.iloc[0]))
        self.assertEqual(lag.iloc[1], 10)
        self.assertTrue(pd.isna(lag.iloc[2]))
        self.assertTrue(at_hour(s, 6).isna().all())
        self.assertEqual(at_hour(s, 1).iloc[0], 20)

    def test_hourly_risk_requires_24_returns(self):
        idx = pd.date_range('2025-08-01', periods=60, freq='h', tz='UTC')
        d = panel(idx)
        built, _ = prepare_data(d)
        self.assertTrue(built.BTC_DOWNSIDE_24.iloc[:24].isna().all())
        self.assertTrue(built.BTC_DOWNSIDE_24.iloc[24:].notna().all())
        sparse, _ = prepare_data(d.drop(index=idx[30]))
        self.assertTrue(sparse.loc[idx[31]:idx[54], 'BTC_DOWNSIDE_24'].isna().all())
        self.assertTrue(pd.notna(sparse.loc[idx[55], 'BTC_DOWNSIDE_24']))

    def test_stale_derived_values_are_not_trusted(self):
        d = panel(pd.date_range('2025-08-01', periods=40, freq='h', tz='UTC'))
        for c in ['USDT_KP', 'USDT_KP_pct', 'RESIDUAL_FINAL', 'DXY_ret',
                  'BTC_DOWNSIDE_24', 'BTC_DOWNSIDE_24_lag1']:
            d[c] = 12345
        built, _ = prepare_data(d)
        self.assertLess(built.USDT_KP.abs().max(), 1e-12)
        self.assertLess(built.USDT_KP_pct.abs().max(), 1e-10)
        self.assertLess(built.RESIDUAL_FINAL.abs().max(), 1e-12)
        self.assertTrue(pd.isna(built.DXY_ret.iloc[0]))
        self.assertTrue(built.BTC_DOWNSIDE_24.iloc[:24].isna().all())

    def test_duplicate_and_naive_times_are_rejected(self):
        idx = pd.date_range('2025-08-01', periods=40, freq='h', tz='UTC')
        d = panel(idx)
        with self.assertRaises(ValueError):
            prepare_data(pd.concat([d, d.iloc[[0]]]))
        d.index = d.index.tz_localize(None)
        with self.assertRaises(ValueError):
            prepare_data(d)

    def test_joint_residual_is_orthogonal_to_both_factors(self):
        rng = np.random.RandomState(4)
        x = rng.normal(size=(100, 2))
        x[:, 1] += .8*x[:, 0]
        y = x @ np.array([2., 3.]) + rng.normal(size=100)
        r = residualize(pd.Series(y), pd.DataFrame(x))
        np.testing.assert_allclose(np.column_stack([np.ones(100), x]).T@r, 0, atol=1e-10)

    def test_quantile_solution_is_scale_invariant(self):
        rng = np.random.RandomState(5)
        x = rng.normal(size=(70, 2))
        y = 3 + x @ np.array([2., -1.]) + rng.normal(size=70)
        X = np.column_stack([np.ones(70), x])
        b = quantile_fit(X, y, .1)
        scaled = X.copy()
        scaled[:, 1] *= 1e-8
        bs = quantile_fit(scaled, y * 1000, .1)
        np.testing.assert_allclose(X@b, scaled@bs/1000, atol=1e-7)
        with self.assertRaises(ValueError):
            quantile_fit(np.column_stack([X, X[:, 1]]), y, .1)

    def test_tick_uses_policy_date(self):
        idx = pd.to_datetime(['2025-07-30T10:00Z', '2025-08-01T10:00Z'])
        ticks = tick_krw(pd.Series([1400.5, 1401.], index=idx))
        np.testing.assert_allclose(ticks, [.5, 1.])
        # Exact intraday switchover was not verified; do not guess on that date.
        uncertain = pd.Series([1400.], index=pd.to_datetime(['2025-07-31T10:00Z']))
        self.assertTrue(tick_krw(uncertain).isna().all())


if __name__ == '__main__':
    unittest.main()
