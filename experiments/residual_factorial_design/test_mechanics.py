"""Synthetic specification tests; no historical performance results are used."""
import unittest

import numpy as np
import pandas as pd

from mechanics import (DESIGN, calibrate_candidate, candidate_specs,
                       choose_past, feature_sets, loss, with_calendar)


def stream(n=100):
    origin = pd.date_range('2025-08-01', periods=n, freq='h', tz='UTC')
    return pd.DataFrame(dict(origin=origin, target_time=origin + pd.Timedelta(hours=1),
                             target=np.arange(n, dtype=float), pred_raw=0., candidate=0))


def selection_rows():
    # Three fully prior months, same labels for every candidate. Candidate 12
    # wins on raw error, while 13 wins after calibration. Selection must use 13.
    origin = pd.to_datetime(['2025-09-10', '2025-10-10', '2025-11-10'], utc=True)
    frames = []
    for spec in candidate_specs():
        c = spec['candidate']
        frames.append(pd.DataFrame(dict(origin=origin,
            target_time=origin + pd.Timedelta(hours=1), target=0.,
            candidate=c, pred_raw=0. if c == 12 else 10.,
            pred_calibrated=0. if c == 13 else 10., calibration_n=60)))
    return pd.concat(frames, ignore_index=True)


class DesignTests(unittest.TestCase):
    def test_nested_information_and_common_calendar(self):
        for mode, sizes in [('available_macro', [5, 15, 23]), ('strict', [5, 12, 20])]:
            f = feature_sets(mode)
            self.assertEqual([len(f[k]) for k in ['R', 'B', 'F']], sizes)
            self.assertLess(set(f['R']), set(f['B']))
            self.assertEqual(set(f['F']) - set(f['B']), set(DESIGN['position_features']))
            for cols in f.values():
                self.assertEqual(len(cols), len(set(cols)))
                self.assertFalse(set(cols) & {'target', 'delta_e', 'target_time'})
                self.assertTrue(set(DESIGN['calendar_in_all_cells_and_cases']) <= set(cols))
        x = pd.DataFrame(index=pd.date_range('2025-08-01', periods=30, freq='h', tz='UTC'))
        np.testing.assert_array_equal(with_calendar(x).weekend,
                                      (x.index.tz_convert('Asia/Seoul').dayofweek >= 5))

    def test_finite_grids_and_pure_ml_pool(self):
        specs = candidate_specs()
        self.assertEqual(len(specs), 60)
        counts = {kind: sum(s['kind'] == kind for s in specs) for kind in DESIGN['families']}
        self.assertEqual(counts, dict(linear=12, qrf=12, boosting=12, threshold=24))
        self.assertEqual(DESIGN['ml_selection_pool'], ['qrf', 'boosting'])
        self.assertEqual(len(DESIGN['primary_comparisons']), 6)

    def test_quantile_direction_and_translation(self):
        np.testing.assert_allclose(loss([-20., -10.], [-10., -20.]), [9., 1.])
        y, pred, current = np.array([-2., 5.]), np.array([-3., 2.]), np.array([8., -9.])
        np.testing.assert_allclose(loss(y, pred), loss(y-current, pred-current))

    def test_calibration_strict_label_boundary_and_order_statistic(self):
        d = stream()
        c = calibrate_candidate(d)
        self.assertEqual(c.calibration_n.iloc[30], 29)
        self.assertEqual(c.calibration_n.iloc[31], 30)
        self.assertEqual(c.correction.iloc[30], 0.)
        self.assertEqual(c.correction.iloc[31], 2.)  # third of errors 0..29
        self.assertEqual(c.calibration_n.iloc[99], 60)
        changed = d.copy()
        changed.loc[39:, 'target'] += 1e9
        np.testing.assert_array_equal(calibrate_candidate(changed).correction.iloc[:41],
                                      c.correction.iloc[:41])

    def test_candidate_histories_cannot_be_mixed(self):
        d = stream()
        d.loc[50:, 'candidate'] = 1
        with self.assertRaisesRegex(ValueError, 'Mixed calibration stream'):
            calibrate_candidate(d)

    def test_old_errors_expire(self):
        d = stream(31)
        d.loc[30, ['origin', 'target_time']] += pd.Timedelta(days=91)
        c = calibrate_candidate(d)
        self.assertEqual(c.calibration_n.iloc[30], 0)
        self.assertEqual(c.correction.iloc[30], 0.)

    def test_choose_calibrated_not_raw_and_exact_tie(self):
        d = selection_rows()
        self.assertEqual(choose_past(d, '2025-12-01T00:00:00Z', 'ml_selected')['candidate'], 13)
        d.pred_calibrated = 0.
        self.assertEqual(choose_past(d, '2025-12-01T00:00:00Z', 'ml_selected')['candidate'], 12)
        self.assertEqual(choose_past(d, '2025-12-01T00:00:00Z', 'linear')['candidate'], 0)

    def test_reject_outer_data_boundary_labels_and_incomplete_pairing(self):
        d = selection_rows()
        for origin in ['2025-12-01T00:00:00Z', '2025-11-30T23:00:00Z', '2025-09-30T23:00:00Z']:
            bad = d.copy()
            bad.loc[0, 'origin'] = pd.Timestamp(origin)
            bad.loc[0, 'target_time'] = pd.Timestamp(origin) + pd.Timedelta(hours=1)
            with self.assertRaises(ValueError):
                choose_past(bad, '2025-12-01T00:00:00Z', 'ml_selected')
        for bad in [d.iloc[1:], d[d.candidate != 0], pd.concat([d, d.iloc[:1]])]:
            with self.assertRaises(ValueError):
                choose_past(bad, '2025-12-01T00:00:00Z', 'ml_selected')
        d.loc[0, 'calibration_n'] = 29
        with self.assertRaisesRegex(ValueError, 'Insufficient calibration'):
            choose_past(d, '2025-12-01T00:00:00Z', 'ml_selected')


if __name__ == '__main__':
    unittest.main(verbosity=2)
