"""Behavioral tests on synthetic values and train-only data assembly."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import core
from core import (SPECS, DESIGN, feature_sets, fit_predict, data_case,
                  purged_inner, legacy, with_calendar, comparison_specs)
from factorial_stats import day_weights, bootstrap_mean, effect, holm
from execution_guard import check_hashes, write_new


def synthetic():
    rng = np.random.default_rng(816)
    index = pd.date_range('2025-06-01', periods=380, freq='h', tz='UTC')
    cols = feature_sets('available_macro')['F']
    d = with_calendar(pd.DataFrame(rng.normal(size=(len(index), len(cols))), index=index, columns=cols))
    d['target'] = .4*d.e_now - .2*d.btc_ret1 + rng.normal(size=len(d))
    d['target_time'] = d.index + pd.Timedelta(hours=1)
    return d.iloc[:320].copy(), d.iloc[320:].copy()


class ExecutionTests(unittest.TestCase):
    def test_removed_information_cannot_affect_refit_or_prediction(self):
        train, test = synthetic()
        full = feature_sets('available_macro')['F']
        for info in ['R', 'B']:
            cols = feature_sets('available_macro')[info]
            changed_train, changed_test = train.copy(), test.copy()
            for col in set(full) - set(cols):
                changed_train[col] += 1e7
                changed_test[col] -= 1e7
            changed_test['target'] += 1e9
            for kind in DESIGN['families']:
                spec = next(s for s in SPECS if s['kind'] == kind)
                _, a = fit_predict(spec, train, test, cols, 20260925)
                _, b = fit_predict(spec, changed_train, changed_test, cols, 20260925)
                np.testing.assert_array_equal(a, b)

    def test_deterministic_seed_cache_is_exact(self):
        train, test = synthetic()
        for kind in ['linear', 'threshold']:
            spec = next(s for s in SPECS if s['kind'] == kind)
            _, a = fit_predict(spec, train, test, feature_sets('available_macro')['F'], 20260925)
            _, b = fit_predict(spec, train, test, feature_sets('available_macro')['F'], 20260928)
            np.testing.assert_array_equal(a, b)

    def test_qrf_quantile_from_independent_leaf_neighbours(self):
        train, test = synthetic()
        cols = feature_sets('available_macro')['R']
        spec = next(s for s in SPECS if s['kind'] == 'qrf')
        model, values = fit_predict(spec, train, test, cols, 20260925)
        forest = model.model.forest
        leaves = forest.apply(train[cols].to_numpy())
        for row in range(5):
            query = forest.apply(test[cols].iloc[[row]].to_numpy())[0]
            weights = np.zeros(len(train))
            for tree in range(leaves.shape[1]):
                members = np.flatnonzero(leaves[:, tree] == query[tree])
                weights[members] += 1 / (len(members)*leaves.shape[1])
            self.assertAlmostEqual(weights.sum(), 1.)
            order = np.argsort(train.target.to_numpy(), kind='stable')
            j = np.flatnonzero(np.cumsum(weights[order]) >= .1-1e-12)[0]
            self.assertEqual(values[row], train.target.to_numpy()[order[j]])

    def test_only_past_three_months_and_no_boundary_labels(self):
        origin = pd.to_datetime(['2025-08-31T23:00Z', '2025-09-01T00:00Z', '2025-09-30T23:00Z',
                                 '2025-11-30T22:00Z', '2025-11-30T23:00Z', '2025-12-01T00:00Z'])
        d = pd.DataFrame(dict(origin=origin, target_time=origin+pd.Timedelta(hours=1)))
        got = purged_inner(d, pd.Timestamp('2025-12-01', tz='UTC'))
        self.assertEqual(list(got.index), [1, 3])

    def test_exact_clock_lag_does_not_jump_missing_hour(self):
        index = pd.to_datetime(['2025-06-01T00:00Z', '2025-06-01T02:00Z', '2025-06-01T03:00Z'])
        result = legacy.a.lead(pd.Series([1., 2., 3.], index=index), 1)
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertEqual(result.iloc[1], 3.)

    def test_monthly_residual_fit_ignores_future_panel_values(self):
        panel = legacy.a.panel('available_macro')
        cutoff = pd.Timestamp('2025-12-01', tz='UTC')
        original = legacy.a.Residualizer('PCA').fit(panel, cutoff)
        fake = panel.copy()
        fake.loc[fake.index >= cutoff, ['y', 'g'] + ['kp_'+c for c in ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']]] += 1e7
        changed = legacy.a.Residualizer('PCA').fit(fake, cutoff)
        self.assertEqual(original.metadata(), changed.metadata())
        train, test, residualizer = data_case('available_macro', 'EQ', cutoff, panel)
        self.assertLess(train.target_time.max(), cutoff)
        self.assertLess(residualizer.fit_end, cutoff)
        self.assertTrue(((test.target_time-test.index) == pd.Timedelta(hours=1)).all())

    def test_paired_bootstrap_against_direct_resampled_rows(self):
        origin = pd.to_datetime(['2025-12-01T01:00Z', '2025-12-01T03:00Z', '2025-12-02T01:00Z',
                                '2025-12-03T01:00Z', '2026-01-01T01:00Z', '2026-01-03T01:00Z', '2026-01-05T01:00Z'])
        frame = pd.DataFrame(dict(origin=origin, fold=origin.strftime('%Y-%m')))
        values = np.array([1., 3., 5., 2., 4., 6., 7.])
        for block in [1, 5, 10]:
            w = day_weights(frame, block)
            rng = np.random.default_rng(20260925+block)
            total = np.zeros(len(w)); count = np.zeros(len(w))
            for _, g in frame.groupby('fold', sort=True):
                days = pd.date_range(g.origin.dt.normalize().min(), g.origin.dt.normalize().max(), freq='D')
                starts = rng.integers(0, len(days), size=(len(w), int(np.ceil(len(days)/block))))
                draws = ((starts[:, :, None]+np.arange(block)) % len(days)).reshape(len(w), -1)[:, :len(days)]
                for k in range(len(w)):
                    for day in days[draws[k]]:
                        ix = g.index[g.origin.dt.normalize() == day]
                        total[k] += values[ix].sum(); count[k] += len(ix)
            np.testing.assert_allclose(bootstrap_mean(w, values), total/count, rtol=0, atol=1e-12)
            e = effect(values, values*.8, w)
            self.assertAlmostEqual(e['improvement_pct'], 20.)
            self.assertAlmostEqual(e['improvement_lo'], 20.)
            self.assertAlmostEqual(e['improvement_hi'], 20.)

    def test_empty_bootstrap_draw_is_not_silently_dropped(self):
        origin = pd.to_datetime(['2025-12-01T00:00Z', '2025-12-20T00:00Z'])
        frame = pd.DataFrame(dict(origin=origin, fold=origin.strftime('%Y-%m')))
        with self.assertRaisesRegex(ValueError, 'Empty bootstrap'):
            day_weights(frame, 1)

    def test_full_selection_assembly_uses_own_candidate_history(self):
        import factorial_run as run
        with tempfile.TemporaryDirectory() as tmp:
            directory = core.Path(tmp)
            (directory/'candidates').mkdir()
            for info, best in [('R', 12), ('B', 24), ('F', 13)]:
                for kind in DESIGN['families']:
                    rows = []
                    for spec in [s for s in SPECS if s['kind'] == kind]:
                        for month in pd.date_range('2025-09-01', '2026-03-01', freq='MS', tz='UTC'):
                            for hour in [1, 2, 3]:
                                origin = month+pd.Timedelta(days=4, hours=hour)
                                value = 0. if spec['candidate'] == best else 2.
                                rows.append(dict(mode='available_macro', definition='EQ', information=info,
                                    kind=kind, candidate=spec['candidate'], seed=20260925,
                                    origin=origin, target_time=origin+pd.Timedelta(hours=1), target=0., e_now=0.,
                                    fold=month.strftime('%Y-%m'), pred_raw=50., pred_calibrated=value,
                                    correction=value-50., calibration_n=60,
                                    latest_calibration_label=origin-pd.Timedelta(hours=1)))
                    pd.DataFrame(rows).to_csv(directory/'candidates'/('__'.join(['available_macro','EQ',info,kind])+'.csv.gz'), index=False)
            with patch.object(run, 'OUT', directory), patch.object(run, 'CASES', [('available_macro', 'EQ')]):
                pred, selections, scores = run.collect_selections()
            self.assertEqual(len(selections), 72)
            self.assertEqual(len(scores), 720)
            for info, best in [('R', 12), ('B', 24), ('F', 13)]:
                own = pred[(pred.information == info) & (pred.strategy == 'ml_selected')]
                matched = pred[(pred.information == info) & (pred.strategy == 'matched_full_spec')]
                self.assertEqual(set(own.candidate), {best})
                self.assertEqual(set(matched.candidate), {13})
                self.assertTrue((matched.pred_calibrated == (0. if info == 'F' else 2.)).all())

    def test_holm_family_and_immutable_file_checks(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
        specs = comparison_specs()
        self.assertEqual(sum(s['primary'] for s in specs), 6)
        self.assertEqual(len({s['id'] for s in specs}), len(specs))
        with tempfile.TemporaryDirectory() as tmp:
            path = core.Path(tmp) / 'marker.json'
            write_new(path, dict(ok=True))
            with self.assertRaises(FileExistsError):
                write_new(path, dict(ok=False))
            with self.assertRaises(RuntimeError):
                check_hashes({str(path): 'invalid_digest'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
