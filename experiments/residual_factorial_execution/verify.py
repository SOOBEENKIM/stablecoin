"""Independently audit arithmetic/selection, then refit January before scores open."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing

import numpy as np
import pandas as pd

from core import (ROOT, HERE, OUT, DESIGN, SPECS, CASES, OUTER_MONTHS, IDENTITY,
    feature_sets, seeds_for, read_predictions, data_case, fit_predict, legacy)
from execution_guard import verify_seal, write_new, sha, now
from factorial_stats import day_weights, bootstrap_mean

JANUARY = {}


def independent_loss(actual, forecast):
    error = np.asarray(actual, float) - np.asarray(forecast, float)
    return np.maximum(.1*error, -.9*error)


def independent_calibration(g, poison=False):
    origins = pd.DatetimeIndex(g.origin).asi8
    labels = pd.DatetimeIndex(g.target_time).asi8
    error = g.target.to_numpy() - g.pred_raw.to_numpy()
    if poison:
        error[len(error)//2:] += 1e8
    corrections, sizes, latest = [], [], []
    for t in origins:
        ix = np.flatnonzero((labels < t) & (labels >= t-90*24*3600*10**9))[-60:]
        sizes.append(len(ix)); latest.append(labels[ix[-1]] if len(ix) else pd.NaT.value)
        corrections.append(float(np.sort(error[ix])[int(np.ceil(.1*len(ix)))-1]) if len(ix) >= 30 else 0.)
    return np.array(corrections), np.array(sizes), np.array(latest)


def audit_candidate_file(path):
    d = read_predictions(path)
    assert not d.duplicated(IDENTITY + ['origin']).any()
    rows = []; maximum = 0.; streams = 0; poisoned = 0
    for key, g in d.groupby(IDENTITY, sort=True):
        g = g.sort_values('origin').reset_index(drop=True)
        assert ((g.target_time-g.origin) == pd.Timedelta(hours=1)).all()
        c, n, latest = independent_calibration(g)
        np.testing.assert_array_equal(n, g.calibration_n)
        np.testing.assert_array_equal(latest, pd.DatetimeIndex(g.latest_calibration_label).asi8)
        np.testing.assert_allclose(c, g.correction, atol=1e-10, rtol=0)
        np.testing.assert_allclose(g.pred_raw+c, g.pred_calibrated, atol=1e-10, rtol=0)
        maximum = max(maximum, float(np.max(np.abs(c-g.correction))))
        assert (g.loc[g.origin >= pd.Timestamp(DESIGN['inner_scoring_start']), 'calibration_n'] >= 30).all()
        if streams == 0:
            changed, _, _ = independent_calibration(g, poison=True)
            stop = len(g)//2+1
            np.testing.assert_array_equal(changed[:stop], c[:stop]); poisoned += 1
        streams += 1
        mode, definition, info, candidate, seed = key
        if seed != DESIGN['candidate_selection_seed']:
            continue
        for cutoff in OUTER_MONTHS:
            lower = cutoff-pd.offsets.MonthBegin(3)
            same_month = g.origin.dt.strftime('%Y-%m') == g.target_time.dt.strftime('%Y-%m')
            inner = g[(g.origin >= lower) & (g.origin < cutoff) & (g.target_time < cutoff) & same_month]
            assert inner.origin.dt.strftime('%Y-%m').nunique() == 3
            l = independent_loss(inner.target, inner.pred_calibrated)
            rows.append(dict(mode=mode, definition=definition, information=info, candidate=int(candidate),
                fold=cutoff.strftime('%Y-%m'), n=len(inner), mean_loss=float(l.mean()), loss_sum=float(l.sum())))
    return rows, maximum, streams, poisoned


def independent_block_means(frame, values, block):
    rng = np.random.default_rng(20260925+block)
    totals = np.zeros((1999, 2))
    d = frame.copy(); d['value'] = values; d['count'] = 1
    for _, g in d.groupby('fold', sort=True):
        days = pd.date_range(g.origin.dt.normalize().min(), g.origin.dt.normalize().max(), freq='D')
        daily = g.groupby(g.origin.dt.normalize())[['value', 'count']].sum().reindex(days, fill_value=0).to_numpy()
        starts = rng.integers(0, len(days), size=(1999, int(np.ceil(len(days)/block))))
        ix = ((starts[:, :, None]+np.arange(block)) % len(days)).reshape(1999, -1)[:, :len(days)]
        totals += daily[ix].sum(axis=1)
    return totals[:, 0]/totals[:, 1]


def refit_job(task):
    mode, definition, info, candidate, seed, expected = task
    train, test, _ = JANUARY[mode, definition]
    spec = SPECS[candidate]; cols = feature_sets(mode)[info]
    fit = legacy.a.window(train, pd.Timestamp('2026-01-01', tz='UTC'), spec['window'])
    model, pred = fit_predict(spec, fit, test, cols, seed)
    np.testing.assert_allclose(pred, expected, rtol=0, atol=1e-10)
    fake_test = test.copy(); fake_test['target'] += 1e8
    np.testing.assert_array_equal(model.predict(fake_test[cols]), pred)
    excluded = set(feature_sets(mode)['F']) - set(cols)
    if excluded:
        fake_fit = fit.copy()
        for col in excluded:
            fake_fit[col] += 1e8
            fake_test[col] -= 1e8
        _, poison_pred = fit_predict(spec, fake_fit, fake_test, cols, seed)
        np.testing.assert_array_equal(poison_pred, pred)
    return float(np.max(np.abs(pred-expected))), bool(excluded)


def audit_input_times():
    counts = {}
    for mode in ['available_macro', 'strict']:
        panel = legacy.a.panel(mode)
        for c, limit in [('fx_age_min', 60.), ('account_age_min', 10.), ('oi_age_min', 10.), ('funding_age_min', 480.)]:
            v = panel[c].dropna()
            assert (v >= 0).all() and (v <= limit).all()
        if mode == 'available_macro':
            for c in ['vix_known_age_hours', 'dxy_known_age_hours']:
                v = panel[c].dropna()
                assert (v >= 0).all() and (v <= 96).all()
        # Reconstruct availability of the derivative feed with an independent
        # searchsorted lookup, including the fixed five-minute release delay.
        feed = pd.read_csv(legacy.a.INPUT_PATHS[-2])
        times = pd.DatetimeIndex(pd.to_datetime(feed.datetime_utc, utc=True, format='mixed')) + pd.Timedelta(minutes=5)
        order = np.argsort(times.asi8); time_ns = times.asi8[order]
        pos = np.searchsorted(time_ns, panel.index.asi8, side='right')-1
        valid = pos >= 0
        valid &= panel.index.asi8-time_ns[np.maximum(pos, 0)] <= 10*60*10**9
        for source, dest in [('ACCOUNT_LS', 'account'), ('OI', 'oi')]:
            expected = np.full(len(panel), np.nan)
            expected[valid] = feed[source].to_numpy()[order][pos[valid]]
            np.testing.assert_allclose(expected, panel[dest], rtol=0, atol=1e-10, equal_nan=True)
        counts[mode] = int(valid.sum())
    return counts


def main(workers):
    stamp, seal = verify_seal()
    if (OUT / 'SCORES_OPENED.json').exists():
        raise RuntimeError('Pre-opening verification must precede score opening')
    input_counts = audit_input_times()
    metadata = json.loads(gzip.decompress((OUT / 'fit_metadata.json.gz').read_bytes()))
    assert len(metadata) == 10080
    keys = set()
    for m in metadata:
        key = tuple(m[k] for k in ['mode', 'definition', 'fold', 'information', 'candidate', 'seed'])
        assert key not in keys; keys.add(key)
        cutoff = pd.Timestamp(m['fold']+'-01', tz='UTC')
        assert pd.Timestamp(m['train_last_label']) < cutoff
        assert pd.Timestamp(m['residualizer']['fit_end']) < cutoff
        assert m['features'] == feature_sets(m['mode'])[m['information']]
        assert m['warnings'] == []
        spec = SPECS[m['candidate']]
        for k in ['kind', 'form', 'window', 'params']:
            assert m[k] == spec[k]
        assert m['seed'] in seeds_for(m['mode'], m['definition'])
        if m['deterministic_seed_cache']:
            assert m['kind'] in ['linear', 'threshold'] and m['seed'] != DESIGN['candidate_selection_seed']
    context = multiprocessing.get_context('fork')
    independent_scores = []; cal_error = 0.; streams = 0; poison_n = 0
    files = sorted((OUT / 'candidates').glob('*.csv.gz'))
    assert len(files) == 48
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for i, (rows, err, count, poisoned) in enumerate(pool.map(audit_candidate_file, files), 1):
            independent_scores.extend(rows); cal_error = max(cal_error, err); streams += count; poison_n += poisoned
            if i % 8 == 0:
                print('INDEPENDENT candidate audit', i, '/', len(files), flush=True)
    scores = pd.DataFrame(independent_scores)
    cols = ['mode', 'definition', 'information', 'fold', 'candidate']
    stored = pd.read_csv(OUT / 'inner_scores.csv', float_precision='round_trip').sort_values(cols).reset_index(drop=True)
    expected = scores.sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(stored[cols+['n']], expected[cols+['n']])
    np.testing.assert_allclose(stored[['mean_loss', 'loss_sum']], expected[['mean_loss', 'loss_sum']], rtol=0, atol=1e-10)
    selections = json.loads((OUT / 'selections.json').read_text())
    assert len(selections) == 288
    for m in selections:
        info = m['choice_source_information']
        s = scores[(scores['mode'] == m['mode']) & (scores.definition == m['definition']) &
                   (scores.information == info) & (scores.fold == m['fold'])]
        kinds = DESIGN['ml_selection_pool'] if m['strategy'] in ['ml_selected', 'matched_full_spec'] else [m['strategy']]
        choice = min((row.mean_loss, int(row.candidate)) for row in s.itertuples() if SPECS[int(row.candidate)]['kind'] in kinds)
        assert choice[1] == m['candidate']
        np.testing.assert_allclose(choice[0], m['inner_score'], atol=1e-12, rtol=0)
    pred = read_predictions(OUT / 'predictions.csv.gz')
    assert not pred.duplicated(['mode', 'definition', 'information', 'strategy', 'seed', 'origin']).any()
    assert (pred.origin >= pd.Timestamp(DESIGN['evaluation_start'])).all()
    assert (pred.target_time < pd.Timestamp(DESIGN['evaluation_end_exclusive'])).all()
    assert ((pred.target_time-pred.origin) == pd.Timedelta(hours=1)).all()
    old = read_predictions(HERE.parent / 'residual_nested_validation/results/predictions.csv.gz')
    counts = {}; join_count = 0
    for mode, definition in CASES:
        ref = old[(old['mode'] == mode) & (old.definition == definition) &
                  (old.origin >= pd.Timestamp(DESIGN['evaluation_start']))]
        ref = ref[['origin', 'target_time', 'target', 'e_now']].drop_duplicates().sort_values('origin')
        for info in DESIGN['information_sets']:
            new = pred[(pred['mode'] == mode) & (pred.definition == definition) & (pred.information == info)]
            for (_, seed), g in new.groupby(['strategy', 'seed']):
                g = g.sort_values('origin')
                assert list(g.origin) == list(ref.origin) and list(g.target_time) == list(ref.target_time)
                np.testing.assert_allclose(g[['target', 'e_now']], ref[['target', 'e_now']], rtol=0, atol=1e-10)
                assert len(g) == DESIGN['expected_outer_n'][mode]
            candidates = pd.concat([read_predictions(OUT / 'candidates' /
                ('__'.join([mode, definition, info, kind])+'.csv.gz')) for kind in DESIGN['families']], ignore_index=True)
            merged = new.merge(candidates, on=['candidate', 'seed', 'origin'], suffixes=('_selected', '_source'), validate='many_to_one')
            assert len(merged) == len(new)
            for col in ['target', 'e_now', 'pred_raw', 'correction', 'pred_calibrated', 'calibration_n']:
                np.testing.assert_array_equal(merged[col+'_selected'], merged[col+'_source'])
            for m in [s for s in selections if (s['mode'], s['definition'], s['information']) == (mode, definition, info)]:
                chosen = new[(new.fold == m['fold']) & (new.strategy == m['strategy'])]
                assert set(chosen.candidate) == {m['candidate']}
                assert set(chosen.seed) == set(seeds_for(mode, definition))
            join_count += len(merged)
        counts[mode+'/'+definition] = len(ref)
        JANUARY[mode, definition] = data_case(mode, definition, pd.Timestamp('2026-01-01', tz='UTC'))
    tasks = []
    jan = pred[pred.fold == '2026-01']
    for key, g in jan.groupby(['mode', 'definition', 'information', 'candidate', 'seed'], sort=True):
        g = g[['origin', 'pred_raw']].drop_duplicates().sort_values('origin')
        assert list(g.origin) == list(JANUARY[key[0], key[1]][1].index)
        tasks.append(tuple(key)+(g.pred_raw.to_numpy(),))
    refit_error = 0.; perturb_refits = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for i, (err, perturbed) in enumerate(pool.map(refit_job, tasks), 1):
            refit_error = max(refit_error, err); perturb_refits += perturbed
            if i % 12 == 0 or i == len(tasks):
                print('JANUARY refit verified', i, '/', len(tasks), flush=True)
    # Independent arithmetic for the six primary paired loss vectors. These
    # diagnostics check equality, not reveal or use their performance scores.
    main_case = pred[(pred['mode'] == DESIGN['primary_case'][0]) & (pred.definition == DESIGN['primary_case'][1])].copy()
    main_case['loss'] = independent_loss(main_case.target, main_case.pred_calibrated)
    maps = {key: g.groupby('origin', as_index=False).agg(loss=('loss', 'mean'), fold=('fold', 'first'))
            for key, g in main_case.groupby(['information', 'strategy'])}
    bootstrap_error = 0.
    for comparison in DESIGN['primary_comparisons']:
        reference, candidate = maps[tuple(comparison['reference'])], maps[tuple(comparison['candidate'])]
        assert reference.origin.equals(candidate.origin)
        for block in [1, 5, 10]:
            weights = day_weights(reference, block)
            for values in [reference.loss.to_numpy(), (reference.loss-candidate.loss).to_numpy()]:
                a = bootstrap_mean(weights, values)
                b = independent_block_means(reference, values, block)
                bootstrap_error = max(bootstrap_error, float(np.max(np.abs(a-b))))
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
    expected_loss = independent_loss(pred.target, pred.pred_calibrated)
    shifted = independent_loss(pred.target-pred.e_now, pred.pred_calibrated-pred.e_now)
    np.testing.assert_allclose(expected_loss, shifted, rtol=0, atol=1e-10)
    write_new(OUT / 'VERIFICATION.json', dict(**stamp, verified_utc=now(),
        predictions_sha256=sha(OUT / 'predictions.csv.gz'), independent_confirmation=False,
        fits_metadata_checked=len(metadata), candidate_streams_independently_calibrated=streams,
        poisoned_candidate_streams=poison_n, candidate_inner_scores_recomputed=len(scores),
        selections_recomputed=len(selections), selected_rows_matched_to_candidates=join_count,
        january_unique_refits=len(tasks), additional_removed_input_refits=perturb_refits,
        counts=counts, derivative_availability_rows=input_counts, max_calibration_error_bp=cal_error,
        max_january_refit_error_bp=refit_error, max_independent_bootstrap_error_bp=bootstrap_error,
        legacy_origins_and_targets_preserved=True, scores_not_used_to_change_design=True))
    print('VERIFIED all candidate histories, selections, paired targets, January refits and arithmetic.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=16)
    main(p.parse_args().workers)
