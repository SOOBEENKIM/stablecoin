"""Generate every candidate stream, then select from earlier calibrated losses.

Use a distinct module name because the preserved experiments also have run.py.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd

from execution_guard import verify_lock, write_new, sha, now
from core import (HERE, ROOT, OUT, DESIGN, SPECS, CASES, MONTHS, OUTER_MONTHS,
    IDENTITY, seeds_for, data_case, fit_predict, feature_sets, calibrate_candidate,
    choose_past, purged_inner, loss, legacy, read_predictions)

DATA = {}
CACHE = ROOT / '.runs/residual_factorial_execution'


def job_name(task):
    return '__'.join(task)


def fit_job(task):
    mode, definition, fold, information, kind = task
    train, test, residualizer = DATA[mode, definition, fold]
    cutoff = pd.Timestamp(fold + '-01', tz='UTC')
    cols = feature_sets(mode)[information]
    frames, metadata = [], []
    fits = 0
    for spec in [s for s in SPECS if s['kind'] == kind]:
        fit = legacy.a.window(train, cutoff, spec['window'])
        deterministic_pred = None
        for seed in seeds_for(mode, definition):
            cached = deterministic_pred is not None and kind in ['linear', 'threshold']
            if cached:
                pred = deterministic_pred
            else:
                _, pred = fit_predict(spec, fit, test, cols, seed)
                deterministic_pred = pred
                fits += 1
            d = test[['target_time', 'target', 'e_now']].reset_index()
            values = dict(mode=mode, definition=definition, fold=fold, information=information,
                          kind=kind, candidate=spec['candidate'], seed=seed)
            for k, v in values.items():
                d[k] = v
            d['pred_raw'] = pred
            frames.append(d)
            metadata.append(dict(**values, window=spec['window'], form=spec['form'], params=spec['params'],
                features=cols, train_n=len(fit), test_n=len(test), train_last_label=fit.target_time.max().isoformat(),
                train_first_origin=fit.index.min().isoformat(), residualizer=residualizer.metadata(),
                deterministic_seed_cache=cached, warnings=[]))
    return pd.concat(frames, ignore_index=True), metadata, fits


def calibrate_job(task):
    mode, definition, information, kind = task
    frames = []
    for cutoff in MONTHS:
        name = job_name((mode, definition, cutoff.strftime('%Y-%m'), information, kind))
        frames.append(pd.read_pickle(CACHE / (name + '.pkl.gz'))['predictions'])
    raw = pd.concat(frames, ignore_index=True)
    cal = pd.concat([calibrate_candidate(g) for _, g in raw.groupby(IDENTITY, sort=True)], ignore_index=True)
    scored = cal.origin >= pd.Timestamp(DESIGN['inner_scoring_start'])
    assert (cal.loc[scored, 'calibration_n'] >= DESIGN['calibration']['minimum']).all()
    assert (cal.loc[scored, 'latest_calibration_label'] < cal.loc[scored, 'origin']).all()
    path = OUT / 'candidates' / (job_name(task) + '.csv.gz')
    cal.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
    return str(path.relative_to(ROOT)), sha(path), len(cal), len(cal.groupby(IDENTITY))


def collect_selections():
    selections, scores, selected_frames = [], [], []
    for mode, definition in CASES:
        streams, choices = {}, {}
        for information in DESIGN['information_sets']:
            streams[information] = pd.concat([read_predictions(OUT / 'candidates' /
                (job_name((mode, definition, information, kind)) + '.csv.gz'))
                for kind in DESIGN['families']], ignore_index=True)
            stream = streams[information]
            first = stream[stream.seed == DESIGN['candidate_selection_seed']]
            for cutoff in OUTER_MONTHS:
                fold = cutoff.strftime('%Y-%m')
                inner = purged_inner(first, cutoff)
                checked = choose_past(inner, cutoff, 'ml_selected')
                rows = []
                for candidate, g in inner.groupby('candidate', sort=True):
                    losses = loss(g.target, g.pred_calibrated)
                    rows.append(dict(mode=mode, definition=definition, information=information, fold=fold,
                        candidate=int(candidate), kind=SPECS[int(candidate)]['kind'], n=len(g),
                        loss_sum=float(losses.sum()), mean_loss=float(losses.mean()),
                        first_origin=g.origin.min().isoformat(), last_label=g.target_time.max().isoformat(),
                        validation_months=','.join(sorted(g.origin.dt.strftime('%Y-%m').unique()))))
                scores.extend(rows)
                for strategy in DESIGN['strategies']:
                    kinds = DESIGN['ml_selection_pool'] if strategy == 'ml_selected' else [strategy]
                    picked = min((r['mean_loss'], r['candidate']) for r in rows if r['kind'] in kinds)
                    spec = SPECS[picked[1]]
                    if strategy == 'ml_selected':
                        assert spec['candidate'] == checked['candidate'] and picked[0] == checked['inner_score']
                    choices[information, fold, strategy] = spec
                    selections.append(dict(mode=mode, definition=definition, information=information,
                        fold=fold, strategy=strategy, **spec, inner_score=picked[0],
                        inner_months=checked['inner_months'], choice_source_information=information))
        for information in DESIGN['information_sets']:
            stream = streams[information]
            for cutoff in OUTER_MONTHS:
                fold = cutoff.strftime('%Y-%m')
                for strategy in DESIGN['strategies'] + ['matched_full_spec']:
                    spec = choices['F', fold, 'ml_selected'] if strategy == 'matched_full_spec' else choices[information, fold, strategy]
                    if strategy == 'matched_full_spec':
                        source = next(s for s in selections if (s['mode'], s['definition'], s['information'],
                            s['fold'], s['strategy']) == (mode, definition, 'F', fold, 'ml_selected'))
                        selections.append(dict(source, information=information, strategy=strategy,
                                               choice_source_information='F'))
                    chosen = stream[(stream.fold == fold) & (stream.candidate == spec['candidate'])].copy()
                    if len(chosen) == 0:
                        raise RuntimeError('Missing selected predictions')
                    chosen['strategy'] = strategy
                    selected_frames.append(chosen)
        print('SELECTED past-only', mode, definition, flush=True)
    return pd.concat(selected_frames, ignore_index=True), selections, pd.DataFrame(scores)


def main(workers, resume):
    stamp = verify_lock()
    if (OUT / 'PREDICTIONS_SEALED.json').exists() or (OUT / 'SCORES_OPENED.json').exists():
        raise RuntimeError('Predictions already sealed; refusing rerun')
    start = time.monotonic()
    if resume:
        previous = json.loads((OUT / 'RUN_STARTED.json').read_text())
        assert previous['lock_sha256'] == stamp['lock_sha256']
    else:
        if CACHE.exists():
            raise RuntimeError('Unowned cache exists')
        write_new(OUT / 'RUN_STARTED.json', dict(**stamp, started_utc=now(), workers=workers,
            existing_data_only=True, independent_confirmation=False))
        CACHE.mkdir(parents=True)
    targets = []
    for mode, definition in CASES:
        panel = legacy.a.panel(mode)
        for cutoff in MONTHS:
            key = mode, definition, cutoff.strftime('%Y-%m')
            DATA[key] = data_case(mode, definition, cutoff, panel)
            d = DATA[key][1].reset_index()
            d['mode'], d['definition'], d['fold'] = key
            targets.append(d)
    pd.concat(targets, ignore_index=True).to_csv(OUT / 'targets.csv.gz', index=False,
                                               compression={'method': 'gzip', 'mtime': 0})
    tasks = [(m, d, cutoff.strftime('%Y-%m'), info, kind) for m, d in CASES for cutoff in MONTHS
             for info in DESIGN['information_sets'] for kind in DESIGN['families']]
    todo, metadata, fits = [], [], 0
    for task in tasks:
        name = job_name(task)
        note = CACHE / (name + '.json')
        if note.exists():
            saved = json.loads(note.read_text())
            assert saved['lock_sha256'] == stamp['lock_sha256']
            assert sha(CACHE / (name + '.pkl.gz')) == saved['sha256']
            metadata.extend(saved['metadata']); fits += saved['fits']
        else:
            todo.append(task)
    context = multiprocessing.get_context('fork')
    completed = len(tasks) - len(todo)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(fit_job, task): task for task in todo}
        for future in as_completed(futures):
            task = futures[future]
            try:
                pred, meta, count = future.result()
            except Exception as exc:
                write_new(OUT / ('FAILURE_' + str(time.time_ns()) + '.json'),
                    dict(task=task, error=repr(exc), **stamp, when=now()))
                raise
            path = CACHE / (job_name(task) + '.pkl.gz')
            pd.to_pickle(dict(predictions=pred), path, compression='gzip')
            write_new(CACHE / (job_name(task) + '.json'),
                dict(**stamp, sha256=sha(path), metadata=meta, fits=count))
            metadata.extend(meta); fits += count; completed += 1
            if completed % 8 == 0 or completed == len(tasks):
                print('FIT jobs', completed, '/', len(tasks), 'actual_fits', fits,
                      'elapsed', round(time.monotonic()-start), flush=True)
    assert len(metadata) == DESIGN['candidate_fit_upper_bound_without_caching']
    with gzip.GzipFile(filename=str(OUT / 'fit_metadata.json.gz'), mode='wb', mtime=0) as f:
        f.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode())
    (OUT / 'candidates').mkdir(exist_ok=True)
    hashes = {}; candidate_rows = 0; stream_count = 0
    cal_tasks = [(m, d, info, kind) for m, d in CASES for info in DESIGN['information_sets'] for kind in DESIGN['families']]
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for i, result in enumerate(pool.map(calibrate_job, cal_tasks), 1):
            path, digest, rows, streams = result
            hashes[path] = digest; candidate_rows += rows; stream_count += streams
            if i % 4 == 0:
                print('CALIBRATE groups', i, '/', len(cal_tasks), flush=True)
    pred, selections, scores = collect_selections()
    pred.sort_values(['mode', 'definition', 'information', 'strategy', 'seed', 'origin']).to_csv(
        OUT / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    write_new(OUT / 'selections.json', selections)
    scores.to_csv(OUT / 'inner_scores.csv', index=False)
    for name in ['targets.csv.gz', 'fit_metadata.json.gz', 'predictions.csv.gz', 'selections.json', 'inner_scores.csv']:
        path = OUT / name
        hashes[str(path.relative_to(ROOT))] = sha(path)
    write_new(OUT / 'PREDICTIONS_SEALED.json', dict(**stamp, sealed_utc=now(), file_sha256=hashes,
        candidate_rows=candidate_rows, candidate_streams=stream_count, requested_fits=len(metadata),
        actual_fits=fits, deterministic_seed_cache_hits=len(metadata)-fits, jobs=len(tasks),
        selections=len(selections), inner_candidate_scores=len(scores), selected_prediction_rows=len(pred),
        warnings=0, elapsed_seconds=round(time.monotonic()-start, 2), independent_confirmation=False))
    print('SEALED every candidate and selected forecast; outer scores remain unopened.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    main(args.workers, args.resume)
