"""Produce and seal all stage-one forecasts without displaying outer scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import warnings
import numpy as np
import pandas as pd
from step1_guard import verify_lock, write_new, sha, now
from step1_core import (HERE, OUT, DESIGN, CASES, SPECS, Benchmark, choose,
                        read_legacy, a, n)


def inner_job(task):
    mode, definition, fold = task
    cutoff = pd.Timestamp(fold + '-01', tz='UTC')
    train, test, r = n.outer_sample(a.panel(mode), mode, definition, cutoff)
    test = n.purge_inner(test, cutoff)
    rows = []
    for spec in SPECS:
        fit = a.window(train, cutoff, spec['window'])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            m = Benchmark(spec).fit(fit)
            pred = m.predict(test[m.columns])
        loss = a.pinball(test.target, pred, .1)
        assert np.isfinite(loss).all()
        rows.append(dict(mode=mode, definition=definition, fold=fold, **spec,
            n=len(test), loss_sum=float(loss.sum()), valid=True, train_n=len(fit),
            first_origin=test.index.min().isoformat(), last_label=test.target_time.max().isoformat(),
            train_last_label=fit.target_time.max().isoformat(), residualizer_fit_end=r.fit_end.isoformat(),
            warnings=[str(w.message) for w in captured]))
    return rows


def outer_job(task):
    mode, definition, fold, rows = task
    cutoff = pd.Timestamp(fold + '-01', tz='UTC')
    train, test, r = n.outer_sample(a.panel(mode), mode, definition, cutoff)
    frames, metadata = [], []
    for strategy in DESIGN['new_strategies']:
        spec = choose(pd.DataFrame(rows), cutoff, strategy)
        fit = a.window(train, cutoff, spec['window'])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model = Benchmark(spec).fit(fit)
            pred = model.predict(test[model.columns])
        assert np.isfinite(pred).all()
        d = test[['target_time', 'target', 'e_now']].reset_index()
        for k, v in dict(mode=mode, definition=definition, fold=fold, strategy=strategy,
                         seed=0, candidate=spec['candidate'], info='dynamics').items():
            d[k] = v
        d['pred_raw'] = pred
        frames.append(d)
        metadata.append(dict(mode=mode, definition=definition, fold=fold, **spec,
            train_n=len(fit), test_n=len(test), features=model.columns,
            train_last_label=fit.target_time.max().isoformat(), first_origin=test.index.min().isoformat(),
            residualizer=r.metadata(), coefficients=model.coefficients(),
            warnings=[str(w.message) for w in captured]))
    return pd.concat(frames, ignore_index=True), metadata


def main(workers):
    stamp = verify_lock()
    write_new(OUT/'RUN_STARTED.json', dict(**stamp, started_utc=now()))
    start = time.monotonic()
    checkpoints = OUT/'checkpoints'
    checkpoints.mkdir()
    months = sorted({m.strftime('%Y-%m') for c in n.FOLDS for m in n.inner_months(c)})
    tasks = [(mode, definition, month) for mode, definition in CASES for month in months]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(inner_job, t): t for t in tasks}
        for i, f in enumerate(as_completed(pending), 1):
            result = f.result(); rows.extend(result)
            write_new(checkpoints/('__'.join(pending[f])+'.json'), result)
            if i % 4 == 0:
                print('INNER', i, '/', len(tasks), 'elapsed', round(time.monotonic()-start), flush=True)
    scores = pd.DataFrame(rows).sort_values(['mode','definition','fold','candidate'])
    scores.to_csv(OUT/'inner_scores.csv', index=False)
    tasks = []
    for mode, definition in CASES:
        for cutoff in n.FOLDS:
            months = [x.strftime('%Y-%m') for x in n.inner_months(cutoff)]
            sub = scores[(scores['mode']==mode)&(scores.definition==definition)&scores.fold.isin(months)]
            tasks.append((mode, definition, cutoff.strftime('%Y-%m'), sub.to_dict('records')))
    frames, metadata = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(outer_job, tasks), 1):
            frame, meta = result; frames.append(frame); metadata.extend(meta)
            if i % 4 == 0: print('OUTER', i, '/', len(tasks), flush=True)
    raw = pd.concat(frames, ignore_index=True)
    fresh = pd.concat([a.calibrate(g) for _, g in raw.groupby(['mode','definition','strategy'])], ignore_index=True)
    fresh['evaluation'] = fresh.origin >= pd.Timestamp(DESIGN['evaluation_start'])
    assert (fresh.loc[fresh.evaluation, 'calibration_n'] >= 30).all()
    old = read_legacy()
    for mode, definition in CASES:
        ref = old[(old['mode']==mode)&(old.definition==definition)&
                  (old.strategy=='selected')&(old.seed==n.SEEDS[0])].sort_values('origin')
        for strategy in DESIGN['new_strategies']:
            g = fresh[(fresh['mode']==mode)&(fresh.definition==definition)&
                      (fresh.strategy==strategy)].sort_values('origin')
            assert list(g.origin)==list(ref.origin)
            assert list(g.target_time)==list(ref.target_time)
            np.testing.assert_allclose(g[['target','e_now']], ref[['target','e_now']], atol=1e-10, rtol=0)
    columns = ['mode','definition','fold','strategy','seed','info','origin','target_time',
               'target','e_now','pred_raw','pred_calibrated','correction','calibration_n',
               'latest_calibration_label','evaluation']
    pred = pd.concat([fresh[columns], old[columns]], ignore_index=True)
    pred = pred.sort_values(['mode','definition','strategy','seed','origin'])
    pred['target_change'] = pred.target-pred.e_now
    pred['pred_change_raw'] = pred.pred_raw-pred.e_now
    pred['pred_change_calibrated'] = pred.pred_calibrated-pred.e_now
    pred.to_csv(OUT/'predictions.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    write_new(OUT/'selected_models.json', metadata)
    pd.DataFrame([{k:v for k,v in m.items() if k not in ['residualizer','warnings']} for m in metadata]).to_csv(OUT/'selected_models.csv', index=False)
    warning_n = sum(len(x) for x in scores.warnings)+sum(len(m['warnings']) for m in metadata)
    write_new(OUT/'PREDICTIONS_SEALED.json', dict(**stamp, sealed_utc=now(),
        predictions_sha256=sha(OUT/'predictions.csv.gz'), selections_sha256=sha(OUT/'selected_models.json'),
        inner_sha256=sha(OUT/'inner_scores.csv'), inner_fits=len(scores), outer_fits=len(metadata),
        warnings=warning_n, elapsed_seconds=round(time.monotonic()-start,2),
        independent_confirmation=False, legacy_predictions_refitted=False))
    print('SEALED. No outer scores displayed.', flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers', type=int, default=8)
    main(p.parse_args().workers)
