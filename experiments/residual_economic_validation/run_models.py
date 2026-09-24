"""Independent monthly fits run in processes; every result is checkpointed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import subprocess
import time
import warnings
import numpy as np
import pandas as pd
from economic import HERE, ADAPTIVE, a, GROUPS, CASES, SEEDS, features, threshold_candidates, make_model

OUT = HERE / 'results'


def job(task):
    mode, definition, fold, kind, info = task
    key = dict(mode=mode, definition=definition, fold=fold, model=kind, info=info)
    cutoff = pd.Timestamp(fold + '-01', tz='UTC')
    vs = cutoff - pd.offsets.MonthBegin(1)
    ti, va, tr, te, ri, ro = a.splits(a.panel(mode), mode, definition, cutoff)
    cols = features(mode, info)
    candidates = threshold_candidates() if kind == 'threshold' else a.candidates('qrf')
    scores, warning_rows = [], []
    for days, form, params in candidates:
        train = a.window(ti, vs, days)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model = make_model(kind, form, params, SEEDS[0]).fit(train, cols)
            pv = model.predict(va)
        loss = float(a.pinball(va.target, pv, .1).mean())
        assert np.isfinite(loss)
        scores.append(dict(**key, window=str(days), form=form, params=json.dumps(params, sort_keys=True),
                           validation_loss_bp=loss, train_n=len(train), validation_n=len(va)))
        warning_rows.extend(str(w.message) for w in captured)
    selected = min(scores, key=lambda s: s['validation_loss_bp'])
    days, form, params = selected['window'], selected['form'], json.loads(selected['params'])
    seeds = SEEDS if mode == 'available_macro' and definition == 'EQ' and kind == 'qrf' else [SEEDS[0]]
    specs = [('retuned', seed, days, form, params) for seed in seeds]
    if kind == 'qrf' and mode == 'available_macro' and definition == 'EQ':
        old = pd.read_csv(ADAPTIVE / 'results/selected_candidates.csv', dtype={'window': str})
        s = old[(old['mode'] == mode) & (old.definition == definition) & (old.fold == fold) &
                (old.model == 'qrf') & (old['info'] == 'history')].iloc[0]
        specs += [('fixed_history', seed, s['window'], s['form'], json.loads(s['params'])) for seed in seeds]
    frames, fitted_metadata = [], []
    for strategy, seed, days, form, params in specs:
        train = a.window(tr, cutoff, days)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model = make_model(kind, form, params, seed).fit(train, cols)
            forecast = model.predict(te)
        warning_rows.extend(str(w.message) for w in captured)
        d = te[['target_time', 'target', 'e_now', 'delta_e', 'downside24', 'btc_vol24', 'account_log']].reset_index()
        for k, v in key.items():
            d[k] = v
        d['strategy'], d['seed'], d['pred_raw'] = strategy, seed, forecast
        assert np.isfinite(forecast).all()
        frames.append(d)
        meta = dict(**key, strategy=strategy, seed=seed, window=str(days), form=form, params=params,
                    features=cols, train_n=len(train), test_n=len(te))
        if kind == 'threshold':
            meta.update(cuts=model.cuts.tolist(), regime_counts=model.counts.tolist())
        fitted_metadata.append(meta)
    audit = dict(**key, inner=ri.metadata(), outer=ro.metadata(), cutoff=cutoff.isoformat(),
        validation_start=vs.isoformat(), inner_max_label=ti.target_time.max().isoformat(),
        outer_max_label=tr.target_time.max().isoformat(), first_test_origin=te.index.min().isoformat())
    return dict(pred=pd.concat(frames, ignore_index=True), selected=selected, candidates=scores,
                fits=fitted_metadata, audit=audit, warnings=warning_rows, fit_count=len(scores) + len(specs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    start = time.time()
    manifest = dict(parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.ROOT, text=True).strip(),
        protocol_sha256=a.sha256(HERE / 'PROTOCOL_KO.md'),
        source_sha256={f.name: a.sha256(f) for f in [HERE / 'economic.py', HERE / 'run_models.py']},
        inherited_source_sha256={str(f.relative_to(a.ROOT)): a.sha256(f) for f in
            [ADAPTIVE / 'adaptive.py', a.LEGACY / 'data.py', a.LEGACY / 'models.py']},
        input_sha256={str(f.relative_to(a.ROOT)): a.sha256(f) for f in a.INPUT_PATHS},
        previous_forecast_sha256={str(f.relative_to(a.ROOT)): a.sha256(f) for f in
            [ADAPTIVE / 'results/predictions.csv.gz', ADAPTIVE / 'results/seed_predictions.csv.gz',
             ADAPTIVE / 'results/selected_candidates.csv']}, workers=args.workers, completed=False)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    tasks = [(mode, definition, cutoff.strftime('%Y-%m'), kind, info)
             for mode, definition in CASES for cutoff in a.FOLDS
             for kind, info in [('qrf', 'without_' + group) for group in GROUPS] +
                               [('threshold', 'base'), ('threshold', 'history')]]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(job, task): task for task in tasks}
        for future in as_completed(pending):
            task = pending[future]
            result = future.result()
            name = '__'.join(task)
            result['pred'].to_csv(checkpoints / (name + '.csv.gz'), index=False,
                                  compression={'method': 'gzip', 'mtime': 0})
            audit_result = {k: v for k, v in result.items() if k != 'pred'}
            (checkpoints / (name + '.json')).write_text(json.dumps(audit_result, indent=2))
            results.append(result)
            print('DONE', len(results), '/', len(tasks), name, 'elapsed', round(time.time() - start, 1), flush=True)
    raw = pd.concat([r['pred'] for r in results], ignore_index=True)
    streams = ['mode', 'definition', 'model', 'info', 'strategy', 'seed']
    pred = pd.concat([a.calibrate(g) for _, g in raw.groupby(streams, sort=True)], ignore_index=True)
    for v in ['raw', 'calibrated']:
        pred['loss_' + v] = a.pinball(pred.target, pred['pred_' + v], .1)
        pred['below_' + v] = (pred.target < pred['pred_' + v]).astype(int)
    pred['evaluation'] = pred.origin >= a.EVALUATION_START
    assert (pred.loc[pred.evaluation, 'calibration_n'] >= 30).all()
    pred.to_csv(OUT / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    for file, key in [('validation_candidates.csv', 'candidates'), ('fitted_models.csv', 'fits')]:
        pd.DataFrame([x for r in results for x in r[key]]).to_csv(OUT / file, index=False)
    pd.DataFrame([r['selected'] for r in results]).to_csv(OUT / 'selected_candidates.csv', index=False)
    (OUT / 'split_audit.json').write_text(json.dumps([r['audit'] for r in results], indent=2))
    (OUT / 'warnings.json').write_text(json.dumps([w for r in results for w in r['warnings']], indent=2))
    manifest.update(completed=True, fit_count=sum(r['fit_count'] for r in results),
        prediction_rows=len(pred), evaluation_rows=int(pred.evaluation.sum()),
        warnings=sum(len(r['warnings']) for r in results), elapsed_seconds=round(time.time() - start, 2))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETE', json.dumps({k:manifest[k] for k in ['fit_count','prediction_rows','warnings','elapsed_seconds']}))


if __name__ == '__main__':
    main()
