"""Execute the fixed post-result design; preserve every candidate and forecast."""
import json
import platform
import subprocess
import time
import warnings
import numpy as np
import pandas as pd
import sklearn
from adaptive import (HERE, LEGACY, ROOT, INPUT_PATHS, BASE, POSITION, HISTORY,
    FOLDS, EVALUATION_START, Residualizer, panel, frame, columns, splits, window,
    candidates, DynamicModel, calibrate, sha256, pinball)

OUT = HERE / 'results'


def diagnostics(panels):
    rows, counts, common, paths = [], [], [], []
    for mode, p in panels.items():
        frozen = Residualizer().fit(p, EVALUATION_START)
        frozen_frame = frame(p, frozen, mode)
        for cutoff in FOLDS[1:]:
            _, _, tr, te, _, r = splits(p, mode, 'EQ', cutoff)
            recent = window(tr, cutoff, 90)
            frozen_test = frozen_frame.loc[te.index]
            rows.append(dict(mode=mode, fold=cutoff.strftime('%Y-%m'),
                train_n=len(tr), recent90_n=len(recent), test_n=len(te),
                train_q10=tr.target.quantile(.1), recent90_q10=recent.target.quantile(.1),
                test_q10=te.target.quantile(.1), frozen_definition_test_q10=frozen_test.target.quantile(.1),
                train_mean=tr.target.mean(), test_mean=te.target.mean(),
                train_delta_q10=tr.delta_e.quantile(.1), test_delta_q10=te.delta_e.quantile(.1)))
        for definition in ['EQ', 'CAP', 'PCA']:
            for cutoff in FOLDS[1:]:
                _, _, train, _, _, r = splits(p, mode, definition, cutoff)
                dn, ls = train.downside24.quantile(.75), train.account_log.quantile(.75)
                ds = {}
                for h in [1, 6, 12]:
                    d = frame(p, r, mode, h)
                    d = d.loc[(d.index >= cutoff) & (d.index < cutoff + pd.offsets.MonthBegin(1))]
                    ds[h] = d
                    counts.append(dict(mode=mode, definition=definition, fold=cutoff.strftime('%Y-%m'),
                        h=h, n=len(d), days=d.index.normalize().nunique()))
                ix = ds[1].index.intersection(ds[6].index).intersection(ds[12].index)
                common.append(dict(mode=mode, definition=definition, fold=cutoff.strftime('%Y-%m'),
                                   common_n=len(ix), days=ix.normalize().nunique()))
                for h, d in ds.items():
                    s = d.loc[ix].copy()
                    s['high_downside'] = (s.downside24 > dn).astype(int)
                    s['high_account'] = (s.account_log > ls).astype(int)
                    s['h'], s['mode'], s['definition'], s['fold'] = h, mode, definition, cutoff.strftime('%Y-%m')
                    paths.append(s.reset_index())
    pd.DataFrame(rows).to_csv(OUT / 'distribution_diagnostics.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'sample_counts.csv', index=False)
    pd.DataFrame(common).to_csv(OUT / 'common_horizon_counts.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(OUT / 'common_origin_paths.csv.gz', index=False,
                                             compression={'method': 'gzip', 'mtime': 0})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = dict(parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        exploratory_post_result=True, python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, sklearn=sklearn.__version__,
        protocol_sha256=sha256(HERE / 'PROTOCOL_KO.md'),
        source_sha256={p.name: sha256(p) for p in HERE.glob('*.py')},
        inherited_source_sha256={p.name: sha256(p) for p in [LEGACY / 'data.py', LEGACY / 'models.py']},
        inputs={str(p.relative_to(ROOT)): sha256(p) for p in INPUT_PATHS},
        candidates={k: candidates(k) for k in ['linear', 'boosting', 'qrf']})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    panels = {m: panel(m) for m in ['strict', 'available_macro']}
    # Verify the expanded sample changes inputs, not price/FX observations or targets.
    for col in ['fx', 'q', 'y', 'g', 'm_EQ', 'm_CAP']:
        np.testing.assert_allclose(panels['strict'][col], panels['available_macro'][col],
                                   equal_nan=True, rtol=0, atol=0)
    diagnostics(panels)
    predictions, choices, selected, audits, warning_rows = [], [], [], [], []
    fit_count = 0
    for mode, p in panels.items():
        for definition in ['EQ', 'CAP', 'PCA']:
            for cutoff in FOLDS:
                vstart = cutoff - pd.offsets.MonthBegin(1)
                ti, va, tr, te, ri, ro = splits(p, mode, definition, cutoff)
                key = dict(mode=mode, definition=definition, fold=cutoff.strftime('%Y-%m'))
                audits.append(dict(**key, validation_start=vstart.isoformat(), cutoff=cutoff.isoformat(),
                    inner=ri.metadata(), outer=ro.metadata(),
                    inner_max_label=ti.target_time.max().isoformat(),
                    max_train_label=tr.target_time.max().isoformat(),
                    first_test_origin=te.index.min().isoformat(), train_n=len(tr), test_n=len(te)))
                for info in ['base', 'full', 'history']:
                    cols = columns(mode, info)
                    for kind in ['linear', 'boosting', 'qrf']:
                        scores = []
                        for days, form, params in candidates(kind):
                            train = window(ti, vstart, days)
                            assert len(train) >= 100
                            with warnings.catch_warnings(record=True) as captured:
                                warnings.simplefilter('always')
                                fitted = DynamicModel(kind, form, params).fit(train, cols)
                                pv = fitted.predict(va)
                            score = float(pinball(va.target, pv, .1).mean())
                            assert np.isfinite(score)
                            scores.append((score, days, form, params))
                            choices.append(dict(**key, info=info, model=kind, window=str(days),
                                form=form, params=json.dumps(params, sort_keys=True), validation_loss_bp=score,
                                train_n=len(train), validation_n=len(va)))
                            for w in captured:
                                warning_rows.append(dict(**key, model=kind, info=info, message=str(w.message)))
                            fit_count += 1
                        loss, days, form, params = min(scores, key=lambda a: a[0])
                        train = window(tr, cutoff, days)
                        fitted = DynamicModel(kind, form, params).fit(train, cols)
                        pred = fitted.predict(te)
                        assert np.isfinite(pred).all()
                        fit_count += 1
                        selected.append(dict(**key, info=info, model=kind, window=str(days), form=form,
                            params=json.dumps(params, sort_keys=True), validation_loss_bp=loss,
                            train_n=len(train), test_n=len(te)))
                        d = te[['target_time', 'target', 'e_now', 'delta_e', 'downside24', 'btc_vol24',
                            'account_log', 'funding_bp', 'oi_ret1'] + HISTORY].copy()
                        for k, v in key.items():
                            d[k] = v
                        d['model'], d['info'], d['window'], d['form'] = kind, info, str(days), form
                        d['pred_raw'] = pred
                        predictions.append(d.reset_index())
                print('DONE', mode, definition, cutoff.strftime('%Y-%m'),
                      'train/test', len(tr), len(te), 'seconds', round(time.time() - started, 1), flush=True)
                # Resumable audit evidence without choosing models using test scores.
                (OUT / 'split_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
                pd.DataFrame(selected).to_csv(OUT / 'selected_candidates.csv', index=False)
                pd.DataFrame(choices).to_csv(OUT / 'validation_candidates.csv', index=False)
    raw = pd.concat(predictions, ignore_index=True)
    combined = []
    for _, g in raw.groupby(['mode', 'definition', 'model', 'info'], sort=True):
        combined.append(calibrate(g))
    pred = pd.concat(combined, ignore_index=True)
    for version in ['raw', 'calibrated']:
        pred['loss_' + version] = pinball(pred.target, pred['pred_' + version], .1)
        pred['below_' + version] = (pred.target < pred['pred_' + version]).astype(int)
    pred['evaluation'] = pred.origin >= EVALUATION_START
    assert (pred.loc[pred.evaluation, 'calibration_n'] >= 30).all()
    pred.to_csv(OUT / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    (OUT / 'warnings.json').write_text(json.dumps(warning_rows, indent=2), encoding='utf-8')
    manifest.update(completed=True, fit_count=fit_count, forecast_rows=len(pred),
        evaluated_forecast_rows=int(pred.evaluation.sum()), warnings=len(warning_rows),
        elapsed_seconds=round(time.time() - started, 2))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('COMPLETE', json.dumps({k: manifest[k] for k in ['fit_count', 'forecast_rows',
           'evaluated_forecast_rows', 'warnings', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
