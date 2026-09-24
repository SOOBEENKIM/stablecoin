"""Run the bounded residual-dynamics experiment specified in PROTOCOL_KO.md."""
from pathlib import Path
import argparse
import json
import platform
import subprocess
import time
import warnings
import numpy as np
import pandas as pd
import scipy
import sklearn
from data import (ROOT, SCENARIOS, INPUT_PATHS, BASE, POSITION, Residualizer,
                  build_panel, make_design, feature_columns, sha256)
from models import ForecastModel, GRIDS, SEED, pinball

HERE = Path(__file__).resolve().parent
FOLDS = pd.date_range('2025-12-01', '2026-03-01', freq='MS', tz='UTC')


def jobs():
    out = []
    for definition in ['EQ', 'CAP', 'PCA']:
        for h in [1, 6, 12]:
            out.append(('NY_delay1', definition, 'sequential', h, .1))
    for h in [1, 6, 12]:
        out.append(('NY_delay1', 'EQ', 'sequential', h, .5))
    for scenario in ['NY_delay0', 'FX_UTC_delay1', 'FX_KST_delay1']:
        out.append((scenario, 'EQ', 'sequential', 1, .1))
    out.append(('NY_delay1', 'EQ', 'joint', 1, .1))
    return out


def configurations(job):
    scenario, definition, projection, h, q = job
    kinds = ['linear', 'interaction', 'spline', 'boosting'] if q == .1 else ['linear', 'boosting']
    infos = ['base', 'full']
    if job == ('NY_delay1', 'EQ', 'sequential', 1, .1):
        infos.append('account')
    return [('constant', 'base'), ('persistence', 'base')] + [(k, i) for k in kinds for i in infos]


def prepare_splits(p, definition, projection, h, cutoff):
    vstart = cutoff - pd.offsets.MonthBegin(1)
    r_inner = Residualizer(definition, projection).fit(p, vstart)
    r_outer = Residualizer(definition, projection).fit(p, cutoff)
    inner = make_design(p, r_inner, h)
    outer = make_design(p, r_outer, h)
    train_inner = inner.loc[inner.target_time < vstart]
    valid = inner.loc[(inner.index >= vstart) & (inner.target_time < cutoff)]
    train = outer.loc[outer.target_time < cutoff]
    test = outer.loc[(outer.index >= cutoff) & (outer.index < cutoff + pd.offsets.MonthBegin(1))]
    if len(train_inner) < 80 or len(valid) < 12 or len(train) < 100 or len(test) < 5:
        raise ValueError('Insufficient split counts: ' + str([len(x) for x in [train_inner, valid, train, test]]))
    assert r_inner.fit_end < vstart and r_outer.fit_end < cutoff
    assert train_inner.target_time.max() < valid.index.min()
    assert train.target_time.max() < test.index.min()
    assert valid.target_time.max() < cutoff
    for d in [train_inner, valid, train, test]:
        assert ((d.target_time - d.index) == pd.Timedelta(hours=h)).all()
    audit = dict(validation_start=vstart.isoformat(), outer_cutoff=cutoff.isoformat(),
        inner_fit=r_inner.metadata(), outer_fit=r_outer.metadata(),
        inner_train_n=len(train_inner), validation_n=len(valid), train_n=len(train), test_n=len(test),
        train_label_max=train.target_time.max().isoformat(), test_origin_min=test.index.min().isoformat(),
        inner_label_max=train_inner.target_time.max().isoformat(), validation_origin_min=valid.index.min().isoformat())
    return train_inner, valid, train, test, audit


def run(out, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = dict(seed=SEED, python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        protocol_sha256=sha256(HERE / 'PROTOCOL_KO.md'),
        source_sha256={p.name: sha256(p) for p in HERE.glob('*.py')},
        inputs={str(p.relative_to(ROOT)): sha256(p) for p in INPUT_PATHS},
        folds=[x.isoformat() for x in FOLDS], grids=GRIDS, smoke=smoke,
        inference='historical chronological evaluation; fixed-prediction block uncertainty')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    panels = {s: build_panel(s) for s in SCENARIOS}
    audit_rows = []
    for scenario, p in panels.items():
        # Temporary early fit is only to assess the complete-case availability.
        r = Residualizer().fit(p, pd.Timestamp('2025-11-01', tz='UTC'))
        for h in [1, 6, 12]:
            d = make_design(p, r, h)
            for month, g in d.groupby(d.index.strftime('%Y-%m')):
                audit_rows.append(dict(scenario=scenario, h=h, month=month,
                    n=len(g), days=g.index.normalize().nunique()))
    pd.DataFrame(audit_rows).to_csv(out / 'sample_counts.csv', index=False)
    first_stages, selections, predictions, failures, warning_rows = [], [], [], [], []
    work = jobs()[:1] if smoke else jobs()
    for job in work:
        scenario, definition, projection, h, q = job
        p = panels[scenario]
        for cutoff in (FOLDS[:1] if smoke else FOLDS):
            key = dict(scenario=scenario, definition=definition, projection=projection,
                       h=h, q=q, fold=cutoff.strftime('%Y-%m'))
            try:
                ti, va, tr, te, audit = prepare_splits(p, definition, projection, h, cutoff)
            except Exception as exc:
                failures.append(dict(**key, stage='splits', error=repr(exc)))
                print('SPLIT FAIL', key, repr(exc), flush=True)
                continue
            first_stages.append(dict(**key, **audit))
            dn = tr.downside24.quantile(.75)
            ls = tr.account_log.quantile(.75)
            for kind, info in configurations(job):
                cols = feature_columns(info)
                candidates = []
                try:
                    for params in GRIDS[kind]:
                        with warnings.catch_warnings(record=True) as captured:
                            warnings.simplefilter('always')
                            model = ForecastModel(kind, q, params).fit(ti[cols], ti.target)
                            score = float(pinball(va.target, model.predict(va[cols]), q).mean())
                            for w in captured:
                                warning_rows.append(dict(**key, model=kind, info=info, message=str(w.message)))
                        candidates.append((score, params))
                        selections.append(dict(**key, model=kind, info=info,
                            params=json.dumps(params, sort_keys=True), validation_pinball_bp=score))
                    _, params = min(candidates, key=lambda x: x[0])
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always')
                        fitted = ForecastModel(kind, q, params).fit(tr[cols], tr.target)
                        pred = fitted.predict(te[cols])
                        for w in captured:
                            warning_rows.append(dict(**key, model=kind, info=info, message=str(w.message)))
                    if not np.isfinite(pred).all():
                        raise ValueError('Nonfinite predictions')
                    saved = te[['target_time', 'target', 'e_now', 'delta_e', 'downside24',
                        'btc_vol24', 'btc_ret1', 'account_log', 'funding_bp', 'oi_ret1',
                        'local_volume_surprise', 'delta_local_bp', 'delta_quote_bp',
                        'delta_fx_bp', 'delta_market_bp', 'delta_g_bp', 'delta_log_volume']].copy()
                    saved['pred'] = pred
                    saved['loss'] = pinball(te.target, pred, q)
                    saved['below'] = (te.target.to_numpy() < pred).astype(int)
                    saved['high_downside'] = (te.downside24 > dn).astype(int)
                    saved['high_account'] = (te.account_log > ls).astype(int)
                    saved['downside_cutoff'] = dn
                    saved['account_cutoff'] = ls
                    saved['model'], saved['info'] = kind, info
                    saved['params'] = json.dumps(params, sort_keys=True)
                    for k, v in key.items():
                        saved[k] = v
                    predictions.append(saved.reset_index())
                except Exception as exc:
                    failures.append(dict(**key, model=kind, info=info, stage='models', error=repr(exc)))
                    print('MODEL FAIL', key, kind, info, repr(exc), flush=True)
            print('DONE', scenario, definition, projection, 'h', h, 'q', q,
                  cutoff.strftime('%Y-%m'), 'n', len(tr), len(te),
                  'seconds', round(time.time() - started, 1), flush=True)
            # Checkpoint every fold so interrupted runs remain inspectable.
            (out / 'first_stages.json').write_text(json.dumps(first_stages, indent=2), encoding='utf-8')
            (out / 'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(out / 'predictions.csv.gz', index=False,
                compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(selections).to_csv(out / 'validation_candidates.csv', index=False)
    (out / 'warnings.json').write_text(json.dumps(warning_rows, indent=2), encoding='utf-8')
    manifest.update(elapsed_seconds=round(time.time() - started, 2),
                    prediction_rows=len(pred), failure_count=len(failures),
                    warning_count=len(warning_rows), completed=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError('Incomplete experiment; see failures.json')
    print('COMPLETE', len(pred), 'predictions', manifest['elapsed_seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE / 'results')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.out, args.smoke)
