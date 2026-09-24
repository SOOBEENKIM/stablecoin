"""Bounded seed diagnostic with previously selected hyperparameters frozen."""
import json
import numpy as np
import pandas as pd
import adaptive as a

OUT = a.HERE / 'results'


def main():
    choices = pd.read_csv(OUT / 'selected_candidates.csv', dtype={'window': str})
    choices = choices[(choices['mode'] == 'available_macro') & (choices.definition == 'EQ') & (choices.model == 'qrf')]
    original = pd.read_csv(OUT / 'predictions.csv.gz')
    for col in ['origin', 'target_time']:
        original[col] = pd.to_datetime(original[col], utc=True)
    original = original[(original['mode'] == 'available_macro') & (original.definition == 'EQ')]
    linear = original[original.evaluation & (original.model == 'linear') & (original['info'] == 'history')].sort_values('origin')
    first = original[original.model == 'qrf'].copy()
    first['seed'] = a.SEED
    frames = [first]
    p = a.panel('available_macro')
    for seed in [20260926, 20260927, 20260928]:
        a.SEED = seed
        rows = []
        for cutoff in a.FOLDS:
            _, _, train, test, _, _ = a.splits(p, 'available_macro', 'EQ', cutoff)
            for info in ['base', 'full', 'history']:
                s = choices[(choices.fold == cutoff.strftime('%Y-%m')) & (choices['info'] == info)].iloc[0]
                m = a.DynamicModel('qrf', s['form'], json.loads(s['params']))
                m.fit(a.window(train, cutoff, s['window']), a.columns('available_macro', info))
                d = test[['target_time', 'target']].reset_index()
                d['pred_raw'] = m.predict(test)
                d['info'], d['fold'], d['seed'] = info, cutoff.strftime('%Y-%m'), seed
                rows.append(d)
        raw = pd.concat(rows, ignore_index=True)
        for info, d in raw.groupby('info'):
            d = a.calibrate(d)
            d['evaluation'] = d.origin >= a.EVALUATION_START
            frames.append(d)
        print('DONE seed', seed, flush=True)
    all_pred = pd.concat(frames, ignore_index=True)
    pred = all_pred[all_pred.evaluation].copy()
    metrics, comparisons = [], []
    for seed, g in pred.groupby('seed'):
        maps = {}
        for info, s in g.groupby('info'):
            s = s.sort_values('origin')
            assert list(s.origin) == list(linear.origin)
            np.testing.assert_allclose(s.target, linear.target, rtol=0, atol=1e-10)
            for version in ['raw', 'calibrated']:
                loss = a.pinball(s.target, s['pred_' + version], .1)
                maps[info, version] = loss.mean()
                metrics.append(dict(seed=seed, info=info, calibration=version, n=len(s),
                    loss_bp=loss.mean(), below_rate=(s.target < s['pred_' + version]).mean()))
        for version in ['raw', 'calibrated']:
            lin = a.pinball(linear.target, linear['pred_' + version], .1).mean()
            for label, ref in [('vs_linear_history', lin), ('vs_qrf_base', maps['base', version]),
                               ('vs_qrf_full', maps['full', version])]:
                comparisons.append(dict(seed=seed, calibration=version, comparison=label,
                    improvement_pct=100 * (1 - maps['history', version] / ref)))
    fields = ['origin', 'target_time', 'target', 'pred_raw', 'pred_calibrated', 'calibration_n',
              'latest_calibration_label', 'seed', 'info', 'fold', 'evaluation']
    all_pred[fields].to_csv(OUT / 'seed_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(metrics).to_csv(OUT / 'seed_metrics.csv', index=False)
    c = pd.DataFrame(comparisons)
    c.to_csv(OUT / 'seed_comparisons.csv', index=False)
    (OUT / 'seed_manifest.json').write_text(json.dumps(dict(seeds=[20260925,20260926,20260927,20260928],
        additional_fits=45, selected_hyperparameters_frozen=True, seed_selection=False,
        protocol_sha256=a.sha256(a.HERE / 'SEED_CHECK_KO.md'),
        source_sha256=a.sha256(a.HERE / 'check_seeds.py'),
        choices_sha256=a.sha256(OUT / 'selected_candidates.csv')), indent=2))
    print(c[c.calibration == 'calibrated'].to_string(index=False))


if __name__ == '__main__':
    main()
