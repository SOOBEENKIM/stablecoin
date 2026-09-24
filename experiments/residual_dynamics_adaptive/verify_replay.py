"""Replay selected future forecasts and check an end-to-end no-lookahead case."""
import json
import numpy as np
import pandas as pd
from adaptive import HERE, panel, columns, splits, window, DynamicModel, QuantileForest, calibrate, sha256

OUT = HERE / 'results'


def main():
    # Two separated conditional distributions must yield different quantiles,
    # not the unconditional quantile or the mean of tree point predictions.
    x = np.r_[np.zeros(100), np.ones(100)].reshape(-1, 1)
    toy = QuantileForest(q=.1, max_depth=1, min_samples_leaf=20, max_features=1.).fit(x, np.arange(200.))
    np.testing.assert_allclose(toy.predict(np.array([[0.], [1.]])), [9., 109.], rtol=0, atol=1e-12)
    selected = pd.read_csv(OUT / 'selected_candidates.csv', dtype={'window': str})
    pred = pd.read_csv(OUT / 'predictions.csv.gz')
    for col in ['origin', 'target_time', 'latest_calibration_label']:
        pred[col] = pd.to_datetime(pred[col], utc=True)
    cutoff = pd.Timestamp('2026-02-01', tz='UTC')
    records = []
    for mode in ['strict', 'available_macro']:
        p = panel(mode)
        for definition in ['EQ', 'CAP', 'PCA']:
            _, _, train, test, _, _ = splits(p, mode, definition, cutoff)
            for kind in ['linear', 'boosting', 'qrf']:
                s = selected[(selected['mode'] == mode) & (selected.definition == definition) &
                    (selected.fold == '2026-02') & (selected.model == kind) & (selected['info'] == 'history')]
                assert len(s) == 1
                s = s.iloc[0]
                tr = window(train, cutoff, s['window'])
                model = DynamicModel(kind, s['form'], json.loads(s['params'])).fit(tr, columns(mode, 'history'))
                y = model.predict(test)
                old = pred[(pred['mode'] == mode) & (pred.definition == definition) &
                    (pred.fold == '2026-02') & (pred.model == kind) & (pred['info'] == 'history')].sort_values('origin')
                assert list(old.origin) == list(test.index)
                np.testing.assert_allclose(old.pred_raw, y, rtol=0, atol=1e-9)
                records.append(dict(mode=mode, definition=definition, model=kind,
                    n=len(y), max_abs_error_bp=float(np.max(np.abs(old.pred_raw.to_numpy() - y)))))
    correction_errors = []
    for key, g in pred.groupby(['mode', 'definition', 'model', 'info']):
        g = g.sort_values('origin').reset_index(drop=True)
        replay = calibrate(g)
        np.testing.assert_allclose(g.pred_calibrated, replay.pred_calibrated, rtol=0, atol=1e-9)
        correction_errors.append(float(np.max(np.abs(g.pred_calibrated - replay.pred_calibrated))))
        # Alter outcomes unavailable at a middle prediction origin; that
        # origin and every previous corrected forecast must remain unchanged.
        i = len(g) // 2
        changed = g.copy()
        changed.loc[changed.target_time >= g.origin.iloc[i], 'target'] += 1e6
        check = calibrate(changed)
        np.testing.assert_allclose(check.loc[:i, 'pred_calibrated'], replay.loc[:i, 'pred_calibrated'],
                                   rtol=0, atol=1e-9)
    result = dict(replay_month='2026-02', model_definition_sample_replays=len(records),
        replay=records, max_calibration_replay_error_bp=max(correction_errors),
        calibration_streams_checked=len(correction_errors),
        conditional_forest_quantiles_verified=True,
        future_outcome_perturbation_passed=True, script_sha256=sha256(HERE / 'verify_replay.py'))
    (OUT / 'REPLAY_VERIFICATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'replay'}, indent=2))


if __name__ == '__main__':
    main()
