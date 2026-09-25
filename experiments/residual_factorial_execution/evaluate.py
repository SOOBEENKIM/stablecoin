"""Open verified, sealed forecasts and report every planned comparison."""
import argparse
import json
import numpy as np
import pandas as pd

from core import OUT, DESIGN, read_predictions, loss, comparison_specs
from execution_guard import verify_seal, write_new, now, sha
from factorial_stats import day_weights, bootstrap_mean, effect, holm


def main(reproduce=False):
    stamp, seal = verify_seal()
    verification = json.loads((OUT / 'VERIFICATION.json').read_text())
    assert verification['predictions_sha256'] == sha(OUT / 'predictions.csv.gz')
    marker = OUT / 'SCORES_OPENED.json'
    if marker.exists():
        if not reproduce:
            raise RuntimeError('Already opened; --reproduce must use the identical sealed forecasts')
        assert json.loads(marker.read_text())['predictions_sha256'] == verification['predictions_sha256']
    else:
        write_new(marker, dict(**stamp, opened_utc=now(),
            predictions_sha256=verification['predictions_sha256'], independent_confirmation=False))
    pred = read_predictions(OUT / 'predictions.csv.gz')
    metrics, monthly, comparisons, seed_rows, monthly_effects = [], [], [], [], []
    for (mode, definition), case in pred.groupby(['mode', 'definition'], sort=True):
        reference_frame = case[['origin', 'fold']].drop_duplicates().sort_values('origin').reset_index(drop=True)
        weights = {b: day_weights(reference_frame, b) for b in [1, 5, 10]}
        for version in ['raw', 'calibrated']:
            streams, maps = {}, {}
            for (info, strategy), g in case.groupby(['information', 'strategy'], sort=True):
                g = g.copy()
                g['loss'] = loss(g.target, g['pred_' + version])
                g['below'] = (g.target < g['pred_' + version]).astype(float)
                d = g.groupby('origin', as_index=False).agg(loss=('loss', 'mean'), below=('below', 'mean'),
                    fold=('fold', 'first'), target=('target', 'first'), seeds=('seed', 'nunique'),
                    abs_correction=('correction', lambda x: np.abs(x).mean()))
                assert d.origin.equals(reference_frame.origin) and d.seeds.nunique() == 1
                maps[info, strategy], streams[info, strategy] = d, g
                key = dict(mode=mode, definition=definition, information=info, strategy=strategy, calibration=version)
                lo, hi = np.quantile(bootstrap_mean(weights[5], d.below), [.025, .975])
                metrics.append(dict(**key, n=len(d), days=d.origin.dt.normalize().nunique(), seeds=int(d.seeds.iloc[0]),
                    loss_bp=d.loss.mean(), below_rate=d.below.mean(), below_lo=lo, below_hi=hi,
                    mean_absolute_correction_bp=d.abs_correction.mean()))
                for fold, group in d.groupby('fold'):
                    monthly.append(dict(**key, fold=fold, n=len(group), loss_bp=group.loss.mean(), below_rate=group.below.mean()))
            for spec in comparison_specs():
                ref, new = maps[tuple(spec['reference'])], maps[tuple(spec['candidate'])]
                np.testing.assert_array_equal(ref.target, new.target)
                key = dict(mode=mode, definition=definition, comparison=spec['id'],
                    primary_comparison=spec['primary'], calibration=version, n=len(ref),
                    reference_information=spec['reference'][0], reference_strategy=spec['reference'][1],
                    candidate_information=spec['candidate'][0], candidate_strategy=spec['candidate'][1])
                month_ref, month_new = ref.groupby('fold').loss.mean(), new.groupby('fold').loss.mean()
                for b, w in weights.items():
                    comparisons.append(dict(**key, block_days=b, **effect(ref.loss, new.loss, w),
                        winning_months=int((month_new < month_ref).sum())))
                for fold in month_ref.index:
                    monthly_effects.append(dict(**key, fold=fold,
                        difference_bp=month_ref[fold]-month_new[fold],
                        improvement_pct=100*(1-month_new[fold]/month_ref[fold])))
                for seed, g in streams[tuple(spec['reference'])].groupby('seed'):
                    a = g.sort_values('origin')
                    b = streams[tuple(spec['candidate'])]
                    b = b[b.seed == seed].sort_values('origin')
                    assert list(a.origin) == list(b.origin)
                    seed_rows.append(dict(**key, seed=int(seed), difference_bp=a.loss.mean()-b.loss.mean(),
                        improvement_pct=100*(1-b.loss.mean()/a.loss.mean())))
    comparisons = pd.DataFrame(comparisons)
    primary = comparisons[(comparisons['mode'] == DESIGN['primary_case'][0]) &
        (comparisons.definition == DESIGN['primary_case'][1]) & comparisons.primary_comparison &
        (comparisons.calibration == 'calibrated') & (comparisons.block_days == 5)].copy()
    assert len(primary) == 6
    primary['p_holm'] = holm(primary.p_centered_boot)
    primary['supported'] = (primary.improvement_pct > 0) & (primary.improvement_lo > 0) & (primary.p_holm < .05)
    for name, frame in [('metrics', pd.DataFrame(metrics)), ('monthly_metrics', pd.DataFrame(monthly)),
        ('comparisons', comparisons), ('primary_tests', primary), ('seed_comparisons', pd.DataFrame(seed_rows)),
        ('monthly_effects', pd.DataFrame(monthly_effects))]:
        frame.to_csv(OUT / (name + '.csv'), index=False)
    selections = json.loads((OUT / 'selections.json').read_text())
    pd.DataFrame(selections).to_csv(OUT / 'algorithm_choices.csv', index=False)
    if not reproduce:
        write_new(OUT / 'EVALUATION_COMPLETE.json', dict(**stamp, completed_utc=now(),
            independent_confirmation=False, primary_tests=primary.to_dict('records')))
    print(primary[['comparison', 'improvement_pct', 'improvement_lo', 'improvement_hi', 'p_holm', 'supported']].to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reproduce', action='store_true')
    main(p.parse_args().reproduce)
