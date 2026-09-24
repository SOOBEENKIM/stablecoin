"""Report all post-result experiments, including unsuccessful comparisons."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptive import HERE, LEGACY, ROOT, sha256, panel, splits, calibrate, pinball
from analyze import day_weights, boot_mean, interval, holm

OUT = HERE / 'results'


def unchanged_original_calibration(original):
    """Controlled supplemental comparison: calibrate the original forecasts.

    December is warm-up because the original run had no November forecasts.
    This cohort is therefore January--March, not the original headline cohort.
    """
    rows = []
    for (definition, model, info), g in original.groupby(['definition', 'model', 'info']):
        if model not in ['linear', 'boosting'] or info not in ['base', 'full']:
            continue
        g = g.rename(columns={'pred': 'pred_raw'}).copy()
        g.target_time = pd.to_datetime(g.target_time, utc=True)
        g = calibrate(g)
        g = g[g.origin >= pd.Timestamp('2026-01-01', tz='UTC')]
        assert (g.calibration_n >= 30).all()
        for version in ['raw', 'calibrated']:
            forecast = g['pred_' + version]
            rows.append(dict(definition=definition, model=model, info=info, calibration=version,
                cohort='2026-01_to_2026-03', n=len(g),
                loss_bp=pinball(g.target, forecast, .1).mean(), below_rate=(g.target < forecast).mean()))
    pd.DataFrame(rows).to_csv(OUT / 'unchanged_original_calibration.csv', index=False)


def compare(a, b, weights, key, label):
    assert a.origin.equals(b.origin)
    np.testing.assert_allclose(a.target, b.target, atol=1e-10, rtol=0)
    diff = a.loss.to_numpy() - b.loss.to_numpy()
    value = diff.mean()
    draws = boot_mean(weights, diff)
    lo, hi = interval(draws)
    plo, phi = interval(-100 * draws / boot_mean(weights, b.loss))
    p = (1 + np.count_nonzero(np.abs(draws - value) >= abs(value))) / (len(draws) + 1)
    monthly = a.groupby('fold').loss.mean() < b.groupby('fold').loss.mean()
    return dict(**key, comparison=label, n=len(a), difference_bp=value,
        difference_lo=lo, difference_hi=hi,
        improvement_pct=100 * (1 - a.loss.mean() / b.loss.mean()),
        improvement_lo=plo, improvement_hi=phi, p_centered_boot=p,
        winning_months=int(monthly.sum()), months=len(monthly))


def model_results(pred, original):
    metrics, monthly, comparisons, sessions, legacy = [], [], [], [], []
    for (mode, definition), sub in pred.groupby(['mode', 'definition']):
        ref = sub[(sub.model == 'linear') & (sub['info'] == 'base')].sort_values('origin').reset_index(drop=True)
        weights = {k: day_weights(ref, k) for k in [1, 5, 10]}
        maps = {}
        old = original[(original.definition == definition) & (original.model == 'linear') &
                       (original['info'] == 'full')].sort_values('origin').reset_index(drop=True)
        for (model, info), raw in sub.groupby(['model', 'info']):
            raw = raw.sort_values('origin').reset_index(drop=True)
            assert raw.origin.equals(ref.origin)
            np.testing.assert_allclose(raw.target, ref.target, rtol=0, atol=1e-10)
            for version in ['raw', 'calibrated']:
                g = raw.copy()
                g['pred'], g['loss'], g['below'] = (g['pred_' + version], g['loss_' + version], g['below_' + version])
                maps[(model, info, version)] = g
                low, high = interval(boot_mean(weights[5], g.below))
                key = dict(mode=mode, definition=definition, model=model, info=info, calibration=version)
                metrics.append(dict(**key, n=len(g), days=g.origin.dt.normalize().nunique(),
                    loss_bp=g.loss.mean(), below_rate=g.below.mean(), below_lo=low, below_hi=high))
                for fold, s in g.groupby('fold'):
                    monthly.append(dict(**key, fold=fold, n=len(s), loss_bp=s.loss.mean(), below_rate=s.below.mean()))
                for name, s in [('original_origins', g[g.origin.isin(old.origin)]),
                                ('additional_origins', g[~g.origin.isin(old.origin)])]:
                    if len(s):
                        sessions.append(dict(**key, subset=name, n=len(s), loss_bp=s.loss.mean(),
                                             below_rate=s.below.mean()))
                if info == 'full':
                    s = g[g.origin.isin(old.origin)].reset_index(drop=True)
                    assert s.origin.equals(old.origin)
                    legacy.append(compare(s, old, day_weights(s, 5), key, 'vs_original_linear_full'))

        pairs = []
        for version in ['raw', 'calibrated']:
            for info in ['base', 'full', 'history']:
                for model in ['boosting', 'qrf']:
                    pairs.append(((model, info, version), ('linear', info, version), 'algorithm_vs_linear'))
            for model in ['linear', 'boosting', 'qrf']:
                pairs += [((model, 'full', version), (model, 'base', version), 'position_levels'),
                          ((model, 'history', version), (model, 'full', version), 'position_changes'),
                          ((model, 'history', version), (model, 'base', version), 'position_all')]
        for model in ['linear', 'boosting', 'qrf']:
            for info in ['base', 'full', 'history']:
                pairs.append(((model, info, 'calibrated'), (model, info, 'raw'), 'online_correction'))
        for a_key, b_key, label in pairs:
            for length in [1, 5, 10]:
                key = dict(mode=mode, definition=definition, model=a_key[0], info=a_key[1],
                    calibration=a_key[2], reference_model=b_key[0], reference_info=b_key[1],
                    reference_calibration=b_key[2], block_days=length)
                comparisons.append(compare(maps[a_key], maps[b_key], weights[length], key, label))
    metrics, monthly, comparisons = map(pd.DataFrame, [metrics, monthly, comparisons])
    main = comparisons[(comparisons['mode'] == 'available_macro') & (comparisons.definition == 'EQ') &
        (comparisons.block_days == 5) & (comparisons.calibration == 'calibrated') &
        (comparisons.model.isin(['boosting', 'qrf'])) & (comparisons['info'] == 'history') &
        comparisons.comparison.isin(['algorithm_vs_linear', 'position_all', 'position_changes'])].copy()
    assert len(main) == 6
    main['p_holm_exploratory'] = holm(main.p_centered_boot)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    monthly.to_csv(OUT / 'monthly_metrics.csv', index=False)
    comparisons.to_csv(OUT / 'comparisons.csv', index=False)
    main.to_csv(OUT / 'primary_exploratory_tests.csv', index=False)
    pd.DataFrame(sessions).to_csv(OUT / 'origin_subset_metrics.csv', index=False)
    pd.DataFrame(legacy).to_csv(OUT / 'original_forecast_comparison.csv', index=False)
    return metrics, monthly, comparisons, main


def adjustment_evidence(pred):
    paths = pd.read_csv(OUT / 'common_origin_paths.csv.gz')
    paths.origin = pd.to_datetime(paths.origin, utc=True)
    states = []
    for key, g in paths.groupby(['mode', 'definition', 'h', 'high_downside', 'high_account']):
        row = dict(zip(['mode', 'definition', 'h', 'high_downside', 'high_account'], key))
        neg = g[g.e_now < 0]
        row.update(n=len(g), days=g.origin.dt.normalize().nunique(),
            mean_start_bp=g.e_now.mean(), future_q10_bp=g.target.quantile(.1),
            mean_future_bp=g.target.mean(), median_delta_bp=g.delta_e.median(),
            mean_delta_bp=g.delta_e.mean(), negative_start_n=len(neg),
            remains_negative_rate=(neg.target < 0).mean() if len(neg) else np.nan)
        for col in ['delta_local_bp', 'delta_quote_bp', 'delta_fx_bp', 'delta_market_bp', 'delta_g_bp']:
            row[col] = g[col].mean()
        states.append(row)
    pd.DataFrame(states).to_csv(OUT / 'matched_state_adjustment.csv', index=False)
    # A fixed, economically interpretable sign split for the added history
    # feature, reported alongside the original level-based state split.
    paths['account_increasing24'] = (paths.account_change24 > 0).astype(int)
    history_states = []
    for key, g in paths.groupby(['mode', 'definition', 'h', 'high_downside', 'account_increasing24']):
        row = dict(zip(['mode', 'definition', 'h', 'high_downside', 'account_increasing24'], key))
        row.update(n=len(g), mean_start_bp=g.e_now.mean(),
            future_q10_bp=g.target.quantile(.1), mean_delta_bp=g.delta_e.mean(),
            median_delta_bp=g.delta_e.median())
        history_states.append(row)
    pd.DataFrame(history_states).to_csv(OUT / 'matched_position_change_states.csv', index=False)
    # Conditional risk calibration on the same origins available at all horizons.
    info = paths[paths.h == 1][['mode', 'definition', 'origin', 'high_downside', 'high_account']]
    d = pred.merge(info, on=['mode', 'definition', 'origin'], how='inner', validate='many_to_one')
    rows = []
    for key, g in d.groupby(['mode', 'definition', 'model', 'info', 'high_downside', 'high_account']):
        row = dict(zip(['mode', 'definition', 'model', 'info', 'high_downside', 'high_account'], key))
        row.update(n=len(g), raw_loss_bp=g.loss_raw.mean(), corrected_loss_bp=g.loss_calibrated.mean(),
                   raw_below_rate=g.below_raw.mean(), corrected_below_rate=g.below_calibrated.mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'common_origin_state_calibration.csv', index=False)
    return paths


def state_forecast_evidence(pred):
    """Describe original positioning/downside states on the full 1h sample.

    State thresholds use the preceding training sample. These are descriptive
    diagnostics, not additional confirmatory tests or causal effects.
    """
    groups, thresholds = [], []
    for mode, p in [(mode, panel(mode)) for mode in ['strict', 'available_macro']]:
        for definition in ['EQ', 'CAP', 'PCA']:
            for fold in sorted(pred.fold.unique()):
                cutoff = pd.Timestamp(fold + '-01', tz='UTC')
                _, _, train, _, _, _ = splits(p, mode, definition, cutoff)
                down, account = train.downside24.quantile(.75), train.account_log.quantile(.75)
                thresholds.append(dict(mode=mode, definition=definition, fold=fold,
                    downside_p75=down, account_log_p75=account, train_n=len(train)))
                g = pred[(pred['mode'] == mode) & (pred.definition == definition) & (pred.fold == fold)].copy()
                g['high_downside'] = (g.downside24 > down).astype(int)
                g['high_account'] = (g.account_log > account).astype(int)
                groups.append(g)
    d = pd.concat(groups, ignore_index=True)
    keys = ['mode', 'definition', 'model', 'info', 'high_downside', 'high_account']
    rows = []
    for key, g in d.groupby(keys):
        for version in ['raw', 'calibrated']:
            rows.append(dict(**dict(zip(keys, key)), calibration=version, n=len(g),
                days=g.origin.dt.normalize().nunique(), mean_start_bp=g.e_now.mean(),
                realized_q10_bp=g.target.quantile(.1), mean_predicted_q10_bp=g['pred_' + version].mean(),
                loss_bp=g['loss_' + version].mean(), below_rate=g['below_' + version].mean()))
    gains = []
    for key, g in d.groupby([k for k in keys if k != 'info']):
        ref = g[g['info'] == 'base'].sort_values('origin')
        for info in ['full', 'history']:
            s = g[g['info'] == info].sort_values('origin')
            assert list(s.origin) == list(ref.origin)
            for version in ['raw', 'calibrated']:
                gains.append(dict(**dict(zip([k for k in keys if k != 'info'], key)),
                    info=info, calibration=version, n=len(s),
                    loss_difference_bp=s['loss_' + version].mean() - ref['loss_' + version].mean(),
                    improvement_pct=100 * (1 - s['loss_' + version].mean() / ref['loss_' + version].mean()),
                    mean_boundary_difference_bp=(s['pred_' + version].to_numpy() - ref['pred_' + version].to_numpy()).mean()))
    pd.DataFrame(rows).to_csv(OUT / 'state_forecast_metrics.csv', index=False)
    pd.DataFrame(gains).to_csv(OUT / 'state_information_gain.csv', index=False)
    pd.DataFrame(thresholds).to_csv(OUT / 'state_thresholds.csv', index=False)


def plot(metrics):
    d = metrics[(metrics['mode'] == 'available_macro') & (metrics.definition == 'EQ') &
                (metrics['info'] == 'history')]
    models = ['linear', 'boosting', 'qrf']
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for j, version in enumerate(['raw', 'calibrated']):
        s = d[d.calibration == version].set_index('model').loc[models]
        axes[0].bar(np.arange(3) + (j - .5) * .36, s.loss_bp, width=.36, label=version)
        axes[1].plot(models, s.below_rate, marker='o', label=version)
    axes[0].set_xticks(range(3), models)
    axes[0].set_ylabel('q10 pinball loss (bp), lower is better')
    axes[1].axhline(.1, ls='--', c='black', label='Nominal 10%')
    axes[1].set_ylabel('Fraction below forecast q10')
    for a in axes:
        a.legend()
    fig.suptitle('Available macro inputs; EQ residual; actual 1h; position history')
    fig.savefig(OUT / 'adaptive_models.png', dpi=180)
    plt.close(fig)
    shifts = pd.read_csv(OUT / 'distribution_diagnostics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mode in zip(axes, ['strict', 'available_macro']):
        s = shifts[shifts['mode'] == mode]
        for col, label in [('train_q10', 'Past training q10'), ('test_q10', 'Realized month q10'),
                           ('frozen_definition_test_q10', 'Realized, fixed Dec residual rule')]:
            ax.plot(s.fold, s[col], marker='o', label=label)
        ax.set_title(mode)
        ax.set_ylabel('Residual q10 (bp)')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(fontsize=8)
    fig.savefig(OUT / 'distribution_diagnostics.png', dpi=180)
    plt.close(fig)


def verification(all_pred, pred, paths):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert manifest['completed']
    for name, digest in manifest['source_sha256'].items():
        assert sha256(HERE / name) == digest, name
    for name, digest in manifest['inherited_source_sha256'].items():
        assert sha256(LEGACY / name) == digest, name
    for name, digest in manifest['inputs'].items():
        assert sha256(ROOT / name) == digest, name
    assert sha256(HERE / 'PROTOCOL_KO.md') == manifest['protocol_sha256']
    assert not all_pred.duplicated(['mode', 'definition', 'model', 'info', 'origin']).any()
    assert np.isfinite(pred[['pred_raw', 'pred_calibrated', 'target']]).all().all()
    assert (all_pred.target_time - all_pred.origin == pd.Timedelta(hours=1)).all()
    assert (pred.latest_calibration_label < pred.origin).all()
    assert (pred.calibration_n >= 30).all()
    audits = json.loads((OUT / 'split_audit.json').read_text())
    for a in audits:
        assert pd.Timestamp(a['max_train_label']) < pd.Timestamp(a['first_test_origin'])
        assert pd.Timestamp(a['inner_max_label']) < pd.Timestamp(a['validation_start'])
        assert pd.Timestamp(a['outer']['fit_end']) < pd.Timestamp(a['cutoff'])
        assert pd.Timestamp(a['inner']['fit_end']) < pd.Timestamp(a['validation_start'])
    for (mode, definition, origin), g in paths.groupby(['mode', 'definition', 'origin']):
        assert set(g.h) == {1, 6, 12}
        assert g.e_now.max() - g.e_now.min() < 1e-9
    parts = ['delta_local_bp', 'delta_quote_bp', 'delta_fx_bp', 'delta_market_bp', 'delta_g_bp']
    error = float(np.max(np.abs(paths[parts].sum(axis=1) - paths.delta_e)))
    assert error < 1e-8
    result = dict(input_and_original_source_hashes_match=True, protocol_unchanged=True,
        training_only_first_stages=True, purged_labels=True, no_future_calibration_labels=True,
        unchanged_residual_targets_checked=True, exact_1h_targets=True,
        common_origin_paths_checked=True, price_identity_max_error_bp=error,
        warmup_and_evaluation_forecasts=len(all_pred), evaluation_forecasts=len(pred),
        model_fits=manifest['fit_count'], warning_count=manifest['warnings'],
        analysis_source_sha256=sha256(HERE / 'analyze_adaptive.py'),
        inference='exploratory fixed-prediction calendar blocks; not refit adaptive paths')
    (OUT / 'VERIFICATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def report(metrics, primary):
    lines = ['# 후속 적응형 실험 자동 결과', '',
        '최초 실패 결과를 본 뒤 추가한 탐색적 실험이다. 주 평가는 동일한 2025-12~2026-03이다.', '',
        '| 표본 | 모형 | 정보 | 보정 | n | 손실(bp) | 하회율 |',
        '|---|---|---|---|---:|---:|---:|']
    for r in metrics[metrics.definition == 'EQ'].itertuples():
        lines.append(f'| {r.mode} | {r.model} | {r.info} | {r.calibration} | {r.n} | {r.loss_bp:.4f} | {r.below_rate:.1%} |')
    lines += ['', '## 미리 정한 후속 주 비교: available_macro / EQ / 보정 후 / history', '',
        '| 모델 | 비교 | 개선율 | 95% 구간 | 개선 월 | 탐색 Holm p |',
        '|---|---|---:|---|---|---:|']
    for r in primary.itertuples():
        lines.append(f'| {r.model} | {r.comparison} | {r.improvement_pct:.2f}% | {r.improvement_lo:.2f}%~{r.improvement_hi:.2f}% | {r.winning_months}/{r.months} | {r.p_holm_exploratory:.4f} |')
    lines += ['', 'CI는 이미 생성된 과거 순차 예측을 날짜 블록으로 재표집한 진단이며 전체 적응·학습 경로를 재학습하지 않았다.',
              '보정 개선을 ML 기여로 해석하지 않는다. 같은 보정을 적용한 선형 대비 차이와 포지셔닝 추가 효과를 각각 확인해야 한다.', '']
    (OUT / 'AUTOMATED_RESULTS_KO.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    all_pred = pd.read_csv(OUT / 'predictions.csv.gz')
    for col in ['origin', 'target_time', 'latest_calibration_label']:
        all_pred[col] = pd.to_datetime(all_pred[col], utc=True)
    pred = all_pred[all_pred.evaluation].copy()
    old = pd.read_csv(LEGACY / 'results/predictions.csv.gz')
    old.origin = pd.to_datetime(old.origin, utc=True)
    old = old[(old.scenario == 'NY_delay1') & (old.projection == 'sequential') &
              (old.h == 1) & (old.q == .1)]
    unchanged_original_calibration(old)
    metrics, monthly, comparisons, primary = model_results(pred, old)
    paths = adjustment_evidence(pred)
    state_forecast_evidence(pred)
    plot(metrics)
    v = verification(all_pred, pred, paths)
    report(metrics, primary)
    print(primary[['model', 'comparison', 'improvement_pct', 'improvement_lo', 'improvement_hi',
                   'winning_months', 'p_holm_exploratory']].to_string(index=False))
    print(json.dumps(v, indent=2))


if __name__ == '__main__':
    main()
