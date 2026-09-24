"""Paired chronological evaluation and descriptive price-adjustment evidence.

Block intervals condition on the already fitted forecasts; they are not full
pipeline inference. Cross-definition losses never select a residual definition.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import SEED
from data import sha256

HERE = Path(__file__).resolve().parent
GROUP = ['scenario', 'definition', 'projection', 'h', 'q']
MODEL = ['model', 'info']
B = 1999


def day_weights(frame, length, B=B, seed=SEED):
    """Circular calendar-day blocks, stratified by prediction month.

    Predictions and exact origin/target pairs are formed before resampling.
    Empty days stay in each calendar grid; no adjacent-row pseudo-days.
    Returns bootstrap frequency weights for the original observations.
    """
    rng = np.random.default_rng(seed + length)
    weights = np.zeros((B, len(frame)), dtype=np.int16)
    for fold in sorted(frame.fold.unique()):
        pos = np.flatnonzero(frame.fold.to_numpy() == fold)
        times = pd.DatetimeIndex(frame.iloc[pos].origin).normalize()
        days = pd.date_range(times.min(), times.max(), freq='D')
        loc = days.get_indexer(times)
        n = len(days)
        starts = rng.integers(0, n, size=(B, int(np.ceil(n / length))))
        sampled = ((starts[:, :, None] + np.arange(length)) % n).reshape(B, -1)[:, :n]
        counts = np.zeros((B, n), dtype=np.int16)
        for b in range(B):
            counts[b] = np.bincount(sampled[b], minlength=n)
        weights[:, pos] = counts[:, loc]
    return weights.astype(float)


def boot_mean(weights, values):
    denom = weights.sum(axis=1)
    return np.divide(weights @ np.asarray(values), denom,
                     out=np.full(len(weights), np.nan), where=denom > 0)


def interval(values):
    return np.nanquantile(values, [.025, .975])


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    corrected = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty(len(p))
    out[order] = np.minimum(1., corrected)
    return out


def summarize(pred, out):
    metrics, monthly, contrasts = [], [], []
    for group, sub in pred.groupby(GROUP, sort=True):
        key = dict(zip(GROUP, group))
        base = sub[(sub.model == 'linear') & (sub['info'] == 'full')].sort_values('origin')
        if not len(base):
            raise ValueError('Missing linear/full comparator')
        base = base.reset_index(drop=True)
        weights = {l: day_weights(base, l) for l in [1, 5, 10]}
        models = {}
        for (model, info), g in sub.groupby(MODEL):
            g = g.sort_values('origin').reset_index(drop=True)
            if not g.origin.equals(base.origin):
                raise AssertionError('Models evaluated on different origins')
            np.testing.assert_allclose(g.target, base.target, rtol=0, atol=1e-12)
            models[(model, info)] = g
            coverage_boot = boot_mean(weights[5], g.below)
            low, high = interval(coverage_boot)
            metrics.append(dict(**key, model=model, info=info, n=len(g),
                days=g.origin.dt.normalize().nunique(), loss_bp=g.loss.mean(),
                below_rate=g.below.mean(), below_lo=low, below_hi=high,
                calibration_abs_error=abs(g.below.mean() - key['q']),
                predicted_q_mean_bp=g.pred.mean(), realized_mean_bp=g.target.mean(),
                mean_start_bp=g.e_now.mean()))
            for fold, f in g.groupby('fold'):
                monthly.append(dict(**key, model=model, info=info, fold=fold,
                    n=len(f), loss_bp=f.loss.mean(), below_rate=f.below.mean()))

        pairs = []
        for info in ['base', 'full', 'account']:
            for model in ['interaction', 'spline', 'boosting']:
                if (model, info) in models and ('linear', info) in models:
                    pairs.append((model, info, 'linear', info, 'algorithm_vs_linear'))
            for model in ['spline', 'boosting']:
                if (model, info) in models and ('interaction', info) in models:
                    pairs.append((model, info, 'interaction', info, 'algorithm_vs_interaction'))
        for model in ['linear', 'interaction', 'spline', 'boosting']:
            if (model, 'full') in models:
                pairs.append((model, 'full', model, 'base', 'position_bundle'))
            if (model, 'account') in models:
                pairs += [(model, 'account', model, 'base', 'account_only'),
                          (model, 'full', model, 'account', 'other_position_given_account')]
        if ('boosting', 'full') in models:
            pairs += [('boosting', 'full', 'persistence', 'base', 'vs_persistence'),
                      ('boosting', 'full', 'constant', 'base', 'vs_constant')]
        for model, info, ref_model, ref_info, label in pairs:
            a, b = models[(model, info)], models[(ref_model, ref_info)]
            diff = a.loss.to_numpy() - b.loss.to_numpy()
            mean = diff.mean()
            for length in [1, 5, 10]:
                boot = boot_mean(weights[length], diff)
                low, high = interval(boot)
                p = (1 + np.count_nonzero(np.abs(boot - mean) >= abs(mean))) / (B + 1)
                improvement = 100 * (1 - a.loss.mean() / b.loss.mean())
                boot_ref = boot_mean(weights[length], b.loss)
                ilo, ihi = interval(-100 * boot / boot_ref)
                af = a.groupby('fold').loss.mean()
                bf = b.groupby('fold').loss.mean()
                contrasts.append(dict(**key, model=model, info=info,
                    reference_model=ref_model, reference_info=ref_info, contrast=label,
                    block_days=length, n=len(a), difference_bp=mean,
                    difference_lo=low, difference_hi=high,
                    improvement_pct=improvement, improvement_lo=ilo, improvement_hi=ihi,
                    p_centered_boot=p, winning_months=int((af < bf).sum()), months=len(af)))
    metrics, monthly, contrasts = map(pd.DataFrame, [metrics, monthly, contrasts])
    primary_mask = ((contrasts.scenario == 'NY_delay1') & (contrasts.definition == 'EQ') &
        (contrasts.projection == 'sequential') & (contrasts.h == 1) & (contrasts.q == .1) &
        (contrasts.model == 'boosting') & (contrasts['info'] == 'full') &
        (contrasts.block_days == 5) &
        (contrasts.contrast.isin(['algorithm_vs_linear', 'algorithm_vs_interaction', 'position_bundle'])))
    primary = contrasts.loc[primary_mask].copy()
    assert len(primary) == 3
    primary['p_holm'] = holm(primary.p_centered_boot)
    metrics.to_csv(out / 'metrics.csv', index=False)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    contrasts.to_csv(out / 'paired_comparisons.csv', index=False)
    primary.to_csv(out / 'primary_tests.csv', index=False)
    return metrics, monthly, contrasts, primary


def state_evidence(pred, out):
    # One model's copy of shared outcomes, to avoid counting predictions as
    # independent market observations. Training thresholds are saved per fold.
    d = pred[(pred.model == 'linear') & (pred['info'] == 'full') & (pred.q == .1)].copy()
    rows = []
    for keyvals, g in d.groupby(GROUP + ['high_downside', 'high_account']):
        key = dict(zip(GROUP + ['high_downside', 'high_account'], keyvals))
        negative = g[g.e_now < 0]
        row = dict(**key, n=len(g), days=g.origin.dt.normalize().nunique(),
            mean_start_bp=g.e_now.mean(), future_q10_bp=g.target.quantile(.1),
            future_median_bp=g.target.median(), mean_delta_bp=g.delta_e.mean(),
            median_delta_bp=g.delta_e.median(), mean_btc_vol_bp=g.btc_vol24.mean(),
            negative_start_n=len(negative),
            remains_negative_rate=(negative.target < 0).mean() if len(negative) else np.nan,
            mean_log_volume_change=g.delta_log_volume.mean())
        for c in ['delta_local_bp', 'delta_quote_bp', 'delta_fx_bp', 'delta_market_bp', 'delta_g_bp']:
            row[c] = g[c].mean()
        rows.append(row)
    states = pd.DataFrame(rows)
    states.to_csv(out / 'state_outcomes.csv', index=False)
    d.to_csv(out / 'shared_outcomes.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    # Longer-horizon sample means are not a response path when their origin
    # times differ. Explicitly diagnose and report the common-origin support.
    support, matched = [], []
    main = d[(d.scenario == 'NY_delay1') & (d.projection == 'sequential')]
    for definition, dd in main.groupby('definition'):
        frames = {h: dd[dd.h == h].set_index('origin').sort_index() for h in [1, 6, 12]}
        for horizons in [(1, 6), (1, 12), (1, 6, 12)]:
            common = frames[horizons[0]].index
            for h in horizons[1:]:
                common = common.intersection(frames[h].index)
            support.append(dict(definition=definition, horizons=','.join(map(str, horizons)),
                n=len(common), days=common.normalize().nunique()))
            if not len(common):
                continue
            start = frames[1].loc[common]
            for h in horizons:
                g = frames[h].loc[common].copy()
                np.testing.assert_allclose(g.e_now, start.e_now, atol=1e-10)
                # Define each origin's state using the same h1 training rule.
                g['high_downside'] = start.high_downside
                g['high_account'] = start.high_account
                for (dn, ls), s in g.groupby(['high_downside', 'high_account']):
                    neg = s[s.e_now < 0]
                    matched.append(dict(definition=definition,
                        horizons=','.join(map(str, horizons)), h=h,
                        high_downside=dn, high_account=ls, n=len(s),
                        mean_start_bp=s.e_now.mean(), mean_future_bp=s.target.mean(),
                        future_q10_bp=s.target.quantile(.1), median_delta_bp=s.delta_e.median(),
                        negative_start_n=len(neg),
                        remains_negative_rate=(neg.target < 0).mean() if len(neg) else np.nan))
    pd.DataFrame(support).to_csv(out / 'horizon_common_support.csv', index=False)
    pd.DataFrame(matched).to_csv(out / 'matched_origin_adjustment.csv', index=False)
    return states


def plots(metrics, monthly, states, out):
    selected = metrics[(metrics.scenario == 'NY_delay1') & (metrics.definition == 'EQ') &
        (metrics.projection == 'sequential') & (metrics.h == 1) & (metrics.q == .1)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    kinds = ['linear', 'interaction', 'spline', 'boosting']
    for j, info in enumerate(['base', 'full']):
        sub = selected[selected['info'] == info].set_index('model').loc[kinds]
        axes[0].bar(np.arange(4) + (j - .5) * .36, sub.loss_bp, width=.36, label=info)
        axes[1].plot(kinds, sub.below_rate, marker='o', label=info)
    axes[0].set_xticks(range(4), kinds)
    axes[0].set_ylabel('Mean q10 pinball loss (bp); lower is better')
    axes[0].legend()
    axes[1].axhline(.1, color='black', ls='--', label='Nominal 10%')
    axes[1].set_ylabel('Observed fraction below predicted q10')
    axes[1].legend()
    fig.suptitle('EQ residual, actual 1h: historical out-of-time evaluation')
    fig.savefig(out / 'primary_models.png', dpi=180)
    plt.close(fig)
    sub = monthly[(monthly.scenario == 'NY_delay1') & (monthly.definition == 'EQ') &
        (monthly.projection == 'sequential') & (monthly.h == 1) & (monthly.q == .1)]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for model, info in [('linear', 'full'), ('interaction', 'full'), ('boosting', 'base'), ('boosting', 'full')]:
        s = sub[(sub.model == model) & (sub['info'] == info)].sort_values('fold')
        ax.plot(s.fold, s.loss_bp, marker='o', label=model + '/' + info)
    ax.set_ylabel('q10 pinball loss (bp)')
    ax.legend()
    fig.savefig(out / 'monthly_primary.png', dpi=180)
    plt.close(fig)


def verify_outputs(pred, metrics, out):
    identity = pred[['delta_local_bp', 'delta_quote_bp', 'delta_fx_bp', 'delta_market_bp', 'delta_g_bp']].sum(axis=1)
    err = float(np.max(np.abs(identity - pred.delta_e)))
    assert err < 1e-8
    assert ((pred.target_time - pred.origin).dt.total_seconds() == pred.h * 3600).all()
    assert not pred.duplicated(GROUP + MODEL + ['origin']).any()
    assert np.isfinite(pred[['target', 'pred', 'loss']]).all().all()
    stages = json.loads((out / 'first_stages.json').read_text())
    for s in stages:
        assert pd.Timestamp(s['outer_fit']['fit_end']) < pd.Timestamp(s['outer_cutoff'])
        assert pd.Timestamp(s['inner_fit']['fit_end']) < pd.Timestamp(s['validation_start'])
        assert pd.Timestamp(s['train_label_max']) < pd.Timestamp(s['test_origin_min'])
        assert pd.Timestamp(s['inner_label_max']) < pd.Timestamp(s['validation_origin_min'])
    result = dict(prediction_rows=len(pred), distinct_model_evaluations=len(metrics),
        successful_fold_designs=len(stages), failures=json.loads((out / 'failures.json').read_text()),
        price_identity_max_error_bp=err, exact_clock_horizons=True, train_only_first_stages=True,
        purged_labels=True, identical_within_definition_targets=True,
        duplicate_predictions=False, finite_predictions=True,
        bootstrap_repetitions=B, bootstrap_scope='fixed forecasts, calendar days stratified by evaluation month',
        analysis_source_sha256=sha256(HERE / 'analyze.py'))
    (out / 'VERIFICATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def report(metrics, primary, states, out):
    main = metrics[(metrics.scenario == 'NY_delay1') & (metrics.definition == 'EQ') &
        (metrics.projection == 'sequential') & (metrics.h == 1) & (metrics.q == .1)]
    lines = ['# 잔차 동학 ML: 첫 번째 고정 설계 실험', '',
        '2026-09-25. 이 문서는 이번 실행의 수치를 자동 생성한다. 가설 지지 여부와 원고용 해석은 상위 RESULTS_KO.md에서 별도로 설명한다.', '',
        '주 분석: NY 시간대 + 거시 관측 1시간 지연, 원고의 순차 EQ 회귀 잔차, 실제 1시간 뒤 q10.',
        '평가: 2025-12~2026-03의 월별 확장 학습. 기존 자료의 역사적 시간 외 평가이며 새 외부 표본은 아니다.', '',
        '| 모형 | 정보 | n | pinball(bp) | 실제 하회율 | 5일 블록 95% 구간 |',
        '|---|---|---:|---:|---:|---|']
    for r in main.itertuples():
        lines.append(f'| {r.model} | {r.info} | {r.n} | {r.loss_bp:.4f} | {r.below_rate:.1%} | {r.below_lo:.1%}–{r.below_hi:.1%} |')
    lines += ['', '## 미리 정한 세 비교', '',
        '개선율은 양수면 부스팅(full)이 우수하다. CI는 고정된 예측의 달력 블록 재표집이며 학습 전체를 재추정하지 않는다.', '',
        '| 비교 기준 | 개선율 | 95% 구간 | 개선된 월 | Holm p |',
        '|---|---:|---|---|---:|']
    for r in primary.itertuples():
        lines.append(f'| {r.reference_model}/{r.reference_info} | {r.improvement_pct:.2f}% | {r.improvement_lo:.2f}%–{r.improvement_hi:.2f}% | {r.winning_months}/{r.months} | {r.p_holm:.4f} |')
    lines += ['', '## 상태별 실제 결과: 설명적 비교', '',
        '하방준분산과 ACCOUNT_LS의 학습 구간 75% 경계를 사용한다. 그룹별 시작 잔차와 변동성이 다르므로 미래 꼬리 차이를 인과적 포지셔닝 효과로 해석하지 않는다.', '',
        '| 하방 상태 | 롱숏 상태 | n | 시작 잔차 평균(bp) | 미래 q10(bp) | 잔차 변화 평균(bp) |',
        '|---|---|---:|---:|---:|---:|']
    s = states[(states.scenario == 'NY_delay1') & (states.definition == 'EQ') &
        (states.projection == 'sequential') & (states.h == 1)]
    for r in s.itertuples():
        lines.append(f'| {r.high_downside} | {r.high_account} | {r.n} | {r.mean_start_bp:.3f} | {r.future_q10_bp:.3f} | {r.mean_delta_bp:.3f} |')
    lines += ['', '## 재현 파일', '',
        '- metrics.csv / monthly_metrics.csv: 모든 모형·정보·시계·민감도 결과.',
        '- paired_comparisons.csv / primary_tests.csv: 대응 손실차와 1·5·10일 블록 불확실성.',
        '- state_outcomes.csv: 상태별 시작값·미래 잔차·가격별 기여·부호 지속성.',
        '- predictions.csv.gz / validation_candidates.csv: 모든 예측과 후보 선택 근거.',
        '- first_stages.json / manifest.json / VERIFICATION.json: 학습 구간·원자료 및 코드 해시·검증.',
        '', '**한계:** 짧은 기간, 거시 자료 시간대·이용 가능 시각 가정, USDC=1 USD 대용가격 가정, 고정 유통량 CAP, 이미 탐색한 표본, 미관측 국내 주문·자금흐름. 잔차는 모델 의존적 상대가격이며 0 이탈은 달러 디페그가 아니다.', '']
    (out / 'AUTOMATED_RESULTS_KO.md').write_text('\n'.join(lines), encoding='utf-8')


def main(out):
    pred = pd.read_csv(out / 'predictions.csv.gz')
    for c in ['origin', 'target_time']:
        pred[c] = pd.to_datetime(pred[c], utc=True)
    metrics, monthly, contrasts, primary = summarize(pred, out)
    states = state_evidence(pred, out)
    plots(metrics, monthly, states, out)
    verification = verify_outputs(pred, metrics, out)
    report(metrics, primary, states, out)
    print(primary[['reference_model', 'reference_info', 'improvement_pct', 'improvement_lo',
                   'improvement_hi', 'winning_months', 'p_holm']].to_string(index=False))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    main(args.out)
