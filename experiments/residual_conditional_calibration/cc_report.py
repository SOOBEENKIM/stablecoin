"""Presentation only: reads sealed/verified scores; never selects models."""
from cc_core import *


def table(d):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.4f}' if np.isfinite(x) else 'NA'
        return str(x)
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
        ['| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)])


def main():
    assert (OUT/'EVALUATION_COMPLETE.json').exists()
    tests=read(OUT/'primary_tests.csv');metrics=read(OUT/'metrics.csv');monthly=read(OUT/'monthly_metrics.csv')
    cmp=read(OUT/'comparisons.csv');states=read(OUT/'state_metrics.csv');choices=read(OUT/'choices.csv')
    seed=read(OUT/'seed_metrics.csv');verification=json.loads((OUT/'VERIFICATION.json').read_text())
    chunks=['# 잔차 상태에 따른 위험 경계 보정: 전체 결과',
        '이 문서는 고정 설계 실행 후 생성한 수치표다. 해석은 `CONCLUSION_KO.md`, 전체 비교 원자료는 `results/`를 참조한다.',
        '기존 12시간, original/F, 2025-12~2026-03의 동일 509시점이다. 평가기간을 이미 여러 번 보았으므로 새 독립 검증으로 해석하지 않는다. 네 시드의 **시점별 손실/의사결정 평균**이며 예측 앙상블이 아니다. CAP는 고정 공급량 시총 대용치다.',
        '## 1. 공정한 공동선택 후 ML 대 기준선',
        '각 모형에 7개 동일 보정 후보를 허용하고 직전 3개월로 모형과 보정을 선택했다. 개선율 양수는 ML에 유리하다. 구간은 5일 블록 95%; p값은 정의/질문별 8개 또는 6개 Holm이며 주 정의는 EQ다. 42개 전체 Holm도 원자료에 공개한다.',
        table(tests[tests.category=='model_advantage'][['definition','reference_family','endpoint','improvement_pct','improvement_lo','improvement_hi','p_holm_within_definition_question','p_holm_all42']]),
        '## 2. 보정 효과와 단순 이력 확대의 효과 구별',
        'fixed_model_selected_cal은 기존 원시 모형 고정 후 보정만 선택한다. joint_selected는 모형도 다시 선택한다. global_window_selected는 전역 보정 이력 60/180개만 허용한 비교군이다.',
        table(tests[tests.category=='calibration'][['definition','reference_policy','new_policy','endpoint','improvement_pct','improvement_lo','improvement_hi','p_holm_within_definition_question']]),
        '## 3. 전체 절대 점수와 위험 경보',
        'pinball 단위는 잔차 bp, cost_10은 가정한 미탐:오경보 9:1의 무차원 비용이다. alarm/fp/fn은 509시점의 시드 평균 개수라 소수일 수 있다. 잔차 10bp 추가 하락은 국내가격 10bp 하락과 다르다.',
        table(metrics[(metrics.policy=='joint_selected')|((metrics.family=='ml_selected')&(metrics.policy=='legacy'))][['definition','policy','family','pinball','below','cost_10','alarm_10','fp_10','fn_10']]),
        '## 4. 3월의 별도 결과',
        '3월은 105시점의 부분 월이다. 이 결과를 보고 선택 설정을 변경하지 않았다. 월별 전체 수치는 monthly_metrics.csv에 있다.',
        table(cmp[(cmp['sample']=='2026-03')&(cmp.definition=='EQ')][['category','reference_policy','reference_family','new_policy','endpoint','improvement_pct','improvement_lo','improvement_hi']]),
        table(monthly[(monthly.fold=='2026-03')&((monthly.policy=='joint_selected')|((monthly.family=='ml_selected')&(monthly.policy=='legacy'))) ][['definition','policy','family','pinball','below','cost_10','fp_10','fn_10']]),
        '## 5. 상태별 위험 경계 정확성',
        'below는 실제 미래 잔차가 예측 하위 10% 경계보다 낮은 비율(목표 .10)이다. 무조건/조건별 보정 정확성과 pinball/경보 비용은 서로 다른 지표다. 조건 오차는 학습 표본으로 나눈 6개 상태 하회율의 |하회율-.10| 가중평균이다. 상태 0/1은 낮은 출발 잔차, 2/3은 중간, 4/5는 높음이며 홀수는 높은 변동성이다. sparse는 30시점 미만을 표시한다.',
        table(metrics[metrics.family=='ml_selected'][['definition','policy','below','coverage_abs_error','conditional_abs_error','pinball','cost_10']]),
        table(states[(states.definition=='EQ')&(states.family=='ml_selected')&(states.policy.isin(['legacy','joint_selected']))][['policy','state','n','below','below_lo','below_hi','q_change','sparse']]),
        '## 6. 선택 결과와 민감도',
        'EQ와 PCA의 선택 설정을 같다고 미리 가정하지 않았다. 모든 선택은 첫 시드의 과거 점수로 확정했다. 추가 시드를 보고 재선택하지 않는다.',
        table(choices[(choices.policy=='joint_selected')][['definition','fold','family','candidate','kind','rule']]),
        table(cmp[(cmp.definition=='EQ')&(cmp.category=='model_advantage')&(cmp['sample'].isin(['full','nonoverlap']))][['reference_family','endpoint','sample','block_days','n','improvement_pct','improvement_lo','improvement_hi']]),
        table(seed[(seed.family=='ml_selected')&(seed.policy.isin(['legacy','joint_selected']))][['definition','policy','seed','pinball','cost_10','below']]),
        '고정 보정별 전체 결과는 `results/fixed_rule_metrics.csv`에 있다. 외부 구간에서 가장 좋은 보정을 선택해 주 분석으로 교체하지 않았다.',
        '## 7. 실행 검증',
        f'사전 검사 7개 통과. 독립 보정 검사 {verification["streams_checked"]}개 후보/시드 스트림, 최대 차이 {verification["calibration_max_error"]:.3g}. 순위 최대 차이 {verification["rank_max_error"]:.3g}. 추가 재학습 재현 {verification["refit_checks"]}건, 최대 차이 {verification["refit_max_error"]:.3g}. 기존 예측/선택도 재현했다.',
        '설계 고정 커밋 `81a8070`; LOCK 및 예측/검증/성능 공개 타임스탬프는 JSON 파일에 보존했다. 국소 보정에 대한 분포 무관 포함률 보장, 포지셔닝 인과 효과 또는 새로운 자료의 검증을 주장하지 않는다.']
    (HERE/'RESULTS_KO.md').write_text('\n\n'.join(chunks)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),constrained_layout=True)
    labels=dict(linear='Linear',threshold='Threshold',historical='Historical quantile',state_hist='State quantile')
    for ax,endpoint,title in zip(axes,['pinball','cost_10'],['Quantile loss','Alarm cost (miss:false alarm = 9:1)']):
        d=tests[(tests.definition=='EQ')&(tests.category=='model_advantage')&(tests.endpoint==endpoint)].copy()
        for y,row in enumerate(d.itertuples()):
            color='#116466' if row.improvement_pct>=0 else '#b35437'
            ax.hlines(y,row.improvement_lo,row.improvement_hi,color=color,lw=2)
            ax.plot(row.improvement_pct,y,'o',color=color)
        ax.set_yticks(range(len(d)));ax.set_yticklabels([labels[x] for x in d.reference_family]);ax.invert_yaxis()
        ax.axvline(0,color='gray',ls='--',lw=1);ax.grid(axis='x',alpha=.2);ax.set_title(title)
        ax.set_xlabel('ML improvement vs benchmark (%)\n95% paired 5-day block interval')
    fig.suptitle('Korean USDT residual: identical past-only calibration search (EQ, 12h)')
    for ext in ['png','svg']:fig.savefig(HERE/f'comparison.{ext}',dpi=180)
    p=HERE/'comparison.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
    plt.close(fig)
    print('Presentation generated from verified scores.')


if __name__=='__main__':main()
