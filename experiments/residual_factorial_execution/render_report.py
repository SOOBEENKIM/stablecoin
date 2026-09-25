"""Post-evaluation presentation only: no model fitting, selection, or new tests."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
LABELS = {
    'nonlinear_R': 'R: ML 선택 vs 선형',
    'nonlinear_F': 'F: ML 선택 vs 선형',
    'market_B_vs_R': '시장 정보 추가: B vs R',
    'position_F_vs_B': '포지셔닝 추가: F vs B',
    'total_F_vs_R': '전체 정보 추가: F vs R',
    'ml_vs_threshold_F': 'F: ML 선택 vs 문턱형',
}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    assert (OUT / 'EVALUATION_COMPLETE.json').exists()
    primary = pd.read_csv(OUT / 'primary_tests.csv').set_index('comparison').loc[list(LABELS)]
    metrics = pd.read_csv(OUT / 'metrics.csv')
    comparisons = pd.read_csv(OUT / 'comparisons.csv')
    verification = json.loads((OUT / 'VERIFICATION.json').read_text())
    main_metrics = metrics[(metrics['mode']=='available_macro') & (metrics.definition=='EQ') &
                           (metrics.calibration=='calibrated')]
    main_cmp = comparisons[(comparisons['mode']=='available_macro') & (comparisons.definition=='EQ') &
                           (comparisons.block_days==5)]
    ptable = table(['비교', '손실 개선율', '개별95% 구간', 'Holm p', '주 기준'],
        [[LABELS[k], f'{r.improvement_pct:+.3f}%', f'[{r.improvement_lo:+.3f}, {r.improvement_hi:+.3f}]%',
          f'{r.p_holm:.4f}', '충족' if r.supported else '미충족'] for k, r in primary.iterrows()])
    mt = main_metrics.pivot(index='information', columns='strategy', values='loss_bp')
    mtable = table(['입력', '선형', 'QRF', '부스팅', '문턱형', 'ML 선택'],
        [[i]+[f'{mt.loc[i,c]:.6f}' for c in ['linear','qrf','boosting','threshold','ml_selected']]
         for i in ['R','B','F']])
    old = pd.read_csv(HERE.parent/'residual_nested_validation/results/metrics.csv')
    old = old[(old['mode']=='available_macro') & (old.definition=='EQ') &
              (old['info']=='history') & (old.calibration=='calibrated')].set_index('strategy')
    change_rows = []
    for name, old_name, new_name in [('선형','linear','linear'), ('선택 절차','selected','ml_selected')]:
        before = float(old.loc[old_name,'loss_bp']); after = float(mt.loc['F',new_name])
        change_rows.append([name, f'{before:.6f}', f'{after:.6f}', f'{100*(1-after/before):.3f}%'])
    old_table = table(['전체 입력의 절차', '이전 손실(bp)', '이번 손실(bp)', '이전 대비 손실 감소'], change_rows)
    month = pd.read_csv(OUT/'monthly_effects.csv')
    month = month[(month['mode']=='available_macro') & (month.definition=='EQ') &
                  (month.calibration=='calibrated') & (month.comparison=='nonlinear_F')]
    monthly_table = table(['평가 월','전체 입력 ML의 선형 대비 개선율'],
                          [[r.fold, f'{r.improvement_pct:+.3f}%'] for r in month.itertuples()])
    diag = main_cmp[(main_cmp.calibration=='calibrated') &
                    main_cmp.comparison.isin(['qrf_vs_linear_F','boosting_vs_linear_F',
                                             'matched_market','matched_position','matched_total'])]
    diag_table = table(['고정 보조 비교','개선율','개별95% 구간','다중비교 보정 전 p'],
        [[r.comparison, f'{r.improvement_pct:+.3f}%',f'[{r.improvement_lo:+.3f}, {r.improvement_hi:+.3f}]%',
          f'{r.p_centered_boot:.4f}'] for r in diag.itertuples()])
    robust = comparisons[(comparisons.calibration=='calibrated') & (comparisons.block_days==5) &
        comparisons.comparison.isin(['nonlinear_F','position_F_vs_B','ml_vs_threshold_F'])]
    robust_table = table(['조건','비교','개선율','개별95% 구간'],
        [[r.mode+'/'+r.definition,LABELS[r.comparison],f'{r.improvement_pct:+.3f}%',
          f'[{r.improvement_lo:+.3f}, {r.improvement_hi:+.3f}]%'] for r in robust.itertuples()])
    raw = main_cmp[(main_cmp.calibration=='raw') & main_cmp.primary_comparison]
    raw_table = table(['동일 후보의 원예측 비교','개선율','개별95% 구간'],
        [[LABELS[r.comparison],f'{r.improvement_pct:+.3f}%',
          f'[{r.improvement_lo:+.3f}, {r.improvement_hi:+.3f}]%'] for r in raw.itertuples()])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = primary.improvement_pct.to_numpy(); y = np.arange(6)[::-1]
    ax.errorbar(x, y, xerr=np.vstack([x-primary.improvement_lo, primary.improvement_hi-x]),
                fmt='o', color='#315c9c', capsize=4, markersize=5)
    ax.axvline(0, color='#444444', linewidth=1, linestyle='--')
    ax.set_yticks(y, ['ML vs linear | R','ML vs linear | F','Market inputs | B vs R',
                     'Position inputs | F vs B','All extra inputs | F vs R','ML vs threshold | F'])
    ax.set_xlabel('Pinball loss improvement (%) — positive favors candidate')
    ax.set_title('Primary EQ sample: paired individual 95% intervals')
    ax.grid(axis='x', alpha=.2)
    fig.text(.5,.01,'No comparison meets the prespecified six-test Holm criterion. Previously seen data.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(OUT/'primary_effects.png',dpi=180)
    fig.savefig(OUT/'primary_effects.svg')
    svg = OUT/'primary_effects.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    report = f'''# 동일 입력·추가 정보 교차 비교 결과

**계획한 전체 비교를 실행하고 검증했다. 주 EQ에서 전체 입력 ML 선택의 선형 대비 손실 개선은3.85%였으나, 사전에 고정한6개 주 비교의 판단 기준을 충족한 항목은 없었다. 직접 시장 정보나 포지셔닝이 추가로 유용하다는 근거도 주 분석에서는 확인하지 못했다.** 평균 손실의 개선 가능성이 없다는 결론이나 두 모형의 동등성 증명은 아니다.

실행 전 잠금 커밋은 `7fa819b6c3c8b617db9e26134ae701d356e01af4`다. [설계](../residual_factorial_design/PROTOCOL_KO.md), [실행 명세](EXECUTION_KO.md), [잠금](LOCK.json), [예측 봉인](results/PREDICTIONS_SEALED.json), [검증](results/VERIFICATION.json), [개봉 기록](results/SCORES_OPENED.json)을 보존했다. 이 문서와 그림은 모든 점수 개봉 후의 표현·해석이며 추가 모형 탐색 결과가 아니다.

## 1. 무엇을 비교했는가

회귀로 공통요인을 고려해 잔차를 산출하는 기존 과정을 유지했다. ML은 실제1시간 뒤 잔차의 조건부 하위10% 분위수를 예측한다. R은 현재 잔차·직전 변화와 공통 달력 통제, B는 R에 직접 시장/거시/거래량 정보 추가, F는 B에 롱숏·펀딩·OI 수준/변화 추가다.

3입력 각각에 선형/QRF/부스팅/문턱형을 재학습했다. `ML 선택`은 직전3개월의 보정 후 손실로 QRF 또는 부스팅의 설정을 선택하는 절차다. 각 후보의 과거 오차만으로 보정한다. 주 표본은 기존과 동일한1,088개 시점이며4seed의 **시점별 손실 평균**으로 평가한다. 학습/평가 목표와 시점은 모든 칸에서 같고12월~3월을 모두 포함한다.

## 2. 주 비교6개

양수는 비교 대상의 손실이 기준보다 낮다는 뜻이다. 예를 들어 F vs B는 포지셔닝을 추가한 F의 손실이 더 낮을 때 양수다.

{ptable}

![주 비교의 효과와 개별 불확실성 구간](results/primary_effects.png)

고정한 기준은 양의 개선·구간 하한>0·6개 비교의 Holm p<.05를 함께 충족하는 것이다. 신뢰구간은 **개별 백분위 구간**, p값은 **절대 손실차이의 중심화 양측 부트스트랩**이며 서로 역산한 동일 검정의 구간이 아니다. 따라서 F의 개별 구간 하한이 약+0.032%라도 Holm p=.429로 주 기준을 충족하지 않는다. 결과를 본 뒤 더 유리한 검정/구간을 선택하지 않았다.

원본: [주 효과·검정](results/primary_tests.csv), [모든 고정 비교와1/5/10일 블록](results/comparisons.csv).

## 3. 같은 입력의 계열별 손실

보정 후 q10 pinball 손실(bp), 낮을수록 좋다. QRF 단독의 외부 성적이 좋아도 이를 사후에 주 선택 절차로 교체하지 않는다.

{mtable}

R의 ML 선택 손실1.683446과 F의1.686902는 표본 평균상 비슷하다. 그러나 이 결과로 '시장 정보가 전혀 필요 없다'거나 '잔차 자체의 구조적 비선형성만이 개선 원인이다'라고 확정할 수 없다. R의 선형 대비 개선도 주 기준을 충족하지 않았으며 두 입력의 동등성 허용폭을 미리 정한 실험도 아니다.

원본: [계열·정보·보정별 손실과 하회율](results/metrics.csv).

## 4. 이전10.65%가 이번3.85%와 다른 이유

평가 시점·정답이 바뀌어서 생긴 차이는 아니다. 이번에는 **보정 후 점수로 내부 선택하고, 후보별 고유 오차로 보정**하도록 절차를 맞췄다. 새 비교의 전체 입력 칸도 모두 재실행했다.

{old_table}

보정 후 ML 선택 손실은 이전보다 약0.27% 낮아 비슷한 수준이다. 반면 선형 손실은 약7.32% 낮아져 ML과의 차이가 줄었다. 이전 전체 입력 선형은12월~2월에 alpha=.05를 선택했으나, 이번에는 같은 달들에서90일/change/alpha=.01을 선택했다.3월의 학습창도 달라졌다. ML 선택은12월QRF·1월부스팅·2월QRF라는 계열은 같지만3월의 학습창이90일에서all로 달라졌다. 보정 이력도 바뀌었다.

이는 **새 규칙에서 더 강한 선형 비교모형이 선택됐다는 관찰**이다. 선택 기준·보정 기록·일부 설정이 함께 바뀌었으므로 각 변경의 원인별 기여율을 분리한 실험이라고 말하지 않는다. 이전10.65%는 당시 절차의 결과로 보존하되, 현재의 강건한 ML 우위를 입증하는 수치로 그대로 제시하면 안 된다.

## 5. 월별·난수·보정 민감도

{monthly_table}

전체 입력 ML의 개선은12월에 컸고1월·3월에는 선형보다 약간 불리했다. 네seed의 전체 기간 개선율은3.69~4.11%로 점 추정의 방향은 같았지만, 포지셔닝 증분은seed별−0.21~+0.56%로 방향도 일정하지 않았다.

주5일 블록 외에1일 블록의 다중비교 보정 전 p값이 작아지는 비교도 있으나, 이를 주 검정으로 교체하지 않는다. 원예측만 보면 다음과 같으며 모든 비교의 구간이0을 포함한다. 이는 **보정 후 기준으로 선택한 같은 후보의 원예측 진단**이지 원예측 기준으로 새로 선택한 실험은 아니다.

{raw_table}

F의 ML 하회율은 원예측19.85%, 보정 후10.32%였고 선형은16.64%에서9.83%였다. 보정으로 전체 하회율이10% 근처에 온다는 사실만으로 상태별 위험 예측이나 경제적 메커니즘이 입증되지는 않는다.

원본: [월별 손실](results/monthly_metrics.csv), [월별 효과](results/monthly_effects.csv), [seed별 효과](results/seed_comparisons.csv).

## 6. 개별 ML과 같은 설정을 적용한 정보 비교

{diag_table}

QRF 단독의 F 선형 대비 개선4.62%는 긍정적인 보조 결과다. 그러나 이 보조 결과의 다중비교 보정 전 p=.029를 사전에 고정한 ML 선택 절차의 주 검정을 대신하는 근거로 바꾸지 않는다. 같은 F 선택 설정을 R/B/F에 적용해도 시장·포지셔닝의 증분은 뚜렷하지 않았다. 따라서 정보 기여가 불명확한 현상을 입력별 모델 선택이 달랐다는 이유만으로 설명하기는 어렵다.

## 7. 다른 잔차 정의와 strict 표본

{robust_table}

CAP 및 strict에서는 선형 대비 개선이 더 크고, PCA에서는 포지셔닝에 긍정적인 보조 결과가 있다. 반면 strict에서는 ML의 문턱형 대비 점 추정이−4.96%이며4개월 모두 문턱형보다 손실이 컸다. **특정 보조 조건의 긍정적 결과를 주 결과로 승격하거나 모든 조건의 우위로 일반화하지 않는다.** strict는467개 시점·첫seed이며 공통 달력 통제3개를 새 설계대로 포함했다. 조건 간 차이는 잔차 정의·표본·관측 조건 등이 함께 달라지는 비교다.

## 8. 실행과 검증이 확인한 범위

- 실행 전18개 검증 통과. 실제 적합7,488회와 검증된 결정적seed 캐시2,592회로 총10,080개 조합을 계산했다. 해결되지 않은 경고/누락 후보는0개다.
- 후보 예측2,735,460행,1,260개 후보 계열, 내부 점수2,880개, 선택288개, 선택 예측125,910행을 저장했다.
- 독립 계산으로 후보 보정과 선택을 대조했다. 보정 최대 차이{verification['max_calibration_error_bp']:.1f}bp,1월84개 고유 모형 재적합 최대 차이{verification['max_january_refit_error_bp']:.1f}bp였다. 제외 입력을 교란한56개 추가 재적합도 일치했다.
- 독립적인 일별 합산과 관측행 가중 재표집의 최대 차이는{verification['max_independent_bootstrap_error_bp']:.3g}bp였다. 기존과 평가 시점/정답도 일치했다.

이는 확인한 구현·산술·시간 경계의 증거다. 원자료 공급자의 시각 표기, 표본 대표성, 모든 경제적 가정, 인과적 해석까지 옳다는 증명은 아니다. 모든 후보와 선택 기록은 [results](results/)에 보존한다.

## 9. 현재 가능한 연구 결론

**이번 자료에서는 비선형 예측 절차의 평균 손실 개선 가능성은 관찰되지만, 사전에 정한 주 비교에서 선형·문턱형 대비 강건한 우위나 포지셔닝의 추가 예측 기여를 확립하지 못했다.** 단순 잔차·달력 입력의 ML과 전체 입력 ML의 평균 손실이 비슷했고, 추가 시장 정보가 개선의 원천이라는 근거도 부족하다.

'ML은 불가능하다'는 결론도 아니고 '포지셔닝이 무관하다'는 결론도 아니다. 다만 현재 결과로 **포지셔닝의 경제적 채널을 ML로 입증했다**거나 **독립된 새 기간에서10.6% 개선을 재확인했다**고 쓰면 안 된다. 같은 기간의 반복 사용 한계는 잠금·재실행으로 사라지지 않는다.

후속 위험 경보·실제 가격 경로 분석은 이번 실행에 포함하지 않았다. 이를 진행하더라도 여기서 드러난 예측 우위/정보 기여의 불확실성을 없어진 것으로 전제할 수 없으며, 고정된 예측을 이용한 별도 실용성·가격 조정 근거를 확인하는 분석으로 명확히 구분해야 한다.
'''
    (HERE/'RESULTS_KO.md').write_text(report)
    print('Rendered complete report and primary comparison figure from sealed evaluation tables.')


if __name__ == '__main__':
    main()
