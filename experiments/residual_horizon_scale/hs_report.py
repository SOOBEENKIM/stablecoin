"""Presentation only; added after model-code lock, never changes predictions."""
import json
import pandas as pd
from hs_core import OUT, HERE, sha, now, write_new, ROOT


def table(frame):
    d=frame.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(lambda x:f'{x:.3f}')
    rows=['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    rows += ['| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def main():
    assert (OUT/'EVALUATION_COMPLETE.json').exists()
    cmp=pd.read_csv(OUT/'development_comparisons.csv')
    met=pd.read_csv(OUT/'metrics.csv')
    cond=pd.read_csv(OUT/'conditional.csv')
    monthly=pd.read_csv(OUT/'monthly_effects.csv')
    allcmp=pd.read_csv(OUT/'comparisons.csv')
    audit=json.loads((OUT/'VERIFICATION.json').read_text())
    seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    maincols=['horizon','comparison','n','improvement_pct','improvement_lo','improvement_hi','p_holm_81','winning_months']
    keynames=['original__nonlinear_F','original__ml_vs_threshold_F','own_scale__nonlinear_F','scale__F__ml_selected',
              'original__position_F_vs_B','own_scale__position_F_vs_B','own_scale__ml_vs_threshold_F']
    sections=['# 결과 이후 진단과 보완: 자체 변동 폭·예측 시차',
      '**탐색적 후속 실험이다. 이전 주 분석을 교체하지 않고, 같은 과거 기간을 다시 사용했다. 전체 교차 비교는 seed 1개이며, 결과 개봉 뒤 12시간 F의 선택된 QRF 설정에 한해 4개 seed 민감도를 추가했다.** '
      '회귀 잔차 계산은 유지하고 예측 단계의 입력과 시차를 교차 비교했다. 수치의 부호가 양수이면 후보의 손실이 더 작다. '
      '새 특징은 과거 72시간에 관측된 실제 1시간 잔차 변화의 RMS로, BTC 변동성이나 평균을 뺀 표준편차와 다르다.',
      '## 읽어야 할 결론\n\n'
      '최근 1시간 주 비교의 비유의를 ML 전체의 불가능성으로 확대할 수 없었다. '
      '**12시간 original/F에서 과거 자료로 선택된 QRF는 선형 대비 14.77%[8.36,19.93], '
      '문턱형 대비 10.17%[4.69,15.95] 낮은 손실**을 보였다. 81개 개발 비교의 Holm p는 각각 .0800/.0405다. '
      '이 내부 보정 통과 여부가 원래 주 분석의 판정을 바꾸거나 과거 탐색을 독립 검증으로 만드는 것은 아니다. '
      '12시간 네 월에서 모두 같은 QRF 설정이 선택됐다.\n\n'
      '이 결과를 자신에게 유리한 표본으로만 설명하지 않도록, 세 시차의 공통 시작점 239개도 비교했다. '
      '12시간의 선형/문턱형 대비 개선은 18.10%/11.29%였고, 1·5·10일 블록 구간도 양의 방향을 유지했다. '
      '다만 3월 보정 후 성능은 선형보다 3.74%, 문턱형보다 3.52% 나빴다. '
      '보정 전의 같은 선택 후보에서는 네 월 모두 개선됐으므로, 향후에는 적응 보정과 월별 상태 변화도 구분해 점검할 필요가 있다. '
      '이 관찰만으로 보정 알고리즘의 구조적 원인을 특정하지 않는다.\n\n'
      '**자체 변동 폭 추가 효과는 별개의 결과다.** F의 ML 선택 손실은 1시간 1.42%, 6시간 2.66% 감소했지만 '
      '12시간에는 0.03% 증가했다. 1시간 선형도 2.92% 개선됐다. '
      'QRF 계열만 비교하면 자체 변동 폭 추가 효과는 1시간 0.69%, 6시간 0.04%로 작고 개별 구간이 0을 포함했다. '
      '선택 기록상 추가 입력은 1시간 1월, 6시간 1·2월의 선택을 부스팅에서 QRF로 바꿨다. '
      '그러므로 ML 선택 절차의 개선을 전부 새 변수 자체의 독립적 예측 정보로 해석하지 않는다.\n\n'
      '**포지셔닝 추가의 안정적 기여는 여전히 확인되지 않았다.** original/F vs B의 개선은 '
      '1/6/12시간에서 +0.21/−1.16/+1.49%, 자체 변동 폭 통제 후에는 +0.46/−0.09/+2.77%이며 '
      '각각의 전체 표본 95% 구간에 0이 포함된다. 12시간 공통 시작점의 긍정적인 보조 수치만으로 전체 표본의 결론을 바꾸지 않는다. '
      '실제 가격 조정/레버리지 인과 메커니즘은 이번 예측 실험으로 입증되지 않는다.\n\n'
      '상태별 진단에서는 학습구간 3분위의 높은 RMS 구간에 평가 관측이 약 87% 몰렸다. '
      '낮은 구간은 1/6/12시간 각각 16/14/7개뿐이므로 그 구간의 위험 경계 하회율 차이를 신뢰할 만한 법칙으로 해석할 수 없다. '
      '모든 상태에서 잘 맞는다는 주장보다 학습·평가 간 상태 분포 변화와 조건별 표본 부족을 먼저 다뤄야 한다.\n\n'
      '12시간 F의 난수 민감도 추가 점검은 아래에 기록했다. 후속 개발의 우선순위는 잔차 정의 민감도, 3월의 보정 후 악화, '
      '실제 이후 가격 경로와의 연결이다. 원래 1시간 주 분석을 사후에 12시간으로 바꾸지 않고, '
      '유망한 탐색 결과로 구분한다. 새 알고리즘을 더 추가하는 것은 그 뒤에도 가능한 선택지다.',
      '## 1. 핵심 비교\n\n'+table(cmp[cmp.comparison.isin(keynames)][maincols]),
      '95% 구간은 개별 백분위 구간이고 Holm p는 81개 개발 비교 전체에 대한 중심화 양측 검정 보정이다. '
      '두 방법은 정확한 역관계가 아니다. 모든 비교를 통과해야 연구가 가능하다는 규칙이 아니며, '
      '원래 6개 주 비교의 결과도 이 표로 바꾸지 않는다. 탐색 이력 전체가 교정되는 것도 아니다.',
      '## 2. 같은 입력의 전체 모형 손실\n\n'+table(met[(met['sample']=='native')&(met.calibration=='calibrated')].pivot(index=['horizon','variant','information'],columns='strategy',values='loss_bp').reset_index()),
      '## 3. 자체 변동 폭을 추가한 효과\n\n'+table(cmp[cmp.comparison.str.startswith('scale__')][maincols]),
      '## 4. 월별 핵심 비교\n\n'+table(monthly[(monthly['sample']=='native')&(monthly.calibration=='calibrated')&monthly.comparison.isin(keynames)].pivot(index=['horizon','comparison'],columns='fold',values='improvement_pct').reset_index()),
      '## 5. 동일 시작점 진단\n\n실제 1·6·12시간 뒤 값이 모두 있는 239개 시작점이다. '
      '사후 미래 자료 가용성에 따른 제한으로, 배치 시점에 적용 가능한 거래/위험 선별 규칙은 아니다. '
      '시차별 절대 손실은 목표 자체가 다르므로 서로의 우열로 해석하지 않는다.\n\n'+table(allcmp[(allcmp['sample']=='common_origins')&(allcmp.calibration=='calibrated')&allcmp.comparison.isin(keynames)][['horizon','comparison','n','improvement_pct','improvement_lo','improvement_hi','winning_months']]),
      '## 6. 잔차 자체 변동 폭 상태별 위험 경계 하회율\n\n각 월 학습 자료의 3분위 경계로 구간을 고정했다. '
      'bin 0/1/2는 낮음/중간/높음이다. q10 경계 아래 실제 잔차가 내려가는 비율은 10%가 기준이다. '
      '전체 비율 10%가 조건별 보정을 보장하지 않는다. 아래 값은 탐색적 기술 통계이며 인과효과가 아니다.\n\n'+table(cond[(cond.calibration=='calibrated')&(cond.state=='e_rms72')&(cond.fold=='all')&cond.strategy.isin(['linear','ml_selected','threshold'])][['horizon','variant','strategy','bin','n','loss_bp','below_rate']]),
      '현재 잔차·직전 변화로 나눈 구간과 모든 월의 수치도 [조건별 전체 표](results/conditional.csv)에 공개했다.',
      '## 7. 구현 검증과 이전 실험의 관계\n\n'
      f"설계/코드 잠금 `{seal['lock_commit']}` 이후 새 적합 {seal['new_fits']:,}회를 실행했다. "
      f"후보 오차 이력 {audit['candidate_streams']:,}개를 별도 산술로 대조하고, "
      f"선택 예측 {audit['selected_rows_checked']:,}행을 후보와 대조했다. "
      f"1월의 {audit['actual_refits']}개 모형을 실제 재적합했다. 보정/재적합 최대 오차는 각각 "
      f"{audit['calibration_max_error']:.3g}/{audit['actual_refit_max_error']:.3g}bp다. "
      '기존 1시간 첫 seed의 예측·선택은 동일하다. 미래 특징/라벨 교란과 정확한 시차·결측 처리는 실행 전 테스트를 통과했다. '
      '이 검증은 코드가 규칙대로 실행된다는 근거이며 모형 설계가 최선이라는 증명은 아니다.\n\n'
      '최초 검증기는 평균 합산의 기계 정밀도 차이를 후보 동점 대조에 반영하지 못해 중단했다. '
      '[검증기 수정](VERIFIER_AMENDMENT_KO.md)은 점수 개봉 전에 추가 잠금했다. '
      '예측·학습·선택·평가 규칙은 그대로이며, 실제 성능 부진의 원인을 찾은 것은 아니다.\n\n'
      '초기 부스팅 연구에도 6·12시간 보조 실험이 있었다. 이번은 최신 보정·선택·입력 교차 비교와 '
      'available_macro 표본을 장기 시차에 적용한 개발이다. 이전 4개 seed 평균과 이번 1개 seed 수치를 혼동하지 않는다.\n\n'
      '환율/거시 자료의 시간대·가용 시각 가정, USDC 달러 대용값, 모형 의존적 잔차, '
      '가격 변화의 회계 분해와 인과 메커니즘의 차이 등 기존 한계도 남는다. '
      '이번 결과만으로 국내 USDT 실제 가격 하락·청산 채널·거래 수익이나 워크숍 게재 가능성이 입증되지는 않는다.',
      '## 8. 자료와 문헌\n\n'
      '[고정 설계](PROTOCOL_KO.md) · [검증 기록](results/VERIFICATION.json) · '
      '[81개 개발 비교](results/development_comparisons.csv) · [원/보정·블록 민감도 전체](results/comparisons.csv)\n\n'
      '현재 QRF의 평균 기반 나무 분할이 구현 오류인 것은 아니다. 다만 평균 외 분포 차이를 학습하는 '
      '[Cevid et al. (2022)](https://jmlr.org/papers/v23/21-0585.html), '
      '[GRF 분위수 forest](https://grf-labs.github.io/grf/reference/quantile_forest.html) 같은 다른 접근이 있으므로 '
      '이번 결과를 모든 ML 접근의 소진으로 해석하지 않는다. 해당 알고리즘은 이번 보완에서는 실행하지 않았다.'
    ]
    if (OUT/'seed_sensitivity/COMPLETE.json').exists():
        seeds=pd.read_csv(OUT/'seed_sensitivity/per_seed.csv')
        averages=pd.read_csv(OUT/'seed_sensitivity/four_seed_mean.csv')
        sections.insert(3,'## 결과 개봉 후 추가한 난수 민감도\n\n'
          '12시간 F에서 선택된 설정을 고정하고 seed 3개를 추가했다. 새 후보 선택이나 성능 최적화를 하지 않았다. '
          'original의 선형 대비 개선은 네 seed에서 14.77~15.17%, 문턱형 대비는 10.17~10.59%였다. '
          '**시점별 네 seed 손실 평균은 선형 대비 14.89%, 문턱형 대비 10.29% 개선**이다. '
          '자체 변동 폭 입력 버전도 선형 대비 13.85~14.22%로 방향은 유지됐다. '
          '이는 모델 난수에 대한 민감도이며 다른 기간/잔차 정의의 검증이나 독립 시장 이력 4개가 아니다. '
          '기존81개 비교의 p값을 이 추가 점검으로 교체하지 않는다.\n\n'+
          table(averages[averages.calibration=='calibrated'])+'\n\n'+
          '[추가 실행 전 기록](SEED_SENSITIVITY_KO.md) · [추가 잠금](SEED_LOCK.json) · '
          '[seed별 전체 결과](results/seed_sensitivity/per_seed.csv) · [월별 결과](results/seed_sensitivity/monthly.csv)')
    (HERE/'RESULTS_KO.md').write_text('\n\n'.join(sections)+'\n')
    print('Wrote descriptive report; interpretation added separately after inspecting all fixed tables.')


if __name__=='__main__':main()
