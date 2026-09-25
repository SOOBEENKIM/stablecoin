"""Presentation only, outside the forecast-code lock."""
import json
import pandas as pd
from r12_core import HERE,OUT


def table(frame):
    d=frame.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(lambda x:f'{x:.4f}')
    lines=['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    lines+=['| '+' | '.join(map(str,r))+' |' for r in d.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    assert (OUT/'DIAGNOSTICS_COMPLETE.json').exists()
    c=pd.read_csv(OUT/'development_comparisons.csv');allc=pd.read_csv(OUT/'comparisons.csv')
    mm=pd.read_csv(OUT/'monthly_comparisons.csv');metrics=pd.read_csv(OUT/'metrics.csv')
    gap=pd.read_csv(OUT/'gap_attribution.csv');cal=pd.read_csv(OUT/'calibration_effects.csv')
    hist=pd.read_csv(OUT/'calibration_history_monthly.csv');states=pd.read_csv(OUT/'state_distributions.csv')
    loo=pd.read_csv(OUT/'march_leave_one_day_out.csv')
    audit=json.loads((OUT/'VERIFICATION.json').read_text());seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    sel=['nonlinear_F','ml_vs_threshold_F','position_F_vs_B']
    sections=['# 12시간 정의별 검증과 3월 악화 진단',
      '[핵심 결론과 해석](CONCLUSION_KO.md)에는 정의별 안정성, 3월 역전의 산술적 경로, 남은 검증 범위를 요약했다.',
      '**같은 과거 자료에서의 탐색적 검증이다. 기존1시간 주 실험과 교수님께 전달한 당시10.6% 수치를 대체하지 않는다.** '
      '세 잔차 정의 모두 같은509개 시점이다. ML은 첫 seed에서 과거 자료로 선택한 설정을 추가3개 seed에서 재학습하고 시점별 손실을 평균했다. '
      '선형/문턱형은 결정적이며, 계열별 QRF/부스팅 부표만 첫 seed다. 원고 회귀 분해와 실제12시간/q10 목표를 유지했다.',
      '## 1. 정의별 핵심 비교\n\n'+table(c[c.comparison.isin(sel)][['definition','variant','comparison','improvement_pct','improvement_lo','improvement_hi','p_holm_36','winning_months']]),
      '양수는 후보의 예측손실이 더 작다는 뜻이다. 95% 구간은 개별 백분위 구간이고 Holm p는36개 개발 비교 전체에 대한 중심화 양측 검정 보정이다. '
      '둘은 같은 검정의 역관계가 아니다. 적합된 예측에 조건부이고 모든 이전 탐색을 교정하지 않는다. '
      'CAP은 고정 유통량 시총가중 대용이며, PCA 제1요인은 매월 과거 자료만으로 적합한다. 정의 간 절대 손실을 비교해 진짜 잔차를 고르지 않는다.',
      '## 2. 모든 입력·모형의 손실\n\n'+table(metrics[(metrics['sample']=='full')&(metrics.calibration=='calibrated')].pivot(index=['definition','variant','information'],columns='strategy',values='loss_bp').reset_index()),
      '## 3. 월별 핵심 개선율\n\n'+table(mm[(mm['sample']=='full')&(mm.calibration=='calibrated')&mm.comparison.isin(sel)].pivot(index=['definition','variant','comparison'],columns='fold',values='improvement_pct').reset_index()),
      '## 4. 3월: 동일 후보의 보정 전후 손실\n\n'+table(cal[cal.fold=='2026-03']),
      'calibration_gain_bp=원손실−보정후손실이다. 양수면 그 모형 자체는 보정으로 개선됐다. '
      'ML의 상대 우위가 줄었다는 것과 ML의 절대 손실이 보정 때문에 커졌다는 것은 구분해야 한다.',
      '## 5. 3월 상대 우위 변화의 정확한 산술 분해\n\n'+table(gap[gap.fold=='2026-03']),
      'G_raw=L_ref_raw−L_ML_raw, G_cal=L_ref_cal−L_ML_cal. '
      'G_cal−G_raw=(L_ref_cal−L_ref_raw)−(L_ML_cal−L_ML_raw)를 각 월에서 직접 확인했다. '
      '이는 기록된 예측에 대한 보정 효과의 산술적 설명이며, 시장/포지셔닝의 경제적 인과 식별은 아니다. '
      'raw 점수로 후보를 다시 고르거나 ML만 보정을 끄는 사후 성과를 제시하지 않는다.',
      '## 6. 부분 월 길이 진단\n\n각 월의20일00:00UTC 이전에 정답이 도착한 관측만 사용했다. '
      '같은 달력 길이일 뿐 관측 수/시장이 같지는 않다.\n\n'+table(allc[(allc['sample']=='first19days')&(allc.calibration=='calibrated')&allc.comparison.isin(sel)][['definition','variant','comparison','n','improvement_pct','improvement_lo','improvement_hi','winning_months']]),
      '## 7. 보정에 실제 사용한 이력\n\n아래는 첫 seed의3월 요약이다. 추가 seed를 포함한18,000여개 개별 예측 이력도 저장했다. '
      'mean_previous_origin_month_fraction은 과거 오류를 낸 모형의 예측 월이 현재 월과 다른 비율이다. '
      'realized_month_raw_error_q10은 해당 월 결과를 본 사후 기술 통계이며 예측 보정에 사용하지 않았다.\n\n'+table(hist[(hist.fold=='2026-03')&(hist.seed==20260925)]),
      '## 8. 학습·평가 상태 분포\n\n각 월의 학습1/50/99%와 평가 요약이다. '
      '분포가 달라진 것이 관찰돼도 그것만으로 성능 악화의 원인을 확정할 수 없다. '
      '전체 변수/월은 [CSV](results/state_distributions.csv)에 기록했다.\n\n'+table(states[(states.fold=='2026-03')&states.feature.isin(['e_now','e_change1','e_rms72','btc_vol24','account_log'])]),
      '## 9. 3월 날짜 집중도\n\n각 날짜를 하나씩 뺐을 때의 개선율 범위를 모두 보고한다. '
      '불리한 날을 빼고 남긴 성과가 아니며, 이 수치를 주 결과로 바꾸지 않는다.\n\n'+table(loo.groupby(['definition','variant','reference'],as_index=False).agg(days=('excluded_day','size'),raw_gain_min=('raw_gain_pct','min'),raw_gain_max=('raw_gain_pct','max'),cal_gain_min=('calibrated_gain_pct','min'),cal_gain_max=('calibrated_gain_pct','max'))),
      '## 10. 실행 검증\n\n'
      f"잠금 `{seal['lock_commit']}` 후 새 적합 {seal['new_fits']:,}회. 후보 이력 {audit['candidate_streams']:,}개 독립 산술 검증, "
      f"선택 예측 {audit['selected_rows_checked']:,}행 대조, 미래 라벨 교란 {audit['future_poison_checks']}개, "
      f"실제1월 재적합 {audit['actual_refits']}개. 보정/재적합 최대 오차는 각각 {audit['calibration_max_error']:.3g}/{audit['actual_refit_max_error']:.3g}bp. "
      '기존 EQ의 첫 seed 예측, F 추가 seed 예측, 네 seed 평균 효과/구간은 모두 기존 저장값과 일치한다. '
      '검증은 구현 충실성의 근거이며 모형 설계가 최선이라는 증명은 아니다. '
      '처음부터 같은509개 정답 시점을 요구했고 실패 후보/경고를 조용히 제외하지 않았다.',
      '[고정 설계](PROTOCOL_KO.md) · [검증 기록](results/VERIFICATION.json) · [모든 비교](results/comparisons.csv) · '
      '[36개 개발 비교](results/development_comparisons.csv) · [seed별 결과](results/seed_comparisons.csv) · '
      '[보정 이력](results/calibration_history.csv.gz) · [날짜별 손실차](results/march_daily.csv)'
    ]
    (HERE/'RESULTS_KO.md').write_text('\n\n'.join(sections)+'\n')
    print('Rendered complete tables; add scoped interpretation after reviewing all diagnostics.')


if __name__=='__main__':main()
