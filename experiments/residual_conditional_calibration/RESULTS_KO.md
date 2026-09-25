# 잔차 상태에 따른 위험 경계 보정: 전체 결과

이 문서는 고정 설계 실행 후 생성한 수치표다. 해석은 `CONCLUSION_KO.md`, 전체 비교 원자료는 `results/`를 참조한다.

기존 12시간, original/F, 2025-12~2026-03의 동일 509시점이다. 평가기간을 이미 여러 번 보았으므로 새 독립 검증으로 해석하지 않는다. 네 시드의 **시점별 손실/의사결정 평균**이며 예측 앙상블이 아니다. CAP는 고정 공급량 시총 대용치다.

## 1. 공정한 공동선택 후 ML 대 기준선

각 모형에 7개 동일 보정 후보를 허용하고 직전 3개월로 모형과 보정을 선택했다. 개선율 양수는 ML에 유리하다. 구간은 5일 블록 95%; p값은 정의/질문별 8개 또는 6개 Holm이며 주 정의는 EQ다. 42개 전체 Holm도 원자료에 공개한다.

| definition | reference_family | endpoint | improvement_pct | improvement_lo | improvement_hi | p_holm_within_definition_question | p_holm_all42 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EQ | linear | pinball | 5.8074 | 0.9845 | 10.0335 | 0.1950 | 1.0000 |
| EQ | linear | cost_10 | 12.2434 | 2.4327 | 22.8422 | 0.1750 | 0.9250 |
| EQ | threshold | pinball | 7.0905 | 0.2113 | 12.9244 | 0.1950 | 1.0000 |
| EQ | threshold | cost_10 | 7.0652 | -12.1302 | 22.4867 | 0.8480 | 1.0000 |
| EQ | historical | pinball | 3.5166 | -1.8862 | 8.2758 | 0.6525 | 1.0000 |
| EQ | historical | cost_10 | -14.2176 | -29.3811 | -2.3372 | 0.1120 | 0.5600 |
| EQ | state_hist | pinball | -0.3444 | -4.3833 | 3.6303 | 0.8755 | 1.0000 |
| EQ | state_hist | cost_10 | -8.8182 | -22.7881 | 1.9122 | 0.5220 | 1.0000 |
| CAP | linear | pinball | 7.7179 | -0.7314 | 15.0148 | 0.3120 | 1.0000 |
| CAP | linear | cost_10 | -1.4259 | -16.3512 | 11.5202 | 1.0000 | 1.0000 |
| CAP | threshold | pinball | 7.8334 | -1.1632 | 15.5609 | 0.3120 | 1.0000 |
| CAP | threshold | cost_10 | -9.3238 | -20.5493 | -0.2465 | 0.2575 | 1.0000 |
| CAP | historical | pinball | 6.8433 | 1.5351 | 11.8644 | 0.0875 | 0.5125 |
| CAP | historical | cost_10 | 0.4664 | -15.1899 | 13.1723 | 1.0000 | 1.0000 |
| CAP | state_hist | pinball | 4.9921 | 1.6421 | 8.2816 | 0.0320 | 0.1680 |
| CAP | state_hist | cost_10 | -19.6188 | -36.7801 | -3.8116 | 0.0875 | 0.5600 |
| PCA | linear | pinball | 6.4147 | 1.5563 | 10.3375 | 0.1160 | 0.5600 |
| PCA | linear | cost_10 | 13.7500 | 2.3637 | 25.8739 | 0.2100 | 1.0000 |
| PCA | threshold | pinball | 7.8960 | -1.9838 | 19.0909 | 0.5860 | 1.0000 |
| PCA | threshold | cost_10 | -1.1207 | -23.9401 | 16.9037 | 1.0000 | 1.0000 |
| PCA | historical | pinball | 4.1306 | -1.7025 | 8.8325 | 0.5860 | 1.0000 |
| PCA | historical | cost_10 | -11.9275 | -25.4309 | -0.7835 | 0.2100 | 1.0000 |
| PCA | state_hist | pinball | -0.2044 | -4.5773 | 4.1748 | 1.0000 | 1.0000 |
| PCA | state_hist | cost_10 | -11.5019 | -27.8459 | 0.3506 | 0.3725 | 1.0000 |

## 2. 보정 효과와 단순 이력 확대의 효과 구별

fixed_model_selected_cal은 기존 원시 모형 고정 후 보정만 선택한다. joint_selected는 모형도 다시 선택한다. global_window_selected는 전역 보정 이력 60/180개만 허용한 비교군이다.

| definition | reference_policy | new_policy | endpoint | improvement_pct | improvement_lo | improvement_hi | p_holm_within_definition_question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EQ | legacy | fixed_model_selected_cal | pinball | -0.5656 | -2.5776 | 1.2547 | 1.0000 |
| EQ | legacy | fixed_model_selected_cal | cost_10 | -1.9197 | -7.4726 | 2.5290 | 1.0000 |
| EQ | legacy | joint_selected | pinball | -0.8461 | -4.4382 | 2.1382 | 1.0000 |
| EQ | legacy | joint_selected | cost_10 | -4.4503 | -20.7164 | 7.6389 | 1.0000 |
| EQ | global_window_selected | joint_selected | pinball | -0.8525 | -3.3622 | 1.2193 | 1.0000 |
| EQ | global_window_selected | joint_selected | cost_10 | -7.3543 | -21.3754 | 1.9598 | 1.0000 |
| CAP | legacy | fixed_model_selected_cal | pinball | 0.4782 | -1.8645 | 2.9944 | 1.0000 |
| CAP | legacy | fixed_model_selected_cal | cost_10 | -4.4204 | -15.1134 | 3.3892 | 1.0000 |
| CAP | legacy | joint_selected | pinball | -2.2279 | -6.2604 | 1.6057 | 1.0000 |
| CAP | legacy | joint_selected | cost_10 | -4.8134 | -15.2259 | 5.7435 | 1.0000 |
| CAP | global_window_selected | joint_selected | pinball | 0.3741 | -0.2188 | 0.9976 | 1.0000 |
| CAP | global_window_selected | joint_selected | cost_10 | -0.2820 | -0.6993 | 0.0000 | 0.6900 |
| PCA | legacy | fixed_model_selected_cal | pinball | 0.1487 | -1.8391 | 1.9113 | 1.0000 |
| PCA | legacy | fixed_model_selected_cal | cost_10 | 1.4580 | -2.1598 | 4.4625 | 1.0000 |
| PCA | legacy | joint_selected | pinball | -0.4489 | -4.9963 | 3.4988 | 1.0000 |
| PCA | legacy | joint_selected | cost_10 | -0.6003 | -18.2548 | 11.2141 | 1.0000 |
| PCA | global_window_selected | joint_selected | pinball | 1.8726 | -1.1118 | 4.3790 | 1.0000 |
| PCA | global_window_selected | joint_selected | cost_10 | -1.2953 | -15.4859 | 9.1173 | 1.0000 |

## 3. 전체 절대 점수와 위험 경보

pinball 단위는 잔차 bp, cost_10은 가정한 미탐:오경보 9:1의 무차원 비용이다. alarm/fp/fn은 509시점의 시드 평균 개수라 소수일 수 있다. 잔차 10bp 추가 하락은 국내가격 10bp 하락과 다르다.

| definition | policy | family | pinball | below | cost_10 | alarm_10 | fp_10 | fn_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EQ | joint_selected | historical | 1.9464 | 0.1198 | 0.0515 | 216.0000 | 127.0000 | 15.0000 |
| EQ | joint_selected | linear | 1.9937 | 0.1198 | 0.0670 | 315.0000 | 224.0000 | 13.0000 |
| EQ | joint_selected | ml_selected | 1.8779 | 0.1218 | 0.0588 | 248.2500 | 159.7500 | 15.5000 |
| EQ | joint_selected | state_hist | 1.8715 | 0.1100 | 0.0540 | 219.0000 | 131.0000 | 16.0000 |
| EQ | joint_selected | threshold | 2.0212 | 0.1159 | 0.0633 | 306.0000 | 214.0000 | 12.0000 |
| EQ | legacy | ml_selected | 1.8622 | 0.1022 | 0.0563 | 270.5000 | 178.5000 | 12.0000 |
| CAP | joint_selected | historical | 1.4657 | 0.1100 | 0.0527 | 229.0000 | 169.0000 | 11.0000 |
| CAP | joint_selected | linear | 1.4796 | 0.1002 | 0.0517 | 244.0000 | 182.0000 | 9.0000 |
| CAP | joint_selected | ml_selected | 1.3654 | 0.1056 | 0.0524 | 217.7500 | 158.7500 | 12.0000 |
| CAP | joint_selected | state_hist | 1.4372 | 0.1061 | 0.0438 | 224.0000 | 160.0000 | 7.0000 |
| CAP | joint_selected | threshold | 1.4815 | 0.1022 | 0.0479 | 205.0000 | 145.0000 | 11.0000 |
| CAP | legacy | ml_selected | 1.3357 | 0.0938 | 0.0500 | 225.5000 | 164.5000 | 10.0000 |
| PCA | joint_selected | historical | 1.9379 | 0.1218 | 0.0515 | 216.0000 | 127.0000 | 15.0000 |
| PCA | joint_selected | linear | 1.9851 | 0.1238 | 0.0668 | 314.0000 | 223.0000 | 13.0000 |
| PCA | joint_selected | ml_selected | 1.8578 | 0.1277 | 0.0576 | 234.7500 | 147.0000 | 16.2500 |
| PCA | joint_selected | state_hist | 1.8540 | 0.1100 | 0.0517 | 207.0000 | 119.0000 | 16.0000 |
| PCA | joint_selected | threshold | 2.0171 | 0.1139 | 0.0570 | 284.0000 | 191.0000 | 11.0000 |
| PCA | legacy | ml_selected | 1.8495 | 0.1012 | 0.0573 | 270.5000 | 179.0000 | 12.5000 |

## 4. 3월의 별도 결과

3월은 105시점의 부분 월이다. 이 결과를 보고 선택 설정을 변경하지 않았다. 월별 전체 수치는 monthly_metrics.csv에 있다.

| category | reference_policy | reference_family | new_policy | endpoint | improvement_pct | improvement_lo | improvement_hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| model_advantage | joint_selected | linear | joint_selected | pinball | -8.9700 | -13.9418 | -1.6083 |
| model_advantage | joint_selected | linear | joint_selected | cost_10 | -1.2821 | -15.7937 | 14.4998 |
| model_advantage | joint_selected | threshold | joint_selected | pinball | -4.1340 | -13.9837 | 2.5118 |
| model_advantage | joint_selected | threshold | joint_selected | cost_10 | -21.5385 | -89.1304 | 8.3392 |
| model_advantage | joint_selected | historical | joint_selected | pinball | -5.5825 | -13.8452 | 4.6930 |
| model_advantage | joint_selected | historical | joint_selected | cost_10 | -8.2192 | -24.5153 | -0.9147 |
| model_advantage | joint_selected | state_hist | joint_selected | pinball | -7.6348 | -17.5319 | 1.7392 |
| model_advantage | joint_selected | state_hist | joint_selected | cost_10 | -23.4375 | -43.9424 | -6.9751 |
| calibration | legacy | ml_selected | fixed_model_selected_cal | pinball | 0.3888 | -7.1440 | 6.1974 |
| calibration | legacy | ml_selected | fixed_model_selected_cal | cost_10 | -2.5547 | -29.3039 | 12.6829 |
| calibration | legacy | ml_selected | joint_selected | pinball | -2.6967 | -11.4031 | 3.4161 |
| calibration | legacy | ml_selected | joint_selected | cost_10 | -15.3285 | -90.3672 | 23.7506 |
| calibration | global_window_selected | ml_selected | joint_selected | pinball | -2.1897 | -8.4291 | 3.9077 |
| calibration | global_window_selected | ml_selected | joint_selected | cost_10 | -23.9216 | -93.1818 | 5.6188 |

| definition | policy | family | pinball | below | cost_10 | fp_10 | fn_10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EQ | joint_selected | historical | 2.0934 | 0.1524 | 0.0695 | 19.0000 | 6.0000 |
| EQ | joint_selected | linear | 2.0283 | 0.1714 | 0.0743 | 15.0000 | 7.0000 |
| EQ | joint_selected | ml_selected | 2.2102 | 0.1524 | 0.0752 | 25.0000 | 6.0000 |
| EQ | joint_selected | state_hist | 2.0535 | 0.1429 | 0.0610 | 19.0000 | 5.0000 |
| EQ | joint_selected | threshold | 2.1225 | 0.2000 | 0.0619 | 29.0000 | 4.0000 |
| EQ | legacy | ml_selected | 2.1522 | 0.1310 | 0.0652 | 41.5000 | 3.0000 |
| CAP | joint_selected | historical | 1.8234 | 0.1905 | 0.0562 | 23.0000 | 4.0000 |
| CAP | joint_selected | linear | 1.5573 | 0.1905 | 0.0543 | 21.0000 | 4.0000 |
| CAP | joint_selected | ml_selected | 1.7010 | 0.1690 | 0.0779 | 18.7500 | 7.0000 |
| CAP | joint_selected | state_hist | 1.8342 | 0.1810 | 0.0486 | 24.0000 | 3.0000 |
| CAP | joint_selected | threshold | 1.6600 | 0.1905 | 0.0676 | 17.0000 | 6.0000 |
| CAP | legacy | ml_selected | 1.6576 | 0.1595 | 0.0624 | 29.5000 | 4.0000 |
| PCA | joint_selected | historical | 2.0938 | 0.1524 | 0.0705 | 20.0000 | 6.0000 |
| PCA | joint_selected | linear | 2.0299 | 0.1905 | 0.0743 | 15.0000 | 7.0000 |
| PCA | joint_selected | ml_selected | 2.1838 | 0.1690 | 0.0714 | 21.0000 | 6.0000 |
| PCA | joint_selected | state_hist | 2.0331 | 0.1429 | 0.0610 | 19.0000 | 5.0000 |
| PCA | joint_selected | threshold | 2.1322 | 0.1905 | 0.0629 | 30.0000 | 4.0000 |
| PCA | legacy | ml_selected | 2.1275 | 0.1286 | 0.0698 | 41.7500 | 3.5000 |

## 5. 상태별 위험 경계 정확성

below는 실제 미래 잔차가 예측 하위 10% 경계보다 낮은 비율(목표 .10)이다. 무조건/조건별 보정 정확성과 pinball/경보 비용은 서로 다른 지표다. 조건 오차는 학습 표본으로 나눈 6개 상태 하회율의 |하회율-.10| 가중평균이다. 상태 0/1은 낮은 출발 잔차, 2/3은 중간, 4/5는 높음이며 홀수는 높은 변동성이다. sparse는 30시점 미만을 표시한다.

| definition | policy | below | coverage_abs_error | conditional_abs_error | pinball | cost_10 |
| --- | --- | --- | --- | --- | --- | --- |
| EQ | fixed_model_selected_cal | 0.1041 | 0.0041 | 0.0350 | 1.8727 | 0.0574 |
| EQ | global_window_selected | 0.1179 | 0.0179 | 0.0330 | 1.8621 | 0.0548 |
| EQ | joint_selected | 0.1218 | 0.0218 | 0.0269 | 1.8779 | 0.0588 |
| EQ | legacy | 0.1022 | 0.0022 | 0.0365 | 1.8622 | 0.0563 |
| CAP | fixed_model_selected_cal | 0.0953 | 0.0047 | 0.0344 | 1.3293 | 0.0522 |
| CAP | global_window_selected | 0.1061 | 0.0061 | 0.0218 | 1.3706 | 0.0523 |
| CAP | joint_selected | 0.1056 | 0.0056 | 0.0213 | 1.3654 | 0.0524 |
| CAP | legacy | 0.0938 | 0.0062 | 0.0359 | 1.3357 | 0.0500 |
| PCA | fixed_model_selected_cal | 0.1051 | 0.0051 | 0.0379 | 1.8468 | 0.0564 |
| PCA | global_window_selected | 0.1243 | 0.0243 | 0.0437 | 1.8933 | 0.0569 |
| PCA | joint_selected | 0.1277 | 0.0277 | 0.0322 | 1.8578 | 0.0576 |
| PCA | legacy | 0.1012 | 0.0012 | 0.0350 | 1.8495 | 0.0573 |

| policy | state | n | below | below_lo | below_hi | q_change | sparse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint_selected | 0 | 11 | 0.0909 | 0.0000 | 0.3333 | -4.9561 | True |
| joint_selected | 1 | 248 | 0.0988 | 0.0530 | 0.1463 | -6.7104 | False |
| joint_selected | 2 | 10 | 0.1000 | 0.0000 | 0.2500 | -10.4715 | True |
| joint_selected | 3 | 98 | 0.1020 | 0.0465 | 0.1610 | -12.0544 | False |
| joint_selected | 4 | 9 | 0.0000 | 0.0000 | 0.0000 | -18.0290 | True |
| joint_selected | 5 | 133 | 0.1917 | 0.1208 | 0.2766 | -21.2376 | False |
| legacy | 0 | 11 | 0.1591 | 0.0000 | 0.3333 | -3.8573 | True |
| legacy | 1 | 248 | 0.0867 | 0.0438 | 0.1304 | -6.9091 | False |
| legacy | 2 | 10 | 0.1000 | 0.0000 | 0.2500 | -9.8721 | True |
| legacy | 3 | 98 | 0.0536 | 0.0136 | 0.0972 | -12.8417 | False |
| legacy | 4 | 9 | 0.0000 | 0.0000 | 0.0000 | -19.1624 | True |
| legacy | 5 | 133 | 0.1692 | 0.1082 | 0.2404 | -22.0819 | False |

## 6. 선택 결과와 민감도

EQ와 PCA의 선택 설정을 같다고 미리 가정하지 않았다. 모든 선택은 첫 시드의 과거 점수로 확정했다. 추가 시드를 보고 재선택하지 않는다.

| definition | fold | family | candidate | kind | rule |
| --- | --- | --- | --- | --- | --- |
| EQ | 2025-12 | linear | 5 | linear | local_shrink180 |
| EQ | 2025-12 | threshold | 59 | threshold | global60 |
| EQ | 2025-12 | ml_selected | 29 | boosting | local_scaled_shrink180 |
| EQ | 2025-12 | historical | 60 | historical | local_scaled180 |
| EQ | 2025-12 | state_hist | 61 | state_hist | local_shrink180 |
| EQ | 2026-01 | linear | 5 | linear | global180 |
| EQ | 2026-01 | threshold | 59 | threshold | local_scaled180 |
| EQ | 2026-01 | ml_selected | 29 | boosting | scaled180 |
| EQ | 2026-01 | historical | 60 | historical | local180 |
| EQ | 2026-01 | state_hist | 61 | state_hist | scaled180 |
| EQ | 2026-02 | linear | 5 | linear | scaled180 |
| EQ | 2026-02 | threshold | 59 | threshold | local_scaled180 |
| EQ | 2026-02 | ml_selected | 29 | boosting | scaled180 |
| EQ | 2026-02 | historical | 60 | historical | local180 |
| EQ | 2026-02 | state_hist | 61 | state_hist | scaled180 |
| EQ | 2026-03 | linear | 5 | linear | local_scaled180 |
| EQ | 2026-03 | threshold | 41 | threshold | local_scaled_shrink180 |
| EQ | 2026-03 | ml_selected | 28 | boosting | local_scaled_shrink180 |
| EQ | 2026-03 | historical | 60 | historical | local180 |
| EQ | 2026-03 | state_hist | 61 | state_hist | local_scaled_shrink180 |
| CAP | 2025-12 | linear | 5 | linear | global60 |
| CAP | 2025-12 | threshold | 47 | threshold | global60 |
| CAP | 2025-12 | ml_selected | 27 | boosting | global60 |
| CAP | 2025-12 | historical | 60 | historical | local180 |
| CAP | 2025-12 | state_hist | 61 | state_hist | global60 |
| CAP | 2026-01 | linear | 5 | linear | scaled180 |
| CAP | 2026-01 | threshold | 47 | threshold | global180 |
| CAP | 2026-01 | ml_selected | 28 | boosting | global180 |
| CAP | 2026-01 | historical | 60 | historical | local180 |
| CAP | 2026-01 | state_hist | 61 | state_hist | global180 |
| CAP | 2026-02 | linear | 5 | linear | scaled180 |
| CAP | 2026-02 | threshold | 47 | threshold | global180 |
| CAP | 2026-02 | ml_selected | 27 | boosting | global60 |
| CAP | 2026-02 | historical | 60 | historical | local180 |
| CAP | 2026-02 | state_hist | 61 | state_hist | local_shrink180 |
| CAP | 2026-03 | linear | 8 | linear | scaled180 |
| CAP | 2026-03 | threshold | 41 | threshold | global180 |
| CAP | 2026-03 | ml_selected | 16 | qrf | scaled180 |
| CAP | 2026-03 | historical | 60 | historical | local_scaled180 |
| CAP | 2026-03 | state_hist | 61 | state_hist | local_shrink180 |
| PCA | 2025-12 | linear | 5 | linear | local_shrink180 |
| PCA | 2025-12 | threshold | 47 | threshold | global180 |
| PCA | 2025-12 | ml_selected | 29 | boosting | local_scaled_shrink180 |
| PCA | 2025-12 | historical | 60 | historical | local_scaled180 |
| PCA | 2025-12 | state_hist | 61 | state_hist | local_shrink180 |
| PCA | 2026-01 | linear | 5 | linear | global180 |
| PCA | 2026-01 | threshold | 59 | threshold | local_scaled180 |
| PCA | 2026-01 | ml_selected | 29 | boosting | scaled180 |
| PCA | 2026-01 | historical | 60 | historical | local180 |
| PCA | 2026-01 | state_hist | 61 | state_hist | scaled180 |
| PCA | 2026-02 | linear | 5 | linear | scaled180 |
| PCA | 2026-02 | threshold | 59 | threshold | local_scaled180 |
| PCA | 2026-02 | ml_selected | 29 | boosting | scaled180 |
| PCA | 2026-02 | historical | 60 | historical | local180 |
| PCA | 2026-02 | state_hist | 61 | state_hist | global180 |
| PCA | 2026-03 | linear | 5 | linear | local_scaled180 |
| PCA | 2026-03 | threshold | 41 | threshold | global180 |
| PCA | 2026-03 | ml_selected | 28 | boosting | local_scaled180 |
| PCA | 2026-03 | historical | 60 | historical | local180 |
| PCA | 2026-03 | state_hist | 61 | state_hist | local_scaled_shrink180 |

| reference_family | endpoint | sample | block_days | n | improvement_pct | improvement_lo | improvement_hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear | pinball | full | 1 | 509 | 5.8074 | -0.3899 | 11.0158 |
| linear | cost_10 | full | 1 | 509 | 12.2434 | -1.6562 | 24.4094 |
| threshold | pinball | full | 1 | 509 | 7.0905 | 0.9890 | 12.1836 |
| threshold | cost_10 | full | 1 | 509 | 7.0652 | -9.9109 | 20.9668 |
| historical | pinball | full | 1 | 509 | 3.5166 | -2.8972 | 9.1304 |
| historical | cost_10 | full | 1 | 509 | -14.2176 | -30.8756 | -2.0528 |
| state_hist | pinball | full | 1 | 509 | -0.3444 | -5.0819 | 3.9174 |
| state_hist | cost_10 | full | 1 | 509 | -8.8182 | -24.2762 | 3.5525 |
| linear | pinball | full | 5 | 509 | 5.8074 | 0.9845 | 10.0335 |
| linear | cost_10 | full | 5 | 509 | 12.2434 | 2.4327 | 22.8422 |
| threshold | pinball | full | 5 | 509 | 7.0905 | 0.2113 | 12.9244 |
| threshold | cost_10 | full | 5 | 509 | 7.0652 | -12.1302 | 22.4867 |
| historical | pinball | full | 5 | 509 | 3.5166 | -1.8862 | 8.2758 |
| historical | cost_10 | full | 5 | 509 | -14.2176 | -29.3811 | -2.3372 |
| state_hist | pinball | full | 5 | 509 | -0.3444 | -4.3833 | 3.6303 |
| state_hist | cost_10 | full | 5 | 509 | -8.8182 | -22.7881 | 1.9122 |
| linear | pinball | full | 10 | 509 | 5.8074 | 1.8898 | 9.4150 |
| linear | cost_10 | full | 10 | 509 | 12.2434 | 2.8410 | 22.7627 |
| threshold | pinball | full | 10 | 509 | 7.0905 | 0.2480 | 13.1428 |
| threshold | cost_10 | full | 10 | 509 | 7.0652 | -10.6103 | 21.6478 |
| historical | pinball | full | 10 | 509 | 3.5166 | -0.9512 | 7.6530 |
| historical | cost_10 | full | 10 | 509 | -14.2176 | -28.7185 | -4.7660 |
| state_hist | pinball | full | 10 | 509 | -0.3444 | -3.4507 | 2.8223 |
| state_hist | cost_10 | full | 10 | 509 | -8.8182 | -19.6514 | -0.9275 |
| linear | pinball | nonoverlap | 5 | 138 | 5.3528 | -2.0097 | 11.3488 |
| linear | cost_10 | nonoverlap | 5 | 138 | -8.1579 | -43.3614 | 14.9354 |
| threshold | pinball | nonoverlap | 5 | 138 | -2.0817 | -14.9752 | 8.6664 |
| threshold | cost_10 | nonoverlap | 5 | 138 | -11.6848 | -55.8900 | 19.9652 |
| historical | pinball | nonoverlap | 5 | 138 | 5.4312 | -3.8797 | 11.9095 |
| historical | cost_10 | nonoverlap | 5 | 138 | -28.4375 | -79.1739 | 0.0000 |
| state_hist | pinball | nonoverlap | 5 | 138 | 2.0710 | -4.6515 | 7.4940 |
| state_hist | cost_10 | nonoverlap | 5 | 138 | -10.4839 | -43.6753 | 8.8032 |

| definition | policy | seed | pinball | cost_10 | below |
| --- | --- | --- | --- | --- | --- |
| EQ | joint_selected | 20260925 | 1.8776 | 0.0599 | 0.1198 |
| EQ | joint_selected | 20260926 | 1.8722 | 0.0578 | 0.1198 |
| EQ | joint_selected | 20260927 | 1.8835 | 0.0599 | 0.1238 |
| EQ | joint_selected | 20260928 | 1.8785 | 0.0576 | 0.1238 |
| EQ | legacy | 20260925 | 1.8647 | 0.0560 | 0.1041 |
| EQ | legacy | 20260926 | 1.8641 | 0.0566 | 0.1002 |
| EQ | legacy | 20260927 | 1.8560 | 0.0558 | 0.1041 |
| EQ | legacy | 20260928 | 1.8639 | 0.0568 | 0.1002 |
| CAP | joint_selected | 20260925 | 1.3623 | 0.0525 | 0.1061 |
| CAP | joint_selected | 20260926 | 1.3693 | 0.0525 | 0.1041 |
| CAP | joint_selected | 20260927 | 1.3640 | 0.0523 | 0.1061 |
| CAP | joint_selected | 20260928 | 1.3661 | 0.0525 | 0.1061 |
| CAP | legacy | 20260925 | 1.3359 | 0.0501 | 0.0943 |
| CAP | legacy | 20260926 | 1.3337 | 0.0501 | 0.0923 |
| CAP | legacy | 20260927 | 1.3369 | 0.0499 | 0.0943 |
| CAP | legacy | 20260928 | 1.3362 | 0.0499 | 0.0943 |
| PCA | joint_selected | 20260925 | 1.8593 | 0.0570 | 0.1297 |
| PCA | joint_selected | 20260926 | 1.8602 | 0.0595 | 0.1297 |
| PCA | joint_selected | 20260927 | 1.8553 | 0.0568 | 0.1238 |
| PCA | joint_selected | 20260928 | 1.8564 | 0.0572 | 0.1277 |
| PCA | legacy | 20260925 | 1.8587 | 0.0589 | 0.1022 |
| PCA | legacy | 20260926 | 1.8423 | 0.0568 | 0.1002 |
| PCA | legacy | 20260927 | 1.8509 | 0.0558 | 0.1041 |
| PCA | legacy | 20260928 | 1.8461 | 0.0576 | 0.0982 |

고정 보정별 전체 결과는 `results/fixed_rule_metrics.csv`에 있다. 외부 구간에서 가장 좋은 보정을 선택해 주 분석으로 교체하지 않았다.

## 7. 실행 검증

사전 검사 7개 통과. 독립 보정 검사 222개 후보/시드 스트림, 최대 차이 0. 순위 최대 차이 0. 추가 재학습 재현 9건, 최대 차이 0. 기존 예측/선택도 재현했다.

설계 고정 커밋 `81a8070`; LOCK 및 예측/검증/성능 공개 타임스탬프는 JSON 파일에 보존했다. 국소 보정에 대한 분포 무관 포함률 보장, 포지셔닝 인과 효과 또는 새로운 자료의 검증을 주장하지 않는다.
