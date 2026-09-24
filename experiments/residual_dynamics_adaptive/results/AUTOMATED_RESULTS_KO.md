# 후속 적응형 실험 자동 결과

최초 실패 결과를 본 뒤 추가한 탐색적 실험이다. 주 평가는 동일한 2025-12~2026-03이다.

| 표본 | 모형 | 정보 | 보정 | n | 손실(bp) | 하회율 |
|---|---|---|---|---:|---:|---:|
| available_macro | boosting | base | raw | 1088 | 1.8190 | 18.3% |
| available_macro | boosting | base | calibrated | 1088 | 1.7568 | 9.7% |
| available_macro | boosting | full | raw | 1088 | 1.8206 | 17.6% |
| available_macro | boosting | full | calibrated | 1088 | 1.7786 | 9.7% |
| available_macro | boosting | history | raw | 1088 | 1.8329 | 17.9% |
| available_macro | boosting | history | calibrated | 1088 | 1.7933 | 10.1% |
| available_macro | linear | base | raw | 1088 | 1.8366 | 14.2% |
| available_macro | linear | base | calibrated | 1088 | 1.8118 | 9.2% |
| available_macro | linear | full | raw | 1088 | 1.8572 | 14.5% |
| available_macro | linear | full | calibrated | 1088 | 1.8316 | 9.3% |
| available_macro | linear | history | raw | 1088 | 1.8413 | 14.4% |
| available_macro | linear | history | calibrated | 1088 | 1.8232 | 9.3% |
| available_macro | qrf | base | raw | 1088 | 1.6914 | 16.3% |
| available_macro | qrf | base | calibrated | 1088 | 1.6697 | 10.3% |
| available_macro | qrf | full | raw | 1088 | 1.7101 | 16.4% |
| available_macro | qrf | full | calibrated | 1088 | 1.6864 | 10.4% |
| available_macro | qrf | history | raw | 1088 | 1.6897 | 17.1% |
| available_macro | qrf | history | calibrated | 1088 | 1.6477 | 10.3% |
| strict | boosting | base | raw | 467 | 1.9809 | 20.6% |
| strict | boosting | base | calibrated | 467 | 1.8212 | 9.2% |
| strict | boosting | full | raw | 467 | 1.9430 | 21.4% |
| strict | boosting | full | calibrated | 467 | 1.8082 | 8.6% |
| strict | boosting | history | raw | 467 | 1.9372 | 21.0% |
| strict | boosting | history | calibrated | 467 | 1.8086 | 8.8% |
| strict | linear | base | raw | 467 | 1.9912 | 19.3% |
| strict | linear | base | calibrated | 467 | 1.8990 | 8.6% |
| strict | linear | full | raw | 467 | 1.9550 | 17.8% |
| strict | linear | full | calibrated | 467 | 1.8918 | 7.9% |
| strict | linear | history | raw | 467 | 2.0483 | 20.6% |
| strict | linear | history | calibrated | 467 | 1.8924 | 9.0% |
| strict | qrf | base | raw | 467 | 1.8105 | 18.4% |
| strict | qrf | base | calibrated | 467 | 1.7754 | 9.0% |
| strict | qrf | full | raw | 467 | 1.7904 | 16.5% |
| strict | qrf | full | calibrated | 467 | 1.7799 | 9.0% |
| strict | qrf | history | raw | 467 | 1.7938 | 17.6% |
| strict | qrf | history | calibrated | 467 | 1.7705 | 8.6% |

## 미리 정한 후속 주 비교: available_macro / EQ / 보정 후 / history

| 모델 | 비교 | 개선율 | 95% 구간 | 개선 월 | 탐색 Holm p |
|---|---|---:|---|---|---:|
| boosting | algorithm_vs_linear | 1.64% | -1.73%~4.93% | 2/4 | 0.3315 |
| qrf | algorithm_vs_linear | 9.63% | 6.79%~12.52% | 4/4 | 0.0030 |
| boosting | position_changes | -0.83% | -1.85%~0.24% | 0/4 | 0.2500 |
| boosting | position_all | -2.08% | -3.81%~-0.68% | 1/4 | 0.0340 |
| qrf | position_changes | 2.29% | 1.10%~3.50% | 4/4 | 0.0030 |
| qrf | position_all | 1.32% | 0.31%~2.43% | 3/4 | 0.0375 |

CI는 이미 생성된 과거 순차 예측을 날짜 블록으로 재표집한 진단이며 전체 적응·학습 경로를 재학습하지 않았다.
보정 개선을 ML 기여로 해석하지 않는다. 같은 보정을 적용한 선형 대비 차이와 포지셔닝 추가 효과를 각각 확인해야 한다.
