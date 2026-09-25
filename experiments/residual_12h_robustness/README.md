# 12시간 잔차 정의 안정성과 3월 악화 진단

[실행 전 설계](PROTOCOL_KO.md) · [검증 테스트](PRE_RUN_TESTS.txt) · [잠금](LOCK.json)

**검증 완료:** 세 정의의 original/F에서 네 seed 평균 ML 손실은 선형 대비 EQ 14.89% / CAP 10.23% / PCA 14.90%, 문턱형 대비 10.29% / 10.27% / 10.34% 낮았다. CAP의 통계적 확실성은 더 약하다. EQ/PCA의 3월 상대 우위 역전은 ML도 보정으로 개선됐지만 비교모형의 개선이 더 컸던 것으로 산술 분해했다. CAP은 원예측부터 선형보다 불리했다. 포지셔닝 증분은 아직 확립되지 않았다. [핵심 결론](CONCLUSION_KO.md) · [전체 결과](RESULTS_KO.md).

새 적합 6,624회, 후보 이력 1,194개 독립 대조, 선택 예측 73,296행 검증, 미래 라벨 교란 72개, 실제 재적합 72개를 완료했다. 기존 EQ의 예측과 네 seed 평균 수치는 그대로다. 원자료/이전 실행 코드는 변경하지 않았다.

같은 과거 자료를 재사용하는 탐색적 후속이다. 기존 EQ의 결과를 유지하면서 CAP/PCA에 동일한 후보·입력·선택·보정 규칙을 적용한다. 3월의 상대 성능 악화는 동일 후보의 보정 전후 손실을 산술 분해하고, 상태 분포와 관측일 집중도를 진단한다. 3월에 맞춘 보정창 변경이나 재튜닝은 포함하지 않는다.

기존 자료의 가격·시간 가정과 순차 회귀를 유지한다. CAP은 고정 유통량 시총가중 대용이고 PCA는 월별 학습 표본에서만 적합한 제1요인이다. 서로 다른 잔차의 절대 손실로 잔차 정의의 우열을 판단하지 않는다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-r12-mpl
python3 experiments/residual_12h_robustness/test_r12.py
# 잠금·commit·push 후 최초 실행:
python3 experiments/residual_12h_robustness/r12_run.py --workers 16
python3 experiments/residual_12h_robustness/r12_verify.py
python3 experiments/residual_12h_robustness/r12_evaluate.py
python3 experiments/residual_12h_robustness/r12_diagnose.py
python3 experiments/residual_12h_robustness/r12_report.py
```

완료 결과를 덮어쓰지 않는다. 적합 재개는 동일 잠금의 `.runs/residual_12h_robustness` 캐시만 허용한다. 첫 seed의 후보 전체를 선택하고, 추가 세 seed는 각 월 선택된 ML 설정을 고정한 민감도다. ML 비교는 시점별 네 seed 손실 평균이며 독립 관측이4배로 늘지 않는다. QRF/부스팅 계열별 표는 첫 seed만이다.

`r12_core.py`: 정의별 자료, 효율화한 과거 점수 계산, 독립 보정 산술. `r12_run.py`: 적합/재사용/추가 seed/봉인. `r12_verify.py`: 독립 검증과 실제 재적합. `r12_evaluate.py`:36개 비교·월별·부분 월·난수 민감도. `r12_diagnose.py`: 보정 효과 항등분해·후보별 오차 이력·상태 이동·날짜 집중도. `r12_report.py`는 점수 개봉 후 표현용이며 선택·예측에 관여하지 않는다.
