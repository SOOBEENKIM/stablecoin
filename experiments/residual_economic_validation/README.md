# 변수군 제거·문턱 기준선·실제 가격 조정 검증

기존 회귀 잔차 이후에 ML을 적용하는 연구의 경제적 해석을 검증한다. 국내 원고 재현본과 앞선 두 ML 실험은 그대로 보존했다. 부모 커밋은 `333c206f1164265907e121c8f3b88b060d616ecc`다.

[연구 결론](CONCLUSION_KO.md) · [전체 결과](RESULTS_KO.md) · [결과 확인 전 고정한 이번 설계](PROTOCOL_KO.md)

## 재현

저장소 루트에서 실행한다. 원자료와 이전 adaptive 예측 결과가 필요하며, Git LFS로 관리되는 자료는 먼저 받아야 한다. 이번 환경은 Python 3.8.10, NumPy 1.23.5, pandas 2.0.3, SciPy 1.10.1, scikit-learn 1.3.2다. 원자료·이전 소스·예측 파일 해시는 `results/manifest.json`에 기록돼 있다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-economic-mpl
python3 -m unittest discover -s experiments/residual_economic_validation -p 'test_*.py'
python3 experiments/residual_economic_validation/run_models.py --workers 8
python3 experiments/residual_economic_validation/report_models.py
python3 experiments/residual_economic_validation/analyze_paths.py
python3 experiments/residual_economic_validation/verify_results.py
```

실행은 이 디렉터리의 결과를 다시 만든다. `run_models.py`는 체크포인트를 저장하지만 기존 체크포인트에서 재개하지 않고 모든 모델을 다시 학습한다. 이번 실행은 후보 검증·외부 재학습을 합쳐 1,885회였으며 학습 경고는 0개였다.

## 분석 단위와 해석

- 기존 `residual_dynamics_adaptive`의 월별 회귀 잔차, 예측 목표, 표본, QRF 및 선형 예측을 계승한다. ML로 김치프리미엄 분해 방식을 다시 바꾸는 실험이 아니다.
- QRF의 account/funding/OI 정보군을 각각 전부 제외하고 재튜닝한다. EQ에서는 원래 전체 모델의 설정을 고정한 제거 재학습도 추가했다.
- 문턱형 분위수 회귀는 현재 잔차의 세 구간별 절편·기울기를 허용한다. 학습 자료에서 문턱·스케일을 정하고 이전 달 검증으로 후보를 선택한다. 선행연구의 문턱 조정 아이디어를 적용한 자체 비교 모형이며 VECM 재현이 아니다.
- 주 분석은 available_macro/EQ, 같은 1,088개 시점이다. 네 QRF 난수 실행의 **시점별 손실 평균**을 비교한다. 예측을 평균낸 앙상블의 손실도 아니고 표본이 4배인 것도 아니다. CAP/PCA 및 strict/EQ는 기본 난수 실행이다.
- 보정은 이미 결과를 관측한 최근 예측오차만 사용한다. 일반적인 시계열에 대해 보장되는 conformal coverage로 주장하지 않는다.
- 매칭은 월·한국시간 6시간대가 같고 출발 잔차·변동성·하방준분산·직전 잔차 변화가 가까운 관측을 일대일로 선택한다. 미래 결과나 예측을 매칭에 쓰지 않는다. 원인 추정용 자연실험 또는 무작위 배정은 아니다.
- 구간은 월별 층화 달력 블록 부트스트랩이다. 예측 비교는 학습된 예측을 고정하고, 경로 비교는 매칭을 고정한다. 전체 모델 재학습/재매칭 불확실성과 연구자의 이전 탐색을 모두 반영하는 확증적 구간은 아니다.
- `matching_coverage.csv`의 `adequate`는 네 매칭 변수의 사전 기준만 충족했다는 뜻이다. 모든 공변량의 균형이나 인과 식별을 뜻하지 않는다. `max_all_base_smd`도 반드시 확인한다.

## 파일

| 파일 | 내용 |
|---|---|
| `economic.py`, `run_models.py` | 변수군, 문턱 모형, 후보 선택 및 월별 재학습 |
| `state_design.py`, `analyze_paths.py` | 미래를 사용하지 않는 매칭, 동일 시작점 경로, 가격별 항등 분해 |
| `report_models.py` | 평균·월별·난수별·상태별 손실, 블록 구간, 네 주 비교의 Holm 보정 |
| `results/predictions.csv.gz` | 이번 추가 모델의 모든 예측과 보정 기록; 기존 모델은 이전 폴더 참조 |
| `results/validation_candidates.csv`, `selected_candidates.csv` | 모든 후보의 내부 검증 점수와 선택 결과 |
| `results/fitted_models.csv`, `checkpoints/`, `split_audit.json` | 학습 설정, 특징, 월별 원예측 및 시간 분할 감사 |
| `results/metrics.csv`, `monthly_metrics.csv`, `comparisons.csv` | 손실·하회율, 월별 결과, 1/5/10일 블록 민감도 |
| `results/primary_tests.csv`, `seed_comparisons.csv`, `state_ablation.csv` | 주 비교·난수·상태별 변수 제거 결과 |
| `results/eligible_paths.csv.gz`, `matched_pairs.csv`, `matched_observations.csv.gz` | 대상 경로, 선택된 쌍, 실제 결과 및 가격 구성요소 |
| `results/matching_audit.csv`, `matching_coverage.csv`, `balance*.csv` | 학습 기준, 쌍 수, 매칭 전후 균형 |
| `results/matched_contrasts.csv` | 실제 변화·하방 사건·가격 구성요소·예측 차이; 불충분한 매칭도 공개 |
| `results/*manifest.json`, `path_verification.json`, `VERIFICATION.json` | 해시·실행 수·수치 항등식·재학습 재현 검증 |
| `results/logs/` | 실제 실행 로그 |

가격별 `delta_*_bp`는 잔차 변화에 대한 부호 있는 항등 기여다. `delta_market_bp`는 공통 프리미엄 자체의 변화가 아니라 회귀계수를 곱해 차감한 기여이고, 국내 가격 기여 역시 원화 가격 변화와 다르다. `local_change_krw`, `local_logreturn_bp`를 별도로 제공한다. 항등 분해가 각 가격의 독립적인 인과효과를 의미하지 않는다.

## 검증 결과

세 무결성 테스트가 통과했다. 2월의 41개 모델/설정/난수 예측 계열을 재학습했으며 저장 예측과 최대 차이는 `3.56e-15 bp` 미만이었다. 41개 보정 계열도 재실행했고 미래 결과를 변경해 과거 예측 보정이 변하지 않는지 확인했다. EQ 2월의 여섯 매칭 구성을 재현하고 미래 가격·결과·예측 변경에 매칭이 변하지 않음을 검증했다. 1·6·12시간은 관측행 이동이 아니라 실제 시계 시간이고, 공통 경로에는 같은 시작점과 같은 쌍을 사용한다.
