# 첫 ML 결과 이후의 원인 진단과 후속 실험

기준 커밋 `3a1e877`의 첫 실험을 보존하고, 실패 원인 후보를 다음과 같이 분리한다.

1. 과거 전체/최근 90일 학습과 수준/변화량 학습: 모든 알고리즘에 동일한 내부 시계열 검증 적용.
2. 포지셔닝의 현재 수준/최근 변화량 정보 비교.
3. Gradient Boosting 외에 조건부 경험분포를 추정하는 Quantile Regression Forest 추가.
4. 각 시점에 이미 결과를 아는 예측오차만으로 온라인 q10 경계를 보정. 선형에도 동일 적용.
5. 예측 입력 VIX/DXY의 마지막 실제 관측값과 나이를 사용한 표본 확장. **환율·시장 가격·미래 목표·회귀 잔차는 보간하지 않는다.** 기존 strict 467개와 확장 1,088개를 구분해 비교한다.

[후속 설계](PROTOCOL_KO.md) · [결과 해석](RESULTS_KO.md) · [전체 자동 결과](results/AUTOMATED_RESULTS_KO.md)

## 실행

저장소 루트, 이전 실험의 [Python 환경](../residual_dynamics_ml/requirements.txt)에서:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s experiments/residual_dynamics_adaptive -p test_integrity.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u experiments/residual_dynamics_adaptive/run_adaptive.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/residual_dynamics_adaptive/analyze_adaptive.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/residual_dynamics_adaptive/verify_replay.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/residual_dynamics_adaptive/check_seeds.py
```

QRF는 RF의 평균 예측이나 트리 평균들의 분위수가 아니다. 각 트리 잎의 모든 원래 학습 관측치에 대한 가중 경험분포를 사용한다. 단일 잎이면 원래 표본의 경험 분위수를 반환하는지 테스트했다. 온라인 보정은 새 관측이 도착하기 전의 예측을 바꾸지 않는지도 검사한다.

## 무엇을 예측하는가

시장 공통 프리미엄과 글로벌 괴리를 기존 순차 회귀로 조정한 잔차를 `e_t`라 한다. 두 학습 표현은 **같은 미래 잔차의 하위 10% 경계**를 예측한다.

- 수준 학습: `q10(e_(t+1) | X_t)`를 직접 학습한다.
- 변화량 학습: `q10(e_(t+1) - e_t | X_t)`를 학습하고 이미 아는 `e_t`를 더한다.

`e_t`가 입력 정보에 포함되므로 조건부 분위수의 이동 성질에 따라 두 목표는 이론상 연결된다. 유한한 트리의 표현과 정규화 방식 때문에 실제 학습 결과가 달라질 수 있다. 변화량 모형은 현재 잔차를 기준으로 다음 움직임을 학습한다. 정규화가 없는 선형 회귀에서는 현재 잔차 계수를 1만큼 바꾸어 같은 함수 집합을 표현할 수 있으므로, 표현 변경 자체를 새로운 경제적 발견으로 부르지 않는다.

예측 경계가 -15bp라면 해당 정보하에서 미래 잔차가 그보다 낮을 가능성을 약 10%로 예측한다는 뜻이다. 경계 하회율이 20%라면 실제 하방 꼬리를 충분히 반영하지 못한 것이다. 이 경계를 개선하는 것, ML이 동일하게 개선한 선형보다 낫다는 것, 포지셔닝에 추가 정보가 있다는 것은 각각 별도의 비교다. 잔차 경계는 한국 USDT 현물가격 자체의 VaR나 매매 수익이 아니다.

## 파일

- `adaptive.py`: 원고 잔차를 계승한 정보 가용성 처리, QRF, 시간순 보정.
- `run_adaptive.py`: 내부 검증 후보와 최종 월별 예측, 원자료 및 코드 해시.
- `analyze_adaptive.py`: 보정 전후/정보 추가/선형 대비 대응 비교와 공통 시작 시점의 가격 조정.
- `results/distribution_diagnostics.csv`: 분포 이동과, 첫 평가 월 이전에 고정한 분해 기준의 비교.
- `results/origin_subset_metrics.csv`: 확장 표본의 예측을 기존 시작 시점과 추가 시점으로 분리한 성능.
- `results/original_forecast_comparison.csv`: 같은 과거 467개 시점에서 최초 선형 예측 대비 비교. 새로운 ML과 새로운 선형의 비교와 다르다.
- `results/unchanged_original_calibration.csv`: 최초 예측값을 고정한 채 보정만 추가한 통제 비교. 준비 기간인 12월을 제외한 1~3월 표본이다.
- `results/common_origin_paths.csv.gz`: 같은 시작점의 실제 1/6/12h 잔차 경로. 장기 예측모형이나 회복시간 생존분석으로 부르지 않는다.
- `results/state_forecast_metrics.csv`, `state_information_gain.csv`: 과거 학습 표본에서 정한 하방 변동성·롱숏 상태별 예측과 포지셔닝 추가 효과. 설명적 진단이며 인과효과나 추가 확증 검정이 아니다.
- `verify_replay.py`: 선택된 모형 재학습, 모든 보정 경로 재생, 아직 알 수 없는 결과를 바꾸었을 때 과거 예측 불변 여부 검사.
- `check_seeds.py`: 주 결과 확인 후, 이미 선택된 설정을 고정한 세 추가 난수 진단. [별도 규칙](SEED_CHECK_KO.md).

상태별 평균 예측 q10과 그 그룹 전체 실제 값의 q10은 동일한 통계량이 아니다. 전자는 관측마다 다른 조건부 분위수 예측의 평균이고, 후자는 조건들을 섞은 분포의 분위수다. 따라서 두 수치의 차이 자체를 보정 오차로 쓰지 않고, 해당 상태의 실제 하회율과 분위수 손실을 함께 확인한다.

본 실험은 이미 결과를 확인한 기간을 재사용하는 탐색적 후속이다. 모든 후보와 실패 결과를 공개한다. 확증적 인과효과·독립 외부 검증·conformal 이론의 무조건 보장을 주장하지 않는다.
