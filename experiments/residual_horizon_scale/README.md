# 잔차 자체의 변동 폭과 예측 시차: 결과 이후의 진단·보완

기존 factorial의 비유의 결과를 ML 일반의 불가능성으로 해석하지 않고, 그 설계가 충분히 다루지 않은 두 축을 제한해서 점검한다. [고정 설계](PROTOCOL_KO.md), [실행 전 검증](PRE_RUN_TESTS.txt), [잠금](LOCK.json)을 보존한다. 결과를 본 과거 기간에서의 탐색적 개발이며 독립 검증이 아니다.

**실행·검증·보고 완료:** 새 적합 7,200회와 후보 오차 이력 1,080개 대조, 72개 실제 재적합을 완료했다. 12시간 original/F에서 QRF의 손실은 선형 대비 14.77%, 문턱형 대비 10.17% 낮았다. 이후 고정 설정으로 추가 난수 세 개를 점검(48회 적합)했으며 네 seed 평균 개선은 14.89%/10.29%였다. 기존 주1시간의 판정을 교체하지 않는다. 3월 보정 후 성능 악화, 포지셔닝 증분의 불확실성도 함께 공개했다. [결과와 해석](RESULTS_KO.md).

난수 민감도는 첫 결과를 확인한 뒤 [별도 범위](SEED_SENSITIVITY_KO.md)를 잠근 후 `python3 experiments/residual_horizon_scale/hs_seed_check.py run`으로 실행했다. 전체 교차 비교가 네 seed로 확대됐다는 뜻은 아니다.

- 회귀 분해를 유지하고, 실제 1·6·12시간 뒤 q10 예측에 ML을 사용한다.
- 원래 R/B/F와 각각에 자체 잔차 변화의 과거 72시간 RMS를 추가한 R/B/F를 비교한다. 선형·QRF·부스팅·문턱형 모두 같은 관측과 입력 묶음으로 재학습한다.
- 이전 1시간 원본 후보 예측은 동일성을 확인하고 재사용한다. 1시간 own_scale 및 6·12시간 두 버전은 새로 적합한다.
- 초기 `residual_dynamics_ml`에도 6·12시간 부스팅 보조 실험은 있었다. 이번은 장기 시차의 최초 실험이 아니라, available_macro 표본과 최근 후보별 보정·중첩 선택·R/B/F 교차 비교를 적용한 후속이다. 기존 표본과 다른 점수의 단순 증감을 성능 개선으로 해석하지 않는다.
- 현재 QRF의 평균 기준 분할은 알려진 방식이며 버그로 확정하지 않는다. 분포/분위수 기준 분할은 별도 방법론 후보로 남아 있고, 이번 실행에는 추가하지 않았다.

실행 환경은 기존 연구의 Python 3.8.10 및 잠긴 의존성을 사용한다. 아래는 최초 실행 순서이며, 이미 봉인·평가된 결과를 덮어쓰는 실행은 거부한다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-hs-mpl
python3 experiments/residual_horizon_scale/test_hs.py
# 설계/실행기/테스트/의존 파일 잠금, 커밋·push 이후:
python3 experiments/residual_horizon_scale/hs_run.py --workers 16
python3 experiments/residual_horizon_scale/hs_verify_v2.py
python3 experiments/residual_horizon_scale/hs_evaluate.py
```

미완료 적합의 재개는 동일 잠금과 해시가 일치하는 `.runs/residual_horizon_scale` 캐시만 허용한다. 새로운 연구 개발은 이 잠금과 결과를 변경하지 않고 별도로 기록한다.

`hs_core.py`는 자료/특징/보정/선택, `hs_run.py`는 후보 적합과 봉인, `hs_verify.py`는 독립 산술 대조와 실제 재적합, `hs_evaluate.py`는 81개 개발 비교와 월별·조건별 결과를 담당한다. 기록된 구간은 적합된 예측의 불확실성이고, 모든 과거 탐색이나 모델 선택의 불확실성을 포괄하지 않는다.

최초 검증기는 NumPy/pandas의 기계 정밀도 합산 차이로 동점 후보를 다르게 고르는 문제를 발견해 중단했다. [검증기 수정 설명](VERIFIER_AMENDMENT_KO.md)과 [추가 잠금](VERIFIER_AMENDMENT_LOCK.json)을 점수 개봉 전에 기록했다. 최종 검증은 `hs_verify_v2.py`이며 학습·선택·예측·평가 규칙은 변경하지 않았다. `hs_report.py`는 표현용 파일이고 모델 선택에 관여하지 않는다.
