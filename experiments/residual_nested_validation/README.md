# 기존 자료의 잠금·중첩 시간순 검증

추가 수집 없이 기존 연구 절차의 모델 선택을 다시 관리한다. **이미 여러 번 사용한 평가 기간이므로 새 독립 검증이 아니다.** 과거 세 내부 월에서 선형을 포함한 84개 후보를 선택한 뒤 이어지는 외부 월에서 평가하는 절차를 고정한다.

[설계](PROTOCOL_KO.md) · [기계 판독 설정](design.json) · [결과](RESULTS_KO.md)

## 선택과 평가의 구분

| 구분 | 기존 후속 | 이번 실행 |
|---|---|---|
| 모델 계열 선택 | 외부 결과를 본 후 QRF 등 추가 검토 | 선형·상호작용·스플라인·부스팅·QRF·문턱을 내부 구간에서 자동 선택 |
| 내부 구간 | 직전 한 달 | 직전 세 달의 순차 예측 |
| 외부 월의 역할 | 시간순 예측 평가, 이후 연구 설계에 참고 | 현재 실행의 선택 함수에 입력하지 않고 전 기간 예측 후 점수 공개 |
| 설계 변경 관리 | 파일로 설계 기록 | 입력·코드·환경 해시, 실행 전 원격 커밋, 예측 봉인, 평가 개봉 기록 |
| 남는 한계 | 같은 과거 평가 구간 반복 사용 | 같은 연구자가 같은 기간을 이미 본 이력은 그대로 남음 |

## 실행

환경은 이전 실험과 같은 Python 3.8.10 / NumPy 1.23.5 / pandas 2.0.3 / SciPy 1.10.1 / scikit-learn 1.3.2 / Matplotlib 3.7.5다. 소스와 데이터의 실제 해시 및 환경은 `LOCK.json`에 저장한다.

최초 실행 전에 합성 테스트, 잠금 생성, Git 커밋/push를 완료한다. 이미 있는 잠금 또는 개봉 기록을 삭제해서 새 검증처럼 실행하지 않는다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-nested-mpl
python3 -m unittest discover -s experiments/residual_nested_validation -p 'test_*.py'
python3 experiments/residual_nested_validation/guard.py verify
python3 experiments/residual_nested_validation/run.py --workers 12
python3 experiments/residual_nested_validation/verify_results.py
python3 experiments/residual_nested_validation/evaluate.py
```

`guard.py create`는 관리자가 최초 잠금 전에만 실행한다. 이 저장소에는 실행한 잠금을 보존하므로 재생성하지 않는다. `run.py`는 외부 점수를 출력하지 않고 모든 내부 선택과 외부 예측을 끝낸다. `evaluate.py --reproduce`는 이미 봉인·개봉한 동일 예측의 점수만 재현한다. 전체 재학습의 재현은 기존 산출물을 유지한 별도 복제본에서 동일 잠금으로 수행하고 독립 실험으로 세지 않는다.

## 결과 파일

- `LOCK.json`: 코드·입력 해시, 패키지 버전, 설계 시점. 자체 Git 감사 기록이지 외부 사전등록이 아니다.
- `results/RUN_STARTED.json`, `PREDICTIONS_SEALED.json`, `SCORES_OPENED.json`: 실행·예측 해시 고정·평가 개봉 순서.
- `results/inner_scores.csv`, `checkpoints/`: 모든 내부 후보 점수, 무효 후보, 학습/검증 목표 시각, 경고.
- `results/selected_models.json`, `selected_models.csv`, `algorithm_choices.csv`: 실제 외부 월별 선택 이력, 특징, 잔차 분해, 설정.
- `results/predictions.csv.gz`: 원예측·시간순 보정·목표·seed. 같은 시점의 seed 손실을 평균하고 표본을 늘리지 않는다.
- `results/metrics.csv`, `monthly_metrics.csv`, `comparisons.csv`, `primary_tests.csv`, `seed_comparisons.csv`: 주·보조 비교를 모두 공개.
- `results/VERIFICATION.json`: 전체 선택 재계산, 실제 시계/분할 검증, 보정 재현·미래 정답 교란 시험, 1월 실제 모델 재학습 대조.

기존 국내 원고, 앞선 ML 결과 및 경제적 해석 검증은 수정하지 않는다. 새로운 데이터를 수집하거나 다른 공급자의 환율·거시 시계열로 교체하지 않는다.
