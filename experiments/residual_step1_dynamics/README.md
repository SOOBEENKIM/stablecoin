# 1단계: 단순 지속성·잔차 조정·시장 변동성 비교

기존 `residual_nested_validation`의 저장된 ML·선형·문턱 예측을 유지하고, 과거 자료만으로 선택한 세 단순 비교모형을 같은 시점에서 평가한다. 네 단계 중 1단계만 실행한다. 새 독립 자료의 검증이 아니다.

[설계와 해석 범위](PROTOCOL_KO.md) · [고정 설정](design.json) · [결과](RESULTS_KO.md)

## 실행 순서

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-step1-mpl
python3 -m unittest discover -s experiments/residual_step1_dynamics -p 'test_*.py' -v
python3 experiments/residual_step1_dynamics/step1_guard.py verify
python3 experiments/residual_step1_dynamics/step1_run.py --workers 8
python3 experiments/residual_step1_dynamics/step1_verify.py
python3 experiments/residual_step1_dynamics/step1_evaluate.py
```

`step1_guard.py create`는 최초 실행 전 한 번만 사용하고 소스·잠금을 commit/push한 뒤 실행한다. 이미 생성된 잠금과 실행/개봉 기록을 삭제하거나 예측을 덮어쓰지 않는다. `step1_evaluate.py --reproduce`는 봉인된 동일 예측의 점수만 재현한다. 저장소에서 결과까지 제공하므로 위 전체 학습 명령을 같은 결과 폴더에 반복 실행하면 의도적으로 거부된다.

## 결과 파일

- `LOCK.json`: 소스, 원자료, 이전 실험의 저장 예측 및 잠금, 환경 해시.
- `results/inner_scores.csv`, `checkpoints/`: 26개 후보의 7개 내부 월×4조건 학습 이력.
- `results/selected_models.json`: 새 모형 월별 선택, 학습 시각, 잔차 분해 계수, 원단위 분위수 회귀 계수.
- `results/predictions.csv.gz`: 기존 세 절차의 예측 복사본과 새 세 절차, 보정 전후, 수준/변화량 표현.
- `results/VERIFICATION.json`: 선택·보정 재계산, 실제 1월 모형 재학습, 기존 예측 불변, 미래 정답 교란 및 pinball 번역 불변성.
- `results/metrics.csv`, `monthly_metrics.csv`, `comparisons.csv`, `primary_tests.csv`, `seed_comparisons.csv`: 주 검정과 사전 지정 보조 결과 전부.
- `results/*STARTED.json`, `*SEALED.json`, `*OPENED.json`, `*COMPLETE.json`, `logs/`: 실행 순서와 기록.

변수군 제거(2단계), 위험 경보(3단계), 실제 가격 조정 경로(4단계)는 이 폴더에서 실행하지 않는다.
