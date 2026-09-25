# 2단계: 직접 예측 변수군의 추가 정보

기존 전체 입력 ML 선택 절차를 보존하고, 글로벌 BTC 시장·국내 USDT 거래량·환율과 거시 그룹을 하나씩 제거해 재학습한다. 현재 잔차와 직전 변화, 잔차 계산식, 평가 표본은 유지한다. 과거 내부 자료로 설정을 다시 고르는 주 분석과 원래 선택 설정을 유지하는 보조 재학습을 함께 보고한다.

[고정 설계](PROTOCOL_KO.md) · [기계 판독 설정](design.json) · [실험 결과](RESULTS_KO.md)

이 기간을 이미 연구에 사용했으므로 새 독립 검증이 아니다. 전체 입력과 제거 절차의 예측력 차이를 경제적 인과효과로 해석하지 않는다. 세 그룹 제거 효과를 더해 이전 10.65% 개선을 배분할 수 없다. 단계 3·4는 실행하지 않는다.

## 최초 실행과 재현

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-step2-mpl
python3 -m unittest discover -s experiments/residual_step2_information -p 'test_*.py' -v
python3 experiments/residual_step2_information/step2_guard.py verify
python3 experiments/residual_step2_information/step2_run.py --workers 12
python3 experiments/residual_step2_information/step2_verify.py
python3 experiments/residual_step2_information/step2_evaluate.py
```

실행 전에 한 번만 `step2_guard.py create`를 사용해 소스·입력·환경을 잠그고 commit/push한다. 기록된 잠금이나 실행/개봉 표식을 삭제하지 않는다. 이미 완료된 결과 폴더의 학습 명령 재실행은 차단된다. `step2_evaluate.py --reproduce`는 봉인한 동일 예측의 점수만 재현한다. 전체 재학습은 결과를 보존한 별도 재현 복제본에서 수행하고 새 검증으로 세지 않는다.

## 기록

- `LOCK.json`, `results/*STARTED.json`, `*SEALED.json`, `*OPENED.json`, `*COMPLETE.json`: 코드·데이터·이전 예측/선택 해시와 실행 순서.
- `results/inner_scores.csv`, `checkpoints/`: 모든 후보 점수·유효성·특성·학습 시각·경고.
- `results/selected_models.json`, `algorithm_choices.csv`: 그룹·조건·월·seed별 선택 및 실제 사용 입력/상호작용.
- `results/predictions.csv.gz`: 전체 입력 및 세 그룹×두 절차의 원예측·보정 예측·정답.
- `results/VERIFICATION.json`, `logs/`: 합성 테스트, 선택 재계산, 삭제 변수·미래 정답 교란, 실제 재학습과 기존 예측 대조.
- `results/primary_tests.csv`, `comparisons.csv`, `metrics.csv`, `monthly_metrics.csv`, `monthly_effects.csv`, `seed_comparisons.csv`: 사전에 정한 주·보조 결과 전체.

`removal_loss_increase_pct`의 분모는 **전체 입력 손실**이다. 양수는 제거하면 나빠짐, 음수는 제거하면 좋아짐이다. `inclusion_gain_pct`는 제거 모형 손실을 분모로 한 별도 표현이므로 혼용하지 않는다.
