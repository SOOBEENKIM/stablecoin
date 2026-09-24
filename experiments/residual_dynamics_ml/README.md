# 원고 후속: 잔차 동학에 ML 적용

기존 순차 회귀의 잔차를 유지하고 이후의 하방 분위수 예측에 ML을 적용한다. 재현 기준 커밋은 `41d1e0c`, 작업 브랜치는 `research/icaif2026-manuscript-development`다. `experiments/manuscript_reproduction/`와 원본 코드는 수정하지 않는다.

- [결과를 보기 전에 정한 설계](PROTOCOL_KO.md)
- [해석과 판단](RESULTS_KO.md)
- [전체 자동 결과표](results/AUTOMATED_RESULTS_KO.md)
- [예측별 원자료](results/predictions.csv.gz)
- [검증 기록](results/VERIFICATION.json)
- [18개 예측 조합의 재실행 대조](results/REPLAY_VERIFICATION.json)

## 실행

저장소 루트, Python 3.8.10 및 [기록된 환경](requirements.txt):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s experiments/residual_dynamics_ml -p test_integrity.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u experiments/residual_dynamics_ml/run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/residual_dynamics_ml/analyze.py
```

결과 확인 후 별도로 추가한 스플라인 진단과 저장 예측의 재실행 검증:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/residual_dynamics_ml/diagnose_spline.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/residual_dynamics_ml/verify_replay.py
```

두 후속 스크립트는 커밋된 기본 `results/` 경로를 사용한다. 최초 실행의 원자료·학습 코드·프로토콜이 변경되면 `verify_replay.py`가 해시 불일치를 알린다.

`run.py --out /tmp/residual-recheck`로 별도 경로에 재실행할 수 있다. `analyze.py`에도 같은 `--out`을 전달한다. 결과 파일을 다시 생성하면 실행 시간 등 메타데이터는 달라질 수 있다. 학습·예측 자체에는 고정 seed와 시간순 구간을 사용한다. `--smoke`는 첫 번째 정의·시계·평가 월만 실행하는 개발 점검이며 전체 연구 결과로 쓰지 않는다.

## 설계

`data.py`는 원자료로부터 이용 가능 시각, 프리미엄, 회귀/PCA 잔차, 정확한 시계의 예측쌍을 만든다. 분해 모형은 각 학습 구간에서 추정해 고정한다. `models.py`는 선형/상호작용/스플라인 분위수 회귀와 분위수 부스팅을 구현한다. `run.py`는 내부 시계열 검증과 월별 외부 평가를 수행하고, `analyze.py`는 동일 목표의 대응 비교·달력 블록 불확실성·상태별 실제 가격 조정을 정리한다.

부스팅이 선형 모형을 이기는지와 포지셔닝이 도움이 되는지는 별개의 비교다. 전체 기간을 보고 가장 좋은 모형만 남기지 않는다. 같은 시점의 잔차 변화는 국내 가격·FX·글로벌 대용가격·공통요인 기여로 정확히 분해하지만 이는 인과분해가 아니다.

1시간이 주 분석이며 6·12시간은 보조다. 각 시계의 관측 시작 시각이 다르므로 평균을 단순 연결해 지속성 곡선으로 해석하지 않는다. `horizon_common_support.csv`와 `matched_origin_adjustment.csv`가 공통 시작 시각의 가용성과 제한된 조정 비교를 기록한다.

평가 기간은 2025년 12월~2026년 3월이다. 같은 기간의 원자료는 이미 다른 분석에서 탐색됐다. 완전히 새로운 외부 기간 검증, USDT 달러 디페깅 모델, 인과적 레버리지 매개효과, 주문장 기반 유동성 또는 투자수익 실험으로 해석하지 않는다.
