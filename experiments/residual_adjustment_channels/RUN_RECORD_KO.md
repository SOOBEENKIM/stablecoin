# 실행 기록

- 환경: Linux, Python3.8.10, numpy1.23.5, pandas2.0.3, sklearn1.3.2. OPENBLAS/OMP/MKL_NUM_THREADS=1, MPLCONFIGDIR=/tmp/stablecoin-channels-mpl.
- 주 설계·소스·입력 준비 고정:9288dfe. 주180fits 봉인→원시 가격3588행·9재학습 검증→주/보조 검정 개봉.
- 주 결과 기록 및 후속 초안:3704309. 후속 lock 생성 시 상대 소스 경로 오류를 발견해 절대 경로로 수정, b11d137에서 lock 고정/push. 후속 학습 전 수정이며 주 결과는 변경 없음.
- 후속60fits: 기존 m/g_total을 고정, 절대 괴리 감소·부호 전환·과도 반전 nuisance 추가. 봉인→9재학습 및 piecewise 항등식 검증→추가 검정 개봉.
- 대칭 평균회귀 대안 고정:f0a35c2.24개 소규모 적합, 과거72h 기준점 전체 평가행 독립 검증, 확률 봉인 뒤 오차 모멘트 검정.
- 원래 주 검정을 본 뒤 설계한 두 후속은 명시적으로 사후 탐색/해석 검증이다. 전체 과거 기간은 이미 재사용됐다.
- 최종 독립 검사: 단순 대안 확률을 closed-form 회귀계수와 대칭 충격의 직접 부등식 평균으로 다시 계산했다. 결과는 FINAL_AUDIT.json.

모듈별 추가 실행 명령(새 복제/재현 목적의 별도 출력 디렉터리에서 실행):

```bash
python3 experiments/residual_adjustment_channels/test_ac.py
python3 experiments/residual_adjustment_channels/ac_run.py --workers 12
python3 experiments/residual_adjustment_channels/ac_verify.py
python3 experiments/residual_adjustment_channels/ac_evaluate.py
python3 experiments/residual_adjustment_channels/test_ac_overshoot.py
python3 experiments/residual_adjustment_channels/ac_overshoot.py run --workers 12
python3 experiments/residual_adjustment_channels/ac_overshoot.py verify
python3 experiments/residual_adjustment_channels/ac_overshoot.py evaluate
python3 experiments/residual_adjustment_channels/test_ac_null.py
python3 experiments/residual_adjustment_channels/ac_symmetric_null.py run
python3 experiments/residual_adjustment_channels/ac_symmetric_null.py evaluate
python3 experiments/residual_adjustment_channels/ac_report.py
python3 experiments/residual_adjustment_channels/ac_final_audit.py
```

원본 결과의 덮어쓰기는 차단한다. 각 실행 전 lock의 source/input SHA256 및 원격 고정 여부를 확인한다. 출판용 그림은 adjustment_summary.pdf, 보고용 그림은 adjustment_summary.png이다. 최종 원본 파일 해시는 FINAL_MANIFEST.json에 보존한다.
