# 회귀 잔차의 교차 예측 비교 실행

[고정 설계](../residual_factorial_design/PROTOCOL_KO.md)를 실제 실행하는 폴더다. [실행 명세](EXECUTION_KO.md)에 적합·검증·봉인·개봉 순서와 재현 명령을 기록한다. 기존 설계와 과거 실험 결과는 보존한다.

- `core.py`: 고정 후보와 변수, 실제 월별 자료·목표, 모형 적합 연결.
- `factorial_run.py`: 모든 후보 적합, 후보별 보정, 과거3개월 선택, 예측 봉인.
- `verify.py`: 독립 보정/손실/선택/블록 산술 대조, 원자료 시각 확인,1월 실제 재적합.
- `evaluate.py`, `factorial_stats.py`: 주6개 비교와 고정 보조 표, 월별·seed별·보정 전후 분석.
- `execution_guard.py`: 기존 설계 보존, 소스/입력/환경 잠금, commit/push 확인.
- `test_execution.py`: 성능 선택과 무관한 실행 동작 검증.

실행 완료 후 결과·검증 수치와 해석을 이 문서에 연결한다.
