# 한국 USDT 잔차 연구 재시작

기존 국내 원고의 회귀 잔차를 비선형 모형으로 재검토하는 별도 연구다.

- [연구 규칙과 단계](PROTOCOL_KO.md)
- [가까운 선행연구와 읽은 범위](RELATED_WORK_KO.md)
- [첫 파일럿 결과](RESULTS_KO.md)

기존 `research/option1_*`와 원자료를 보존한다. 새 결과는 이 폴더의 `results/`에 기록한다.

실행 환경은 저장소의 `environments/requirements-analysis.txt` 중 numpy, pandas, scipy, scikit-learn, lightgbm이다. 과거 신경망 예측 LFS 파일은 이번 단계에 필요하지 않다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/stablecoin-residual-mpl python3 experiments/nonlinear_residual/run_pilot.py
```

실행은 원자료 해시·시각·표본 대응·보관 파일 보존 검사를 함께 수행한다. 이 명령은 이 폴더의 첫 파일럿 결과를 덮어쓴다. 새로운 실험 설정은 별도 프로토콜과 결과 경로로 기록해야 한다.
