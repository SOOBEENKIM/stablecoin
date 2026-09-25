# 잔차 예측의 경보 유용성과 가격 경로

[실행 전 설계](PROTOCOL_KO.md) · [선행연구와 차별화 범위](REFERENCES_KO.md) · [사전 검사](PRE_RUN_TESTS.txt)

**완료:** [핵심 결론과 기여 후보](CONCLUSION_KO.md) · [전체 표와 그림](RESULTS_KO.md). EQ의10bp 추가 하락 경보에서 선형 대비 비용22.98% 감소(고정 비용비9:1)를 확인했고, 그 개선은 오경보 감소가 컸다. 단순 상태별 기준선의 경쟁력,3월과 중첩 제거 민감도, 국내가격 해석의 한계도 함께 공개했다. 이 수치는 기존1시간10.6% 또는12시간 pinball14.89%와 다른 평가다.

기존 `residual_12h_robustness`의 ML/선형/문턱형 예측을 보존한다. 같은 12시간 예측을 현재 잔차 대비 변화로 환산해, 상태별 오류와 5/10/20bp 추가 하락 경보를 평가하고 실제 가격 구성요소와 연결한다. 새로운 ML 튜닝은 없다. 과거 자료의 탐색적 후속이며 독립 확인 연구가 아니다.

첫 실행용 순서(기존 완료 결과는 덮어쓰지 않음):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-decisions-mpl
python3 experiments/residual_decision_value/test_dv.py
# 이미 기록된 설계/보완 해시와 원본 예측을 검증:
python3 experiments/residual_decision_value/dv_prepare_v2.py
python3 experiments/residual_decision_value/dv_verify_v3.py
python3 experiments/residual_decision_value/dv_evaluate.py
python3 experiments/residual_decision_value/dv_paths_v2.py
python3 experiments/residual_decision_value/dv_report.py
```

실행 전 설계는 `92f68b5`에 고정했다. 결과 개봉 전 독립 검증에서 발견한 세 실행상의 문제와 보완은 각각 별도로 기록했다.

- [준비 단계](PREPARATION_AMENDMENT_KO.md): 기존 target 파일의 내부 월을 최종 평가 시점 비교에서 제외. 509개 평가 시점은 유지.
- [검증기](VERIFIER_AMENDMENT_KO.md): 백분율 변환의 반올림 때문에 바뀔 수 있는 분위수 경계의 동점 처리를 직접 보간으로 일치시킴. 예측 수정 없음.
- [시각 정렬](ALIGNMENT_AMENDMENT_KO.md): EQ/F 추가 seed의 현재 잔차 메타데이터 3,054행을 정의·시각·월과 목표 값까지 대조해 연결. 원래 예측·보정값은 유지.

`dv_core.py`는 고정 경보 비용과 상태·단순 기준선을, `dv_evaluate.py`는 주12개 비용 비교와 보조 표를, `dv_paths.py`는 동일 시작점의 관찰적 가격 분해를 구현한다. `*_v2.py`, `*_v3.py` 진입점은 위 결과 개봉 전 보완을 적용한다. 기존 잠금 파일은 그대로다. `dv_report.py`는 결과 개봉 뒤 표·그림을 만드는 표현용 코드로 예측이나 평가 기준에 관여하지 않는다.

경보 비용은 미탐:오경보=9:1인 가상 의사결정의 단위 없는 값이다. 투자수익·원화 손실·실제 거래비용을 측정하지 않는다. 첫 seed의 경로 집단과 네 seed 평균 비용을 구분한다. 잔차 하락과 국내 USDT 원화가격 하락도 구분한다.
