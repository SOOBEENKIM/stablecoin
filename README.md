# 한국 USDT 잔차 연구 — 원고 재현과 ML 동학 확장

현재 브랜치는 `research/icaif2026-manuscript-development`입니다. 국내 원고 재현본을 보존하고, 수정된 가격·시간 계산에 기반한 **잔차의 미래 하방 예측**으로 연구를 확장했습니다.

**2026-09-26 후속 안정성·3월 진단 완료:** 같은 509개 예측 시점에서 12시간 ML 선택의 선형 대비 손실 개선은 EQ **14.89%**, 시총가중 대용 CAP **10.23%**, PCA1 **14.90%**였고, 문턱형 대비는 각각 **10.29%, 10.27%, 10.34%**였습니다. 개선 방향은 세 정의에서 유지됐지만 CAP의 다중비교 결과와 포지셔닝의 추가 기여는 불확실합니다. EQ/PCA의 3월에는 보정이 ML도 개선했으나 비교모형을 더 크게 개선해 상대 우위가 역전됐습니다. CAP은 원예측부터 선형보다 불리해 같은 설명을 일반화할 수 없습니다. 아래 이전 실험은 보존하며, 이번 결과 역시 같은 과거 자료를 재사용한 탐색적 검증입니다.

[최신 결론과 3월 해석](experiments/residual_12h_robustness/CONCLUSION_KO.md) · [전체 수치표](experiments/residual_12h_robustness/RESULTS_KO.md) · [실행 전 설계](experiments/residual_12h_robustness/PROTOCOL_KO.md)

**2026-09-26 진단·보완 실험 완료:** 기존 결과를 보존하고 잔차 자체의 최근 변동 폭과 실제1·6·12시간 예측을 교차 비교했습니다. 12시간의 전체 입력 QRF는 같은 입력의 선형·문턱형 대비 개선을 보였고, 결과 개봉 후 고정 설정의 난수 민감도를 추가한 네 seed 평균 손실 개선은 각각 **14.89%, 10.29%**였습니다. 자체 변동 폭 입력은1·6시간 ML 선택에 작은 개선을 줬으나 일부는 모델 선택 변화와 연결됐습니다. **3월 보정 후 성능 악화와 포지셔닝의 추가 기여는 미해결**입니다. 같은 과거 자료의 탐색적 후속이며, 기존 주1시간 분석을 대체하거나 경제적 메커니즘을 입증했다고 주장하지 않습니다.

[새 결과와 해석](experiments/residual_horizon_scale/RESULTS_KO.md) · [설계·검증 기록](experiments/residual_horizon_scale/README.md) · [난수 민감도](experiments/residual_horizon_scale/SEED_SENSITIVITY_KO.md)

**2026-09-25 교차 비교 실행 완료:** 고정한 R/B/F 입력 조건과 선형/QRF/부스팅/문턱형을 모두 실행했습니다. 주 EQ의 전체 입력 ML 선택은 선형보다 손실이 **3.85% 낮았지만**,6개 주 비교의 고정 판단 기준은 모두 충족하지 못했습니다. 시장 정보 추가는−0.45%, 포지셔닝 추가는+0.25%로 뚜렷하지 않았습니다. 이전10.65%와 비교하면 ML의 보정 후 손실은 비슷하고, 새 선택·보정 규칙에서 선형 손실이 낮아져 차이가 줄었습니다. **ML 우위나 포지셔닝의 경제적 채널을 확립했다는 주장은 아직 뒷받침되지 않습니다.** 전체 후보·보조 결과·검증을 공개하며 기존 수치는 보존합니다.

[교차 비교 결과와 이전 수치와의 차이](experiments/residual_factorial_execution/RESULTS_KO.md) · [실행·검증](experiments/residual_factorial_execution/README.md) · [주6개 비교](experiments/residual_factorial_execution/results/primary_tests.csv)

**2026-09-25 교차 비교 설계 기록:** R(잔차·달력), B(+시장 정보), F(+포지셔닝) 각각에 선형/QRF/분위수 부스팅/문턱형을 적용하도록 설계를 구체화했습니다. 후보마다 자기 과거 오차로 보정하고, 최종 평가와 같은 **보정 후 손실**로 과거3개월에서 설정을 선택합니다. 동일 입력의 알고리즘 비교와 정보 추가 비교, 문턱 기준선 비교를 포함한6개 주 비교를 고정했습니다. 이 설계 커밋 당시에는 합성 규칙 검사·자료 가용성만 점검했고 성능 실험은 아직 실행하지 않았습니다. 이후 실행은 위의 별도 폴더에 기록했습니다.

[교차 실험 설계](experiments/residual_factorial_design/PROTOCOL_KO.md) · [설정·점검 안내](experiments/residual_factorial_design/README.md)

**2026-09-25 2단계 후 감사:** 기존 평가 함수를 사용하지 않는 별도 계산으로 56개 손실 요약·49개 보정 계열·주 검정을 대조했고, 확인 범위에서 수치 오류를 발견하지 못했습니다. 다만 한 그룹씩 제거하는 실험만으로 전체 개선의 출처를 분리할 수 없고, 현재 선택 기준은 보정 전 손실인 반면 주 평가는 보정 후 손실이라는 설계 보완점이 있습니다. **3단계로 곧바로 넘어가기보다 동일 입력의 선형/비선형 비교와 입력 묶음 추가 비교를 먼저 보완할 것을 권고합니다.** 해당 보완 실험과 3·4단계는 아직 실행하지 않았습니다.

[계산 감사와 설계 보완 제안](experiments/residual_step2_audit/README.md)

**2026-09-25 후속 2단계 — 변수군 제거·재학습:** 현재 잔차와 직전 변화를 유지하고 글로벌 BTC 시장·국내 USDT 거래량·환율과 거시 입력을 각각 제거했습니다. 과거 세 내부 월에서 같은 84개 후보를 다시 선택한 주 EQ 분석의 손실 증가는 **0.27%, 0.31%, 0.80%**였고, 세 비교 모두 불확실성 구간에 0이 포함됐습니다. **기존 예측 개선은 유지되지만, 그 개선을 특정 정보군의 추가 기여로 설명할 근거는 아직 부족합니다.** PCA의 긍정적 보조 결과와 설정 유지 시 일부 반대 결과도 함께 공개했습니다. 이번 후속의 3단계 경보·4단계 가격 경로는 실행하지 않았습니다.

[2단계 결과](experiments/residual_step2_information/RESULTS_KO.md) · [실행 전 설계](experiments/residual_step2_information/PROTOCOL_KO.md) · [코드·검증 안내](experiments/residual_step2_information/README.md)

**2026-09-25 후속 1단계 — 단순 동학 비교:** 기존 ML 예측을 유지하고 지속성·잔차 수준 조정·조정과 시장 변동성을 반영한 비교모형을 추가했습니다. 주 EQ에서 ML의 보정 후 하방 예측손실은 각각 **19.10%, 11.35%, 7.81% 낮았고**, 세 비교가 고정한 판정 기준을 통과했습니다. 문턱형 대비 우위는 여전히 불확실했으며 조정·변동성 대비 개선도 CAP/strict 조건과 미보정 결과에서는 통계적으로 뚜렷하지 않았습니다. **주 분석의 개선은 단순 지속성만으로 설명되지 않지만, 모든 정의·표본에서의 우위나 경제적 원인까지 확인한 것은 아닙니다.** 기존 선형 대비 10.65% 결과는 그대로이며, 새 개선율과 더하지 않습니다. 2단계 변수군 분석, 3단계 경보, 4단계 가격 경로는 이번 후속에서 실행하지 않았습니다.

[1단계 결과](experiments/residual_step1_dynamics/RESULTS_KO.md) · [실행 전 설계](experiments/residual_step1_dynamics/PROTOCOL_KO.md) · [코드·검증 안내](experiments/residual_step1_dynamics/README.md)

**2026-09-25 잠금·중첩 시간순 검증:** 추가 수집 없이 기존 자료로 모델 선택 절차를 다시 관리했습니다. 실행 전 설계·소스·입력을 커밋 `519682a`로 고정하고, 과거 세 달에서 선형을 포함한 84개 후보를 선택한 뒤 이어지는 월을 평가했습니다. 주 EQ에서 선택 절차의 선형 대비 손실 개선은 **10.65%[7.14, 14.00]**, 세 주 비교의 Holm p=0.0015였습니다. 문턱 대비 우위와 포지셔닝 증분은 주 기준을 통과하지 못했습니다. **이번 실행의 외부 성적에 의한 모델 선택은 차단했지만, 이미 사용한 기간이므로 독립된 새 데이터 검증은 아닙니다.**

[중첩 검증 결과](experiments/residual_nested_validation/RESULTS_KO.md) · [실행 전 고정 설계](experiments/residual_nested_validation/PROTOCOL_KO.md) · [실행·변경 기록](experiments/residual_nested_validation/README.md)

**2026-09-25 경제적 해석 검증:** 변수군 제거 재학습, 문턱형 분위수 기준선, 유사한 출발 상태의 실제 1·6·12시간 가격 경로를 비교했습니다. 주 EQ에서 네 난수 평균 QRF 손실은 선형보다 **9.10% 낮았지만**, 롱숏 정보의 추가 개선은 **0.94%[−0.13, 2.06]**로 불확실했습니다. 펀딩·OI의 추가 개선은 확인되지 않았습니다. 문턱 기준선 대비 평균 개선은 4.84%였으나 다중 비교와 엄격한 표본까지 고려한 우위는 확증되지 않았습니다. 실제 조정 경로도 포지셔닝 메커니즘을 뒷받침하지 못했습니다. **현재 결론은 이 표본의 하방 예측 개선이며, 레버리지의 경제적 원인까지 입증한 것은 아닙니다.**

[연구 결론](experiments/residual_economic_validation/CONCLUSION_KO.md) · [전체 결과](experiments/residual_economic_validation/RESULTS_KO.md) · [설계](experiments/residual_economic_validation/PROTOCOL_KO.md) · [실행·검증](experiments/residual_economic_validation/README.md)

**2026-09-25 후속 실험:** 학습 창·수준/변화량 표현·과거 오차 보정을 비교하고 Quantile Regression Forest(QRF)를 추가했습니다. 확장 표본의 주 EQ 분석에서 같은 조건의 선형 대비 QRF의 예측손실이 **9.63% 감소**, 네 평가 월 모두 개선됐습니다. 난수 네 개에서는 8.68~9.63% 개선됐습니다. 포지셔닝 수준과 변화량의 추가 효과는 주 실행 1.32%, 난수별 0.15~1.32%로 작았고, 보정 여부와 잔차 정의에 민감했습니다. 같은 과거 기간을 재사용한 탐색적 후속으로, 원래 경제적 메커니즘의 입증이나 독립 외부 검증으로 주장하지 않습니다.

[후속 결과와 해석](experiments/residual_dynamics_adaptive/RESULTS_KO.md) · [후속 설계](experiments/residual_dynamics_adaptive/PROTOCOL_KO.md) · [실행 안내](experiments/residual_dynamics_adaptive/README.md)

**2026-09-25 ML 첫 실험:** 선형·상호작용·스플라인·분위수 부스팅을 EQ/CAP/PCA 잔차와 같은 미래 평가 구간에서 비교했습니다. 주 분석에서 부스팅의 예측손실은 선형보다 12.37% 컸고, 포지셔닝 추가 효과는 확인되지 않았습니다. 이 첫 실험은 당시 제안한 기여를 뒷받침하지 못했습니다.

[ML 실험 결과와 한계](experiments/residual_dynamics_ml/RESULTS_KO.md) · [사전에 정한 설계](experiments/residual_dynamics_ml/PROTOCOL_KO.md) · [실행 안내](experiments/residual_dynamics_ml/README.md)

## 보존된 국내 원고 재현 기준선

현재 연구 브랜치에 추가했던 ML 분해 실험(`experiments/nonlinear_residual/`)은 삭제했습니다. 원본 데이터·원고·코드는 보존하고, 새로 실행한 결과와 원고의 일치·불일치를 함께 정리했습니다.

[재현 결과](experiments/manuscript_reproduction/RESULTS_KO.md) · [실행 안내](experiments/manuscript_reproduction/README.md) · [원고 PDF](research/reference/submitted/경영과학학술지_김수빈_V2_stablecoin.pdf)

[워크스테이션 폴더 직접 대조](experiments/manuscript_reproduction/WORKSTATION_SOURCE_CHECK_KO.md) 결과, 실행한 원본 코드·전처리 데이터는 실제 `stablecoin_v4`와 바이트 단위로 같다. 첨부 최종 원고는 v4 폴더의 초안보다 나중에 편집된 문서다.

## 재현 범위와 결과

보존된 715개 관측치의 전처리 데이터에서 다음 원본 분석을 재실행했습니다.

1. 시장 공통 김치프리미엄 회귀 → 1차 잔차 → 글로벌 괴리 회귀 → 최종 잔차.
2. 분위수 회귀, 기간별 반응, 수요·레버리지 변수 비교, M0/M1/M2 통제 분석.
3. EQ·시총가중·PCA 정의 비교, 스트레스일 제외·기간 분할, 경제적 크기 환산.

**최종 잔차는 최대 오차 약 1.13×10⁻¹⁶로 재계산됐고, 핵심 분위수 회귀표(원고 표 3)는 전부 일치했습니다.** 원본 기본값 B=300에서 표 1–6의 숫자 185개 중 173개가 원고의 표시 자릿수와 일치합니다. 표본 분할표는 별도 B=200 실행에서 모두 일치합니다.

원고 전체를 완벽히 재현한 상태는 아닙니다. 일부 p값·CAP 계수와 원고 그림의 신뢰구간·오차막대는 보존된 코드로 동일하게 생성되지 않습니다. [표별 비교와 남은 차이](experiments/manuscript_reproduction/RESULTS_KO.md)를 확인할 수 있습니다. 원자료 전처리를 처음부터 재실행한 결과로 주장하지 않습니다.

**2026-09-25 원인 추적:** 기본값과 다른 12개 숫자 중 10개는 회귀 반복 횟수와 표별 부트스트랩 횟수 차이로 재현 확인했습니다. 표 5는 회귀 반복 60회·B=150, 표 6은 B=200에서 원고와 일치합니다. 나머지 2개의 작성 경위는 미확인입니다. [원인별 증거와 통제 실험](experiments/manuscript_reproduction/difference_audit/RESULTS_KO.md)을 보존했습니다.

## 실행

저장소 루트에서 [기록된 Python 환경](experiments/manuscript_reproduction/requirements.txt)을 사용합니다.

```bash
python3 experiments/manuscript_reproduction/run_reproduction.py
```

원본 코드 5개를 수정 없이 복사·실행하고, CSV·그림·로그·입출력 해시와 원고 대조표를 생성합니다. B=300 기본 실행과 과거 CSV를 대조하는 B=40·200 실행을 구분해 보관합니다.

## 브랜치와 자료 구분

- `research/icaif2026-nonlinear-residual`: ML 추가분을 제거하고 이번 원고 재현 결과를 기록하는 기존 연구 브랜치.
- `research/icaif2026-manuscript-development`: 원고 수치 불일치의 원인 추적과, 회귀 잔차 이후의 ML 동학 실험을 기록한 현재 개발 브랜치.
- `research/reference/original_v4/`: 수정하지 않은 기존 코드와 전처리 데이터.
- `experiments/manuscript_reproduction/`: 이번에 추가한 재현 실행기와 새 실행 결과.
- `experiments/residual_dynamics_ml/`: 가격·시간 수정, 학습 구간별 회귀 잔차, 미래 분위수 예측 및 포지셔닝 증분 비교. 과거 ML 분해나 `main` 호가 변형 연구와 다른 실험.
- `experiments/residual_dynamics_adaptive/`: 첫 결과 이후 학습 표현·QRF·시간순 보정·정보 가용성·포지셔닝 변화량을 점검한 후속. 최초 실험은 보존.
- `experiments/residual_economic_validation/`: 변수군 제거, 문턱형 기준선, 출발 조건 매칭과 실제 가격 경로로 예측 개선의 경제적 해석을 검증한 후속.
- `experiments/residual_nested_validation/`: 기존 자료만으로 모델 계열·설정 선택을 과거 세 내부 월에 한정하고, 실행 전 잠금과 평가 개봉 기록을 남긴 중첩 시간순 검증.
- `experiments/residual_step1_dynamics/`: 기존 ML 예측을 보존하고 단순 지속성·잔차 수준 조정·시장 변동성 모형을 추가한 1단계 비교. 2단계 결과는 다음 폴더에 별도로 기록.
- `experiments/residual_step2_information/`: 현재 잔차·직전 변화·표본을 유지한 세 정보군 제거 실험. 과거 내부 자료의 재선택과 기존 설정 유지 재학습을 비교하며 3·4단계는 미실행.
- `experiments/residual_step2_audit/`: 별도 계산 구현으로 저장된 2단계 결과를 감사하고, 정보 출처를 설명하기 위해 필요한 후속 비교를 정리. 새로운 예측 성능 실험은 아님.
- `experiments/residual_factorial_design/`: 감사 후 동일 입력의 알고리즘 비교·정보 묶음 추가·후보별 보정/선택을 구체화한 설계. 자료 가용성과 합성 규칙만 점검했으며 새 성능 실험은 미실행.
- `experiments/residual_factorial_execution/`: 해당 고정 설계의 실행기·독립 산술 감사·실제 재적합·전체 후보 예측·주/보조 결과. 기존 설계 폴더를 수정하지 않고 실행·해석을 별도로 기록.
- `experiments/residual_horizon_scale/`: factorial 이후 자체 잔차 변동 폭과 실제1·6·12시간의 제한된 개발 실험. 원본 대비 추가 입력 효과, 선형/ML/문턱형, 조건별 위험 경계와 포지셔닝 증분을 함께 공개. 검증기 동점 수정과12시간 F의 후속 난수 민감도도 별도 잠금·기록.
- `experiments/residual_12h_robustness/`: 동일한 12시간 설계를 EQ/CAP/PCA1 잔차에 적용한 안정성 검증과 3월 진단. 보정 전후 상대 손실의 정확한 분해, 부분 월·날짜 민감도, 상태 분포와 후보별 과거 오차를 공개하며 경제적 인과와 구분.
- `research/`, `docs/`, `context/`의 나머지 파일: `main`에서 물려받은 이전 연구 기록. 이번 원고 재현 결과와 구분합니다.

`main`의 호가·AI 연구는 별도 방향의 과거 기록입니다. 해당 안내는 [기존 연구 흐름](docs/STUDY_GUIDE_KO.md)에 남아 있습니다. 삭제한 ML 분해 실험의 이전 상태는 Git 커밋 `b051620e86bfac07efc90d6b7a257c0ba9d86d1d`에서 확인할 수 있습니다.
