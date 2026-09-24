# 한국 USDT 잔차 연구 — 원고 재현과 ML 동학 확장

현재 브랜치는 `research/icaif2026-manuscript-development`입니다. 국내 원고 재현본을 보존하고, 수정된 가격·시간 계산에 기반한 **잔차의 미래 하방 예측**으로 연구를 확장했습니다.

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
- `research/`, `docs/`, `context/`의 나머지 파일: `main`에서 물려받은 이전 연구 기록. 이번 원고 재현 결과와 구분합니다.

`main`의 호가·AI 연구는 별도 방향의 과거 기록입니다. 해당 안내는 [기존 연구 흐름](docs/STUDY_GUIDE_KO.md)에 남아 있습니다. 삭제한 ML 분해 실험의 이전 상태는 Git 커밋 `b051620e86bfac07efc90d6b7a257c0ba9d86d1d`에서 확인할 수 있습니다.
