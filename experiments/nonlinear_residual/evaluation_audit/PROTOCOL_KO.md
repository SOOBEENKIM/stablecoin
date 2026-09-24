# A5: 잔차 크기, 분리의 타당성, 후속 연구 유용성의 구분

이 문서는 A5 결과 생성 전에 고정한다. A1–A4 결과를 이미 본 뒤 설계한 **탐색적 추가 분석**이며 새로운 독립 검증 표본은 아니다. A4의 고정 Ridge(alpha=1) readout 결과를 재사용한다. 원자료와 기존 결과를 덮어쓰지 않는다.

## 무엇을 비교하는가

`e_t = y_t - f_hat(X_t)`이고 기존 RMSE는 `sqrt(mean(e_t²))`이다. 진짜 USDT 고유 성분은 관측되지 않으므로 이는 **시점 t 프리미엄 설명오차/잔차 RMS**이며 고유 성분 추정오차가 아니다. 과거 자료로 학습했어도 현재 X_t를 쓰므로 미래 예측오차라고 부르지 않는다. AE의 다섯 기준 코인 재구성 MSE도 별개의 목표다.

주 비교: EQ5 순차 OLS, PCA1, PCA2, 과거 검증자료로 선택한 AE. 추가 진단: 같은 m,g를 동시에 넣은 joint OLS. 순차 회귀는 두 설명변수가 상관되어 있으면 첫 회귀의 직교성이 유지되지 않을 수 있다. joint OLS는 이 차이를 점검하는 통제군이다. A4에서 다른 모델을 추가 선택하지 않는다.

## 공통 평가 설계

- 다섯 시간 정렬 시나리오, 주 평가 2026년 1–3월. 중복된 시나리오는 독립 증거로 세지 않는다.
- 각 월 이전 자료만 학습. 후속 예측은 목표 시각도 학습 마감 이전인 행만 사용한다. h=6은 정확히 6시간이다.
- 3일/7일 월별 층화 달력 블록 재표집, 고정 예측의 차이는 2,000회 구간. 시계열 중복과 결측일을 보존한다. 월별 결과도 제시한다.
- 여러 시도 후 남은 표본이므로 구간은 탐색적·선택 조건부이며 확증적 유의성이나 인과효과를 주장하지 않는다.

## 평가 1: 잔차에 남은 기준 자산 정보

월별 과거 out-of-sample 잔차를 목표로 probe를 학습하여 다음 월 잔차를 설명한다. 과거 잔차 평균을 기준으로 잔차 MSE 감소를 측정한다. 입력은 (a) m,g, (b) 다섯 코인 프리미엄,g. (a)는 평균 시장 지표, (b)는 코인 간 차이까지 포함하며 전부 공통 요인이라고 해석하지 않는다.

고정 probe: 표준화 Ridge(alpha=1), LightGBM(100 trees, depth=2, leaves=4, leaf minimum=100, learning rate=.03, lambda=1). 성능을 본 뒤 변경하지 않는다. 잔여 예측 가능성이 없다는 결과도 독립성/정확한 구조 분리를 입증하지 않는다.

## 평가 2: 같은 실제 결과에 대한 후속 예측

모든 분해 방식에서 동일한 관측 목표 `y(t+6h)-y(t)`를 사용한다. 기본 정보는 현재 y, 다섯 코인 프리미엄, g, downside, btc_vol, btc_ret. 여기에 각 잔차를 추가한 표현의 유용성을 본다. 같은 원자료의 함수이므로 새 정보가 생기는 것은 아니다. 레버리지 정보 추가(LS 및 downside×LS)도 각각 비교한다.

고정 예측기: 표준화 QuantileRegressor(alpha=.01)와 quantile LightGBM(100 trees, depth=3, leaves=7, leaf minimum=50, learning rate=.03, lambda=1). q=.1 및 .5. 기준은 과거 목표의 해당 분위수. Pinball loss, 기준 대비 skill, 하회율(목표 .1/.5), 월별 일관성, paired block 구간을 보고한다. 모형 복잡도와 선택 결과에 조건부인 비교이다.

잔차 자체의 h=6 예측도 같은 두 예측기 및 LS 추가/제외로 실시한다. 이 경우 모델마다 목표가 다르므로 **원시 pinball loss를 분해 모델 간 순위에 사용하지 않는다**. 각 잔차의 자체 과거 분위수 기준 skill 및 LS 추가의 개선 여부를 본다.

## 평가 3: 원래 연구의 관계와 불확실성

2026년 1–3월 M2 모형에서 h=6의 q=.1/.5, 같은 origin을 가진 h=1/6/12의 q=.1을 계산한다. downside, LS 및 현재 잔차의 계수와 q10−q50/h6−h1/h12−h1 대비를 보고한다. 레버리지 지표는 계좌 long/short 비율이며 실제 차입 규모/청산 흐름은 아니다.

7일 달력 블록 199회: 매 반복마다 과거 자료의 표준화/PCA/AE 인코더/USDT readout/g 회귀를 다시 학습하고 다음 시점 잔차를 다시 생성한다. AE는 A4에서 해당 월에 선택된 구조·규제와 세 seed를 유지한다. 후보 전체 재선택은 하지 않으므로 선택 불확실성 전체를 포함하지 않는다. 3일 블록 199회는 인코더를 고정한 후속 회귀 민감도로 별도 표기한다. 중복 UTC 정렬은 공통 core 분석을 재사용한다. 실패 반복/수렴 경고를 기록하고 불리한 결과를 숨기지 않는다.

## 판정 규칙

잔차 RMS 최저, 조건부 관계가 원하는 부호, p값 최소 중 어느 하나만으로 모델을 선택하지 않는다. 설명 성능, 요인 해석(특히 DOGE에 집중된 PC2), 잔여 의존성, 동일 관측 목표의 미래 성능, 후속 경제 관계의 안정성을 각각 판정한다. 지표가 엇갈리면 전체 승자를 선언하지 않는다. 동일 미래 목표에서 AE가 우세하더라도 진짜 고유 성분이나 인과 메커니즘을 식별했다는 결론은 별도 근거가 필요하다.

## 참고문헌 적용 범위

- Gu, Kelly & Xiu (2021), *Autoencoder asset pricing models*, Journal of Econometrics. 동시 설명력(total R²), 미래 예측력(predictive R²), 가격결정 오차를 구분한다. 주식 조건부 요인모형이며 한국 USDT 분해의 직접 복제가 아니다. DOI: https://doi.org/10.1016/j.jeconom.2020.07.009
- Nechvátalová et al. (2025), *Autoencoder asset pricing models and economic restrictions — international evidence*, International Review of Financial Analysis. 표본 외 total/predictive R²와 경제적 성과를 구분하는 최신 확장. https://doi.org/10.1016/j.irfa.2025.104642
- Spilak & Härdle (2022 **v1**), *Does non-linear factorization of financial returns help build better and stabler portfolios?* 재구성과 최종 포트폴리오/꼬리위험 성과를 함께 평가. 암호화폐를 포함하지만 김프 분해 논문은 아니다. 후속 v2는 제목·내용이 달라 버전을 고정한다. https://arxiv.org/pdf/2204.02757v1
- Seo, Koo & Yang (2024), *Nonlinear dynamics of Kimchi premium*. 임계 자기회귀와 비선형 검정/평균회귀를 다루며 ML 잔차 분리의 우수성을 보인 논문은 아니다. https://doi.org/10.1016/j.econmod.2024.106726
- Kang et al. (2025), *Stablecoin and cross-border crypto market integration*. BTC/USDT 프리미엄의 장기 관계와 조정 속도를 평가. 출판사 초록/본문 미리보기 확인 범위이며 전체 원문 확보로 표시하지 않는다. https://doi.org/10.1016/j.econlet.2025.112704
