# 어떤 선행연구의 어떤 평가를 참고했는가

## 1. 현재 설명력이 낮아도 최종 성과는 더 좋을 수 있는가

**가능하며 실제 예가 있다.** Gu, Kelly & Xiu (2021), *Autoencoder asset pricing models*, Journal of Econometrics의 출판사 본문에서 3요인 모형의 표본 외 total R²는 AE 12.6%, IPCA 13.3%인데 predictive R²는 AE 0.50%, IPCA 0.23%라고 보고한다. 같은 주식 수익률을 얼마나 설명하는가와 미래 수익률을 얼마나 예측하는가의 순위가 다르다. 예측에 기반한 포트폴리오와 pricing error도 평가한다. 따라서 첫 단계의 제곱오차만으로 ML을 자동 탈락시키는 것은 이 문헌의 평가 방식과도 맞지 않는다. 다만 주식의 조건부 beta 모형이므로 한국 USDT의 고유 성분을 정확히 분리했다는 직접 근거는 아니다. 우리 AE는 이 논문 전체를 복제한 모형도 아니다. [출판사 초록·서론·방법 및 결과 미리보기](https://www.sciencedirect.com/science/article/pii/S0304407620301998).

## 2. 암호화폐를 포함한 비선형 요인 연구

Spilak & Härdle (2022 **v1**), *Does non-linear factorization of financial returns help build better and stabler portfolios?*는 PCA/NMF/제약 AE와 암호화폐를 포함한 자산 배분을 검토한다. 재구성 오차뿐 아니라 월별 재학습 후 Sharpe, VaR/ES, 최대낙폭, 회전율, 분산투자와 꼬리위험을 평가한다. 비선형 AE가 선형 NMF의 포트폴리오 성과를 자동으로 넘어선 것은 아니고, 일부 성과는 짧은 검증기간 때문에 통계적 확신이 제한된다. 우리가 가져올 것은 **최종 목적의 평가와 안정성 검증**이다. 주가 수익률 포트폴리오 연구를 김프 분해의 직접 선행연구라고 부르지 않는다. 공개 PDF의 모형·표본 외 평가·결론을 확인했으며, 내용이 다른 후속 v2와 혼합하지 않는다. [v1 원문, 특히 §3.6](https://arxiv.org/pdf/2204.02757v1).

## 3. 최신 2025년 검증: 통계적 성능과 경제적 유용성은 별개

Nechvátalová (2025), *Autoencoder asset pricing models and economic restrictions — international evidence*, International Review of Financial Analysis는 미국/국제 주식에서 CAE의 total/predictive R²와 실제 제약하의 포트폴리오를 나눠 평가한다. 유동성 필터와 거래 비용이 성과에 큰 영향을 주며, 조건 없이 보인 수익성이 그대로 경제적 기여가 되지 않는다. 우리도 ‘잔차가 작다’와 ‘의미 있는 위험 지표다’를 나눠야 한다. 거래 비용·호가·실제 청산자료 없는 현재 분석에서 수익성이나 유동성 공급 효익을 입증했다고 말할 수 없다. [2025 출판사 초록·서론·결과 미리보기](https://www.sciencedirect.com/science/article/pii/S105752192500729X).

읽기 범위: 공개 2025 최종 PDF는 접근에 실패했다. 모형·시계열 분할·통계 평가 정의·비용 계산은 [저자 소속기관의 2024 working paper 원문](https://ies.fsv.cuni.cz/sites/default/files/uploads/files/wp_2024_26_nechvatalova.pdf)에서 추가 확인했다. 2024의 수치와 최종 2025 수치를 혼합하지 않는다. 프로토콜의 ‘et al.’은 오기이며 단독 저자다.

## 4. 최신 2026년 예: 요인의 경제적 근거도 별도 평가

Chen, Wang & Huang (2026), *Variational Autoencoder Asset Pricing models with economic restrictions*, International Review of Economics & Finance는 거시 상태와의 경제적 제약을 붙인 VAE를 제안한다. predictive R², IC, Sharpe 및 모형이 내포하는 pricing kernel과 거시 상태의 관계 등을 평가한다. 요인이 경제적으로 의미 있다는 주장은 이름을 붙이는 데서 끝나지 않고 외부 경제 변수와의 관계로 뒷받침한다는 점이 관련된다. 이 논문은 미국 주식이며 한국 김프를 다루지 않는다. 출판사에서 검색 가능한 초록·방법·추가분석을 확인한 범위로 기록하고 전체 원문을 읽었다고 표시하지 않는다. [출판사](https://www.sciencedirect.com/science/article/pii/S1059056026005150), [DOI](https://doi.org/10.1016/j.iref.2026.105402).

## 5. 한국 김프·스테이블코인 연구에서 가져올 기준

Seo, Koo & Yang (2024), *Nonlinear dynamics of Kimchi premium*, Economic Modelling은 임계 자기회귀를 통해 프리미엄 크기에 따라 조정 양상이 달라지는지 검토하고 거래 마찰과 연결한다. ‘비선형 현상이 있을 수 있다’는 근거이지만 **USDT와 공통 김프의 동시점 관계가 비선형이며 AE 분해가 더 낫다**는 증거가 아니다. [공개 원문](https://researchmgt.monash.edu/ws/portalfiles/portal/590976616/586624035_oa.pdf).

Kang et al. (2025), *Stablecoin and cross-border crypto market integration*, Economics Letters는 BTC/USDT 프리미엄의 장기 관계와 오차수정 속도를 검토한다. 경제적 가설이 맞으면 어떤 관계와 조정 속도가 관찰되어야 하는지를 검사한다는 점이 중요하다. 우리도 분해 후 하방 반응·지속성·LS의 추가 정보를 동일한 표본과 시차로 검증해야 한다. 실제 자금 이동의 인과경로를 직접 관측한 것과는 구분한다. 전체 PDF를 확보하지 못했으므로 출판사 초록·서론·분석 미리보기 확인 범위다. [출판사](https://www.sciencedirect.com/science/article/pii/S0165176525005415).

## 우리 평가에 적용한 결론

1. 설명력 비교는 필요한 진단이다. 진짜 고유 성분의 정답 검증은 아니다.
2. 첫 단계 RMSE가 더 큰 ML도 같은 미래 관측 목표에서 더 유용할 수 있으므로 직접 확인한다.
3. 잔차별 목표가 달라질 때는 원시 예측손실을 직접 비교하지 않는다.
4. 원하는 계수 부호나 작은 p값으로 분해 방법을 선택하지 않는다.
5. 같은 목표의 예측 개선, 재추정에도 유지되는 관계, 요인 해석 근거를 구분해 보고하며 어느 하나도 자동으로 인과 식별을 제공하지 않는다.

‘모든 관련 문헌을 빠짐없이 읽었다’거나 ‘위 모든 모형을 구현했다’는 뜻이 아니다. 실제 구현 범위는 A4/A5 프로토콜과 실행 기록에 한정된다.
