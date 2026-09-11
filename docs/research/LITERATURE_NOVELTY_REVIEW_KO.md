> GitHub 탐색용 사본입니다. [체크섬이 보존된 원본](../../research/LITERATURE_NOVELTY_REVIEW_KO.md)의 내용과 과거 시점 표기를 유지하고 문서 링크만 변환했습니다. 실행은 [현재 재현 안내](../REPRODUCING.md)를 따릅니다.

# 워크숍 확장안의 선행연구 중복 검토

> 최신 후속 실험: 2026-09-09 첫 예측 실험에서 ML 우위와 포지셔닝 정보의 시험기간 개선은 확인되지 않았다. 가까운 선행연구에 대한 아래 검토와 함께 [실제 예측 결과·워크숍 방향 재판정](pilot_20260909/RESULTS_KO.md)을 참조한다.

> 후속 업데이트(2026-09-09): 이 문서 이후 실제 가격·시간축 민감도 재계산과 추가 문헌 검토를 완료했다. 기존의 강한 롱숏 결과와 6~12시간 증폭은 유지된다고 볼 수 없다. 최신 판단과 상세 설계는 [종합 재평가 보고서](reanalysis_20260909/RESEARCH_REASSESSMENT_KO.md)를 기준으로 한다. 아래는 앞선 검토 시점의 기록이다.

검토일: 2026-09-09. 대상은 기존 국내 원고와 별도로 제안한 ‘한국 USDT 상대 가격 괴리의 상태별 꼬리위험 예측’ 확장안이다. 분석 코드·데이터·제출 원고는 변경하지 않았다.

**판정: 가까운 선행연구가 여러 편 있다. ‘AI로 스테이블코인 꼬리위험을 더 잘 예측한다’는 일반적 주장만으로는 독창성이 충분하지 않다. 이번 검색에서 한국 USDT의 공통요인 조정 후 괴리, 파생상품 포지셔닝·체결 압력, 시간 단위 표본 외 분위수 예측을 모두 함께 다룬 동일 연구는 확인하지 못했다. 그러나 미발견은 부재의 증명이 아니며, 이 조합 자체가 충분한 기여라는 보장도 없다.**

## 확인 범위

영문·국문으로 Korean USDT/stablecoin premium, kimchi premium, residual, positioning/long-short, order flow/imbalance, quantile/tail forecasting, machine learning 등을 조합해 검색했다. 출판사·저자 소속 대학·학회·NBER·저자 공개 저장소를 근거로 사용했다. 검색 결과의 수집일을 게재일로 해석하지 않았다.

아래 ‘공개 본문 일부’는 출판사 검색에 노출된 초록·소개·절별 발췌를 읽었다는 뜻이다. 구독 제한 논문의 전체 본문·부록·실험 코드를 검증한 것은 아니다. 검색으로 찾은 연구의 보고 결과를 요약한 것이며, 그 결과를 독립 재현한 것은 아니다.

## 직접 비교해야 할 문헌

| 연구 | 확인된 내용 | 제안과 겹치는 지점 및 차이 | 확인 수준 |
|---|---|---|---|
| [Stablecoin risk – a hybrid Copula-GARCH–QT framework for early warning, tail quantiles, and co-depeg dynamics](https://www.sciencedirect.com/science/article/pii/S1062940826000768), Ming Che Lee, 2026, North American Journal of Economics and Finance, DOI 10.1016/j.najef.2026.102654 | 여러 USD 스테이블코인의 달러 페그 위험을 Q-Transformer와 Copula-GARCH로 분석. 다음 날 꼬리 분위수를 예측하고 기존 모형들과 비교하며 스트레스 국면·변수 기여를 해석한다. | **AI + 스테이블코인 꼬리위험 + 국면별 해석은 이미 연구됨.** 확인된 대상은 달러 페그 위험으로, 제안한 한국 원화시장의 상대 가격 괴리와는 구별된다. | 출판사 초록·공개 본문 일부 |
| [Stablecoin and cross-border crypto market integration](https://www.sciencedirect.com/science/article/pii/S0165176525005415), 2025, Economics Letters 257, 112704, DOI 10.1016/j.econlet.2025.112704 | 2025년 1~8월 한국 BTC·USDT 프리미엄을 10분 자료와 VECM으로 분석. 두 프리미엄의 장기 관계와 이탈 후 조정을 보고한다. | **한국 USDT와 BTC 공통 프리미엄·상대 괴리·조정이라는 금융 질문이 매우 가깝다.** 다섯 코인으로 기준을 바꾸는 것만으로 충분한 차별화라고 보기 어렵다. | 출판사 초록·공개 본문 일부 |
| [Nonlinear dynamics of Kimchi premium](https://researchmgt.monash.edu/ws/portalfiles/portal/590976616/586624035_oa.pdf), Myung Hwan Seo·Bonsoo Koo·Yangzhuoran Fin Yang, 2024, Economic Modelling 135, 106726 | BTC·ETH 김치프리미엄의 문턱회귀. 괴리 크기와 거래 마찰에 따라 조정이 달라지는 비선형 동학을 분석한다. | **김치프리미엄의 상태별 비선형성 자체도 새롭지 않다.** 선형 모형에만 이기는 비교로는 충분하지 않을 수 있다. 여기서 언급하는 leverage effect를 선물 계정 롱숏비율과 동일시하면 안 된다. | 대학 저장소의 출판 PDF, 관련 본문·모형 확인 |
| [The capital-access premium and the won: Korean stablecoin premia as a leading indicator of exchange-rate pressure](https://www.sciencedirect.com/science/article/abs/pii/S0165176526003484), 2026, Economics Letters 268, 113152 | 한국 스테이블코인 프리미엄을 원화 환율 압력과 연결하고 단기 예측력·위험 국면에 따른 차이를 보고한다. | **한국 스테이블코인 프리미엄의 예측 정보와 위험 국면이라는 설명이 겹친다.** 예측 대상이 원화 환율이라는 점은 USDT 잔차 자체를 예측하는 제안과 다르다. | 출판사 초록·공개 본문 일부 |
| [국내 거래소 신규 상장 알트코인의 단기 급등 및 프리미엄 예측: 해외 시세와 소셜 신호를 활용한 기계학습 기반 분석](https://journal.kci.go.kr/jksci/archive/articlePdf?artiId=ART003305943), Eun Hong Park·Yeong-In Lee·Ha Young Kim, 2026, 한국컴퓨터정보학회논문지 31(2), 87–96 | 해외 시세·소셜 신호로 업비트 신규 상장 알트코인의 급등과 프리미엄 발생을 예측한다. | **한국 거래소 프리미엄 + 해외 정보 + ML 예측도 이미 존재한다.** 신규 상장 이벤트의 분류가 중심이므로 USDT 상대 괴리의 연속적 꼬리 예측과는 다르다. | 학회 PDF 초록·관련 방법 부분 확인 |
| [Leverage and Stablecoin Pegs](https://www.nber.org/papers/w30796), Gorton 외, NBER 2022; 최종 JFQA 2026, 61(1), 99–136 | 레버리지 거래자의 스테이블코인 차입 수요, 대여 수익과 페그 유지의 관계를 다룬다. | **레버리지와 스테이블코인 가격을 연결하는 이론 자체는 선행연구다.** 한국의 상대 괴리에 대한 추가 예측 정보를 검증하는 것과 구별해야 한다. | NBER 초록·최종 게재정보 |

## 추가로 추적할 자료

- [Market Microstructure of Stablecoins: Evidence from USDT](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6778618), Baulkaran·Jain·Sobanski, 2026 SSRN 작업논문: 여러 거래소의 10분 자료로 시간대별 유동성·시장 품질·페그 조정을 다룬다. 초록 확인. ‘USDT에 유동성·시간대 효과가 있다’만으로는 차별화하기 어렵다.
- [Tether Premiums and Exchange Rates: Evidence from South Korea](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5169056), JungJae Park, 2025 SSRN 작업논문: 한국 테더 프리미엄과 다음 날 원화 환율의 관계. 초록 확인. 위 2026년 Economics Letters 논문과 동일 연구라고 단정하지 않는다.
- [Onchain Insights](https://github.com/QuantLet/onchain-insights), Owen Chaffard, 2026으로 표기된 공개 연구 프로젝트: Uniswap USDC-USDT 유동성 자료를 이용한 트리 기반 디페그 조기경보와 분위수 함수·신경망 예측 관련 코드 설명이 있다. README 확인이며 학술지 게재 상태는 미확인이다. 유동성 정보 + ML 조기경보라는 구현도 이미 공개되어 있다.
- [김치프리미엄과 환율 변동 예측을 활용한 가상화폐의 통계적 차익거래 연구](https://www.kais99.org/jkais/journal/Vol24No10/vol24no10p041.pdf), 조기정·박종현·안현철, 2023: Upbit·Binance BTC, 환율 예측 LSTM과 통계적 차익거래. 학회 PDF 초록·방법 부분 확인. 국내외 가격 차이에 딥러닝을 결합하는 것 자체는 새롭지 않다.
- [Southampton 학위논문 PDF 검색 결과](https://eprints.soton.ac.uk/504578/1/Final_Thesis_for_archive.pdf)에서 ‘Stablecoin Mispricing: Cross-Exchanges Arbitrage’ 장과 스프레드·깊이·주문 불균형 관련 서술을 찾았다. 원문 접근은 403으로 실패했으므로 저자·연도·모형·결과 대조 전까지 직접 중복 판정의 근거로 사용하지 않는다.

## 기존 제안에서 바뀌어야 할 점

1. 일반적인 ‘AI가 스테이블코인 꼬리위험 예측을 개선한다’는 문장은 독창성 주장에서 제외한다. 이는 가능한 실험 결과의 요약이며 이미 가까운 문헌이 있다.
2. 후보 B의 직접·간접 KRW/USDT 가격 차이는 측정 후보로 남기되 그 공식이나 상대 괴리 개념 자체가 새롭다고 주장하지 않는다. 원래 OLS 잔차와도 다르다.
3. 비교 기준을 보강한다. 지속성·선형 분위수회귀 외에 상대 프리미엄의 오차수정 구조와 문턱·국면별 조정을 반영한 비교가 필요하다. 각 모형은 같은 목표·정보 시점·예측시계·분위수 손실로 공정하게 맞춘다. 평균 예측 VECM과 분위수 모형의 손실을 그대로 비교하지 않는다.
4. 금융 질문을 ‘이미 알려진 공통 프리미엄·조정 특성을 고려한 뒤에도 포지셔닝/체결 정보가 한국 USDT 상대 괴리의 미래 꼬리에 추가 정보를 주는가’로 좁힌다. 단순히 한국 자료나 LightGBM을 썼다는 이유를 기여로 삼지 않는다.
5. 국내 USDT 체결 압력과 글로벌 현물·선물 지표를 구분한다. 국내 체결 자료가 확보되지 않으면 글로벌 지표의 정보 가치까지만 주장하며 국내 유동성·수요 메커니즘을 입증한 것으로 쓰지 않는다.
6. 오류 수정 후 결론이 유지되는지 먼저 확인한다. 외생변수의 시간대·정보 이용 가능 시각을 해결하지 않은 상태의 예측 개선은 기여의 증거가 아니다. 잔차 생성부터 학습 기간 안에서 수행해야 한다.

## 현재 조건에서 검토할 연구 질문

> 한국 USDT와 시장 공통 프리미엄 사이의 조정 특성을 고려한 뒤에도, 글로벌 파생상품 포지셔닝은 향후 상대 괴리의 꼬리위험에 추가 예측 정보를 제공하는가? 그 정보는 시장 상태에 따라 달라지는가?

이는 신규성이 확정된 최종 질문이 아니라, 위 문헌과 대조하면서 검증할 후보이다. 같은 기간의 국내 체결 자료를 확보하면 ‘그 정보가 국내 체결 압력을 고려한 뒤에도 남는가’를 추가할 수 있다. 수집 기간 연장이나 과거 호가 자료 확보를 필수 전제로 하지 않는다.

모형·시계·변수 선택을 통해 원하는 유의성이나 성능 우위를 찾는 것이 목적이 아니다. 알려진 조정 구조와 강한 기준선으로 설명되지 않는 추가 정보가 있고, 이후 기간에서도 확인되며, 금융적 해석이 분명할 때 확장 기여를 주장할 수 있다. 정보가 남지 않거나 성능 개선이 불안정하면 제안의 매력이 약해지며 제출 방향을 재평가해야 한다. 이번 검색만으로 제출 가능·불가능 또는 채택 여부를 확정하지 않는다.
