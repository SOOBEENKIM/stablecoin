# 후속 실험의 근거와 차별화 범위

2026-09-26 공개 원문/출판사 공개 부분 재확인. 아래 비교는 확인 가능한 범위에 한정하며 모든 관련 논문을 완독했거나 세계 최초임을 확인했다는 뜻이 아니다.

| 연구 | 확인된 접근 | 이번 실험에서 구별할 질문 | 열람 범위 |
|---|---|---|---|
| Kang et al. (2025), [Stablecoin and cross-border crypto market integration](https://www.sciencedirect.com/science/article/pii/S0165176525005415), Economics Letters 257,112704 | 한국 BTC/USDT 프리미엄의 공적분과 VECM 조정. 10분 자료에서 빠른 정렬을 분석 | 평균 조정이 있다는 사실을 반복하기보다, 공통요인 조정 후 남은 미래 하방 변화의 조건부 위험과 경보 가치를 검증 | 출판사 초록·방법/결론 공개 발췌. 전체 유료 본문 확보 아님 |
| Seo, Koo & Yang (2024), [Nonlinear dynamics of Kimchi premium](https://doi.org/10.1016/j.econmod.2024.106726), Economic Modelling 135,106726 | 문턱 밖 평균회귀와 문턱 안 동학, 거래 마찰 | 비선형성 자체는 새롭지 않음. 문턱 분위수 및 단순 상태별 경험분위수와 비교해 ML의 추가 실용성을 확인 | 출판사 공개 초록·방법 발췌; 이번 세션 PDF 접근 실패. 기존 문헌 검토 기록도 참고 |
| Lee (2026), [Stablecoin risk – a hybrid Copula-GARCH–QT framework for early warning, tail quantiles, and co-depeg dynamics](https://www.sciencedirect.com/science/article/pii/S1062940826000768), NAJEF,102654 | 달러 페그 중심 Q-Transformer 꼬리 예측과 Copula–GARCH 의존성, 조기경보 | AI+스테이블코인 위험은 이미 존재. 지역 시장의 회귀 잔차 변화라는 다른 목표가 실제로 어떤 가격 움직임을 경고하는지 검증 | 출판사 초록·구조 공개 발췌. 모델을 그대로 복제하거나 직접 우열 검정한 것은 아님 |
| Ehm, Gneiting, Jordan & Krüger (2016), [Of Quantiles and Expectiles: Consistent Scoring Functions, Choquet Representations, and Forecast Rankings](https://arxiv.org/abs/1503.08195), JRSS B | 분위수 손실의 elementary score 혼합 표현과 Murphy diagram을 통한 의사결정 문턱별 비교 | 평균 pinball 개선이 실제 관심 하락 문턱에서도 비용 개선을 뜻하는지 진단. 이미 존재하는 평가 이론을 적용하며 새 평가 알고리즘이라고 부르지 않음 | 공개 [원문](https://arxiv.org/html/1503.08195), 2.2/2.3 및 비교 해석 확인 |

좁은 기여 후보: **한국 USDT의 공통요인 조정 잔차에서 비선형 하방 예측을 비교하고, 그 개선을 상태별 오류·추가 하락 경보·가격 항등분해로 연결해 유용성과 해석 가능 범위를 함께 검증한다.** 실증 결과가 이 연결을 뒷받침해야 한다. 단순히 ML 이름, 한국 자료, 다섯 코인, 더 작은 손실을 나열하는 것으로 기여가 자동 성립하지 않는다.
