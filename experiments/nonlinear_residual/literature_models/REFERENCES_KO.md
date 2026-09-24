# A3의 문헌 근거와 실제 적용 범위

2026-09-24 원출처 재확인. 아래 문헌의 방법을 모두 그대로 복제했다는 뜻은 아니다.

| 원출처와 확인 범위 | 원래 연구와 배울 점 | 이번 적용 |
|---|---|---|
| [Lee·Chiu·Hsieh, Stablecoin depegging risk prediction, Pacific-Basin Finance Journal 90, 2025](https://doi.org/10.1016/j.pacfin.2024.102640), [저자 기관의 초록](https://ah.lib.nccu.edu.tw/item?item_id=176299), [공개 저자 초록](https://doi.org/10.2139/ssrn.4700764). 출판사 미리보기·초록 확인, 전문 실험표 미확보. | Logistic/RF/XGBoost로 네 스테이블코인의 디페깅 사건을 분류하고 거래량 기반 가변 문턱을 사용한다. | 빠졌던 XGBoost 계열을 추가한다. 여기서는 사건 분류가 아닌 공통 관계의 제곱손실 회귀로 적용하므로 직접 재현이나 같은 성능을 주장하지 않는다. |
| [Gu·Kelly·Xiu, Empirical Asset Pricing via Machine Learning, RFS 2020](https://doi.org/10.1093/rfs/hhaa009). 공개 본문의 1.1·1.5~1.8·2.1절을 확인. | 선형·스플라인·트리·신경망을 시간순 학습/검증/평가로 비교한다. 비선형 상호작용과 얕은 신경망, 여러 초기화 예측의 평균을 검토한다. | 작은 MLP와 상호작용 스플라인, 시간순 선택을 추가한다. 원 논문의 대규모 주식 패널·깊은 네트워크·학습법 전체를 복제하지 않는다. 2개 입력의 짧은 시계열에 맞춰 크기와 복잡도를 제한한다. |
| [FinTSB, Hu 외, 2025 공개본](https://arxiv.org/html/2502.18834v3), [ICAIF 2025 Rethinking Financial Time-Series의 채택 목록](https://icaif-25-rtfs.github.io/#accepted-papers). 공개본 데이터 구성·4.1/4.2·실험표 및 공식 워크숍 목록 확인. | 주식 시계열의 여러 시장 상황에서 XGBoost·LightGBM·신경망 등 공통 평가 체계를 제시한다. 시간축 분할과 동일 비교 조건이 중요하다. 해당 제목은 워크숍 Best Paper 목록에 있다. | 모든 후보의 동일 표본과 월별 결과를 유지한다. 벤치마크의 수익률 예측을 한국 USDT 잔차화의 직접 선행 결과로 인용하지 않는다. |
| [Na 외, Probabilistic forecasting of high-frequency realized cryptocurrency volatility via CEEMDAN-integrated autoregressive recurrent neural network, Physica A 2026](https://doi.org/10.1016/j.physa.2026.131364). 출판사 초록·서론·방법 미리보기 확인, 전문 수치 검증 미완료. 관련 저자·유사 제목은 위 2025 워크숍 목록에 있으나 두 원고가 동일하다고 단정하지 않는다. | 이동창 분해 성분을 DeepAR의 입력으로 사용하여 암호화폐 변동성 분포를 예측한다. 전체 시계열 선분해의 미래 정보 혼입을 피하도록 설계했다고 설명한다. | 분해와 최종 예측을 따로 평가하고 분해도 과거에 적합한다는 설계를 참고한다. CEEMDAN은 주파수 분해이고 우리 공통 시장 요인의 경제적 제거와 다르므로 그대로 대체하지 않는다. |
| [Kang 외, Stablecoin and cross-border crypto market integration, Economics Letters 2025](https://doi.org/10.1016/j.econlet.2025.112704). 출판사 초록·서론·데이터 미리보기 확인. | 한국 BTC·USDT 프리미엄의 공적분과 VECM 조정 경로를 연구한다. | 시장 공통 관계가 강할 수 있다는 경제적 비교 대상이다. 이 논문이 ML 잔차화를 표준으로 확립했다거나 비선형이 반드시 우월하다고 주장했다고 인용하지 않는다. |

**설계상 판단:** 공통 요인을 더 유연하게 조정하는 것과 기존 하방·포지셔닝 결과가 재현되는 것은 별도 평가다. 분해 모델은 공통 요인에 대한 이후 구간 적합으로 선택하고, 이후 경제적 분석은 선택에 사용하지 않는다. 선형+비선형 보정의 강도를 과거 검증으로 축소하는 비교는 이번 구현의 추가 설계이며 위 문헌이 동일한 한국 잔차 알고리즘을 제시했다고 쓰지 않는다.
