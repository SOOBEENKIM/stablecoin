# 보정 설계의 근거와 차이

- Guan (2023), *Localized conformal prediction: a generalized inference framework for conformal prediction*, Biometrika 110(1), 33–50. [논문](https://doi.org/10.1093/biomet/asac040), [공개 전문](https://arxiv.org/html/2106.08460). 예측 대상의 공변량과 가까운 점수에 가중치를 주는 국소화가 이질적 분산·오차 상황에 유용할 수 있다는 근거다. 해당 논문의 보장에는 조정된 분위수 수준과 별도 조건이 필요하다. 여기서는 최근 이력에서 학습 표본 상대순위가 가까운 60개 오류를 사용하는 경험적 변형이며 그 이론적 보장을 가져오지 않는다.
- Gibbs & Candès (2021), *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS. [공식 논문](https://papers.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html). 분포 변화에서 오차 빈도를 온라인으로 관리한다는 문제 설정을 참고한다. 본 실험은 ACI 학습률/오류 업데이트 알고리즘을 구현한 것이 아니다.
- *Reliable value at risk estimation with conformal prediction* (2026), Risk Management. [저널 논문](https://link.springer.com/article/10.1057/s41283-026-00243-6). 금융 위험 예측에서 예측기와 사후 불확실성 보정을 구분하는 최근 사례다. 해당 연구의 VaR 추정치 주변 구간과 여기서 실현 잔차의 한쪽 분위수 경계는 다른 대상이다. 알고리즘 복제 또는 성능 직접 비교라고 주장하지 않는다.

본 연구의 기여 후보는 보정 기법 자체의 발명이 아니라 한국 USDT 공통요인 제거 잔차에서 상태 의존적 위험 경계가 단순 기준선·오경보·월별 변화에 갖는 실증적 의미다. 이 기여를 인정할지는 실제 결과와 비교군에 따라 결정한다.
