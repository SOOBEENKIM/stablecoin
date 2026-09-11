# stablecoin_workshop 3안

**현재 상태: 3안 재설계·자료 준비·시간 정렬 검증 완료. 새 AI 적합은 아직 실행하지 않음.**

- [연구 방향과 차별성 판단](/home/ssd-990/soobeenkim/stablecoin_workshop/workshop3_20260909/RESEARCH_DESIGN_KO.md)
- [고정된 후속 실험 사양](/home/ssd-990/soobeenkim/stablecoin_workshop/workshop3_20260909/EXPERIMENT_PROTOCOL_KO.md)
- [실제 관측 수](/home/ssd-990/soobeenkim/stablecoin_workshop/workshop3_20260909/feasibility_counts.csv)
- [자료 검증 기록](/home/ssd-990/soobeenkim/stablecoin_workshop/workshop3_20260909/FEASIBILITY_MANIFEST.json)

1안은 측정 진단 중심의 실증 후보로 보존한다. 2안은 강한 로지스틱을 넘어서는 신경망 우위가 확인되지 않아 추가 모델 탐색을 멈춘다. 3안은 세 비교 기준의 서로 다른 미래 꼬리를 함께 예측하고, 가장 취약한 기준의 성능을 줄이는 다중 과제 학습을 검증하도록 설계했다. 기존 방법을 적용하는 응용 연구 후보이며, 새 알고리즘이나 긍정적 결과가 이미 확보됐다는 뜻이 아니다.

prepare_design.py는 원시 Upbit·Binance 시간봉에서 27특징의 연속 24시간 이력과 정확히 6시간 뒤 세 목표를 만든다. design_data.npz는 6,665개 이력, 평가 1,814개·77일을 포함한다. 같은 시장 기간이 기존 실험에서 재사용됐으므로 외부 검증이라고 부르지 않는다. PRESERVED_OPTIONS_SHA256.json은 작업 시작과 끝에 동일했던 1·2안 기존 산출물 119개의 해시다.
