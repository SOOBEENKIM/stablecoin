> GitHub 탐색용 사본입니다. [체크섬이 보존된 원본](../../../../context/audit_20260908/proj_review/REVIEW_KO.md)의 내용과 과거 시점 표기를 유지하고 문서 링크만 변환했습니다. 실행은 [현재 재현 안내](../../../REPRODUCING.md)를 따릅니다.

# PROJ 자료의 용도·전처리·스테이블코인 연구 적합성 검토

검토일: 2026-09-08

검토 대상: `/home/ssd-990/soobeenkim/stablecoin_v4/PROJ`

**판정: 이 폴더는 기존 비트코인 변동성 예측 프로젝트의 자료다. 현재 스테이블코인 논문의 누락 원자료에 해당하지 않는다. 따라서 사용자의 “맞다면 workshop에 넣어 달라”는 조건에 부합하는 데이터는 없으며, PROJ 데이터를 workshop에 복사하지 않았다.**

폴더 이름은 사용자가 말씀하신 `stablecoin_4`가 아니라 실제 서버의 `stablecoin_v4`에서 확인했다. 현재 위치나 최근 복사 시각만으로 연구의 버전을 판단하면 안 된다.

## 1. 확인한 범위

- 프로젝트 최상위 연구 파일 213개: CSV 133개, 노트북 25개, Excel 54개, 작업공간 설정 1개. `.git`과 Windows 가상환경 `myenv`는 연구 데이터에서 제외했다.
- CSV 전체의 열·행 수·결측·무한대·날짜를 조사했다. 날짜 열을 식별한 CSV는 115개이며, 스테이블코인 표본 기간에 속하는 관측치는 없었다.
- 노트북 25개 전체의 소스에서 연구 대상과 입출력 계보를 검색했고, 대표 전처리 및 최종/NEW 모형의 코드를 직접 대조했다.
- Excel 54개는 모두 `Prediction_y`, `Actual_y` 두 열과 데이터 554행으로 구성된 예측 결과 파일이었다. 원시 가격·환율 데이터가 아니다.
- 일별 거시변수 → 4시간 간격 확장 → BTC와 병합 → 로그수익률·미래 변동성 계산을 저장 CSV에서 독립 재현했다.
- 외부 공급자의 과거 시세를 재다운로드하여 진위를 검증하거나 GARCH/LSTM을 재학습한 검토는 아니다. 아래의 “일치”는 파일 사이의 수치와 전처리 계보가 일치한다는 뜻이다.

근거: 전체 CSV 목록 (historical source path; see archive notes), 열·간격 상세 (historical source path; see archive notes), 노트북 목록 (historical source path; see archive notes), Excel 목록 (historical source path; see archive notes).

## 2. 어떤 연구의 어떤 데이터인가

노트북 제목은 “비트코인 가격/수익률 변동성 예측_GARCH(1,1) + LSTM”이며, 이후 EGARCH 및 GARCH·LSTM 결합 모형을 비교한 실험이 이어진다.

| 분류 | 대표 파일 | 내용·기간·단위 |
|---|---|---|
| BTC 기초 가격 | `BTC.csv` | 업비트 KRW-BTC 4시간봉 계보. 2020-01-01 01:00~2024-04-09 05:00, 9,361행. 원화 OHLC·거래량·거래대금 |
| 더 긴 BTC 가격 | `BTC_1.csv`, `BTC_2.csv` | 2019-12-21 01:00~2024-05-01 05:00, 9,559행. `BTC_2`는 날짜 열 정리 후 저장 |
| 거시 기초자료 | `CBOE_VIX.csv`, `KOSPI200.csv`, `USD_KRW.csv` | 2019-12-20~2024-05-02의 일별 자료. 각각 1,123·1,077·1,140행. 환율 파일은 환율 수준 |
| 거시 중간자료 | 번호가 붙은 `CBOE_VIX_*`, `KOSPI200_*`, `USD_KRW_*` 및 `*_expand.csv` | 날짜 정렬, 휴일 결측 채우기, 일별 값을 하루 여섯 시점에 반복한 자료. 구버전 계보도 함께 존재 |
| 통합 학습자료 | `BTC_CBOE_VIX_KOSPI200_USDKRW.csv` | BTC 4시간봉과 거시변수를 합친 9,559행·13열. 로그수익률·역사적 변동성·미래 변동성 포함 |
| 가공 특징·모형 산출물 | `GARCH_*`, `para_*`, `real_*`, `final*`, `BTC_normalized.csv` | 변동성 추정, GARCH 계수, 스케일링 및 실험 중간 결과. 원시 가격으로 사용할 수 없음 |
| 예측 평가 결과 | `LSTM_*.xlsx`, `GARCH*_pred_actual_values*`, `EGARCH*` | 예측값과 실제값의 비교 결과 |
| 별도 예제 | `model.ipynb`, `financial_data.csv`, `financial_data_cleaned.csv` | 주가 변동성 예제 계보. `financial_data.csv`는 1990~2020 날짜가 열에 놓인 전치 자료. 현재 스테이블코인 입력과 관계를 확인하지 못함 |

수집 코드에는 `ticker='KRW-BTC'`, `interval='minute240'`이 명시돼 있다. 저장된 BTC 가격 간격도 대부분 4시간이다. CSV 시각에는 시간대 정보가 없으므로 이를 UTC라고 바로 간주하면 안 된다. 수집 코드 (historical source path; see archive notes)

## 3. 현재 스테이블코인 연구와 일치하지 않는 이유

| 비교 항목 | 스테이블코인 논문·v4 | PROJ |
|---|---|---|
| 실제 회귀 표본 기간 | 2025-06-03~2026-03-19, 715개 관측치 | 주요 BTC·거시자료가 2019~2024년. 시간대 조정으로 해소할 수 없는 기간 차이 |
| 연구 대상 | 한국 USDT 프리미엄 잔차 | BTC 수익률·가격 변동성 예측 |
| 암호자산 자료 | Binance·Upbit의 BTC/ETH/XRP/SOL/DOGE/USDT, USDCUSDT | 업비트 KRW-BTC 중심. 필요한 양 거래소·다자산 패널 없음 |
| 필요한 시간 해상도 | 실제 1시간 가격과 거시 관측·시차 | BTC 4시간봉, 거시는 일별 값 반복 |
| 주요 거시변수 | USDKRW·DXY·VIX 계보, SPY·금 자료 | USD_KRW·CBOE_VIX·KOSPI200 중심. DXY 없음 |
| 주요 분석 | 순차 잔차화·분위수회귀·LP·포지셔닝 채널 | GARCH·EGARCH·LSTM |

CSV 열 이름과 노트북 소스에서 USDT·USDC·DXY·Binance·RESIDUAL_FINAL 관련 항목을 찾지 못했다. 기존 전처리 노트북이 읽는 `USDKRW.csv`, `VIXY.csv`, `DXY.csv`, `SPY US.csv`, `XAUUSD.csv`, `exogenous_full_aligned.csv`, 2025~2026 Binance/Upbit 시간봉 파일도 없었다.

특히 `USD_KRW.csv`와 기존 연구의 `USDKRW.csv`는 이름이 비슷해도 기간·주기·전처리 계보가 다르다. 파일명을 바꿔서 대신 넣을 수 없다. PROJ의 CBOE_VIX 자료 역시 기존 연구의 `VIXY_clean_1.csv`가 어떤 상품인지 확인해 주는 자료는 아니다.

## 4. 실제 전처리를 대조한 결과

### 재현되는 부분

대표 계보는 다음과 같다.

```text
업비트 KRW-BTC 4시간봉
  → BTC_1.csv → 날짜 열 정리 → BTC_2.csv

일별 CBOE_VIX / KOSPI200 / USD_KRW
  → Date·Price 선택 → 이름 변경·날짜 정렬
  → 달력 날짜로 확장 → 휴일·결측을 이전 값으로 채움(ffill)
  → 동일한 일별 값을 01·05·09·13·17·21시에 각각 배치
  → 앞 6행·뒤 10행 제거 → 날짜 형식 통일

BTC와 거시변수의 datetime 내부 병합
  → BTC_CBOE_VIX_KOSPI200_USDKRW.csv
  → 로그수익률, 과거 42개 수익률의 표준편차,
    현재부터 앞으로 42개 수익률의 표준편차 생성
  → GARCH/EGARCH 특징 → LSTM 실험 → 예측·실제값 저장
```

- `CBOE_VIX_9.csv`, `KOSPI200_7.csv`, `USD_KRW_8.csv`는 각각 **9,560행 전부** 일별 값 정렬·ffill·당일 반복으로 재현됐다. 최대 수치 오차 0이다.
- 이들을 BTC와 합치면 저장된 통합 파일과 **9,559개 타임스탬프가 전부 일치**한다. 가격·거시 수준은 오차 0, 거래대금의 최대 차이는 약 0.0000305원으로 CSV 부동소수점 정밀도 수준이다.
- 로그수익률 최대 오차 약 1.00e−16, 미래 42개 수익률 변동성 최대 오차 약 1.87e−16이다.
- 대표 BTC 파일은 타임스탬프 중복과 OHLC 상하관계 위반이 없었다. 다만 **2023-12-04 01:00**의 4시간봉 한 개가 빠져 있어 한 구간은 8시간 간격이다. 원천에서 미생성된 봉인지 수집 누락인지는 확인하지 못했다.

근거: [전처리 대조 수치](../../../../context/audit_20260908/proj_review/preprocessing_checks.json), 재현 스크립트 (historical source path; see archive notes).

### “전처리가 정확하므로 그대로 사용 가능”이라고 판단할 수 없는 부분

1. **일별 값을 반복한 자료는 실제 시간별 관측치가 아니다.** 세 거시변수의 같은 날짜 내 고유 값은 최대 1개다. 4시간마다 실제로 변동한 VIX·환율을 관측한 것으로 해석할 수 없다.
2. **관측 가능 시점이 반영되지 않았다.** 같은 날짜의 일별 값을 새벽 01시부터 부여하고 시간대를 명시하지 않는다. 당일 종가를 당일 새벽 예측에 쓰면 미래정보가 포함될 수 있다. 원천 시간대·발표/마감 시각을 확인하고 그 이후부터 사용할 수 있게 정렬해야 한다. 노트북 소스에서 `tz_localize`·`tz_convert`를 찾지 못했다.
3. **환율 수준과 변동성은 다르다.** 일부 주석은 원래의 `USD_KRW`를 “환율변동성 지수”라고 부르지만 값은 원/달러 환율 수준이다. 이후 별도로 생성한 `USD_KRW_vol`과 구분해야 한다.
4. **미래 변동성의 정의를 정확히 구분해야 한다.** `Next_7_Days_Volatility`는 역순 rolling 42로 계산하므로 현재 수익률부터 미래 41개 수익률까지 포함한다. 이를 설명변수로 넣으면 미래정보를 사용하게 된다. 목표변수로 쓰는 경우에는 예측 기준시점과 구간을 명시해야 한다. 한 개의 봉 누락 때문에 모든 42행이 항상 같은 실제 시간을 의미하지도 않는다. 계산 코드 (historical source path; see archive notes)
5. **일부 가공 데이터는 정리되지 않은 실험 산출물이다.** `BTC_normalized.csv`는 9,361행·192열이고 수치형 **무한대 값이 410,899개**다. 그대로 학습용 입력으로 사용할 수 없다. 이것이 원시 BTC 가격 파일까지 잘못됐다는 뜻은 아니다.
6. **final/NEW라는 이름이 단일 최종 모형을 보장하지 않는다.** `BTC_PROJ_Team2_NEW_1.ipynb`의 첫 실험은 `historical_volatility`를 목표로 사용한다. 다른 셀들의 `Next_7_Days_Volatility`와는 다른 대상이다. 학습 2022년 말까지·검증 2023년·테스트 2024년 분할과 학습자료에만 scaler를 적합하는 구현은 확인되지만, 모든 버전이 동일한 실험은 아니다. NEW_1의 데이터·분할 정의 (historical source path; see archive notes)
7. **전체 노트북의 순차 실행 재현성은 별도로 확인해야 한다.** 예를 들어 final 노트북에는 DataFrame으로 정의한 `train_df`를 `last_obs=train_df-1`에 쓰는 코드가 남아 있다. 이번에는 해당 모델들을 재학습하지 않았으므로 저장된 모든 GARCH/LSTM 결과가 노트북 전체 실행과 일치한다고 인증하지 않는다. 해당 코드 (historical source path; see archive notes)

## 5. 파일 처리 및 폴더의 의미

**PROJ는 현재 스테이블코인 연구 자료로 반입하지 않았다. 원본 PROJ와 기존 연구 코드·데이터는 수정하지 않았다.** 검토용 코드·목록·수치·보고서만 `stablecoin_audit_20260908/proj_review/`에 저장했다.

| 폴더 | 역할 |
|---|---|
| `stablecoin_v4` | 기존 연구의 코드·데이터·원고 묶음. 새로 추가한 PROJ는 별도 BTC 연구 자료 |
| `stablecoin_workshop` | v4를 복사해 둔 향후 수정 작업 공간. 사용자 지시에 따라 본 분석의 수정·재계산은 보류 상태 |
| `stablecoin_audit_20260908` | 이번 대화에서 생성한 대조 검토 기록. 논문 추출본, 독립 검증 코드, 기존 코드 재실행 결과, 민감도 분석, 보고서가 들어 있음 |

audit의 날짜가 최근이라는 것은 **검토를 최근에 수행했다는 뜻**이다. audit의 민감도 계산은 일부 정의를 바꾸어 영향만 확인한 것이므로 최신 확정 연구 데이터나 수정 완료 논문의 결과로 사용하면 안 된다.

재검토 실행:

```bash
python3 /home/ssd-990/soobeenkim/stablecoin_audit_20260908/proj_review/inspect_proj.py
python3 /home/ssd-990/soobeenkim/stablecoin_audit_20260908/proj_review/verify_preprocessing.py
```

검토 당시 원본 213개 파일의 해시는 source_manifest.json (historical source path; see archive notes)에 저장했다.
