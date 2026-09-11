
# CELL 0
# %pip install pandas

# CELL 1
import pandas as pd

# CELL 2
# 1. CSV 불러오기
df = pd.read_csv('USDKRW.csv', header=None)

# 2. 컬럼명 지정
df.columns = ['datetime', 'price']

# 3. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 4. CSV로 저장
df.to_csv('USDKRW_clean.csv', index=False)

# CELL 3
# 1. CSV 불러오기
df = pd.read_csv('VIXY.csv', header=None)

# 2. 컬럼명 지정
df.columns = ['datetime', 'price']

# 3. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 4. CSV로 저장
df.to_csv('VIXY_clean.csv', index=False)

# CELL 4
# 1. CSV 불러오기
df = pd.read_csv('SPY US.csv', header=None)

# 2. 컬럼명 지정
df.columns = ['datetime', 'open', 'high', 'low', 'close']

# 3. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 4. CSV로 저장
df.to_csv('SPY US_clean.csv', index=False)

# CELL 5
# 1. CSV 불러오기
df = pd.read_csv('DXY.csv', header=None)

# 2. 컬럼명 지정
df.columns = ['datetime', 'price']

# 3. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 4. CSV로 저장
df.to_csv('DXY_clean.csv', index=False)

# CELL 6
# 1. CSV 불러오기
df = pd.read_csv('XAUUSD.csv', header=None)

# 2. 컬럼명 지정
df.columns = ['datetime', 'open', 'high', 'low', 'close']

# 3. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 4. CSV로 저장
df.to_csv('XAUUSD_clean.csv', index=False)

# CELL 7
# 1. clean 파일 불러오기
df = pd.read_csv('USDKRW_clean.csv')

# 2. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 3. 정렬
df = df.sort_values('datetime')

# 4. index 설정
df = df.set_index('datetime')

# 5. 전체 1시간 단위 index 생성
full_index = pd.date_range(start=df.index.min(),
                           end=df.index.max(),
                           freq='1h')

# 6. reindex -> NaN 행 생성
df = df.reindex(full_index)

# 7. 다시 datetime 컬럼으로 돌리기
df = df.reset_index()
df = df.rename(columns={'index': 'datetime'})
df.to_csv('USDKRW_clean.csv', index=False)

# CELL 8
# 1. 파일 불러오기
df = pd.read_csv('VIXY_clean.csv')

# 2. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# 3. datetime이 비어 있는 행 제거
df = df.dropna(subset=['datetime'])

# 4. 정렬
df = df.sort_values('datetime')

# 5. index 설정
df = df.set_index('datetime')

# 6. 시작 시각의 분(minute) 기준 잡기
minute_offset = df.index.min().minute
start = df.index.min().floor('h') + pd.Timedelta(minutes=minute_offset)

if start < df.index.min():
    start += pd.Timedelta(hours=1)

# 7. 전체 1시간 간격 index 생성
full_index = pd.date_range(
    start=start,
    end=df.index.max(),
    freq='1h'
)

# 8. 없는 시간 NaN 행 생성
df = df.reindex(full_index)

# 9. 다시 datetime 컬럼으로 복구
df = df.reset_index().rename(columns={'index': 'datetime'})

# 10. 저장
df.to_csv('VIXY_clean_1.csv', index=False)

# CELL 9
# 1. 파일 불러오기
df = pd.read_csv('SPY US_clean.csv')

# 2. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# 3. datetime이 비어 있는 행 제거
df = df.dropna(subset=['datetime'])

# 4. 중복 datetime 제거
df = df.drop_duplicates(subset=['datetime'], keep='last')

# 5. 정렬
df = df.sort_values('datetime')

# 6. index 설정
df = df.set_index('datetime')

# 7. 시작 시각의 분(minute) 기준 잡기
minute_offset = df.index.min().minute
start = df.index.min().floor('h') + pd.Timedelta(minutes=minute_offset)

if start < df.index.min():
    start += pd.Timedelta(hours=1)

# 8. 전체 1시간 간격 index 생성
full_index = pd.date_range(
    start=start,
    end=df.index.max(),
    freq='1h'
)

# 9. 없는 시간 NaN 행 생성
df = df.reindex(full_index)

# 10. 다시 datetime 컬럼으로 복구
df = df.reset_index().rename(columns={'index': 'datetime'})

# 11. 저장
df.to_csv('SPY_clean_1.csv', index=False)

# CELL 10
# 1. 파일 불러오기
df = pd.read_csv('DXY_clean.csv')

# 2. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# 3. datetime이 비어 있는 행 제거
df = df.dropna(subset=['datetime'])

# 4. 중복 datetime 제거
df = df.drop_duplicates(subset=['datetime'], keep='last')

# 5. 정렬
df = df.sort_values('datetime')

# 6. index 설정
df = df.set_index('datetime')

# 7. 시작 시각의 분(minute) 기준 잡기
minute_offset = df.index.min().minute
start = df.index.min().floor('h') + pd.Timedelta(minutes=minute_offset)

if start < df.index.min():
    start += pd.Timedelta(hours=1)

# 8. 전체 1시간 간격 index 생성
full_index = pd.date_range(
    start=start,
    end=df.index.max(),
    freq='1h'
)

# 9. 없는 시간 NaN 행 생성
df = df.reindex(full_index)

# 10. 다시 datetime 컬럼으로 복구
df = df.reset_index().rename(columns={'index': 'datetime'})

# 11. 저장
df.to_csv('DXY_clean_1.csv', index=False)

# CELL 11
# 1. clean 파일 불러오기
df = pd.read_csv('XAUUSD_clean.csv')

# 2. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# 3. datetime 비어있거나 깨진 행 제거
df = df.dropna(subset=['datetime'])

# 4. 같은 datetime 중복 제거
df = df.drop_duplicates(subset='datetime', keep='last')

# 5. 정렬
df = df.sort_values('datetime')

# 6. index 설정
df = df.set_index('datetime')

# 7. 전체 1시간 단위 index 생성
full_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq='1h'
)

# 8. reindex -> NaN 행 생성
df = df.reindex(full_index)

# 9. 다시 datetime 컬럼으로 돌리기
df = df.reset_index().rename(columns={'index': 'datetime'})

# 10. 저장
df.to_csv('XAUUSD_clean_1.csv', index=False)

# CELL 12
df = pd.read_csv('XAUUSD_clean_1.csv')

# 1. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 2. 정렬
df = df.sort_values('datetime')

# 3. index 설정
df = df.set_index('datetime')

print("NaN 개수:")
print(df.isna().sum())
print("중복 datetime 존재 여부:", df.index.duplicated().any())
time_diff = df.index.to_series().diff().dropna()
print(time_diff.value_counts().head())
print(time_diff[time_diff != pd.Timedelta(hours=1)].head(20))
print("시작:", df.index.min())
print("끝:", df.index.max())

is_nan = df['close'].isna()

groups = (is_nan != is_nan.shift()).cumsum()
nan_blocks = df[is_nan].groupby(groups).size()

print("연속 NaN 구간:")
print(nan_blocks.head())

print(df[df['close'].isna()].head(50))

# CELL 13
df = pd.read_csv('DXY_clean_1.csv')

# 1. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 2. 정렬
df = df.sort_values('datetime')

# 3. index 설정
df = df.set_index('datetime')

print("NaN 개수:")
print(df.isna().sum())
print("중복 datetime 존재 여부:", df.index.duplicated().any())

time_diff = df.index.to_series().diff().dropna()
print(time_diff.value_counts().head())

print(time_diff[time_diff != pd.Timedelta(hours=1)].head(20))

print("시작:", df.index.min())
print("끝:", df.index.max())

is_nan = df['price'].isna()

groups = (is_nan != is_nan.shift()).cumsum()
nan_blocks = df[is_nan].groupby(groups).size()

print("연속 NaN 구간:")
print(nan_blocks.head())

print(df[df['price'].isna()].head(50))

# CELL 14
df = pd.read_csv('SPY_clean_1.csv')

# 1. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 2. 정렬
df = df.sort_values('datetime')

# 3. index 설정
df = df.set_index('datetime')

print("NaN 개수:")
print(df.isna().sum())
print("중복 datetime 존재 여부:", df.index.duplicated().any())

time_diff = df.index.to_series().diff().dropna()
print(time_diff.value_counts().head())

print(time_diff[time_diff != pd.Timedelta(hours=1)].head(20))

print("시작:", df.index.min())
print("끝:", df.index.max())

is_nan = df['close'].isna()

groups = (is_nan != is_nan.shift()).cumsum()
nan_blocks = df[is_nan].groupby(groups).size()

print("연속 NaN 구간:")
print(nan_blocks.head())

print(df[df['close'].isna()].head(50))

# CELL 15
df = pd.read_csv('VIXY_clean_1.csv')

# 1. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 2. 정렬
df = df.sort_values('datetime')

# 3. index 설정
df = df.set_index('datetime')

print("NaN 개수:")
print(df.isna().sum())
print("중복 datetime 존재 여부:", df.index.duplicated().any())

time_diff = df.index.to_series().diff().dropna()
print(time_diff.value_counts().head())

print(time_diff[time_diff != pd.Timedelta(hours=1)].head(20))

print("시작:", df.index.min())
print("끝:", df.index.max())

is_nan = df['price'].isna()

groups = (is_nan != is_nan.shift()).cumsum()
nan_blocks = df[is_nan].groupby(groups).size()

print("연속 NaN 구간:")
print(nan_blocks.head())

print(df[df['price'].isna()].head(50))

# CELL 16
df = pd.read_csv('USDKRW_clean.csv')

# 1. datetime 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 2. 정렬
df = df.sort_values('datetime')

# 3. index 설정
df = df.set_index('datetime')

print("NaN 개수:")
print(df.isna().sum())
print("중복 datetime 존재 여부:", df.index.duplicated().any())

time_diff = df.index.to_series().diff().dropna()
print(time_diff.value_counts().head())

print(time_diff[time_diff != pd.Timedelta(hours=1)].head(20))

print("시작:", df.index.min())
print("끝:", df.index.max())

is_nan = df['price'].isna()

groups = (is_nan != is_nan.shift()).cumsum()
nan_blocks = df[is_nan].groupby(groups).size()

print("연속 NaN 구간:")
print(nan_blocks.head())

print(df[df['price'].isna()].head(50))

# CELL 25
import numpy as np

# CELL 26
# def preprocess(path, col_name):
#     df = pd.read_csv(path)
#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df = df.sort_values('datetime')

#     # 어떤 가격 컬럼을 쓸지 결정
#     if 'price' in df.columns:
#         value_col = 'price'
#     elif 'close' in df.columns:
#         value_col = 'close'
#     else:
#         raise ValueError(f"{path} 에서 사용할 가격 컬럼(price 또는 close)을 찾지 못함. 현재 컬럼: {list(df.columns)}")

#     df = df[['datetime', value_col]].dropna()
#     df = df.set_index('datetime')

#     # 1시간 bin의 마지막 관측값 사용
#     df = df.resample('1h').last()

#     # 최종 컬럼명 통일
#     df = df.rename(columns={value_col: col_name})

#     return df
def preprocess(path, col_name):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime')

    if 'price' in df.columns:
        value_col = 'price'
    elif 'close' in df.columns:
        value_col = 'close'
    else:
        raise ValueError(
            f"{path} 에서 사용할 가격 컬럼(price 또는 close)을 찾지 못함. "
            f"현재 컬럼: {list(df.columns)}"
        )

    df = df[['datetime', value_col]].dropna()
    df = df.set_index('datetime')

    # 동일 timestamp 중복 제거
    df = df[~df.index.duplicated(keep='last')]

    # 1시간 bin 마지막 관측값 사용
    df = df.resample('1h').last()

    df = df.rename(columns={value_col: col_name})
    return df

# CELL 27
# usdkrw = preprocess("USDKRW_clean.csv", "USDKRW")
# vix    = preprocess("VIXY_clean_1.csv", "VIX")
# spy    = preprocess("SPY_clean_1.csv", "SPY")
# dxy    = preprocess("DXY_clean_1.csv", "DXY")
# xau    = preprocess("XAUUSD_clean_1.csv", "XAUUSD")

# df = pd.concat([usdkrw, vix, spy, dxy, xau], axis=1, sort=True)
# df = df.sort_index()

# df = df.loc["2025-06-01 18:00":"2026-03-19 16:00"]
# data_baseline = df.dropna()
# df_ffill = df.copy()

# cols = ['USDKRW','VIX','SPY','DXY','XAUUSD']

# for col in cols:
#     df_ffill[f'{col}_stale'] = df_ffill[col].isna().astype(int)

# df_ffill[cols] = df_ffill[cols].ffill(limit=1)

# data_robust = df_ffill.dropna()

usdkrw = preprocess("USDKRW_clean.csv", "USDKRW")
vix    = preprocess("VIXY_clean_1.csv", "VIX")
spy    = preprocess("SPY_clean_1.csv", "SPY")
dxy    = preprocess("DXY_clean_1.csv", "DXY")
xau    = preprocess("XAUUSD_clean_1.csv", "XAUUSD")

# 공통 panel 생성
df = pd.concat([usdkrw, vix, spy, dxy, xau], axis=1, sort=True)
df = df.sort_index()

# 분석 표본 구간 제한
start = "2025-06-01 18:00:00+00:00"
end   = "2026-03-19 16:00:00+00:00"
df = df.loc[start:end]

print(df.head())
print(df.columns.tolist())
print(df.shape)

# baseline: overlap-only
data_baseline = df.dropna().copy()

# robustness: bounded ffill + stale flag
FFILL_LIMIT = 1
cols = ["USDKRW", "VIX", "SPY", "DXY", "XAUUSD"]

df_ffill = df.copy()

for col in cols:
    df_ffill[f"{col}_stale_flag"] = df_ffill[col].isna().astype(int)

df_ffill[cols] = df_ffill[cols].ffill(limit=FFILL_LIMIT)
data_robust = df_ffill.dropna().copy()

print("baseline shape:", data_baseline.shape)
print("robust shape  :", data_robust.shape)

# CELL 29
print(df.columns)
print(df.head())

# CELL 30
print(usdkrw.columns)
print(vix.columns)
print(spy.columns)
print(dxy.columns)
print(xau.columns)

# CELL 31
print("Non-missing counts by variable:")
print(df.notna().sum())

print("\nMissing ratio by variable:")
print(df.isna().mean().sort_values())

print("\nBaseline period:")
print(data_baseline.index.min(), "~", data_baseline.index.max())

print("\nRobust period:")
print(data_robust.index.min(), "~", data_robust.index.max())

# CELL 32
tmp = df.copy()
for col in df.columns:
    print(col, tmp.dropna(subset=[col]).shape[0])

# CELL 33
print("df shape:", df.shape)
print("baseline shape:", data_baseline.shape)
print("robust shape:", data_robust.shape)

print("\nMissing values in df:")
print(df.isna().sum())

print("\nBaseline start/end:")
print(data_baseline.index.min(), data_baseline.index.max())

print("\nRobust start/end:")
print(data_robust.index.min(), data_robust.index.max())

print("\nHead of df:")
print(df.head())

print("\nHead of baseline:")
print(data_baseline.head())

print("\nHead of robust:")
print(data_robust.head())

# CELL 35
save_path = "./"  # 원하는 경로로 바꿔도 됨

# 1. 전체 정렬된 데이터 (NaN 포함)
df.to_csv(save_path + "exogenous_full_aligned.csv", index=True)

# 2. baseline (논문 메인 데이터)
data_baseline.to_csv(save_path + "exogenous_baseline_overlap.csv", index=True)

# 3. robust (ffill + stale 포함)
data_robust.to_csv(save_path + "exogenous_robust_ffill.csv", index=True)

print("CSV 저장 완료")

# CELL 37
data_baseline.between_time("00:00","06:00")

# CELL 38
data_baseline['SPY'].diff().value_counts()

# CELL 40
# CSV 불러오기
df = pd.read_csv("exogenous_full_aligned.csv", parse_dates=['datetime'], index_col='datetime')
data_baseline = pd.read_csv("exogenous_baseline_overlap.csv", parse_dates=['datetime'], index_col='datetime')
data_robust = pd.read_csv("exogenous_robust_ffill.csv", parse_dates=['datetime'], index_col='datetime')

print("데이터 로드 완료")

cols = ['USDKRW','VIX','SPY','DXY','XAUUSD']

stale_df = pd.DataFrame({
    col: data_robust[f'{col}_stale_flag'].mean()
    for col in cols
}, index=['stale_ratio'])

print("\n=== Stale Ratio ===")
print(stale_df.T)

print("\n=== Observation Count ===")
print("Baseline obs:", len(data_baseline))
print("Robust obs:", len(data_robust))

print("\n=== Time Range ===")
print("Baseline:", data_baseline.index.min(), "~", data_baseline.index.max())
print("Robust  :", data_robust.index.min(), "~", data_robust.index.max())

# CELL 44
# %pip install requests

# CELL 45
import requests
import time

# 공통 설정
START_UTC = pd.Timestamp("2025-06-01 18:00:00", tz="UTC")
END_UTC   = pd.Timestamp("2026-03-19 16:00:00", tz="UTC")

# Binance spot
BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_LIMIT = 1000  # official max 1000

BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "USDT": "USDCUSDT",
}

# Upbit KRW markets + stablecoin
UPBIT_URL = "https://api.upbit.com/v1/candles/minutes/60"
UPBIT_LIMIT = 200  # official max 200

UPBIT_MARKETS = {
    "BTC": "KRW-BTC",
    "ETH": "KRW-ETH",
    "XRP": "KRW-XRP",
    "SOL": "KRW-SOL",
    "DOGE": "KRW-DOGE",
    "USDT": "KRW-USDT",
}

session = requests.Session()

def get_binance_klines_exact(symbol: str,
                             start_utc: pd.Timestamp,
                             end_utc: pd.Timestamp,
                             interval: str = "1h") -> pd.DataFrame:
    """
    Binance spot 1h klines, exact UTC window.
    Returns OHLCV indexed by open_time (UTC).
    """
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    rows = []

    while start_ms <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": BINANCE_LIMIT,
        }

        r = session.get(BINANCE_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        rows.extend(data)

        last_open_ms = data[-1][0]
        next_start_ms = last_open_ms + 60 * 60 * 1000  # 다음 1시간 봉 시작
        if next_start_ms <= start_ms:
            break
        start_ms = next_start_ms

        time.sleep(0.2)

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].astype(float)

    # 정확한 공통 구간으로 한 번 더 클립
    df = df.loc[start_utc:end_utc]

    # 안전하게 hourly grid 강제
    full_index = pd.date_range(start=start_utc, end=end_utc, freq="1h", tz="UTC")
    df = df.reindex(full_index)

    return df

def get_upbit_klines_exact(market: str,
                           start_utc: pd.Timestamp,
                           end_utc: pd.Timestamp) -> pd.DataFrame:
    """
    Upbit 60-minute candles in exact UTC window.
    Upbit candles exist only when trades occurred.
    Returns OHLCV indexed by candle_date_time_utc (UTC).
    """
    dfs = []
    to_cursor = end_utc

    while True:
        params = {
            "market": market,
            "count": UPBIT_LIMIT,
            "to": to_cursor.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        r = session.get(UPBIT_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list) or len(data) == 0:
            break

        df = pd.DataFrame(data)

        df["candle_date_time_utc"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)
        df = df.sort_values("candle_date_time_utc")
        dfs.append(df)

        oldest = df["candle_date_time_utc"].min()
        if oldest <= start_utc:
            break

        # 중복 방지용으로 oldest 직전 1초로 이동
        to_cursor = oldest - pd.Timedelta(seconds=1)
        time.sleep(0.2)

    if not dfs:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = pd.concat(dfs, axis=0)
    out = out.drop_duplicates(subset=["candle_date_time_utc"]).sort_values("candle_date_time_utc")
    out = out.set_index("candle_date_time_utc")

    out = out[[
        "opening_price",
        "high_price",
        "low_price",
        "trade_price",
        "candle_acc_trade_volume"
    ]].rename(columns={
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    }).astype(float)

    # 정확한 공통 구간으로 클립
    out = out.loc[start_utc:end_utc]

    # Upbit는 거래 없으면 캔들이 아예 없으므로, 여기선 NaN 유지
    full_index = pd.date_range(start=start_utc, end=end_utc, freq="1h", tz="UTC")
    out = out.reindex(full_index)

    return out

# CELL 46
# Binance 수집
binance_data = {}

for name, symbol in BINANCE_SYMBOLS.items():
    print(f"[Binance] Downloading {name} ({symbol}) ...")
    df_b = get_binance_klines_exact(symbol, START_UTC, END_UTC, interval="1h")
    df_b.columns = [f"{name}_BINANCE_{c.upper()}" for c in df_b.columns]
    binance_data[name] = df_b
    print(name, df_b.shape, df_b.index.min(), df_b.index.max())

df_binance = pd.concat(binance_data.values(), axis=1, sort=True).sort_index()

# CELL 48
# Upbit 수집
upbit_data = {}

for name, market in UPBIT_MARKETS.items():
    print(f"[Upbit] Downloading {name} ({market}) ...")
    df_u = get_upbit_klines_exact(market, START_UTC, END_UTC)
    df_u.columns = [f"{name}_UPBIT_{c.upper()}" for c in df_u.columns]
    upbit_data[name] = df_u
    print(name, df_u.shape, df_u.index.min(), df_u.index.max())

df_upbit = pd.concat(upbit_data.values(), axis=1, sort=True).sort_index()

# 저장
df_binance.index.name = "datetime_utc"
df_upbit.index.name = "datetime_utc"

df_binance.to_csv("binance_1h_2025-06-01_2026-03-19.csv")
df_upbit.to_csv("upbit_1h_2025-06-01_2026-03-19.csv")

print("저장 완료")
print("Binance shape:", df_binance.shape)
print("Upbit shape:", df_upbit.shape)

# CELL 49
print("\n Binance columns")
print(df_binance.columns)

print("\n Upbit columns")
print(df_upbit.columns)

# CELL 50
binance = pd.read_csv(
    "binance_1h_2025-06-01_2026-03-19.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)

upbit = pd.read_csv(
    "upbit_1h_2025-06-01_2026-03-19.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)

exog = pd.read_csv(
    "exogenous_baseline_overlap.csv",
    parse_dates=["datetime"],
    index_col="datetime"
)
# exog = pd.read_csv(
#     "exogenous_baseline_overlap.csv",
#     parse_dates=["datetime_utc"],
#     index_col="datetime_utc"
# )

# timezone 통일
binance.index = pd.to_datetime(binance.index, utc=True)
upbit.index = pd.to_datetime(upbit.index, utc=True)
exog.index = pd.to_datetime(exog.index, utc=True)

print("binance:", binance.shape)
print("upbit  :", upbit.shape)
print("exog   :", exog.shape)

# CELL 51
# close price만 사용

coins = ["BTC", "ETH", "XRP", "SOL", "DOGE"]

bin_close = pd.DataFrame(index=binance.index)
up_close = pd.DataFrame(index=upbit.index)

for c in coins:
    bin_close[f"{c}_BINANCE_CLOSE"] = binance[f"{c}_BINANCE_CLOSE"]
    up_close[f"{c}_UPBIT_CLOSE"] = upbit[f"{c}_UPBIT_CLOSE"]

# USDT (업비트)
# up_close["USDT_UPBIT_CLOSE"] = upbit["USDT_UPBIT_CLOSE"]
bin_close["USDT_BINANCE_CLOSE"] = binance["USDT_BINANCE_CLOSE"]
up_close["USDT_UPBIT_CLOSE"] = upbit["USDT_UPBIT_CLOSE"]

# FX
fx = exog[["USDKRW"]].copy()

# CELL 52
# baseline exog 시간축 기준
# panel = exog[["USDKRW"]].join(bin_close, how="left").join(up_close, how="left")
panel = bin_close.join(up_close, how="outer").sort_index()

print(panel.shape)
print(panel.head())
# print("\nUSDKRW NaN count:", panel["USDKRW"].isna().sum())

# CELL 54
# exog baseline에서 FX만 가져오기
fx = exog[["USDKRW"]].copy()

# crypto panel + FX
panel = panel.join(fx, how="left").sort_index()

print(panel.shape)
print(panel[["USDKRW"]].head())
print("\nUSDKRW NaN count:", panel["USDKRW"].isna().sum())

# CELL 55
for c in coins:
    panel[f"{c}_KP"] = (
        panel[f"{c}_UPBIT_CLOSE"] /
        (panel[f"{c}_BINANCE_CLOSE"] * panel["USDKRW"])
        - 1
    )

# panel["USDT_DEV"] = panel["USDT_UPBIT_CLOSE"] / panel["USDKRW"] - 1
panel["USDT_KP"] = (
    panel["USDT_UPBIT_CLOSE"] /
    (panel["USDT_BINANCE_CLOSE"] * panel["USDKRW"])
    - 1
)

# CELL 56
# kp_cols = [f"{c}_KP" for c in coins] + ["USDT_DEV"]
kp_coin_cols = [f"{c}_KP" for c in coins]
panel["MKT_KP_EQ"] = panel[kp_coin_cols].mean(axis=1)

panel["RESIDUAL_RAW"] = panel["USDT_KP"] - panel["MKT_KP_EQ"]

check_cols = kp_coin_cols + ["USDT_KP", "MKT_KP_EQ", "RESIDUAL_RAW"]

# print(panel[kp_cols].head())
# print("\nNaN counts:")
# print(panel[kp_cols].isna().sum())

# print("\nSummary stats:")
# print(panel[kp_cols].describe())
print("\nHead:")
print(panel[check_cols].head())

print("\nNaN counts:")
print(panel[check_cols].isna().sum())

print("\nSummary stats:")
print(panel[check_cols].describe())

# CELL 58
# for c in coins:
#     panel[f"{c}_KP_pct"] = panel[f"{c}_KP"] * 100

# panel["USDT_DEV_pct"] = panel["USDT_DEV"] * 100
for c in coins:
    panel[f"{c}_KP_pct"] = panel[f"{c}_KP"] * 100

panel["USDT_KP_pct"] = panel["USDT_KP"] * 100
panel["MKT_KP_EQ_pct"] = panel["MKT_KP_EQ"] * 100
panel["RESIDUAL_RAW_pct"] = panel["RESIDUAL_RAW"] * 100

# CELL 59
panel.index.name = "datetime_utc"

panel.to_csv("crypto_kp_panel_baseline.csv")

print("저장 완료: crypto_kp_panel_baseline.csv")

# CELL 61
# exog_baseline = pd.read_csv(
#     "exogenous_baseline_overlap.csv",
#     parse_dates=["datetime"],
#     index_col="datetime"
# )

# exog_baseline.index = pd.to_datetime(exog_baseline.index, utc=True)

# # 기존 panel은 이미 USDKRW 포함이라 나머지만 붙이기
# exog_vars = exog_baseline.drop(columns=["USDKRW"])

# final_df = panel.join(exog_vars)

# print(final_df.shape)
# print(final_df.head())
# print(final_df[["USDKRW", "VIX", "SPY", "DXY", "XAUUSD"]].isna().sum())
# 1. exogenous baseline load
exog_baseline = pd.read_csv(
    "exogenous_baseline_overlap.csv",
    parse_dates=["datetime"],
    index_col="datetime"
)

# 2. index 완전 통일
panel.index = pd.to_datetime(panel.index, utc=True)
exog_baseline.index = pd.to_datetime(exog_baseline.index, utc=True)

# 3. index 이름도 통일
panel.index.name = "datetime_utc"
exog_baseline.index.name = "datetime_utc"

# 4. panel은 이미 USDKRW 포함이므로 나머지 exog만 사용
exog_vars = exog_baseline.drop(columns=["USDKRW"])

# 5. merge
final_df = panel.join(exog_vars)

# 6. 확인
print(final_df.shape)
print(final_df.head())
print(final_df[["USDKRW", "VIX", "SPY", "DXY", "XAUUSD"]].isna().sum())

# 7. 저장
final_df.index.name = "datetime_utc"
final_df.to_csv("final_dataset_baseline.csv")

print("최종 저장 완료: final_dataset_baseline.csv")

# CELL 62
# # index 완전 통일
# panel.index = pd.to_datetime(panel.index, utc=True)
# exog_baseline.index = pd.to_datetime(exog_baseline.index, utc=True)

# # 이름도 통일
# panel.index.name = "datetime_utc"
# exog_baseline.index.name = "datetime_utc"
# final_df = panel.join(exog_vars, how="left")

# CELL 63
# final_df.index.name = "datetime_utc"

# final_df.to_csv("final_dataset_baseline.csv")

# print("최종 저장 완료: final_dataset_baseline.csv")

# CELL 64
# # 1. final dataset load
# final_df = pd.read_csv(
#     "final_dataset_baseline.csv",
#     parse_dates=["datetime_utc"],
#     index_col="datetime_utc"
# )

# # timezone 유지/통일
# final_df.index = pd.to_datetime(final_df.index, utc=True)

# print("final_df shape:", final_df.shape)
# print(final_df.head())
# 1. final dataset load
final_df = pd.read_csv(
    "final_dataset_baseline.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)

final_df.index = pd.to_datetime(final_df.index, utc=True)

print("final_df shape:", final_df.shape)
print(final_df.head())

# 2. strict baseline usable sample 먼저 추출
baseline_df = final_df.dropna(subset=["USDKRW", "VIX", "SPY", "DXY", "XAUUSD"]).copy()

print("baseline_df shape:", baseline_df.shape)

# 3. 외생변수 log return 생성
exog_cols = ["VIX", "SPY", "DXY", "XAUUSD"]

for col in exog_cols:
    baseline_df[f"{col}_ret"] = np.log(baseline_df[col]).diff()

print(baseline_df[[*exog_cols, "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret"]].head())

# CELL 65
# # 2. 외생변수 변환
# # level -> log return
# exog_cols = ["VIX", "SPY", "DXY", "XAUUSD"]

# for col in exog_cols:
#     final_df[f"{col}_ret"] = np.log(final_df[col]).diff()

# print(final_df[[*exog_cols, "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret"]].head())

# CELL 66
# # 3. 회귀용 clean dataset 생성
# reg_cols = [
#     "BTC_KP", "ETH_KP", "XRP_KP", "SOL_KP", "DOGE_KP",
#     "USDT_DEV",
#     "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret"
# ]

# reg_df = final_df[reg_cols].dropna().copy()

# print("reg_df shape:", reg_df.shape)
# print(reg_df.head())
# 3. 회귀용 clean dataset 생성 (baseline_df 기준)
reg_cols = [
    "BTC_KP", "ETH_KP", "XRP_KP", "SOL_KP", "DOGE_KP",
    "USDT_KP",
    "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret"
]

reg_df = baseline_df[reg_cols].dropna().copy()

print("reg_df shape:", reg_df.shape)
print(reg_df.head())

# CELL 67
# # index 이름 명확히
# final_df.index.name = "datetime_utc"
# reg_df.index.name = "datetime_utc"

# # 1. 전체 데이터 (KP + exog + returns 포함)
# final_df.to_csv("final_dataset_baseline_with_returns.csv")

# # 2. 회귀용 clean dataset (dropna 완료된 것)
# reg_df.to_csv("regression_dataset_baseline.csv")

# print("저장 완료")
# print("1) final_dataset_baseline_with_returns.csv")
# print("2) regression_dataset_baseline.csv")
# index 이름 명확히
final_df.index.name = "datetime_utc"
baseline_df.index.name = "datetime_utc"
reg_df.index.name = "datetime_utc"

# 저장
final_df.to_csv("final_dataset_baseline_with_returns.csv")
baseline_df.to_csv("baseline_dataset_clean.csv")
reg_df.to_csv("regression_dataset_baseline.csv")

print("저장 완료")
print("1) final_dataset_baseline_with_returns.csv")
print("2) baseline_dataset_clean.csv")
print("3) regression_dataset_baseline.csv")

# CELL 69
print(baseline_df.columns)

# CELL 70
# print(reg_df.isna().sum())
# print(reg_df.describe())

# CELL 73
import pandas as pd
import numpy as np

# final_df = pd.read_csv(
#     "final_dataset_baseline.csv",
#     parse_dates=["datetime_utc"],
#     index_col="datetime_utc"
# )

# binance = pd.read_csv(
#     "binance_1h_2025-06-01_2026-03-19.csv",
#     parse_dates=["datetime_utc"],
#     index_col="datetime_utc"
# )

# upbit = pd.read_csv(
#     "upbit_1h_2025-06-01_2026-03-19.csv",
#     parse_dates=["datetime_utc"],
#     index_col="datetime_utc"
# )

# # timezone 통일
# final_df.index = pd.to_datetime(final_df.index, utc=True)
# binance.index = pd.to_datetime(binance.index, utc=True)
# upbit.index = pd.to_datetime(upbit.index, utc=True)

# print("final_df:", final_df.shape)
# print("binance :", binance.shape)
# print("upbit   :", upbit.shape)

baseline_df = pd.read_csv(
    "baseline_dataset_clean.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)

binance = pd.read_csv(
    "binance_1h_2025-06-01_2026-03-19.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)

upbit = pd.read_csv(
    "upbit_1h_2025-06-01_2026-03-19.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)

# timezone 통일
baseline_df.index = pd.to_datetime(baseline_df.index, utc=True)
binance.index = pd.to_datetime(binance.index, utc=True)
upbit.index = pd.to_datetime(upbit.index, utc=True)

print("baseline_df:", baseline_df.shape)
print("binance    :", binance.shape)
print("upbit      :", upbit.shape)

# # =========================
# # 2) exogenous returns 다시 생성
# # =========================
# exog_cols = ["VIX", "SPY", "DXY", "XAUUSD"]

# for col in exog_cols:
#     final_df[f"{col}_ret"] = np.log(final_df[col]).diff()

# print("\nExogenous returns added:")
# print(final_df[[*exog_cols, "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret"]].head())

# # =========================
# # 3) micro variables 생성
# # =========================
# coins = ["BTC", "ETH", "XRP", "SOL", "DOGE"]

# micro = pd.DataFrame(index=final_df.index)

# # (a) BTC global return
# micro["BTC_RET"] = np.log(
#     binance["BTC_BINANCE_CLOSE"].reindex(final_df.index)
# ).diff()

# # (b) BTC rolling volatility (24 observations)
# micro["BTC_VOL_24"] = micro["BTC_RET"].rolling(window=24, min_periods=24).std()

# # (c) coin-specific volume ratios
# #     raw ratio + log ratio 둘 다 생성
# for c in coins:
#     bin_vol = binance[f"{c}_BINANCE_VOLUME"].reindex(final_df.index)
#     up_vol = upbit[f"{c}_UPBIT_VOLUME"].reindex(final_df.index)

#     micro[f"{c}_VOL_RATIO"] = bin_vol / up_vol
#     micro[f"{c}_VOL_RATIO_LOG"] = np.log(bin_vol / up_vol)

# print("\nMicro variables head:")
# print(micro.head())

# print("\nNaN counts in micro:")
# print(micro.isna().sum())

# # =========================
# # 4) merge
# # =========================
# final_micro_df = final_df.join(micro)

# print("\nMerged shape:", final_micro_df.shape)
# print(final_micro_df.head())

# # =========================
# # 5) regression dataset 생성
# # =========================
# reg_cols_micro = [
#     "BTC_KP", "ETH_KP", "XRP_KP", "SOL_KP", "DOGE_KP",
#     "USDT_DEV",
#     "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret",
#     "BTC_RET", "BTC_VOL_24",
#     "BTC_VOL_RATIO_LOG", "ETH_VOL_RATIO_LOG", "XRP_VOL_RATIO_LOG",
#     "SOL_VOL_RATIO_LOG", "DOGE_VOL_RATIO_LOG"
# ]

# # 존재하지 않는 컬럼 체크
# missing_cols = [c for c in reg_cols_micro if c not in final_micro_df.columns]
# print("\nMissing columns:", missing_cols)

# reg_micro_df = final_micro_df[reg_cols_micro].dropna().copy()

# print("\nreg_micro_df shape:", reg_micro_df.shape)
# print(reg_micro_df.head())

# # =========================
# # 6) 저장
# # =========================
# final_micro_df.index.name = "datetime_utc"
# reg_micro_df.index.name = "datetime_utc"

# final_micro_df.to_csv("final_dataset_baseline_with_micro.csv")
# reg_micro_df.to_csv("regression_dataset_baseline_with_micro.csv")

# print("\n저장 완료")
# print("1) final_dataset_baseline_with_micro.csv")
# print("2) regression_dataset_baseline_with_micro.csv")

# CELL 74
# 2) micro variables 생성 (수정 버전)
coins = ["BTC", "ETH", "XRP", "SOL", "DOGE"]

micro = pd.DataFrame(index=baseline_df.index)

# --------------------------------------------------
# (A) BTC micro variables는 Binance full hourly 기준으로 먼저 계산
# --------------------------------------------------
btc_full = pd.DataFrame(index=binance.index.copy())
btc_full["BTC_CLOSE"] = binance["BTC_BINANCE_CLOSE"]

# 진짜 1시간 수익률
btc_full["BTC_RET"] = np.log(btc_full["BTC_CLOSE"]).diff()

# 진짜 24시간 rolling volatility
btc_full["BTC_VOL_24"] = btc_full["BTC_RET"].rolling(window=24, min_periods=24).std()

# 비대칭 shock 변수
btc_full["BTC_RET_POS"] = btc_full["BTC_RET"].clip(lower=0)
btc_full["BTC_RET_NEG"] = btc_full["BTC_RET"].clip(upper=0)
btc_full["BTC_RET_ABS"] = btc_full["BTC_RET"].abs()

# downside proxy: 최근 24시간 하방 반응
btc_full["BTC_DOWNSIDE_24"] = (
    btc_full["BTC_RET"].clip(upper=0) ** 2
).rolling(window=24, min_periods=24).mean()

# 큰 충격 dummy는 full hourly BTC_RET 기준으로 계산
up_thr = btc_full["BTC_RET"].quantile(0.95)
down_thr = btc_full["BTC_RET"].quantile(0.05)

btc_full["BTC_BIG_UP"] = (btc_full["BTC_RET"] >= up_thr).astype(int)
btc_full["BTC_BIG_DOWN"] = (btc_full["BTC_RET"] <= down_thr).astype(int)

# baseline sample index에 맞춰 가져오기
btc_cols = [
    "BTC_RET", "BTC_VOL_24",
    "BTC_RET_POS", "BTC_RET_NEG", "BTC_RET_ABS",
    "BTC_DOWNSIDE_24", "BTC_BIG_UP", "BTC_BIG_DOWN"
]

micro[btc_cols] = btc_full[btc_cols].reindex(baseline_df.index)

# --------------------------------------------------
# (B) coin-specific volume ratios
# --------------------------------------------------
for c in coins:
    bin_vol = binance[f"{c}_BINANCE_VOLUME"].reindex(baseline_df.index)
    up_vol = upbit[f"{c}_UPBIT_VOLUME"].reindex(baseline_df.index)

    # 0 또는 음수 로그 방지
    bin_vol = bin_vol.where(bin_vol > 0)
    up_vol = up_vol.where(up_vol > 0)

    ratio = bin_vol / up_vol
    micro[f"{c}_VOL_RATIO"] = ratio
    micro[f"{c}_VOL_RATIO_LOG"] = np.log(ratio)

# --------------------------------------------------
# (C) 추가 권장 변수 1: USDKRW return
# --------------------------------------------------
baseline_df["USDKRW_ret"] = np.log(baseline_df["USDKRW"]).diff()

# --------------------------------------------------
# (D) 추가 권장 변수 2: USDT-specific volume ratio
# --------------------------------------------------
usdt_bin_vol = binance["USDT_BINANCE_VOLUME"].reindex(baseline_df.index)
usdt_up_vol = upbit["USDT_UPBIT_VOLUME"].reindex(baseline_df.index)

usdt_bin_vol = usdt_bin_vol.where(usdt_bin_vol > 0)
usdt_up_vol = usdt_up_vol.where(usdt_up_vol > 0)

micro["USDT_VOL_RATIO"] = usdt_bin_vol / usdt_up_vol
micro["USDT_VOL_RATIO_LOG"] = np.log(micro["USDT_VOL_RATIO"])

print("\nMicro variables head:")
print(micro.head())

print("\nNaN counts in micro:")
print(micro.isna().sum())

# 3) merge
baseline_micro_df = baseline_df.join(micro, how="left")

print("\nMerged shape:", baseline_micro_df.shape)
print(baseline_micro_df.head())

# CELL 77
# 4) 회귀용 dataset 생성

# (A) KP 설명용 dataset
# reg_cols_kp = [
#     "USDT_KP",
#     "MKT_KP_EQ",
#     "BTC_KP", "ETH_KP", "XRP_KP", "SOL_KP", "DOGE_KP",
#     "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret",
#     "BTC_RET", "BTC_VOL_24",
#     "BTC_RET_POS", "BTC_RET_NEG", "BTC_RET_ABS",
#     "BTC_DOWNSIDE_24", "BTC_BIG_UP", "BTC_BIG_DOWN",
#     "BTC_VOL_RATIO_LOG", "ETH_VOL_RATIO_LOG", "XRP_VOL_RATIO_LOG",
#     "SOL_VOL_RATIO_LOG", "DOGE_VOL_RATIO_LOG"
# ]
reg_cols_kp = [
    "USDT_KP",
    "MKT_KP_EQ",
    "BTC_KP", "ETH_KP", "XRP_KP", "SOL_KP", "DOGE_KP",
    "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret", "USDKRW_ret",
    "BTC_RET", "BTC_VOL_24",
    "BTC_RET_POS", "BTC_RET_NEG", "BTC_RET_ABS",
    "BTC_DOWNSIDE_24", "BTC_BIG_UP", "BTC_BIG_DOWN",
    "BTC_VOL_RATIO_LOG", "ETH_VOL_RATIO_LOG", "XRP_VOL_RATIO_LOG",
    "SOL_VOL_RATIO_LOG", "DOGE_VOL_RATIO_LOG",
    "USDT_VOL_RATIO_LOG"
]

missing_kp = [c for c in reg_cols_kp if c not in baseline_micro_df.columns]
print("\nMissing columns in reg_cols_kp:", missing_kp)

reg_kp_df = baseline_micro_df[reg_cols_kp].dropna().copy()

print("\nreg_kp_df shape:", reg_kp_df.shape)
print(reg_kp_df.head())

# (B) Residual 설명용 dataset
# reg_cols_resid = [
#     "RESIDUAL_RAW",
#     "USDT_KP",
#     "MKT_KP_EQ",
#     "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret",
#     "BTC_RET", "BTC_VOL_24",
#     "BTC_RET_POS", "BTC_RET_NEG", "BTC_RET_ABS",
#     "BTC_DOWNSIDE_24", "BTC_BIG_UP", "BTC_BIG_DOWN",
#     "BTC_VOL_RATIO_LOG", "ETH_VOL_RATIO_LOG", "XRP_VOL_RATIO_LOG",
#     "SOL_VOL_RATIO_LOG", "DOGE_VOL_RATIO_LOG"
# ]
reg_cols_resid = [
    "RESIDUAL_RAW",
    "USDT_KP",
    "MKT_KP_EQ",
    "VIX_ret", "SPY_ret", "DXY_ret", "XAUUSD_ret", "USDKRW_ret",
    "BTC_RET", "BTC_VOL_24",
    "BTC_RET_POS", "BTC_RET_NEG", "BTC_RET_ABS",
    "BTC_DOWNSIDE_24", "BTC_BIG_UP", "BTC_BIG_DOWN",
    "BTC_VOL_RATIO_LOG", "ETH_VOL_RATIO_LOG", "XRP_VOL_RATIO_LOG",
    "SOL_VOL_RATIO_LOG", "DOGE_VOL_RATIO_LOG",
    "USDT_VOL_RATIO_LOG"
]

missing_resid = [c for c in reg_cols_resid if c not in baseline_micro_df.columns]
print("\nMissing columns in reg_cols_resid:", missing_resid)

reg_resid_df = baseline_micro_df[reg_cols_resid].dropna().copy()

print("\nreg_resid_df shape:", reg_resid_df.shape)
print(reg_resid_df.head())

# 5) 저장
baseline_micro_df.index.name = "datetime_utc"
reg_kp_df.index.name = "datetime_utc"
reg_resid_df.index.name = "datetime_utc"

baseline_micro_df.to_csv("baseline_dataset_with_micro.csv")
reg_kp_df.to_csv("regression_dataset_kp.csv")
reg_resid_df.to_csv("regression_dataset_residual.csv")

print("\n저장 완료")
print("1) baseline_dataset_with_micro.csv")
print("2) regression_dataset_kp.csv")
print("3) regression_dataset_residual.csv")

# CELL 79
print("\nMissing columns in reg_cols_kp:", missing_kp)
print("\nMissing columns in reg_cols_resid:", missing_resid)

# CELL 83
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv(
    "regression_dataset_residual.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)

# Step 1: USDT_KP ~ MKT_KP_EQ
X = sm.add_constant(df["MKT_KP_EQ"])
y = df["USDT_KP"]

model = sm.OLS(y, X).fit()

print(model.summary())

# Step 2: residual
df["RESIDUAL_OLS"] = model.resid

# CELL 87
import statsmodels.api as sm

# X 구성 (핵심 변수만 먼저)
X = df[
    [
        "VIX_ret",
        "SPY_ret",
        "DXY_ret",
        "XAUUSD_ret",
        "BTC_RET",
        "BTC_VOL_24",
        "BTC_RET_NEG",
        "BTC_DOWNSIDE_24",
        "BTC_BIG_DOWN"
    ]
]

X = sm.add_constant(X)
y = df["RESIDUAL_OLS"]

# model = sm.OLS(y, X).fit()
model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

print(model.summary())

# CELL 91
# Cell 1: 현재 df에서 depeg 후보 컬럼 확인
print("df columns:")
print(df.columns.tolist())

candidate_cols = [
    c for c in df.columns
    if ("DEPEG" in c.upper())
    or ("USDC" in c.upper())
    or ("GLOBAL" in c.upper())
    or ("USDTUSD" in c.upper())
    or ("USDCUSDT" in c.upper())
    or ("USDTUSDC" in c.upper())
]

print("\nPossible depeg-related columns:")
print(candidate_cols)

# CELL 92
# Cell 2: 현재 메모리에 있는 원천 데이터에서 depeg 후보 찾기

for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame):
        cols = [c for c in obj.columns if ("USDT" in c.upper()) or ("USDC" in c.upper())]
        if len(cols) > 0:
            print(f"\n[{name}]")
            print(cols)

# CELL 93
# Cell 3: USDT_BINANCE_CLOSE 값 범위 확인
print("USDT_BINANCE_CLOSE head:")
print(baseline_micro_df["USDT_BINANCE_CLOSE"].head(10))

print("\nUSDT_BINANCE_CLOSE summary:")
print(baseline_micro_df["USDT_BINANCE_CLOSE"].describe())

# CELL 95
# Cell 4: baseline_micro_df에서 depeg 변수 붙이기

df = df.join(baseline_micro_df[["USDT_BINANCE_CLOSE"]], how="left")

df["DEPEG_GLOBAL"] = (df["USDT_BINANCE_CLOSE"] - 1.0).abs()

print(df[["USDT_BINANCE_CLOSE", "DEPEG_GLOBAL"]].head())
print("\nMissing DEPEG_GLOBAL:", df["DEPEG_GLOBAL"].isna().sum())
print("\nDEPEG summary:")
print(df["DEPEG_GLOBAL"].describe())

# CELL 96
# Cell 5: Stage 2 residual (depeg 제거)

import statsmodels.api as sm

X = sm.add_constant(df["DEPEG_GLOBAL"])
y = df["RESIDUAL_OLS"]

model_depeg = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

print(model_depeg.summary())

# 최종 residual
df["RESIDUAL_FINAL"] = model_depeg.resid

print("\nRESIDUAL_FINAL summary:")
print(df["RESIDUAL_FINAL"].describe())

# CELL 99
# X = df[
#     [
#         "DXY_ret",
#         "BTC_VOL_24",
#         "BTC_DOWNSIDE_24"
#     ]
# ]

# X = sm.add_constant(X)
# y = df["RESIDUAL_OLS"]

# model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
# print(model.summary())
# Cell 6: 진짜 구조 분석 (Stage 3)

import statsmodels.api as sm

X = df[
    [
        "DXY_ret",
        "BTC_VOL_24",
        "BTC_DOWNSIDE_24"
    ]
]

X = sm.add_constant(X)

# 여기만 바뀜
y = df["RESIDUAL_FINAL"]

model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

print(model.summary())

# CELL 103
plot_df = df[["RESIDUAL_FINAL", "BTC_DOWNSIDE_24"]].dropna().copy()

plot_df = (plot_df - plot_df.mean()) / plot_df.std()

plot_df.plot(figsize=(12,5), title="Standardized: Residual vs Downside")

# CELL 105
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1) 비교용 데이터 준비
compare_df = df[["RESIDUAL_FINAL", "BTC_DOWNSIDE_24"]].dropna().copy()

# 표준화
compare_df["RESIDUAL_Z"] = (
    compare_df["RESIDUAL_FINAL"] - compare_df["RESIDUAL_FINAL"].mean()
) / compare_df["RESIDUAL_FINAL"].std()

compare_df["DOWNSIDE_Z"] = (
    compare_df["BTC_DOWNSIDE_24"] - compare_df["BTC_DOWNSIDE_24"].mean()
) / compare_df["BTC_DOWNSIDE_24"].std()

# 2) 차이(Residual - Downside)
compare_df["Z_DIFF"] = compare_df["RESIDUAL_Z"] - compare_df["DOWNSIDE_Z"]
compare_df["Z_SUM"]  = compare_df["RESIDUAL_Z"] + compare_df["DOWNSIDE_Z"]

print("Head:")
print(compare_df[["RESIDUAL_Z", "DOWNSIDE_Z", "Z_DIFF"]].head())

print("\nSummary:")
print(compare_df[["RESIDUAL_Z", "DOWNSIDE_Z", "Z_DIFF"]].describe())

# 3) 전체 상관계수
corr = compare_df["RESIDUAL_Z"].corr(compare_df["DOWNSIDE_Z"])
print("\nCorrelation(Residual_Z, Downside_Z):", corr)

# 4) 단순 회귀: 표준화 residual ~ 표준화 downside
X = sm.add_constant(compare_df["DOWNSIDE_Z"])
y = compare_df["RESIDUAL_Z"]

model_std = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print("\nRegression: RESIDUAL_Z ~ DOWNSIDE_Z")
print(model_std.summary())

# 5) rolling correlation
window = 24
compare_df["ROLL_CORR_24"] = (
    compare_df["RESIDUAL_Z"].rolling(window).corr(compare_df["DOWNSIDE_Z"])
)

# 6) plot
plt.figure(figsize=(12, 5))
plt.plot(compare_df.index, compare_df["Z_DIFF"], label="Z_DIFF = Residual_Z - Downside_Z")
plt.axhline(0, linestyle="--")
plt.title("Standardized Difference: Residual - Downside")
plt.xlabel("datetime_utc")
plt.ylabel("Z-score difference")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(compare_df.index, compare_df["ROLL_CORR_24"], label="24h Rolling Corr")
plt.axhline(0, linestyle="--")
plt.title("24h Rolling Correlation: Residual vs Downside")
plt.xlabel("datetime_utc")
plt.ylabel("Correlation")
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(compare_df["DOWNSIDE_Z"], compare_df["RESIDUAL_Z"], alpha=0.5)
plt.xlabel("DOWNSIDE_Z")
plt.ylabel("RESIDUAL_Z")
plt.title("Scatter: Residual_Z vs Downside_Z")
plt.show()

# CELL 106
# 차이가 큰 구간 상위 10개
top_gap = compare_df["Z_DIFF"].abs().sort_values(ascending=False).head(10)
print("\nTop 10 absolute gaps:")
print(compare_df.loc[top_gap.index, ["RESIDUAL_Z", "DOWNSIDE_Z", "Z_DIFF"]])

# CELL 108


# CELL 109
# import matplotlib.pyplot as plt

# plt.figure(figsize=(6,5))
# plt.scatter(df["BTC_DOWNSIDE_24"], df["RESIDUAL_FINAL"], alpha=0.3)
# plt.xlabel("BTC_DOWNSIDE_24")
# plt.ylabel("RESIDUAL_FINAL")
# plt.title("Residual vs Downside (Scatter)")
# plt.show()

# CELL 110
# thr = df["BTC_DOWNSIDE_24"].quantile(0.9)

# subset = df[df["BTC_DOWNSIDE_24"] >= thr]

# subset[["RESIDUAL_FINAL", "BTC_DOWNSIDE_24"]].plot(
#     figsize=(12,5),
#     title="Top 10% Downside Periods"
# )

# CELL 111
df.loc[df["BTC_DOWNSIDE_24"].nlargest(10).index, 
       ["RESIDUAL_FINAL", "BTC_DOWNSIDE_24", "DXY_ret", "DEPEG_GLOBAL"]]

# CELL 112
# X = df[
#     [
#         "BTC_DOWNSIDE_24",
#         "DXY_ret"
#     ]
# ]

# X = sm.add_constant(X)
# y = df["RESIDUAL_OLS"]

# model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
# print(model.summary())
X = df[
    [
        "BTC_DOWNSIDE_24",
        "DXY_ret"
    ]
]

X = sm.add_constant(X)
y = df["RESIDUAL_FINAL"]

model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(model.summary())

# CELL 115
# import statsmodels.formula.api as smf

# # 90% quantile (premium 높은 구간)
# model_q90 = smf.quantreg(
#     "RESIDUAL_OLS ~ BTC_DOWNSIDE_24 + DXY_ret",
#     df
# ).fit(q=0.9)

# print(model_q90.summary())
import statsmodels.formula.api as smf

model_q10 = smf.quantreg(
    "RESIDUAL_FINAL ~ BTC_DOWNSIDE_24 + DXY_ret",
    df
).fit(q=0.1)

print(model_q10.summary())

# CELL 116
model_q50 = smf.quantreg(
    "RESIDUAL_FINAL ~ BTC_DOWNSIDE_24 + DXY_ret",
    df
).fit(q=0.5)

print(model_q50.summary())

# CELL 117
model_q90 = smf.quantreg(
    "RESIDUAL_FINAL ~ BTC_DOWNSIDE_24 + DXY_ret",
    df
).fit(q=0.9)

print(model_q90.summary())

# CELL 121
# Cell 1: quantile sweep (수정 버전)
import pandas as pd
import statsmodels.formula.api as smf

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
results = []

for q in quantiles:
    mod = smf.quantreg(
        "RESIDUAL_FINAL ~ BTC_DOWNSIDE_24 + DXY_ret",
        df
    ).fit(q=q)

    results.append({
        "q": q,
        "beta_downside": mod.params["BTC_DOWNSIDE_24"],
        "p_downside": mod.pvalues["BTC_DOWNSIDE_24"],
        "beta_dxy": mod.params["DXY_ret"],
        "p_dxy": mod.pvalues["DXY_ret"],
    })

res_df = pd.DataFrame(results)
print(res_df)

# CELL 123
# Cell 2: quantile coefficient plot
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.plot(res_df["q"], res_df["beta_downside"], marker="o")
plt.axhline(0, linestyle="--")
plt.xlabel("Quantile")
plt.ylabel("Coefficient on BTC_DOWNSIDE_24")
plt.title("Quantile Regression Coefficients: RESIDUAL_FINAL")
plt.show()

# CELL 124
# model_raw_q10 = smf.quantreg(
#     "USDT_KP ~ BTC_DOWNSIDE_24 + DXY_ret",
#     df
# ).fit(q=0.1)

# model_raw_q90 = smf.quantreg(
#     "USDT_KP ~ BTC_DOWNSIDE_24 + DXY_ret",
#     df
# ).fit(q=0.9)

# print(model_raw_q10.summary())
# print(model_raw_q90.summary())

# CELL 125
# print("q=0.1 params")
# print(model_raw_q10.params)
# print(model_raw_q10.pvalues)

# print("\nq=0.9 params")
# print(model_raw_q90.params)
# print(model_raw_q90.pvalues)

# CELL 126
# raw_res = pd.DataFrame({
#     "q": [0.1, 0.9],
#     "beta_downside": [model_raw_q10.params["BTC_DOWNSIDE_24"],
#                       model_raw_q90.params["BTC_DOWNSIDE_24"]],
#     "p_downside": [model_raw_q10.pvalues["BTC_DOWNSIDE_24"],
#                    model_raw_q90.pvalues["BTC_DOWNSIDE_24"]],
#     "beta_dxy": [model_raw_q10.params["DXY_ret"],
#                  model_raw_q90.params["DXY_ret"]],
#     "p_dxy": [model_raw_q10.pvalues["DXY_ret"],
#               model_raw_q90.pvalues["DXY_ret"]],
# })
# print(raw_res)

# CELL 127
# import statsmodels.formula.api as smf
# import pandas as pd

# model_raw_q10 = smf.quantreg(
#     "USDT_KP ~ BTC_DOWNSIDE_24 + DXY_ret",
#     df
# ).fit(q=0.1)

# model_raw_q50 = smf.quantreg(
#     "USDT_KP ~ BTC_DOWNSIDE_24 + DXY_ret",
#     df
# ).fit(q=0.5)

# model_raw_q90 = smf.quantreg(
#     "USDT_KP ~ BTC_DOWNSIDE_24 + DXY_ret",
#     df
# ).fit(q=0.9)

# raw_quantile_res = pd.DataFrame({
#     "q": [0.1, 0.5, 0.9],
#     "beta_downside": [
#         model_raw_q10.params["BTC_DOWNSIDE_24"],
#         model_raw_q50.params["BTC_DOWNSIDE_24"],
#         model_raw_q90.params["BTC_DOWNSIDE_24"],
#     ],
#     "p_downside": [
#         model_raw_q10.pvalues["BTC_DOWNSIDE_24"],
#         model_raw_q50.pvalues["BTC_DOWNSIDE_24"],
#         model_raw_q90.pvalues["BTC_DOWNSIDE_24"],
#     ],
#     "beta_dxy": [
#         model_raw_q10.params["DXY_ret"],
#         model_raw_q50.params["DXY_ret"],
#         model_raw_q90.params["DXY_ret"],
#     ],
#     "p_dxy": [
#         model_raw_q10.pvalues["DXY_ret"],
#         model_raw_q50.pvalues["DXY_ret"],
#         model_raw_q90.pvalues["DXY_ret"],
#     ],
# })

# print(raw_quantile_res)

# CELL 129
# Cell 1: VIX level 붙이고 HIGH_STRESS 만들기

# df에는 현재 VIX_ret만 있고 VIX level은 없으므로 baseline_micro_df에서 붙임
if "VIX" not in df.columns:
    df = df.join(baseline_micro_df[["VIX"]], how="left")

print("Missing VIX after join:", df["VIX"].isna().sum())

# high-stress threshold: sample 75th percentile
vix_thr = df["VIX"].quantile(0.75)
df["HIGH_STRESS"] = (df["VIX"] >= vix_thr).astype(int)

print("VIX threshold:", vix_thr)
print(df["HIGH_STRESS"].value_counts(dropna=False))

# CELL 131
# Cell 2: regime sample 확인

df_high = df[df["HIGH_STRESS"] == 1].copy()
df_low  = df[df["HIGH_STRESS"] == 0].copy()

print("HIGH_STRESS shape:", df_high.shape)
print("LOW_STRESS shape :", df_low.shape)

print("\nHIGH_STRESS VIX summary")
print(df_high["VIX"].describe())

print("\nLOW_STRESS VIX summary")
print(df_low["VIX"].describe())

# CELL 132
# Cell 3: regime split OLS

import statsmodels.api as sm

def run_regime_ols(data, label):
    tmp = data[["RESIDUAL_FINAL", "BTC_DOWNSIDE_24", "DXY_ret"]].dropna().copy()

    X = tmp[["BTC_DOWNSIDE_24", "DXY_ret"]]
    X = sm.add_constant(X)
    y = tmp["RESIDUAL_FINAL"]

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    print(f"\n===== {label} =====")
    print(model.summary())

run_regime_ols(df_high, "HIGH STRESS")
run_regime_ols(df_low, "LOW STRESS")

# CELL 133
# Cell 4: interaction regression

tmp = df[["RESIDUAL_FINAL", "BTC_DOWNSIDE_24", "DXY_ret", "HIGH_STRESS"]].dropna().copy()
tmp["DOWN_x_STRESS"] = tmp["BTC_DOWNSIDE_24"] * tmp["HIGH_STRESS"]

X = tmp[["BTC_DOWNSIDE_24", "HIGH_STRESS", "DOWN_x_STRESS", "DXY_ret"]]
X = sm.add_constant(X)
y = tmp["RESIDUAL_FINAL"]

model_inter = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(model_inter.summary())

# CELL 134
# HIGH_STRESS가 언제 발생했는지 시간 확인
stress_dates = df.loc[df["HIGH_STRESS"] == 1].index.to_series()

print("First 20 HIGH_STRESS timestamps:")
print(stress_dates.head(20))

print("\nLast 20 HIGH_STRESS timestamps:")
print(stress_dates.tail(20))

# 날짜별 개수 집계
stress_by_day = stress_dates.dt.date.value_counts().sort_index()
print("\nHIGH_STRESS counts by day:")
print(stress_by_day.tail(30))

# CELL 137


# CELL 138
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12,3))
# plt.plot(df.index, df["HIGH_STRESS"], drawstyle="steps-post")
# plt.title("HIGH_STRESS over time")
# plt.xlabel("datetime_utc")
# plt.ylabel("HIGH_STRESS")
# plt.show()

# CELL 141
# df["USDT_KP"].describe()

# CELL 143
# df["BTC_DOWNSIDE_24_std"] = (
#     df["BTC_DOWNSIDE_24"] - df["BTC_DOWNSIDE_24"].mean()
# ) / df["BTC_DOWNSIDE_24"].std()
# standardized downside
df["BTC_DOWNSIDE_24_std"] = (
    df["BTC_DOWNSIDE_24"] - df["BTC_DOWNSIDE_24"].mean()
) / df["BTC_DOWNSIDE_24"].std()

# CELL 144
# import statsmodels.formula.api as smf

# quantiles = [0.1, 0.5, 0.9]

# results_std = []

# for q in quantiles:
#     mod = smf.quantreg(
#         "USDT_KP ~ BTC_DOWNSIDE_24_std + DXY_ret",
#         df
#     ).fit(q=q)

#     results_std.append({
#         "q": q,
#         "beta_downside_std": mod.params["BTC_DOWNSIDE_24_std"],
#         "p_downside_std": mod.pvalues["BTC_DOWNSIDE_24_std"],
#         "beta_dxy": mod.params["DXY_ret"],
#         "p_dxy": mod.pvalues["DXY_ret"],
#     })

# res_std_df = pd.DataFrame(results_std)
# print(res_std_df)

#beta_downside_std 음수나와야함
import statsmodels.formula.api as smf

quantiles = [0.1, 0.5, 0.9]
results_q = []

for q in quantiles:
    mod = smf.quantreg(
        "RESIDUAL_FINAL ~ BTC_DOWNSIDE_24_std + DXY_ret",
        df
    ).fit(q=q)

    results_q.append({
        "q": q,
        "beta_downside_std": mod.params["BTC_DOWNSIDE_24_std"],
        "p_downside_std": mod.pvalues["BTC_DOWNSIDE_24_std"],
        "beta_dxy": mod.params["DXY_ret"],
        "p_dxy": mod.pvalues["DXY_ret"],
    })

res_q_df = pd.DataFrame(results_q)
print(res_q_df)

# CELL 145
# Cell 1: standardized downside가 제대로 만들어졌는지 확인
print(df[["BTC_DOWNSIDE_24", "BTC_DOWNSIDE_24_std"]].describe())

corr_val = df[["BTC_DOWNSIDE_24", "BTC_DOWNSIDE_24_std"]].corr().iloc[0,1]
print("\nCorrelation:", corr_val)

# CELL 146
# Cell 2: 같은 종속변수(RESIDUAL_FINAL)로 standardized downside OLS 재확인
import statsmodels.formula.api as smf

mod_check = smf.ols(
    "RESIDUAL_FINAL ~ BTC_DOWNSIDE_24_std + DXY_ret",
    data=df
).fit()

print(mod_check.summary())

# CELL 147
# Cell 3: ETH full-hourly return / downside 생성
import pandas as pd
import numpy as np

eth_full = pd.DataFrame(index=binance.index.copy())
eth_full["ETH_CLOSE"] = binance["ETH_BINANCE_CLOSE"]

eth_full["ETH_RET"] = np.log(eth_full["ETH_CLOSE"]).diff()

eth_full["ETH_DOWNSIDE_24"] = (
    eth_full["ETH_RET"].clip(upper=0) ** 2
).rolling(window=24, min_periods=24).mean()

# 현재 분석 표본 df index에 맞춰 붙이기
df["ETH_DOWNSIDE_24"] = eth_full["ETH_DOWNSIDE_24"].reindex(df.index)

print(df[["ETH_DOWNSIDE_24"]].head())
print("\nMissing ETH_DOWNSIDE_24:", df["ETH_DOWNSIDE_24"].isna().sum())

# CELL 148
# Cell 4: ETH downside standardized
df["ETH_DOWNSIDE_24_std"] = (
    df["ETH_DOWNSIDE_24"] - df["ETH_DOWNSIDE_24"].mean()
) / df["ETH_DOWNSIDE_24"].std()

print(df[["ETH_DOWNSIDE_24", "ETH_DOWNSIDE_24_std"]].describe())

# CELL 149
# Cell 5: ETH downside quantile regression
import statsmodels.formula.api as smf
import pandas as pd

quantiles = [0.1, 0.5, 0.9]
results_eth = []

for q in quantiles:
    mod = smf.quantreg(
        "RESIDUAL_FINAL ~ ETH_DOWNSIDE_24_std + DXY_ret",
        df
    ).fit(q=q)

    results_eth.append({
        "q": q,
        "beta_eth_std": mod.params["ETH_DOWNSIDE_24_std"],
        "p_eth_std": mod.pvalues["ETH_DOWNSIDE_24_std"],
        "beta_dxy": mod.params["DXY_ret"],
        "p_dxy": mod.pvalues["DXY_ret"],
    })

res_eth_df = pd.DataFrame(results_eth)
print(res_eth_df)

# CELL 150
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df["RESIDUAL_FINAL"].dropna())

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Used lags:", adf_result[2])
print("N obs:", adf_result[3])
print("Critical values:")
for k, v in adf_result[4].items():
    print(f"  {k}: {v}")

# CELL 151
from statsmodels.tsa.stattools import kpss

kpss_result = kpss(df["RESIDUAL_FINAL"].dropna(), regression="c", nlags="auto")

print("KPSS Statistic:", kpss_result[0])
print("p-value:", kpss_result[1])
print("Used lags:", kpss_result[2])
print("Critical values:")
for k, v in kpss_result[3].items():
    print(f"  {k}: {v}")

# CELL 152
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(df["RESIDUAL_FINAL"].dropna(), lags=[5, 10, 20], return_df=True)
print(lb)

# CELL 153
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df["RESIDUAL_FINAL"].dropna(), lags=30, ax=axes[0])
plot_pacf(df["RESIDUAL_FINAL"].dropna(), lags=30, ax=axes[1], method="ywm")
axes[0].set_title("ACF of RESIDUAL_FINAL")
axes[1].set_title("PACF of RESIDUAL_FINAL")
plt.tight_layout()
plt.show()

# CELL 154


# CELL 155
# Cell 1: VAR용 데이터셋
var_df = df[["BTC_DOWNSIDE_24_std", "RESIDUAL_FINAL"]].dropna().copy()

print(var_df.shape)
print(var_df.head())
print(var_df.describe())

# CELL 156
# Cell 2: 단순 상관 확인
print(var_df.corr())

# CELL 157
# Cell 3: VAR lag selection
from statsmodels.tsa.api import VAR

var_model = VAR(var_df)
lag_sel = var_model.select_order(10)

print(lag_sel.summary())

# CELL 158
# Cell 4: VAR 적합 (메인 스펙: BIC 기준 lag 4)
p = 4

var_res = var_model.fit(p)

print(var_res.summary())

# CELL 159
from statsmodels.stats.diagnostic import acorr_ljungbox

resid = var_res.resid

print(acorr_ljungbox(resid["RESIDUAL_FINAL"], lags=[4,8,12], return_df=True))
print(acorr_ljungbox(resid["BTC_DOWNSIDE_24_std"], lags=[4,8,12], return_df=True))

# CELL 161
print("Is stable:", var_res.is_stable(verbose=True))

# CELL 162
irf = var_res.irf(10)

# CELL 163
irf.plot(orth=False)

# CELL 166
# irf.plot(impulse="BTC_DOWNSIDE_24_std", response="RESIDUAL_FINAL", orth=False)

# CELL 167


# CELL 168


# CELL 169


# CELL 171
# import matplotlib.pyplot as plt

# plt.plot(res_std_df["q"], res_std_df["beta_downside_std"], marker='o')
# plt.axhline(0, linestyle='--')

# plt.xlabel("Quantile")
# plt.ylabel("Coefficient (standardized downside)")
# plt.title("Quantile Regression (Standardized)")

# plt.show()

# CELL 172
# from statsmodels.tsa.api import VAR

# var_df = df[["RESIDUAL_OLS", "BTC_DOWNSIDE_24"]].dropna()

# model = VAR(var_df)
# results = model.fit(1)

# print(results.summary())

# CELL 174
# irf = results.irf(10)   # 10시간(또는 10 step) 반응

# irf.plot(orth=False)

# CELL 176
# model = VAR(var_df)

# lag_order = model.select_order(maxlags=10)
# print(lag_order.summary())

# CELL 178
# results_var2 = model.fit(2)
# print(results_var2.summary())

# irf2 = results_var2.irf(10)
# irf2.plot(orth=False)

# CELL 181
df_dyn = df.copy()

df_dyn["RESIDUAL_FINAL_lag1"] = df_dyn["RESIDUAL_FINAL"].shift(1)
df_dyn["BTC_DOWNSIDE_24_std_lag1"] = df_dyn["BTC_DOWNSIDE_24_std"].shift(1)
df_dyn["DXY_ret_lag1"] = df_dyn["DXY_ret"].shift(1)

df_dyn = df_dyn.dropna(subset=[
    "RESIDUAL_FINAL",
    "RESIDUAL_FINAL_lag1",
    "BTC_DOWNSIDE_24_std_lag1",
    "DXY_ret_lag1"
])

print("df_dyn shape:", df_dyn.shape)
print(df_dyn[[
    "RESIDUAL_FINAL",
    "RESIDUAL_FINAL_lag1",
    "BTC_DOWNSIDE_24_std_lag1",
    "DXY_ret_lag1"
]].head())

# CELL 182
import statsmodels.formula.api as smf

mod_q10_dyn = smf.quantreg(
    "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 + BTC_DOWNSIDE_24_std_lag1 + DXY_ret_lag1",
    df_dyn
).fit(q=0.1, max_iter=5000)

mod_q50_dyn = smf.quantreg(
    "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 + BTC_DOWNSIDE_24_std_lag1 + DXY_ret_lag1",
    df_dyn
).fit(q=0.5, max_iter=5000)

mod_q90_dyn = smf.quantreg(
    "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 + BTC_DOWNSIDE_24_std_lag1 + DXY_ret_lag1",
    df_dyn
).fit(q=0.9, max_iter=5000)

print(mod_q10_dyn.summary())
print(mod_q50_dyn.summary())
print(mod_q90_dyn.summary())

# CELL 183
import pandas as pd

horizons = [1, 2, 3, 6, 12]
quantiles = [0.1, 0.5, 0.9]

results_lp = []

for h in horizons:
    temp = df_dyn.copy()
    temp[f"RESIDUAL_lead_{h}"] = temp["RESIDUAL_FINAL"].shift(-h)

    temp = temp.dropna(subset=[f"RESIDUAL_lead_{h}"])

    print(f"h={h}, sample size={len(temp)}")

    for q in quantiles:
        mod = smf.quantreg(
            f"RESIDUAL_lead_{h} ~ RESIDUAL_FINAL + BTC_DOWNSIDE_24_std_lag1 + DXY_ret_lag1",
            temp
        ).fit(q=q, max_iter=5000)

        results_lp.append({
            "h": h,
            "q": q,
            "beta": mod.params["BTC_DOWNSIDE_24_std_lag1"],
            "p_beta": mod.pvalues["BTC_DOWNSIDE_24_std_lag1"],
            "n": len(temp),
        })

lp_df = pd.DataFrame(results_lp)

print(lp_df)
print(lp_df.pivot(index="h", columns="q", values="beta"))

# CELL 186
# # 1) 결과를 DataFrame으로 정리
# lp_df = pd.DataFrame(results_lp)

# print(lp_df)

# CELL 187
# # 2) 분위수별 계수 경로 표
# lp_pivot = lp_df.pivot(index="h", columns="q", values="beta")
# print(lp_pivot)

# CELL 188
# horizons = [1, 2, 3, 6, 12]
# quantiles = [0.1, 0.5, 0.9]

# results_lp = []
# sample_sizes = []

# for h in horizons:
#     temp = df_dyn.copy()
#     temp[f"RESIDUAL_lead_{h}"] = temp["RESIDUAL_OLS"].shift(-h)

#     temp = temp.dropna(subset=[
#         f"RESIDUAL_lead_{h}",
#         "BTC_DOWNSIDE_24_lag1"
#     ])

#     sample_sizes.append({"h": h, "n": len(temp)})

#     for q in quantiles:
#         mod = smf.quantreg(
#             f"RESIDUAL_lead_{h} ~ BTC_DOWNSIDE_24_lag1",
#             temp
#         ).fit(q=q, max_iter=5000)

#         results_lp.append({
#             "h": h,
#             "q": q,
#             "beta": mod.params["BTC_DOWNSIDE_24_lag1"],
#             "intercept": mod.params["Intercept"],
#             "p_beta": mod.pvalues["BTC_DOWNSIDE_24_lag1"],
#             "n": len(temp),
#         })

# lp_df = pd.DataFrame(results_lp)
# n_df = pd.DataFrame(sample_sizes)

# print("Sample sizes by horizon:")
# print(n_df)

# print("\nLocal projection quantile results:")
# print(lp_df)

# print("\nPivot table of beta:")
# print(lp_df.pivot(index="h", columns="q", values="beta"))

# CELL 189
# import matplotlib.pyplot as plt

# pivot_beta = lp_df.pivot(index="h", columns="q", values="beta")

# for q in pivot_beta.columns:
#     plt.plot(pivot_beta.index, pivot_beta[q], marker="o", label=f"q={q}")

# plt.axhline(0, linestyle="--")
# plt.xlabel("Horizon")
# plt.ylabel("Beta on BTC_DOWNSIDE_24_lag1")
# plt.title("Quantile Local Projection Coefficients")
# plt.legend()
# plt.show()

# CELL 190
# =========================================
# Timeline figure: event windows + residual jumps
# Assumption: df already exists in memory
# Required columns:
#   - RESIDUAL_FINAL
# Optional column:
#   - BTC_DOWNSIDE_24_std  (if exists, second panel plots it)
# Index:
#   - datetime_utc (UTC) or convertible datetime index
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 0) Basic prep
# -----------------------------
plot_df = df.copy()

# Ensure datetime index
if not isinstance(plot_df.index, pd.DatetimeIndex):
    if "datetime_utc" in plot_df.columns:
        plot_df["datetime_utc"] = pd.to_datetime(plot_df["datetime_utc"], utc=True)
        plot_df = plot_df.set_index("datetime_utc")
    else:
        raise ValueError("DatetimeIndex가 없고 'datetime_utc' 컬럼도 없습니다.")

# Ensure UTC
if plot_df.index.tz is None:
    plot_df.index = plot_df.index.tz_localize("UTC")
else:
    plot_df.index = plot_df.index.tz_convert("UTC")

# Keep needed columns only
needed_cols = ["RESIDUAL_FINAL"]
optional_cols = [c for c in ["BTC_DOWNSIDE_24_std"] if c in plot_df.columns]
plot_df = plot_df[needed_cols + optional_cols].sort_index().dropna(subset=["RESIDUAL_FINAL"]).copy()

print("plot_df shape:", plot_df.shape)
print("sample start:", plot_df.index.min())
print("sample end  :", plot_df.index.max())

# -----------------------------
# 1) Define event windows
# -----------------------------
event_windows = {
    "2025-10-10/11 crypto crash": (
        pd.Timestamp("2025-10-10 00:00:00", tz="UTC"),
        pd.Timestamp("2025-10-11 23:59:59", tz="UTC")
    ),
    "2026-01-31~2026-02-05 crypto slump/liquidation": (
        pd.Timestamp("2026-01-31 00:00:00", tz="UTC"),
        pd.Timestamp("2026-02-05 23:59:59", tz="UTC")
    ),
}

# -----------------------------
# 2) Detect residual jump points
#    Rule: top 1% absolute hourly changes
# -----------------------------
plot_df["RESIDUAL_DIFF"] = plot_df["RESIDUAL_FINAL"].diff()
plot_df["ABS_RESIDUAL_DIFF"] = plot_df["RESIDUAL_DIFF"].abs()

jump_threshold = plot_df["ABS_RESIDUAL_DIFF"].quantile(0.99)
jump_df = plot_df.loc[plot_df["ABS_RESIDUAL_DIFF"] >= jump_threshold, ["RESIDUAL_FINAL", "RESIDUAL_DIFF", "ABS_RESIDUAL_DIFF"]].copy()

print("\nJump threshold (top 1% abs change):", jump_threshold)
print("\nResidual jump points:")
print(jump_df.sort_values("ABS_RESIDUAL_DIFF", ascending=False).head(20))

# For labeling on the figure: use top 8 largest jumps
top_jumps = jump_df.sort_values("ABS_RESIDUAL_DIFF", ascending=False).head(8).copy()

# -----------------------------
# 3) Make figure
# -----------------------------
use_two_panels = "BTC_DOWNSIDE_24_std" in plot_df.columns

if use_two_panels:
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    ax1, ax2 = axes
else:
    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax2 = None

# ---- Top panel: residual
ax1.plot(plot_df.index, plot_df["RESIDUAL_FINAL"], linewidth=1.2, label="RESIDUAL_FINAL")
ax1.scatter(top_jumps.index, top_jumps["RESIDUAL_FINAL"], s=35, zorder=5, label="Top residual jumps")

# Shade event windows
for label, (start, end) in event_windows.items():
    ax1.axvspan(start, end, alpha=0.18)
    # place label near the top of current y-range
    y_top = ax1.get_ylim()[1]
    ax1.text(start, y_top, label, fontsize=9, va="top", ha="left", rotation=90)

# Annotate jump timestamps
for ts, row in top_jumps.iterrows():
    ax1.annotate(
        ts.strftime("%Y-%m-%d %H:%M"),
        xy=(ts, row["RESIDUAL_FINAL"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

ax1.set_title("Sample Timeline: Korean USDT Residual, Event Windows, and Residual Jump Points")
ax1.set_ylabel("RESIDUAL_FINAL")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.3)

# ---- Bottom panel: downside proxy if available
if use_two_panels:
    ax2.plot(plot_df.index, plot_df["BTC_DOWNSIDE_24_std"], linewidth=1.1, label="BTC_DOWNSIDE_24_std")
    for label, (start, end) in event_windows.items():
        ax2.axvspan(start, end, alpha=0.18)
    ax2.set_ylabel("BTC downside (std)")
    ax2.set_xlabel("UTC time")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)
else:
    ax1.set_xlabel("UTC time")

plt.tight_layout()
plt.show()

# -----------------------------
# 4) Event-window summary table
# -----------------------------
summary_rows = []
for label, (start, end) in event_windows.items():
    sub = plot_df.loc[(plot_df.index >= start) & (plot_df.index <= end)].copy()
    if len(sub) == 0:
        summary_rows.append({
            "event": label,
            "n_obs": 0,
            "residual_mean": np.nan,
            "residual_min": np.nan,
            "residual_max": np.nan,
            "abs_diff_max": np.nan,
        })
    else:
        summary_rows.append({
            "event": label,
            "n_obs": len(sub),
            "residual_mean": sub["RESIDUAL_FINAL"].mean(),
            "residual_min": sub["RESIDUAL_FINAL"].min(),
            "residual_max": sub["RESIDUAL_FINAL"].max(),
            "abs_diff_max": sub["ABS_RESIDUAL_DIFF"].max(),
        })

event_summary = pd.DataFrame(summary_rows)
print("\nEvent-window summary:")
print(event_summary)

# -----------------------------
# 5) Which jump points fall inside event windows?
# -----------------------------
def which_event(ts):
    hits = []
    for label, (start, end) in event_windows.items():
        if start <= ts <= end:
            hits.append(label)
    return ", ".join(hits) if hits else "Outside event windows"

jump_event_map = top_jumps.copy()
jump_event_map["event_window"] = [which_event(ts) for ts in jump_event_map.index]

print("\nTop jump points mapped to event windows:")
print(jump_event_map[["RESIDUAL_FINAL", "RESIDUAL_DIFF", "ABS_RESIDUAL_DIFF", "event_window"]])

# CELL 191
# -----------------------------
# 1) Downside regime 정의
# -----------------------------
df_reg = df.copy()

# datetime index 정리 (혹시 필요하면)
if not isinstance(df_reg.index, pd.DatetimeIndex):
    df_reg["datetime_utc"] = pd.to_datetime(df_reg["datetime_utc"], utc=True)
    df_reg = df_reg.set_index("datetime_utc")

# downside threshold (상위 10%)
threshold = df_reg["BTC_DOWNSIDE_24_std"].quantile(0.9)

# regime dummy
df_reg["DOWNSIDE"] = (df_reg["BTC_DOWNSIDE_24_std"] >= threshold).astype(int)

print("Downside threshold:", threshold)
print(df_reg["DOWNSIDE"].value_counts())

# CELL 192
# -----------------------------
# 2) Residual 통계 비교
# -----------------------------
summary = df_reg.groupby("DOWNSIDE")["RESIDUAL_FINAL"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max",
    skew="skew"
)

print("\nDownside vs Normal summary:")
print(summary)

# CELL 193
# -----------------------------
# 3) Quantile 비교 (tail)
# -----------------------------
quantiles = df_reg.groupby("DOWNSIDE")["RESIDUAL_FINAL"].quantile([0.01, 0.05, 0.1, 0.5, 0.9]).unstack()

print("\nQuantile comparison:")
print(quantiles)

# CELL 194
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.boxplot(x="DOWNSIDE", y="RESIDUAL_FINAL", data=df_reg)
plt.title("Residual Distribution: Downside vs Normal")
plt.xlabel("Downside (1 = stress)")
plt.ylabel("RESIDUAL_FINAL")
plt.grid(alpha=0.3)
plt.show()

# CELL 195
from scipy.stats import ttest_ind

res_down = df_reg[df_reg["DOWNSIDE"] == 1]["RESIDUAL_FINAL"]
res_norm = df_reg[df_reg["DOWNSIDE"] == 0]["RESIDUAL_FINAL"]

t_stat, p_val = ttest_ind(res_down, res_norm, equal_var=False)

print("\nT-test:")
print("t-stat:", t_stat)
print("p-value:", p_val)

# CELL 196
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# =========================================================
# 0) 기본 준비
# =========================================================
df_plot = df.copy()

if not isinstance(df_plot.index, pd.DatetimeIndex):
    if "datetime_utc" in df_plot.columns:
        df_plot["datetime_utc"] = pd.to_datetime(df_plot["datetime_utc"], utc=True)
        df_plot = df_plot.set_index("datetime_utc")
    else:
        raise ValueError("DatetimeIndex가 없고 datetime_utc 컬럼도 없습니다.")

if df_plot.index.tz is None:
    df_plot.index = df_plot.index.tz_localize("UTC")
else:
    df_plot.index = df_plot.index.tz_convert("UTC")

needed = ["RESIDUAL_FINAL", "BTC_DOWNSIDE_24_std"]
missing = [c for c in needed if c not in df_plot.columns]
if missing:
    raise ValueError(f"필요 컬럼 누락: {missing}")

df_plot = df_plot[needed].dropna().sort_index().copy()

# =========================================================
# 1) downside regime 정의
#    상위 10%를 downside stress로 정의
# =========================================================
threshold = df_plot["BTC_DOWNSIDE_24_std"].quantile(0.90)
df_plot["DOWNSIDE"] = (df_plot["BTC_DOWNSIDE_24_std"] >= threshold).astype(int)

print("Downside threshold:", threshold)
print(df_plot["DOWNSIDE"].value_counts())

# =========================================================
# 2) Figure 1: downside indicator 시계열
#    - threshold 선
#    - downside 구간 shading
# =========================================================
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(df_plot.index, df_plot["BTC_DOWNSIDE_24_std"], linewidth=1.2, label="BTC_DOWNSIDE_24_std")
ax.axhline(threshold, linestyle="--", linewidth=1.2, label=f"90th percentile threshold = {threshold:.3f}")

# downside 구간 shading
down_mask = df_plot["DOWNSIDE"] == 1
start = None
for i, is_down in enumerate(down_mask):
    if is_down and start is None:
        start = df_plot.index[i]
    elif (not is_down) and start is not None:
        end = df_plot.index[i-1]
        ax.axvspan(start, end, alpha=0.18)
        start = None
if start is not None:
    ax.axvspan(start, df_plot.index[-1], alpha=0.18)

ax.set_title("Figure 1. Downside Stress Indicator over the Sample")
ax.set_ylabel("BTC_DOWNSIDE_24_std")
ax.set_xlabel("UTC time")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# 3) Table 1: downside vs normal 통계 비교
# =========================================================
summary = df_plot.groupby("DOWNSIDE")["RESIDUAL_FINAL"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    q01=lambda x: x.quantile(0.01),
    q05=lambda x: x.quantile(0.05),
    q10=lambda x: x.quantile(0.10),
    median="median",
    q90=lambda x: x.quantile(0.90),
    max="max",
    skew="skew"
)

summary.index = ["Normal", "Downside"]
print("\nTable 1. Residual distribution: Downside vs Normal")
print(summary.round(6))

# =========================================================
# 4) t-test
# =========================================================
res_norm = df_plot.loc[df_plot["DOWNSIDE"] == 0, "RESIDUAL_FINAL"]
res_down = df_plot.loc[df_plot["DOWNSIDE"] == 1, "RESIDUAL_FINAL"]

t_stat, p_val = ttest_ind(res_down, res_norm, equal_var=False)

print("\nT-test (Downside vs Normal)")
print("t-stat :", round(t_stat, 6))
print("p-value:", p_val)

# =========================================================
# 5) Figure 2: boxplot
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5))

data_to_plot = [
    df_plot.loc[df_plot["DOWNSIDE"] == 0, "RESIDUAL_FINAL"],
    df_plot.loc[df_plot["DOWNSIDE"] == 1, "RESIDUAL_FINAL"]
]

ax.boxplot(data_to_plot, labels=["Normal", "Downside"])
ax.set_title("Figure 2. Residual Distribution: Downside vs Normal")
ax.set_ylabel("RESIDUAL_FINAL")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# CELL 197


# CELL 198


# CELL 199


# CELL 201
print(df.columns.tolist())

# CELL 202
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# regime 분석용 master panel
df_regime = pd.read_csv(
    "baseline_dataset_with_micro.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)

df_regime.index = pd.to_datetime(df_regime.index, utc=True)

print(df_regime.shape)
print(df_regime.columns.tolist())

# CELL 203
import statsmodels.api as sm

# Step 1에서 했던 residual 생성 다시
X = sm.add_constant(df_regime["MKT_KP_EQ"])
y = df_regime["USDT_KP"]

model_resid = sm.OLS(y, X, missing="drop").fit()

# index 맞춰서 저장
df_regime["RESIDUAL_OLS"] = np.nan
df_regime.loc[model_resid.resid.index, "RESIDUAL_OLS"] = model_resid.resid

# CELL 204
# stress regime: VIX upper 20%
vix_thr = df_regime["VIX"].quantile(0.8)

df_regime["HIGH_STRESS"] = (df_regime["VIX"] >= vix_thr).astype(int)
df_regime["LOW_STRESS"] = (df_regime["VIX"] < vix_thr).astype(int)

print("VIX threshold:", vix_thr)
print(df_regime["HIGH_STRESS"].value_counts(dropna=False))

# CELL 205
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# 1) master panel load
# =========================
df_regime = pd.read_csv(
    "baseline_dataset_with_micro.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)
df_regime.index = pd.to_datetime(df_regime.index, utc=True)

# =========================
# 2) RESIDUAL_OLS 생성 (없으면)
# =========================
if "RESIDUAL_OLS" not in df_regime.columns:
    X = sm.add_constant(df_regime["MKT_KP_EQ"])
    y = df_regime["USDT_KP"]

    model_resid = sm.OLS(y, X, missing="drop").fit()

    df_regime["RESIDUAL_OLS"] = np.nan
    df_regime.loc[model_resid.resid.index, "RESIDUAL_OLS"] = model_resid.resid

# =========================
# 3) HIGH_STRESS 생성
# =========================
vix_thr = df_regime["VIX"].quantile(0.8)
df_regime["HIGH_STRESS"] = (df_regime["VIX"] >= vix_thr).astype(int)

# =========================
# 4) lag 변수 생성
# =========================
df_dyn = df_regime.copy()

df_dyn["RESIDUAL_OLS_lag1"] = df_dyn["RESIDUAL_OLS"].shift(1)
df_dyn["BTC_DOWNSIDE_24_lag1"] = df_dyn["BTC_DOWNSIDE_24"].shift(1)
df_dyn["DXY_ret_lag1"] = df_dyn["DXY_ret"].shift(1)
df_dyn["HIGH_STRESS_lag1"] = df_dyn["HIGH_STRESS"].shift(1)

df_dyn = df_dyn.dropna(subset=[
    "RESIDUAL_OLS",
    "RESIDUAL_OLS_lag1",
    "BTC_DOWNSIDE_24_lag1",
    "DXY_ret_lag1",
    "HIGH_STRESS_lag1"
])

print("df_dyn shape:", df_dyn.shape)

# =========================
# 5) regime interaction local projection (corrected)
# =========================
horizons = [1, 2, 3, 6, 12]
results_regime = []

for h in horizons:
    temp = df_dyn.copy()
    temp[f"RESIDUAL_lead_{h}"] = temp["RESIDUAL_OLS"].shift(-h)

    temp = temp.dropna(subset=[
        f"RESIDUAL_lead_{h}",
        "RESIDUAL_OLS_lag1",
        "BTC_DOWNSIDE_24_lag1",
        "DXY_ret_lag1",
        "HIGH_STRESS_lag1"
    ])

    for q in [0.1, 0.5, 0.9]:
        mod = smf.quantreg(
            f"""
            RESIDUAL_lead_{h} ~ 
            RESIDUAL_OLS_lag1
            + DXY_ret_lag1
            + HIGH_STRESS_lag1
            + BTC_DOWNSIDE_24_lag1
            + BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1
            """,
            temp
        ).fit(q=q, max_iter=5000)

        beta_normal = mod.params["BTC_DOWNSIDE_24_lag1"]
        interaction_name = "BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1"
        beta_interaction = mod.params[interaction_name]
        p_interaction = mod.pvalues[interaction_name]

        results_regime.append({
            "h": h,
            "q": q,
            "n": len(temp),
            "beta_normal": beta_normal,
            "beta_stress": beta_normal + beta_interaction,
            "interaction": beta_interaction,
            "p_interaction": p_interaction,
            "beta_regime_main": mod.params["HIGH_STRESS_lag1"],
            "p_regime_main": mod.pvalues["HIGH_STRESS_lag1"],
            "beta_resid_lag1": mod.params["RESIDUAL_OLS_lag1"],
            "p_resid_lag1": mod.pvalues["RESIDUAL_OLS_lag1"],
        })

df_regime_lp = pd.DataFrame(results_regime)

print("\nRegime LP results:")
print(df_regime_lp)

print("\nPivot: beta_normal")
print(df_regime_lp.pivot(index="h", columns="q", values="beta_normal"))

print("\nPivot: beta_stress")
print(df_regime_lp.pivot(index="h", columns="q", values="beta_stress"))

print("\nPivot: interaction")
print(df_regime_lp.pivot(index="h", columns="q", values="interaction"))

print("\nPivot: p_interaction")
print(df_regime_lp.pivot(index="h", columns="q", values="p_interaction"))

# CELL 206
import matplotlib.pyplot as plt

for q in [0.1, 0.5, 0.9]:
    subset = df_regime_lp[df_regime_lp["q"] == q]

    plt.plot(subset["h"], subset["beta_normal"], label="Normal")
    plt.plot(subset["h"], subset["beta_stress"], label="Stress")

    plt.title(f"IRF (q={q})")
    plt.legend()
    plt.show()

# CELL 207
horizons = [1, 2, 3, 6, 12]
results_dollar = []

for h in horizons:
    temp = df_dyn.copy()
    temp[f"RESIDUAL_lead_{h}"] = temp["RESIDUAL_OLS"].shift(-h)

    temp = temp.dropna(subset=[
        f"RESIDUAL_lead_{h}",
        "RESIDUAL_OLS_lag1",
        "BTC_DOWNSIDE_24_lag1",
        "DXY_ret_lag1",
        "HIGH_STRESS_lag1"
    ])

    for q in [0.1, 0.5, 0.9]:
        mod = smf.quantreg(
            f"""
            RESIDUAL_lead_{h} ~ 
            RESIDUAL_OLS_lag1
            + HIGH_STRESS_lag1
            + BTC_DOWNSIDE_24_lag1
            + BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1
            + DXY_ret_lag1
            + DXY_ret_lag1:HIGH_STRESS_lag1
            """,
            temp
        ).fit(q=q, max_iter=5000)

        results_dollar.append({
            "h": h,
            "q": q,
            "beta_dxy": mod.params["DXY_ret_lag1"],
            "beta_dxy_stress": mod.params["DXY_ret_lag1"] 
                               + mod.params["DXY_ret_lag1:HIGH_STRESS_lag1"],
            "p_dxy_interaction": mod.pvalues["DXY_ret_lag1:HIGH_STRESS_lag1"]
        })

df_dollar = pd.DataFrame(results_dollar)

print(df_dollar)
print(df_dollar.pivot(index="h", columns="q", values="beta_dxy_stress"))

# CELL 210
core_cols = [
    "RESIDUAL_OLS",
    "RESIDUAL_OLS_lag1",
    "HIGH_STRESS_lag1",
    "BTC_DOWNSIDE_24_lag1",
    "DXY_ret_lag1",
    "FUNDING_lag1",
    "OI_lag1",
    "LIQUIDATION_lag1"
]

existing_cols = [c for c in core_cols if c in df_dyn.columns]
missing_cols = [c for c in core_cols if c not in df_dyn.columns]

print("=== Existing columns ===")
print(existing_cols)

print("\n=== Missing columns ===")
print(missing_cols)

print("\n=== Missing ratio ===")
print(df_dyn[existing_cols].isna().mean().sort_values())

print("\n=== Summary stats ===")
print(df_dyn[existing_cols].describe().T)

# CELL 211
check_cols = [
    "RESIDUAL_OLS",
    "RESIDUAL_OLS_lag1",
    "HIGH_STRESS_lag1",
    "BTC_DOWNSIDE_24_lag1",
    "DXY_ret_lag1"
]

print(df_dyn[check_cols].isna().mean().sort_values())
print(df_dyn[check_cols].describe().T)

# CELL 212
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

horizons = [1, 2, 3, 6, 12]
quantiles = [0.1, 0.5, 0.9]

results_base = []

for h in horizons:
    temp = df_dyn.copy()
    temp[f"RESIDUAL_lead_{h}"] = temp["RESIDUAL_OLS"].shift(-h)

    need_cols = [
        f"RESIDUAL_lead_{h}",
        "RESIDUAL_OLS_lag1",
        "HIGH_STRESS_lag1",
        "BTC_DOWNSIDE_24_lag1",
        "DXY_ret_lag1"
    ]
    temp = temp.dropna(subset=need_cols)

    for q in quantiles:
        mod = smf.quantreg(
            f"""
            RESIDUAL_lead_{h} ~
            RESIDUAL_OLS_lag1
            + HIGH_STRESS_lag1
            + BTC_DOWNSIDE_24_lag1
            + BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1
            + DXY_ret_lag1
            + DXY_ret_lag1:HIGH_STRESS_lag1
            """,
            temp
        ).fit(q=q, max_iter=5000)

        results_base.append({
            "h": h,
            "q": q,
            "n": len(temp),
            "beta_dxy": mod.params.get("DXY_ret_lag1", np.nan),
            "beta_dxy_inter": mod.params.get("DXY_ret_lag1:HIGH_STRESS_lag1", np.nan),
            "beta_dxy_stress_total": mod.params.get("DXY_ret_lag1", 0.0)
                                     + mod.params.get("DXY_ret_lag1:HIGH_STRESS_lag1", 0.0),
            "p_dxy": mod.pvalues.get("DXY_ret_lag1", np.nan),
            "p_dxy_inter": mod.pvalues.get("DXY_ret_lag1:HIGH_STRESS_lag1", np.nan)
        })

df_base = pd.DataFrame(results_base)
print(df_base.round(4))

# CELL 215
[c for c in df_dyn.columns if "KRW" in c.upper() or "USD" in c.upper() or "FX" in c.upper() or "EXCHANGE" in c.upper()]

# CELL 216
import numpy as np

df_dyn = df_dyn.copy()
df_dyn["USDKRW_ret"] = np.log(df_dyn["USDKRW"]).diff()

# CELL 217
df_dyn["USDKRW_ret_lag1"] = df_dyn["USDKRW_ret"].shift(1)

# CELL 218
check_cols_fx = [
    "USDKRW",
    "USDKRW_ret",
    "USDKRW_ret_lag1",
    "RESIDUAL_OLS",
    "RESIDUAL_OLS_lag1",
    "HIGH_STRESS_lag1",
    "BTC_DOWNSIDE_24_lag1"
]

print(df_dyn[check_cols_fx].isna().mean().sort_values())
print(df_dyn[check_cols_fx].describe().T)

# CELL 219
import pandas as pd
import statsmodels.formula.api as smf

horizons = [1, 2, 3, 6, 12]
quantiles = [0.1, 0.5, 0.9]

results_fx = []

for h in horizons:
    temp = df_dyn.copy()
    temp[f"RESIDUAL_lead_{h}"] = temp["RESIDUAL_OLS"].shift(-h)

    need_cols = [
        f"RESIDUAL_lead_{h}",
        "RESIDUAL_OLS_lag1",
        "HIGH_STRESS_lag1",
        "BTC_DOWNSIDE_24_lag1",
        "USDKRW_ret_lag1"
    ]
    temp = temp.dropna(subset=need_cols)

    for q in quantiles:
        mod = smf.quantreg(
            f"""
            RESIDUAL_lead_{h} ~
            RESIDUAL_OLS_lag1
            + HIGH_STRESS_lag1
            + BTC_DOWNSIDE_24_lag1
            + BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1
            + USDKRW_ret_lag1
            + USDKRW_ret_lag1:HIGH_STRESS_lag1
            """,
            temp
        ).fit(q=q, max_iter=5000)

        results_fx.append({
            "h": h,
            "q": q,
            "n": len(temp),
            "beta_fx": mod.params.get("USDKRW_ret_lag1", np.nan),
            "beta_fx_inter": mod.params.get("USDKRW_ret_lag1:HIGH_STRESS_lag1", np.nan),
            "beta_fx_stress_total": mod.params.get("USDKRW_ret_lag1", 0.0)
                                    + mod.params.get("USDKRW_ret_lag1:HIGH_STRESS_lag1", 0.0),
            "p_fx": mod.pvalues.get("USDKRW_ret_lag1", np.nan),
            "p_fx_inter": mod.pvalues.get("USDKRW_ret_lag1:HIGH_STRESS_lag1", np.nan)
        })

df_fx = pd.DataFrame(results_fx)
print(df_fx.round(4))

# CELL 221
compare_fx_dxy = df_base.merge(df_fx, on=["h", "q"], how="outer")
print(compare_fx_dxy.round(4))

# CELL 222
cols_compare = [
    "h", "q",
    "beta_dxy", "p_dxy", "beta_dxy_inter", "p_dxy_inter", "beta_dxy_stress_total",
    "beta_fx", "p_fx", "beta_fx_inter", "p_fx_inter", "beta_fx_stress_total"
]

print(compare_fx_dxy[cols_compare].round(4))

# CELL 224
import requests
import pandas as pd
import numpy as np
import time

BASE = "https://fapi.binance.com"

def get_json(url, params=None, timeout=20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# 1) Funding rate history
# 공식: GET /fapi/v1/fundingRate
# 최대 1000개, startTime/endTime(ms) 지원
def fetch_funding_rate(symbol="BTCUSDT", start_ms=None, end_ms=None, limit=1000, sleep=0.2):
    out = []
    cur = start_ms

    while True:
        params = {"symbol": symbol, "limit": limit}
        if cur is not None:
            params["startTime"] = cur
        if end_ms is not None:
            params["endTime"] = end_ms

        data = get_json(f"{BASE}/fapi/v1/fundingRate", params=params)
        if not data:
            break

        out.extend(data)

        # 마지막 fundingTime 다음부터 이어받기
        last_t = data[-1]["fundingTime"]
        if len(data) < limit:
            break
        if end_ms is not None and last_t >= end_ms:
            break

        cur = last_t + 1
        time.sleep(sleep)

    df = pd.DataFrame(out)
    if df.empty:
        return df

    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["symbol", "fundingTime", "fundingRate"]].drop_duplicates()


# 2) Current open interest snapshot
# 공식: GET /fapi/v1/openInterest
def fetch_open_interest_snapshot(symbol="BTCUSDT"):
    data = get_json(f"{BASE}/fapi/v1/openInterest", params={"symbol": symbol})
    df = pd.DataFrame([data])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
    return df[["symbol", "time", "openInterest"]]


# 3) Historical open interest statistics
# 공식: GET /futures/data/openInterestHist
# period: "5m","15m","30m","1h","2h","4h","6h","12h","1d"
def fetch_open_interest_hist(symbol="BTCUSDT", period="1h", start_ms=None, end_ms=None, limit=500):
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit,
    }
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    data = get_json(f"{BASE}/futures/data/openInterestHist", params=params)
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Binance 응답 필드명은 문서/버전별로 약간 다를 수 있어 방어적으로 처리
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "time" in df.columns:
        ts_col = "time"
    else:
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    df["datetime"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)

    # 흔히 sumOpenInterest, sumOpenInterestValue가 옴
    for c in ["sumOpenInterest", "sumOpenInterestValue", "CMCCirculatingSupply"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = ["symbol", "datetime"]
    for c in ["sumOpenInterest", "sumOpenInterestValue", "CMCCirculatingSupply"]:
        if c in df.columns:
            keep.append(c)

    return df[keep].drop_duplicates().sort_values("datetime")

# CELL 225
start = pd.Timestamp("2025-06-02 09:00:00", tz="UTC")
end   = pd.Timestamp("2026-03-19 12:00:00", tz="UTC")

start_ms = int(start.timestamp() * 1000)
end_ms   = int(end.timestamp() * 1000)

df_funding = fetch_funding_rate(
    symbol="BTCUSDT",
    start_ms=start_ms,
    end_ms=end_ms,
    limit=1000
)

print(df_funding.head())
print(df_funding.tail())
print(len(df_funding))

# CELL 227
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================================================
# Corrected Step 0: Load existing collected/preprocessed data
# =========================================================
# Data collection cells are intentionally left untouched.
# This file already contains KP, exogenous variables, micro variables, and USDT_BINANCE_CLOSE.
df_main = pd.read_csv(
    "baseline_dataset_with_micro.csv",
    parse_dates=["datetime_utc"],
    index_col="datetime_utc"
)
df_main.index = pd.to_datetime(df_main.index, utc=True)

required_cols = [
    "USDT_KP", "MKT_KP_EQ", "USDT_BINANCE_CLOSE",
    "BTC_DOWNSIDE_24", "DXY_ret", "USDKRW_ret", "VIX"
]
missing = [c for c in required_cols if c not in df_main.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

df_main = df_main.dropna(subset=required_cols).copy()

# =========================================================
# Corrected Step 1: Remove market-common Korean premium
# =========================================================
step1 = df_main[["USDT_KP", "MKT_KP_EQ"]].dropna().copy()
model_market = sm.OLS(
    step1["USDT_KP"],
    sm.add_constant(step1["MKT_KP_EQ"])
).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

df_main["RESIDUAL_OLS"] = np.nan
df_main.loc[step1.index, "RESIDUAL_OLS"] = model_market.resid

# =========================================================
# Corrected Step 2: Remove global stablecoin depeg proxy
# =========================================================
# In the collection block, USDT_BINANCE_CLOSE is Binance USDCUSDT.
# Treat abs(USDCUSDT - 1) as a Binance stablecoin-pair depeg proxy.
df_main["DEPEG_GLOBAL"] = (df_main["USDT_BINANCE_CLOSE"] - 1.0).abs()

step2 = df_main[["RESIDUAL_OLS", "DEPEG_GLOBAL"]].dropna().copy()
model_depeg = sm.OLS(
    step2["RESIDUAL_OLS"],
    sm.add_constant(step2["DEPEG_GLOBAL"])
).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

df_main["RESIDUAL_FINAL"] = np.nan
df_main.loc[step2.index, "RESIDUAL_FINAL"] = model_depeg.resid

# =========================================================
# Corrected Step 3: Lagged variables for main identification
# =========================================================
df_main["RESIDUAL_FINAL_lag1"] = df_main["RESIDUAL_FINAL"].shift(1)
df_main["BTC_DOWNSIDE_24_lag1"] = df_main["BTC_DOWNSIDE_24"].shift(1)
df_main["DXY_ret_lag1"] = df_main["DXY_ret"].shift(1)
df_main["USDKRW_ret_lag1"] = df_main["USDKRW_ret"].shift(1)

# VIX stress regime is a comparison benchmark, not the main identification device.
vix_threshold = df_main["VIX"].quantile(0.80)
df_main["HIGH_STRESS"] = (df_main["VIX"] >= vix_threshold).astype(int)
df_main["HIGH_STRESS_lag1"] = df_main["HIGH_STRESS"].shift(1)

main_cols = [
    "RESIDUAL_FINAL", "RESIDUAL_FINAL_lag1",
    "BTC_DOWNSIDE_24_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1",
    "HIGH_STRESS_lag1"
]
df_emp = df_main.dropna(subset=main_cols).copy()

print("df_emp shape:", df_emp.shape)
print("sample:", df_emp.index.min(), "~", df_emp.index.max())
print("\nStep 1: USDT_KP ~ MKT_KP_EQ")
print(model_market.summary())
print("\nStep 2: RESIDUAL_OLS ~ DEPEG_GLOBAL")
print(model_depeg.summary())

# Save the corrected master dataset for reproducibility.
df_main.index.name = "datetime_utc"
df_main.to_csv("baseline_dataset_with_residual_final.csv")
print("\nSaved: baseline_dataset_with_residual_final.csv")


# CELL 228
# =========================================================
# Corrected main tests
# =========================================================
# Research question: Is the final residual a structured lower-tail response,
# rather than a simple average stress-regime effect?

# 1) Mean response with HAC standard errors
main_formula = (
    "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 "
    "+ BTC_DOWNSIDE_24_lag1 + DXY_ret_lag1 + USDKRW_ret_lag1"
)
model_main_ols = smf.ols(main_formula, data=df_emp).fit(
    cov_type="HAC", cov_kwds={"maxlags": 5}
)
print("===== Main HAC OLS: lagged downside on RESIDUAL_FINAL =====")
print(model_main_ols.summary())

# 2) Tail asymmetry: quantile regression across residual distribution
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
quantile_results = []

for q in quantiles:
    mod = smf.quantreg(main_formula, df_emp).fit(q=q, max_iter=5000)
    quantile_results.append({
        "q": q,
        "beta_downside_lag1": mod.params["BTC_DOWNSIDE_24_lag1"],
        "p_downside_lag1": mod.pvalues["BTC_DOWNSIDE_24_lag1"],
        "beta_dxy_lag1": mod.params["DXY_ret_lag1"],
        "p_dxy_lag1": mod.pvalues["DXY_ret_lag1"],
        "beta_fx_lag1": mod.params["USDKRW_ret_lag1"],
        "p_fx_lag1": mod.pvalues["USDKRW_ret_lag1"],
        "n": int(mod.nobs),
    })

quantile_results_df = pd.DataFrame(quantile_results)
print("\n===== Quantile response: tail asymmetry =====")
print(quantile_results_df.round(6))
quantile_results_df.to_csv("corrected_quantile_tail_results.csv", index=False)

# 3) Stress-regime comparison: does VIX high-stress explain more than tail position?
regime_formula = (
    "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 "
    "+ BTC_DOWNSIDE_24_lag1 + HIGH_STRESS_lag1 "
    "+ BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1 "
    "+ DXY_ret_lag1 + USDKRW_ret_lag1"
)
regime_results = []

for q in [0.10, 0.50, 0.90]:
    mod = smf.quantreg(regime_formula, df_emp).fit(q=q, max_iter=5000)
    interaction = "BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1"
    regime_results.append({
        "q": q,
        "beta_downside_normal": mod.params["BTC_DOWNSIDE_24_lag1"],
        "beta_downside_stress_total": mod.params["BTC_DOWNSIDE_24_lag1"] + mod.params[interaction],
        "p_downside_stress_interaction": mod.pvalues[interaction],
        "n": int(mod.nobs),
    })

regime_results_df = pd.DataFrame(regime_results)
print("\n===== Stress-regime comparison =====")
print(regime_results_df.round(6))
regime_results_df.to_csv("corrected_regime_tail_compare.csv", index=False)

# 4) Local projection version for persistence over horizons
horizons = [1, 2, 3, 6, 12]
lp_results = []

for h in horizons:
    temp = df_emp.copy()
    temp[f"RESIDUAL_FINAL_lead_{h}"] = temp["RESIDUAL_FINAL"].shift(-h)
    temp = temp.dropna(subset=[f"RESIDUAL_FINAL_lead_{h}"])

    lp_formula = (
        f"RESIDUAL_FINAL_lead_{h} ~ RESIDUAL_FINAL_lag1 "
        "+ BTC_DOWNSIDE_24_lag1 + DXY_ret_lag1 + USDKRW_ret_lag1"
    )

    for q in [0.10, 0.50, 0.90]:
        mod = smf.quantreg(lp_formula, temp).fit(q=q, max_iter=5000)
        lp_results.append({
            "h": h,
            "q": q,
            "beta_downside_lag1": mod.params["BTC_DOWNSIDE_24_lag1"],
            "p_downside_lag1": mod.pvalues["BTC_DOWNSIDE_24_lag1"],
            "n": int(mod.nobs),
        })

lp_results_df = pd.DataFrame(lp_results)
print("\n===== Quantile local projection =====")
print(lp_results_df.round(6))
lp_results_df.to_csv("corrected_quantile_local_projection.csv", index=False)

print("\nSaved:")
print("- corrected_quantile_tail_results.csv")
print("- corrected_regime_tail_compare.csv")
print("- corrected_quantile_local_projection.csv")
