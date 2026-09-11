"""
================================================================================
 economic_magnitude.py — 잔차의 경제적 크기 환산 (리뷰 ① 대응)
================================================================================
"통계적 유의 ≠ 경제적 유의" 비판에 답하기 위해, 잔차와 하락국면 효과를
basis point(bp)·원(KRW)·호가단위(tick)·김프 대비 비율로 환산한다.

 입력 : corrected_outputs/baseline_dataset_with_residual_final.csv
 실행 : python economic_magnitude.py
 출력 : 콘솔 + paper_outputs/economic_magnitude.csv
================================================================================
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "corrected_outputs" / "baseline_dataset_with_residual_final.csv"
OUT = HERE / "paper_outputs"; OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["datetime_utc"], index_col="datetime_utc").sort_index()
thr = df["BTC_DOWNSIDE_24"].quantile(0.90)
df["DOWNSIDE"] = (df["BTC_DOWNSIDE_24"] >= thr).astype(int)

BP = 1e4  # 분수 -> basis point

# --- USDT 가격(원) 및 호가단위(tick) 추정 ---
px = df["USDT_UPBIT_CLOSE"].dropna()
px_mean = px.mean()
diffs = px.diff().abs()
tick = diffs[diffs > 0].min()                 # 최소 양(+)의 가격변화 = 호가단위
tick_bp = tick / px_mean * BP

# --- 잔차 규모 ---
res = df["RESIDUAL_FINAL"].dropna()
res_std_bp = res.std() * BP
g = df[["RESIDUAL_FINAL", "DOWNSIDE"]].dropna()
n_mean = g[g.DOWNSIDE == 0]["RESIDUAL_FINAL"].mean()
d_mean = g[g.DOWNSIDE == 1]["RESIDUAL_FINAL"].mean()
gap_bp = (n_mean - d_mean) * BP                # 하락국면에서 정상 대비 잔차 하락폭(bp)
gap_krw = (n_mean - d_mean) * px_mean          # 원 환산
# 하단 꼬리(q05/q10) 잔차 깊이
d_q10_bp = g[g.DOWNSIDE == 1]["RESIDUAL_FINAL"].quantile(0.10) * BP
n_q10_bp = g[g.DOWNSIDE == 0]["RESIDUAL_FINAL"].quantile(0.10) * BP

# --- 김프(프리미엄) 대비 ---
kp = df["USDT_KP"].dropna()
kp_mean_bp = kp.mean() * BP
kp_std_bp = kp.std() * BP
var_share = res.var() / kp.var()              # 분산 기준 (=1-R²_step1)
std_share = res.std() / kp.std()              # 표준편차 기준

rows = [
 ("USDT 평균가격(KRW)", round(px_mean, 1)),
 ("호가단위 tick(KRW)", round(float(tick), 4)),
 ("호가단위 tick(bp)", round(tick_bp, 2)),
 ("잔차 표준편차(bp)", round(res_std_bp, 2)),
 ("잔차 std / tick (배)", round(res_std_bp / tick_bp, 2)),
 ("하락국면 잔차 확대폭(bp)", round(gap_bp, 2)),
 ("하락국면 잔차 확대폭(KRW)", round(gap_krw, 3)),
 ("하락국면 확대폭 / tick (배)", round(gap_bp / tick_bp, 2)),
 ("하락국면 q10 잔차깊이(bp)", round(d_q10_bp, 2)),
 ("정상국면 q10 잔차깊이(bp)", round(n_q10_bp, 2)),
 ("USDT 김프 평균(bp)", round(kp_mean_bp, 1)),
 ("USDT 김프 표준편차(bp)", round(kp_std_bp, 2)),
 ("잔차/김프 분산비(%) (=1-R²)", round(var_share * 100, 2)),
 ("잔차/김프 표준편차비(%)", round(std_share * 100, 2)),
]
out = pd.DataFrame(rows, columns=["지표", "값"])
out.to_csv(OUT / "economic_magnitude.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
print("\n[saved]", OUT / "economic_magnitude.csv")
