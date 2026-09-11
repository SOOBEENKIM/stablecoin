"""
================================================================================
 Experiment 2 — Sub-sample / leave-event-out stability
================================================================================
짧은 표본(약 10개월)이 한 에피소드에 끌려간 것은 아닌지 점검.
핵심 주장(레버리지 채널의 M2 생존)과 메인 분위수 반응을 표본 컷별로 재추정.

 표본 컷:
  FULL            전체
  FIRST/SECOND_HALF  시간 분할(검정력 확인용)
  DROP_TOP1_DAY   최대 stress 1일 제거 (leave-largest-event-out)
  DROP_TOP5_DAYS  최대 stress 5일 제거

 해석 가이드:
  - DROP_TOP*_DAYS 에서도 살아남으면 '단일 사건 의존' 아님(robust).
  - HALF 분할에서 약해지면 검정력 한계(짧은 표본) → sub-period 안정성은 미확립으로 보고.

출력: paper_outputs/exp2_subsample_stability.csv  (기존 파일 미변경)
실행: python experiment2_subsample_stability.py
================================================================================
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)
B_BOOT = int(os.environ.get("B_BOOT", 300))
HERE = Path(__file__).resolve().parent
DATA = HERE / "corrected_outputs" / "baseline_dataset_with_residual_final.csv"
OUT = HERE / "paper_outputs"; OUT.mkdir(exist_ok=True)

BX = ["RESIDUAL_FINAL_lag1", "BTC_DOWNSIDE_24_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1"]
DEC = ["RESIDUAL_FINAL_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1", "BTC_DOWNSIDE_24_lag1",
       "BTC_VOL_24_lag1", "BTC_RET_lag1", "VIX_ret_lag1", "ACCOUNT_LS_lag1"]


def addc(X): return np.column_stack([np.ones(X.shape[0]), X])
def qr(X, y, tau, it=60, eps=1e-7):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ b; w = np.where(r >= 0, tau, 1 - tau) / np.maximum(np.abs(r), eps); XtW = X.T * w
        try: bn = np.linalg.solve(XtW @ X, XtW @ y)
        except np.linalg.LinAlgError: bn = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
        if np.max(np.abs(bn - b)) < 1e-9: return bn
        b = bn
    return b
def boot(data, yc, xc, tau, tg, day, B=B_BOOT):
    Z = data[[yc] + xc].values.astype(float); yv = Z[:, 0]; Xf = addc(Z[:, 1:]); pos = {c: i + 1 for i, c in enumerate(xc)}
    ud = pd.unique(day); dr = {u: np.where(day == u)[0] for u in ud}; dd = {t: [] for t in tg}
    for _ in range(B):
        rr = np.concatenate([dr[u] for u in np.random.choice(ud, len(ud), True)])
        try:
            bb = qr(Xf[rr], yv[rr], tau, it=30)
            for t in tg: dd[t].append(bb[pos[t]])
        except Exception: pass
    pt = qr(Xf, yv, tau); out = {}
    for t in tg:
        a = np.asarray(dd[t]); p = 2 * min((a > 0).mean(), (a < 0).mean()) if len(a) else np.nan
        out[t] = (pt[pos[t]], p)
    return out


def main():
    df = pd.read_csv(DATA, parse_dates=["datetime_utc"], index_col="datetime_utc").sort_index()
    df["RESIDUAL_FINAL_lag1"] = df["RESIDUAL_FINAL"].shift(1)
    for c in ["BTC_VOL_24", "BTC_RET", "VIX_ret"]: df[c + "_lag1"] = df[c].shift(1)
    df["day"] = df.index.normalize()
    dayrank = df.groupby("day")["BTC_DOWNSIDE_24"].sum().sort_values(ascending=False)
    top1, top5 = set(dayrank.index[:1]), set(dayrank.index[:5])
    tmid = df.index[len(df) // 2]
    cuts = {"FULL": df, "FIRST_HALF": df[df.index < tmid], "SECOND_HALF": df[df.index >= tmid],
            "DROP_TOP1_DAY": df[~df["day"].isin(top1)], "DROP_TOP5_DAYS": df[~df["day"].isin(top5)]}
    rows = []
    for name, sub in cuts.items():
        d = sub[["RESIDUAL_FINAL"] + BX + ["day"]].dropna(); res = {}
        for q in [0.10, 0.25]:
            res[q] = boot(d, "RESIDUAL_FINAL", BX, q, ["BTC_DOWNSIDE_24_lag1"], d["day"].values)["BTC_DOWNSIDE_24_lag1"]
        dz = sub[["RESIDUAL_FINAL"] + DEC + ["day"]].dropna(); cols = ["RESIDUAL_FINAL"] + DEC
        z = (dz[cols] - dz[cols].mean()) / dz[cols].std(); z["day"] = dz["day"].values
        r = boot(z, "RESIDUAL_FINAL", DEC, 0.10, ["ACCOUNT_LS_lag1"], z["day"].values)["ACCOUNT_LS_lag1"]
        rows.append(dict(sample=name, n=len(d), q10_dn=round(res[0.10][0], 2), q10_p=round(res[0.10][1], 3),
                         q25_dn=round(res[0.25][0], 2), q25_p=round(res[0.25][1], 3),
                         M2_leverage_beta=round(r[0], 3), M2_leverage_p=round(r[1], 3),
                         M2_leverage="survives" if r[1] < 0.05 else "no"))
    out = pd.DataFrame(rows); out.to_csv(OUT / "exp2_subsample_stability.csv", index=False)
    pd.set_option("display.width", 180)
    print("top-5 stress days:", [str(d.date()) for d in dayrank.index[:5]])
    print(out.to_string(index=False))
    print(f"\n[done] saved to {OUT / 'exp2_subsample_stability.csv'}")


if __name__ == "__main__":
    main()
