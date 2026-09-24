"""
================================================================================
 Kimchi-Premium Definition Robustness  (EQ / CAP / PCA)
================================================================================
시장 대표 김치프리미엄(MKT_KP) 정의를 3가지로 바꿔가며 잔차를 재구성하고,
핵심 결과(잔차의 하단 tail 반응 + 레버리지 포지셔닝 매개/생존)가
정의에 무관하게 유지되는지 점검한다.

 EQ  : 5개 코인 김프 동일가중 평균 (= 기존 MKT_KP_EQ)
 CAP : USD 시가총액 가중 (가격 x 고정 reference 유통량)
 PCA : 5개 코인 김프의 제1주성분

출력: paper_outputs/robustness_kp_main_quantile.csv, robustness_kp_mediation.csv
실행: python kp_robustness.py    (B_BOOT=300 기본, 환경변수로 단축 가능)
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

COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
KP = [c + "_KP" for c in COINS]
# 공개 유통량(대략, 2025 mid). 시총가중의 '고정 reference share' 방식 — 천천히 변하므로 고정 근사.
SUPPLY = {"BTC": 19.9e6, "ETH": 120.5e6, "XRP": 59.5e9, "SOL": 590e6, "DOGE": 149e9}

BX = ["RESIDUAL_FINAL_lag1", "BTC_DOWNSIDE_24_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1"]


def ols_resid(y, x):
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X @ b
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
    # CAP
    mcap = pd.DataFrame({c: df[c + "_BINANCE_CLOSE"] * SUPPLY[c] for c in COINS})
    w = mcap.div(mcap.sum(axis=1), axis=0)
    df["MKT_KP_CAP"] = (w.values * df[KP].values).sum(axis=1)
    # PCA
    K = df[KP].dropna(); Z = (K - K.mean()) / K.std()
    ev, evec = np.linalg.eigh(np.corrcoef(Z.values.T)); v1 = evec[:, -1]
    if v1.sum() < 0: v1 = -v1
    df.loc[K.index, "MKT_KP_PCA"] = Z.values @ v1
    print("CAP weights:", {c: round(float(w[c].mean()), 3) for c in COINS})
    print("PCA PC1 explained:", round(ev[-1] / ev.sum(), 3))

    for c in ["BTC_VOL_24", "BTC_RET", "VIX_ret"]: df[c + "_lag1"] = df[c].shift(1)
    defs = {"EQ": "MKT_KP_EQ", "CAP": "MKT_KP_CAP", "PCA": "MKT_KP_PCA"}
    main_rows, med_rows = [], []
    for name, col in defs.items():
        d = df.copy()
        s = d[["USDT_KP", col]].dropna(); r1 = pd.Series(np.nan, index=d.index)
        r1.loc[s.index] = ols_resid(s["USDT_KP"].values, s[col].values); d["R_OLS"] = r1
        s2 = d[["R_OLS", "DEPEG_GLOBAL"]].dropna(); rf = pd.Series(np.nan, index=d.index)
        rf.loc[s2.index] = ols_resid(s2["R_OLS"].values, s2["DEPEG_GLOBAL"].values); d["RF"] = rf
        d["RESIDUAL_FINAL_lag1"] = d["RF"].shift(1); d["day"] = d.index.normalize()
        dd = d[["RF"] + BX + ["day"]].dropna()
        for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
            b, p = boot(dd, "RF", BX, q, ["BTC_DOWNSIDE_24_lag1"], dd["day"].values)["BTC_DOWNSIDE_24_lag1"]
            main_rows.append(dict(definition=name, q=q, beta_downside=round(b, 3), p=round(p, 3)))
        for ch in ["ACCOUNT_LS_lag1", "TOPTRADER_ACCOUNT_LS_lag1"]:
            cols = ["RF"] + BX + [ch]; dz = d[cols + ["day"]].dropna()
            z = (dz[cols] - dz[cols].mean()) / dz[cols].std(); z["day"] = dz["day"].values
            base = boot(z, "RF", BX, 0.1, ["BTC_DOWNSIDE_24_lag1"], z["day"].values)["BTC_DOWNSIDE_24_lag1"]
            r = boot(z, "RF", BX + [ch], 0.1, ["BTC_DOWNSIDE_24_lag1", ch], z["day"].values)
            dn, cc = r["BTC_DOWNSIDE_24_lag1"], r[ch]
            genuine = (cc[1] < 0.05) and (dn[1] >= 0.05) and (abs(dn[0]) < 0.6 * abs(base[0]))
            med_rows.append(dict(definition=name, channel=ch.replace("_lag1", ""), base_dn=round(base[0], 3),
                                 dn_beta=round(dn[0], 3), dn_p=round(dn[1], 3), ch_p=round(cc[1], 3),
                                 result="YES" if genuine else ("artifact" if cc[1] < 0.05 and dn[1] >= 0.05 else "no")))
        nd = ["RF", "RESIDUAL_FINAL_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1", "BTC_DOWNSIDE_24_lag1",
              "BTC_VOL_24_lag1", "BTC_RET_lag1", "VIX_ret_lag1", "ACCOUNT_LS_lag1"]
        dz = d[nd + ["day"]].dropna(); z = (dz[nd] - dz[nd].mean()) / dz[nd].std(); z["day"] = dz["day"].values
        r = boot(z, "RF", nd[1:], 0.1, ["ACCOUNT_LS_lag1"], z["day"].values)["ACCOUNT_LS_lag1"]
        med_rows.append(dict(definition=name, channel="ACCOUNT_LS (decisive M2)", base_dn="-",
                             dn_beta=round(r[0], 3), dn_p="-", ch_p=round(r[1], 3),
                             result="survives" if r[1] < 0.05 else "no"))
    main = pd.DataFrame(main_rows); med = pd.DataFrame(med_rows)
    main.to_csv(OUT / "robustness_kp_main_quantile.csv", index=False)
    med.to_csv(OUT / "robustness_kp_mediation.csv", index=False)
    print("\n=== main quantile beta_downside ===")
    print(main.pivot(index="q", columns="definition", values="beta_downside").to_string())
    print("\n=== mediation / survival (q=0.10) ===")
    print(med.to_string(index=False))
    print(f"\n[done] saved to {OUT}")


if __name__ == "__main__":
    main()
