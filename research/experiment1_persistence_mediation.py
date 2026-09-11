"""
================================================================================
 Experiment 1 — Is the persistent lower-tail residual response mediated by
                crypto leverage positioning?
================================================================================
메인 결과 두 기둥을 하나로 묶는 검정:
 (a) 하단 tail(q=0.10) 잔차의 BTC-downside 반응이 horizon h에 걸쳐 커진다(지속/증폭).
 (b) 그 반응이 레버리지 포지셔닝(계정 롱/숏)으로 매개되는가?

각 horizon h에서 로컬프로젝션 잔차(residual_{t+h})에 대해:
 base : YL ~ RESID_lag1 + BTC_DOWNSIDE_lag1 + DXY_lag1 + USDKRW_lag1
 +lev : base + (ACCOUNT_LS_lag1 또는 TOPTRADER_ACCOUNT_LS_lag1)
 → 레버리지 추가 시 BTC_DOWNSIDE 계수가 축소/비유의화되고 채널이 유의하면 '매개'.

표준화 계수 + 일(day) 블록 부트스트랩 추론. 기존 파일/결과는 건드리지 않음.
출력: paper_outputs/exp1_persistence_mediation.csv
실행: python experiment1_persistence_mediation.py   (B_BOOT 환경변수로 단축 가능)
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

BASE = ["RESIDUAL_FINAL_lag1", "BTC_DOWNSIDE_24_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1"]
MEDIATORS = ["TOPTRADER_ACCOUNT_LS_lag1", "ACCOUNT_LS_lag1"]
HORIZONS = [1, 3, 6, 12]
QUANTILE = 0.10


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
    df["RESIDUAL_FINAL_lag1"] = df["RESIDUAL_FINAL"].shift(1); df["day"] = df.index.normalize()
    rows = []
    for ch in MEDIATORS:
        for h in HORIZONS:
            tmp = df.copy(); tmp["YL"] = tmp["RESIDUAL_FINAL"].shift(-h)
            cols = ["YL"] + BASE + [ch]; d = tmp[cols + ["day"]].dropna()
            z = (d[cols] - d[cols].mean()) / d[cols].std(); z["day"] = d["day"].values
            bse = boot(z, "YL", BASE, QUANTILE, ["BTC_DOWNSIDE_24_lag1"], z["day"].values)["BTC_DOWNSIDE_24_lag1"]
            r = boot(z, "YL", BASE + [ch], QUANTILE, ["BTC_DOWNSIDE_24_lag1", ch], z["day"].values)
            dn, cc = r["BTC_DOWNSIDE_24_lag1"], r[ch]
            shrink = 1 - abs(dn[0]) / abs(bse[0]) if bse[0] != 0 else np.nan
            med = (cc[1] < 0.05) and (dn[1] >= 0.05) and (abs(dn[0]) < 0.6 * abs(bse[0]))
            rows.append(dict(mediator=ch.replace("_lag1", ""), h=h, q=QUANTILE,
                             base_dn=round(bse[0], 3), base_p=round(bse[1], 3),
                             dn_after=round(dn[0], 3), dn_p_after=round(dn[1], 3),
                             shrink_pct=round(100 * shrink, 0), ch_beta=round(cc[0], 3), ch_p=round(cc[1], 3),
                             mediates="YES" if med else "no"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "exp1_persistence_mediation.csv", index=False)
    pd.set_option("display.width", 170)
    print(out.to_string(index=False))
    print(f"\n[done] saved to {OUT / 'exp1_persistence_mediation.csv'}")


if __name__ == "__main__":
    main()
