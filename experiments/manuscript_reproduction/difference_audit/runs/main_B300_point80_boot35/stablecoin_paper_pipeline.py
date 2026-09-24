"""
================================================================================
 Korean USDT Residual — Reproducible Pipeline (TABLES + FIGURES, one run)
================================================================================
한 번 실행하면 논문용 표(Table 1~7)와 그림(그림 1·3·4·5)이 '같은 실행·같은
결과'에서 모두 생성된다. 그림은 표 결과를 그대로 사용하므로 항상 일치한다.

 입력 : corrected_outputs/baseline_dataset_with_residual_final.csv
 출력 : paper_outputs/table1~7.csv  +  paper_outputs/fig1_timeseries.png,
        fig3_localprojection.png, fig4_mediation.png, fig5_decisive.png

분위수 회귀·블록 부트스트랩은 numpy로 직접 구현(검증됨). 정상성(ADF/KPSS/
Ljung-Box)만 statsmodels, 그림은 matplotlib를 쓰며 없으면 자동 skip.
시드(42) 고정 → 매번 동일 결과.   실행: python stablecoin_paper_pipeline.py
================================================================================
"""
import math, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    HAVE_MPL = True
    plt.rcParams.update({"savefig.dpi": 300, "font.size": 10,
                         "axes.grid": True, "grid.alpha": .3, "axes.axisbelow": True})
except Exception:
    HAVE_MPL = False

SEED = 42
B_BOOT = int(os.environ.get("B_BOOT", 300))
QUANTILES_MAIN = [0.10, 0.25, 0.50, 0.75, 0.90]
HORIZONS_LP = [1, 2, 3, 6, 12]
DOWNSIDE_PCTL = 0.90
HAC_LAGS = 5
NAVY, TEAL, GREY = "#1f4e79", "#2a9d8f", "#888888"

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "corrected_outputs" / "baseline_dataset_with_residual_final.csv"
OUT = HERE / "paper_outputs"; OUT.mkdir(exist_ok=True)
np.random.seed(SEED)

BX = ["RESIDUAL_FINAL_lag1", "BTC_DOWNSIDE_24_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1"]
DEMAND = ["USDT_VOL_RATIO_LOG", "COIN_VOL_RATIO_LOG_MEAN", "USDT_RELATIVE_VOLUME_PRESSURE", "VOL_RATIO_DISPERSION",
          "FUNDING_lag1", "OI_lag1", "OI_VALUE_lag1", "OI_RET_lag1", "OI_VALUE_RET_lag1",
          "TOPTRADER_ACCOUNT_LS_lag1", "TOPTRADER_POSITION_LS_lag1", "ACCOUNT_LS_lag1", "TAKER_LONG_SHORT_VOL_RATIO_lag1"]


# ------------------------------------------------------------- 통계 도구
def normal_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def add_const(X): return np.column_stack([np.ones(X.shape[0]), X])

def ols_hac(y, X, L=HAC_LAGS):
    Xc = add_const(X); XtX_inv = np.linalg.inv(Xc.T @ Xc)
    beta = XtX_inv @ (Xc.T @ y); u = y - Xc @ beta
    S = (Xc * u[:, None]).T @ (Xc * u[:, None])
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0); g = (Xc[l:] * u[l:, None]).T @ (Xc[:-l] * u[:-l, None]); S += w * (g + g.T)
    cov = XtX_inv @ S @ XtX_inv; se = np.sqrt(np.diag(cov)); t = beta / se
    p = np.array([2 * (1 - normal_cdf(abs(ti))) for ti in t]); r2 = 1 - (u @ u) / (((y - y.mean()) ** 2).sum())
    return beta, se, t, p, r2, len(y)

def quantile_fit(X, y, tau, n_iter=80, eps=1e-7):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - X @ b; w = np.where(r >= 0, tau, 1 - tau) / np.maximum(np.abs(r), eps); XtW = X.T * w
        try: bn = np.linalg.solve(XtW @ X, XtW @ y)
        except np.linalg.LinAlgError: bn = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
        if np.max(np.abs(bn - b)) < 1e-9: return bn
        b = bn
    return b

def block_bootstrap(data, ycol, xcols, tau, targets, day, B=B_BOOT):
    Z = data[[ycol] + xcols].values.astype(float); yv = Z[:, 0]; Xf = add_const(Z[:, 1:])
    pos = {c: i + 1 for i, c in enumerate(xcols)}
    ud = pd.unique(day); dr = {u: np.where(day == u)[0] for u in ud}; dd = {t: [] for t in targets}
    for _ in range(B):
        rr = np.concatenate([dr[u] for u in np.random.choice(ud, len(ud), True)])
        try:
            b = quantile_fit(Xf[rr], yv[rr], tau, n_iter=35)
            for t in targets: dd[t].append(b[pos[t]])
        except Exception: pass
    pt = quantile_fit(Xf, yv, tau); out = {}
    for t in targets:
        a = np.asarray(dd[t]); p = 2 * min((a > 0).mean(), (a < 0).mean()) if len(a) else np.nan
        out[t] = (pt[pos[t]], p)
    return out

def zscore(df, cols): s = df[cols].copy(); return (s - s.mean()) / s.std()


# ------------------------------------------------------------- 데이터/잔차
def load_and_build():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime_utc"], index_col="datetime_utc").sort_index()
    s1 = df[["USDT_KP", "MKT_KP_EQ"]].dropna()
    b1, *_, r2_1, _ = ols_hac(s1["USDT_KP"].values, s1[["MKT_KP_EQ"]].values)
    df["RESIDUAL_OLS"] = np.nan
    df.loc[s1.index, "RESIDUAL_OLS"] = s1["USDT_KP"].values - add_const(s1[["MKT_KP_EQ"]].values) @ b1
    df["DEPEG_GLOBAL"] = (df["USDT_BINANCE_CLOSE"] - 1.0).abs()
    s2 = df[["RESIDUAL_OLS", "DEPEG_GLOBAL"]].dropna()
    b2, *_, r2_2, _ = ols_hac(s2["RESIDUAL_OLS"].values, s2[["DEPEG_GLOBAL"]].values)
    df["RESIDUAL_FINAL_REBUILD"] = np.nan
    df.loc[s2.index, "RESIDUAL_FINAL_REBUILD"] = s2["RESIDUAL_OLS"].values - add_const(s2[["DEPEG_GLOBAL"]].values) @ b2
    chk = df[["RESIDUAL_FINAL", "RESIDUAL_FINAL_REBUILD"]].dropna()
    max_dev = (chk["RESIDUAL_FINAL"] - chk["RESIDUAL_FINAL_REBUILD"]).abs().max()
    df["RESIDUAL_FINAL_lag1"] = df["RESIDUAL_FINAL"].shift(1)
    for c in ["BTC_VOL_24", "BTC_RET", "VIX_ret"]: df[c + "_lag1"] = df[c].shift(1)
    thr = df["BTC_DOWNSIDE_24"].quantile(DOWNSIDE_PCTL)
    df["DOWNSIDE"] = (df["BTC_DOWNSIDE_24"] >= thr).astype(int)
    df["day"] = df.index.normalize()
    return df, dict(step1_r2=r2_1, step1_beta=b1[1], step2_r2=r2_2, rebuild_max_dev=max_dev, downside_thr=thr)


# ------------------------------------------------------------- 표 1~7
def table1_descriptive(df):
    g = df[["RESIDUAL_FINAL", "DOWNSIDE"]].dropna(); rows = []
    for lab, sub in [("Normal", g[g.DOWNSIDE == 0]), ("Downside", g[g.DOWNSIDE == 1])]:
        r = sub["RESIDUAL_FINAL"]
        rows.append(dict(regime=lab, n=len(r), mean=r.mean(), std=r.std(), q05=r.quantile(.05),
                         q10=r.quantile(.10), median=r.median(), q90=r.quantile(.90), skew=r.skew()))
    a = g[g.DOWNSIDE == 1]["RESIDUAL_FINAL"].values; b = g[g.DOWNSIDE == 0]["RESIDUAL_FINAL"].values
    tstat = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t1 = pd.DataFrame(rows); t1.to_csv(OUT / "table1_residual_descriptive.csv", index=False)
    return t1, tstat

def table2_stationarity(df):
    y = df["RESIDUAL_FINAL"].dropna()
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.stats.diagnostic import acorr_ljungbox
        adf = adfuller(y); kp = kpss(y, regression="c", nlags="auto")
        lb = acorr_ljungbox(y, lags=[5, 10, 20], return_df=True)
        out = pd.DataFrame({"test": ["ADF", "KPSS", "LjungBox(5)", "LjungBox(10)", "LjungBox(20)"],
                            "stat": [adf[0], kp[0], *lb["lb_stat"].tolist()],
                            "pvalue": [adf[1], kp[1], *lb["lb_pvalue"].tolist()]})
    except Exception as e:
        out = pd.DataFrame({"note": [f"statsmodels 미설치로 건너뜀: {e}"]})
    out.to_csv(OUT / "table2_stationarity.csv", index=False); return out

def table3_main_quantile(df):
    d = df[["RESIDUAL_FINAL"] + BX + ["day"]].dropna(); rows = []
    for q in QUANTILES_MAIN:
        b, p = block_bootstrap(d, "RESIDUAL_FINAL", BX, q, ["BTC_DOWNSIDE_24_lag1"], d["day"].values)["BTC_DOWNSIDE_24_lag1"]
        rows.append(dict(q=q, beta_downside=b, p_boot=p, n=len(d)))
    t3 = pd.DataFrame(rows); t3.to_csv(OUT / "table3_main_quantile_tail.csv", index=False); return t3

def table4_local_projection(df):
    rows = []
    for h in HORIZONS_LP:
        tmp = df.copy(); tmp["YL"] = tmp["RESIDUAL_FINAL"].shift(-h)
        d = tmp[["YL"] + BX + ["day"]].dropna()
        for q in [0.10, 0.50, 0.90]:
            b, p = block_bootstrap(d, "YL", BX, q, ["BTC_DOWNSIDE_24_lag1"], d["day"].values)["BTC_DOWNSIDE_24_lag1"]
            rows.append(dict(h=h, q=q, beta_downside=b, p_boot=p, n=len(d)))
    t4 = pd.DataFrame(rows); t4.to_csv(OUT / "table4_quantile_local_projection.csv", index=False); return t4

def table5_stress_regime(df):
    d0 = df.copy(); vix_thr = d0["VIX"].quantile(0.80)
    d0["HIGH_STRESS_lag1"] = (d0["VIX"] >= vix_thr).astype(int).shift(1)
    d0["DN_x_STRESS"] = d0["BTC_DOWNSIDE_24_lag1"] * d0["HIGH_STRESS_lag1"]
    xc = ["RESIDUAL_FINAL_lag1", "BTC_DOWNSIDE_24_lag1", "HIGH_STRESS_lag1", "DN_x_STRESS", "DXY_ret_lag1", "USDKRW_ret_lag1"]
    d = d0[["RESIDUAL_FINAL"] + xc + ["day"]].dropna(); rows = []
    for q in [0.10, 0.50, 0.90]:
        r = block_bootstrap(d, "RESIDUAL_FINAL", xc, q, ["BTC_DOWNSIDE_24_lag1", "DN_x_STRESS"], d["day"].values)
        rows.append(dict(q=q, beta_downside_normal=r["BTC_DOWNSIDE_24_lag1"][0], p_normal=r["BTC_DOWNSIDE_24_lag1"][1],
                         beta_interaction=r["DN_x_STRESS"][0], p_interaction=r["DN_x_STRESS"][1], n=len(d)))
    t5 = pd.DataFrame(rows); t5.to_csv(OUT / "table5_stress_regime.csv", index=False); return t5

def table6_demand_channel(df):
    chans = [c for c in DEMAND if c in df.columns]
    d = df[["RESIDUAL_FINAL"] + BX + chans + ["day"]].dropna()
    z = zscore(d, ["RESIDUAL_FINAL"] + BX + chans); z["day"] = d["day"].values
    base = {q: block_bootstrap(z, "RESIDUAL_FINAL", BX, q, ["BTC_DOWNSIDE_24_lag1"], z["day"].values)["BTC_DOWNSIDE_24_lag1"] for q in [0.10, 0.50]}
    rows = []
    for ch in chans:
        for q in [0.10, 0.50]:
            r = block_bootstrap(z, "RESIDUAL_FINAL", BX + [ch], q, ["BTC_DOWNSIDE_24_lag1", ch], z["day"].values)
            dn, cc = r["BTC_DOWNSIDE_24_lag1"], r[ch]
            genuine = (cc[1] < 0.05) and (dn[1] >= 0.05) and (abs(dn[0]) < 0.6 * abs(base[q][0]))
            flag = "YES" if genuine else ("artifact" if (cc[1] < 0.05 and dn[1] >= 0.05) else "no")
            rows.append(dict(channel=ch, q=q, base_dn=round(base[q][0], 3), dn_beta=round(dn[0], 3),
                             dn_p=round(dn[1], 3), ch_beta=round(cc[0], 3), ch_p=round(cc[1], 3), mediates=flag))
    t6 = pd.DataFrame(rows).sort_values(["q", "dn_p"]); t6.to_csv(OUT / "table6_demand_channel_mediation.csv", index=False); return t6

def table7_decisive(df):
    need = ["RESIDUAL_FINAL"] + BX + ["BTC_VOL_24_lag1", "BTC_RET_lag1", "VIX_ret_lag1", "ACCOUNT_LS_lag1", "day"]
    d = df[need].dropna(); z = zscore(d, [c for c in need if c != "day"]); z["day"] = d["day"].values
    bc = ["RESIDUAL_FINAL_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1"]; glob = ["BTC_VOL_24_lag1", "BTC_RET_lag1", "VIX_ret_lag1"]
    models = {"M0_base+downside": bc + ["BTC_DOWNSIDE_24_lag1"], "M1_+global_state": bc + ["BTC_DOWNSIDE_24_lag1"] + glob,
              "M2_+ACCOUNT_LS": bc + ["BTC_DOWNSIDE_24_lag1"] + glob + ["ACCOUNT_LS_lag1"]}
    rows = []
    for q in [0.10, 0.25, 0.50]:
        for m, xc in models.items():
            tg = [c for c in ["BTC_DOWNSIDE_24_lag1", "BTC_VOL_24_lag1", "ACCOUNT_LS_lag1"] if c in xc]
            r = block_bootstrap(z, "RESIDUAL_FINAL", xc, q, tg, z["day"].values)
            row = dict(q=q, model=m, n=len(d))
            for v, key in [("BTC_DOWNSIDE_24_lag1", "downside"), ("BTC_VOL_24_lag1", "btc_vol"), ("ACCOUNT_LS_lag1", "account_ls")]:
                if v in r: row[key] = round(r[v][0], 3); row["p_" + key] = round(r[v][1], 3)
            rows.append(row)
    t7 = pd.DataFrame(rows); t7.to_csv(OUT / "table7_decisive_robustness.csv", index=False); return t7


# ------------------------------------------------------------- 그림 (표 결과 사용)
def make_figures(df, t4, t6, t7):
    if not HAVE_MPL: print("[fig] matplotlib 미설치 → 그림 건너뜀"); return
    g = df.dropna(subset=["RESIDUAL_FINAL"])
    # 그림 1: 잔차 시계열 + 하락 국면 음영
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.plot(g.index, g["RESIDUAL_FINAL"] * 100, color=NAVY, lw=.7); ax.axhline(0, color="k", lw=.5)
    for t in g.index[g["DOWNSIDE"] == 1]: ax.axvspan(t, t + pd.Timedelta(hours=1), color=TEAL, alpha=.18, lw=0)
    ax.set_ylabel("USDT residual (%)"); plt.tight_layout(); plt.savefig(OUT / "fig1_timeseries.png"); plt.close()
    # 그림 3: 로컬 프로젝션
    fig, ax = plt.subplots(figsize=(6.6, 3.6)); col = {0.1: NAVY, 0.5: TEAL, 0.9: GREY}
    lab = {0.1: "lower tail (q=0.10)", 0.5: "median (q=0.50)", 0.9: "upper tail (q=0.90)"}
    for q in [0.1, 0.5, 0.9]:
        s = t4[t4["q"] == q].sort_values("h"); ax.plot(s["h"], s["beta_downside"], "o-", color=col[q], lw=1.8, label=lab[q])
    ax.axhline(0, color="k", lw=.6); ax.set_xlabel("Horizon h (hours ahead)")
    ax.set_ylabel(r"$\beta$ (downside on residual$_{t+h}$)"); ax.legend(frameon=False, fontsize=8.5)
    plt.tight_layout(); plt.savefig(OUT / "fig3_localprojection.png"); plt.close()
    # 그림 4: 수요 채널 매개
    sub = t6[t6["q"] == 0.1].copy(); order = sub.iloc[::-1].reset_index(drop=True); base = sub["base_dn"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.4)); yy = np.arange(len(order) + 1)[::-1]; ax.plot(base, yy[0], "o", color=NAVY, ms=7)
    for i, r in order.iterrows():
        face = {"YES": TEAL, "artifact": "white", "no": GREY}[r["mediates"]]; edge = TEAL if r["mediates"] in ("YES", "artifact") else GREY
        ax.plot(r["dn_beta"], yy[i + 1], "o", mfc=face, mec=edge, ms=7)
    ax.axvline(0, color="k", lw=.6); ax.set_yticks(yy)
    ax.set_yticklabels(["(base)"] + ["+ " + r["channel"].replace("_lag1", "") for _, r in order.iterrows()], fontsize=7.5)
    ax.set_xlabel(r"$\beta$ on BTC downside (q=0.10, standardized)"); plt.tight_layout(); plt.savefig(OUT / "fig4_mediation.png"); plt.close()
    # 그림 5: 결정적 강건성
    s = t7[t7["q"] == 0.1].set_index("model"); fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xp = {"M0_base+downside": 0, "M1_+global_state": 1, "M2_+ACCOUNT_LS": 2}
    for key, c, lb in [("downside", NAVY, "BTC downside"), ("btc_vol", GREY, "BTC volatility"), ("account_ls", TEAL, "Account L/S (leverage)")]:
        xs, ys = [], []
        for m in xp:
            if key in s.columns and pd.notna(s.loc[m, key]): xs.append(xp[m]); ys.append(s.loc[m, key])
        if xs: ax.plot(xs, ys, "o-", color=c, lw=1.8, label=lb)
    ax.axhline(0, color="k", lw=.6); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["M0\nbase", "M1\n+global", "M2\n+leverage"])
    ax.set_ylabel(r"$\beta$ at q=0.10 (standardized)"); ax.legend(frameon=False, fontsize=8.5)
    plt.tight_layout(); plt.savefig(OUT / "fig5_decisive.png"); plt.close()
    print("[fig] 4 paper figures saved to", OUT)


def main():
    print("=" * 64 + "\n Korean USDT Residual — pipeline (tables + figures)\n" + "=" * 64)
    df, meta = load_and_build()
    print(f"[load] n={len(df.dropna(subset=['RESIDUAL_FINAL']))}  step1_r2={meta['step1_r2']:.4f}")
    print(f"[check] rebuilt vs saved RESIDUAL_FINAL max|dev|={meta['rebuild_max_dev']:.2e}")
    t1, ts = table1_descriptive(df); print(f"[Table1] Welch t={ts:.2f}")
    table2_stationarity(df)
    t3 = table3_main_quantile(df); print("[Table3]\n" + t3.round(3).to_string(index=False))
    t4 = table4_local_projection(df)
    table5_stress_regime(df)
    t6 = table6_demand_channel(df)
    print("[Table6 q=0.10 mediators]\n" + t6[t6.q == 0.10][["channel", "dn_beta", "dn_p", "ch_p", "mediates"]].to_string(index=False))
    t7 = table7_decisive(df)
    print("[Table7 q=0.10]\n" + t7[t7.q == 0.10][["model", "downside", "p_downside", "btc_vol", "account_ls", "p_account_ls"]].to_string(index=False))
    make_figures(df, t4, t6, t7)
    print(f"\n[done] tables + figures -> {OUT}")


if __name__ == "__main__":
    main()
