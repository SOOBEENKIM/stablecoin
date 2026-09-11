"""
================================================================================
 make_paper_figures.py — 논문 그림(그림 1·3·4·5) 재현 스크립트
================================================================================
stablecoin_paper_pipeline.py 가 만든 결과표(table1/table4/table6/table7)와
원자료(baseline_dataset_with_residual_final.csv)를 읽어, 논문에 삽입된 그림을
그대로 재생성한다. 즉 모든 그림은 '네 코드가 산출한 결과값'에서 나온다.

선행: 먼저 python stablecoin_paper_pipeline.py 를 실행해 paper_outputs/ 의
      table*.csv 가 생성되어 있어야 한다.
출력: paper_outputs/fig1_timeseries.png, fig3_localprojection.png,
      fig4_mediation.png, fig5_decisive.png
실행: python make_paper_figures.py
================================================================================
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"savefig.dpi": 300, "font.size": 10,
                     "axes.grid": True, "grid.alpha": .3, "axes.axisbelow": True})
NAVY, TEAL, GREY = "#1f4e79", "#2a9d8f", "#888888"

HERE = Path(__file__).resolve().parent
DATA = HERE / "corrected_outputs" / "baseline_dataset_with_residual_final.csv"
P = HERE / "paper_outputs"


def main():
    df = pd.read_csv(DATA, parse_dates=["datetime_utc"], index_col="datetime_utc").sort_index()
    thr = df["BTC_DOWNSIDE_24"].quantile(.9)
    df["DOWNSIDE"] = (df["BTC_DOWNSIDE_24"] >= thr).astype(int)
    g = df.dropna(subset=["RESIDUAL_FINAL"])

    # 그림 1: 잔차 시계열 + 하락 국면 음영
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(g.index, g["RESIDUAL_FINAL"] * 100, color=NAVY, lw=.7); ax.axhline(0, color="k", lw=.5)
    for t in g.index[g["DOWNSIDE"] == 1]:
        ax.axvspan(t, t + pd.Timedelta(hours=1), color=TEAL, alpha=.18, lw=0)
    ax.set_ylabel("USDT residual (%)")
    plt.tight_layout(); plt.savefig(P / "fig1_timeseries.png"); plt.close()

    # 그림 3: 로컬 프로젝션 (table4)
    t4 = pd.read_csv(P / "table4_quantile_local_projection.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    col = {0.1: NAVY, 0.5: TEAL, 0.9: GREY}
    lab = {0.1: "lower tail (q=0.10)", 0.5: "median (q=0.50)", 0.9: "upper tail (q=0.90)"}
    for q in [0.1, 0.5, 0.9]:
        s = t4[t4["q"] == q].sort_values("h")
        ax.plot(s["h"], s["beta_downside"], "o-", color=col[q], lw=1.8, label=lab[q])
    ax.axhline(0, color="k", lw=.6); ax.set_xlabel("Horizon h (hours ahead)")
    ax.set_ylabel(r"$\beta$ (downside on residual$_{t+h}$)"); ax.legend(frameon=False, fontsize=8.5)
    plt.tight_layout(); plt.savefig(P / "fig3_localprojection.png"); plt.close()

    # 그림 4: 수요 채널 매개 forest (table6, q=0.10)
    t6 = pd.read_csv(P / "table6_demand_channel_mediation.csv"); sub = t6[t6["q"] == 0.1].copy()
    order = sub.iloc[::-1].reset_index(drop=True); base = sub["base_dn"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.4)); yy = np.arange(len(order) + 1)[::-1]
    ax.plot(base, yy[0], "o", color=NAVY, ms=7)
    for i, r in order.iterrows():
        face = {"YES": TEAL, "artifact": "white", "no": GREY}[r["mediates"]]
        edge = TEAL if r["mediates"] in ("YES", "artifact") else GREY
        ax.plot(r["dn_beta"], yy[i + 1], "o", mfc=face, mec=edge, ms=7)
    ax.axvline(0, color="k", lw=.6); ax.set_yticks(yy)
    ax.set_yticklabels(["(base)"] + ["+ " + r["channel"].replace("_lag1", "") for _, r in order.iterrows()], fontsize=7.5)
    ax.set_xlabel(r"$\beta$ on BTC downside (q=0.10, standardized)")
    plt.tight_layout(); plt.savefig(P / "fig4_mediation.png"); plt.close()

    # 그림 5: 결정적 강건성 (table7, q=0.10)
    t7 = pd.read_csv(P / "table7_decisive_robustness.csv"); s = t7[t7["q"] == 0.1].set_index("model")
    fig, ax = plt.subplots(figsize=(6.6, 3.6)); xp = {"M0_base+downside": 0, "M1_+global_state": 1, "M2_+ACCOUNT_LS": 2}
    for key, c, lb in [("downside", NAVY, "BTC downside"), ("btc_vol", GREY, "BTC volatility"), ("account_ls", TEAL, "Account L/S (leverage)")]:
        xs, ys = [], []
        for m in xp:
            if key in s.columns and pd.notna(s.loc[m, key]):
                xs.append(xp[m]); ys.append(s.loc[m, key])
        if xs: ax.plot(xs, ys, "o-", color=c, lw=1.8, label=lb)
    ax.axhline(0, color="k", lw=.6); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["M0\nbase", "M1\n+global", "M2\n+leverage"])
    ax.set_ylabel(r"$\beta$ at q=0.10 (standardized)"); ax.legend(frameon=False, fontsize=8.5)
    plt.tight_layout(); plt.savefig(P / "fig5_decisive.png"); plt.close()

    print("[done] saved 4 paper figures to", P)


if __name__ == "__main__":
    main()
