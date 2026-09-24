"""Summarize every prespecified A3 family without choosing by downstream signs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
SCENARIOS=["clock_UTC_quote","clock_NY_quote","clock_FXUTC_USNY_quote","clock_FXKST_USNY_quote","clock_NY_macro_end1h_quote"]
LABELS=["UTC/UTC","NY/NY","UTC/NY","KST/NY","NY/NY+1h"]
NAMES={"seq_xgb":"순차 XGBoost","seq_mlp":"순차 MLP","seq_spline":"순차 spline",
    "joint_xgb":"공동 XGBoost","joint_mlp":"공동 MLP","joint_tensor":"공동 tensor spline",
    "seq_selected":"순차: 과거 최선 ML","seq_shrunk":"순차: 과거 축소 선택",
    "joint_selected":"공동: 과거 최선 ML","joint_shrunk":"공동: 과거 축소 선택",
    "sequential_ols":"순차 OLS","joint_ols":"공동 OLS","pca":"PCA 순차"}


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers)]+["| "+" | ".join(map(str,r))+" |" for r in rows])


def main():
    manifest=json.loads((OUT/"RUN_MANIFEST.json").read_text())
    scores=pd.read_csv(OUT/"scores.csv")
    ci=pd.read_csv(OUT/"paired_block_intervals.csv")
    coef=pd.read_csv(OUT/"coefficients.csv")
    selected=pd.concat([pd.DataFrame(json.loads((OUT/s/"selection.json").read_text())).assign(scenario=s) for s in SCENARIOS])
    eval_selected=selected.loc[selected.month.isin(["2026-01","2026-02","2026-03"])]
    notes=["# A3 결과: 문헌 기반 후보 확대와 원고 순차 분해 비교","",
      "2026-09-24. 실행 전 코드·규칙 커밋: `%s`. **XGBoost·소규모 신경망·스플라인의 28개 비선형 설정을 추가 실행했다.** "
      "세 선형/PCA 대조를 포함해 31개 설정, 5개 시각 가정, 9개 학습 기준월에서 총 %d회 적합했다. "
      "MLP 한 적합은 세 초기화의 평균이다. 독립 실험 횟수라는 뜻은 아니다."%(manifest["source_commit"][:7],manifest["fit_count"]),"",
      "**현재 결과:** 원고 순서를 유지한 MLP의 1~3월 RMSE는 순차 OLS보다 0.09~0.82% 높아 수치상 가까웠다. "
      "이는 통계적 동등성의 입증은 아니다. 순차 스플라인은 KST/NY에서 0.34% 감소했지만 "
      "차이의 7일 구간은 [-0.128, +0.055]bp로 0을 포함했다. 추가 후보에서 여러 조건에 걸친 안정적인 우위는 확보되지 않았다.","",
      "## 기존 연구에서 무엇을 유지했나","",
      "연구 질문은 한국 USDT 프리미엄의 공통 성분을 제거한 잔차의 하방 관계·지속성·포지셔닝 연관성이다. "
      "이번 비교 안에서는 데이터·가격 산식·변수·관측 조건을 고정했다. main의 호가 교란 연구로 목표를 바꾸지 않았다.","",
      "- 순차 확장: `y=f(m)+r1` 다음 `r1=a+b*g+e`. **공통 김프 제거의 첫 단계만 ML로 바꾸고 글로벌 요인 제거는 OLS로 유지**했다.",
      "- 공동 대조: `y=f(m,g)+e`. 같은 입력의 공동 OLS와 비선형 보정을 비교했다.",
      "- 각 구조에서 XGBoost·MLP·스플라인을 비교했다. 공동 스플라인은 두 입력의 상호작용 기저를 포함한다.",
      "- 설정과 보정 강도는 과거 검증 오차로 선택했다. 후속 하방·포지셔닝 계수의 부호를 보고 선택하지 않았다.","",
      "[문헌과 적용 근거](REFERENCES_KO.md): 스테이블코인 위험 예측의 XGBoost, 금융 자산가격 연구의 신경망·상호작용, "
      "ICAIF 2025 FinTSB의 공통 평가 설계 등을 참고했다. 각 논문의 목적과 이번 적용 차이, 전문 접근 범위를 명시했다.","",
      "## 1. 분해 오차: 같은 2026년 1~3월 평가","",
      "학습 이후 자료의 동시점 m·g로 y를 설명하는 **조건부 평균 적합**이다. 미래 가격 예측이나 참된 경제적 공통 성분의 관측이 아니다. "
      "아래는 시간대별 RMSE 변화율의 최소~최대이며 **음수는 개선, 양수는 악화**다. 월별 결과는 [전체 점수](results/scores.csv)에 있다. "
      "UTC/UTC와 UTC/NY는 분해 입력이 같으므로 독립 재현 두 번이 아니다.",""]
    methods=list(NAMES)[:10]
    rows=[]
    for method in methods:
        own="sequential_ols" if method.startswith("seq_") else "joint_ols"
        a=ci.loc[(ci.model==method)&(ci.baseline==own)&(ci.block_days==7)]
        b=ci.loc[(ci.model==method)&(ci.baseline=="joint_ols")&(ci.block_days==7)]
        rows.append([NAMES[method],"%+.2f~%+.2f%%"%(a.relative_rmse_pct.min(),a.relative_rmse_pct.max()),
            "%+.2f~%+.2f%%"%(b.relative_rmse_pct.min(),b.relative_rmse_pct.max()),
            "%d/5"%int((a.delta_rmse_bp<0).sum()),"%d/5"%int((a.rmse_ci_high<0).sum())])
    notes += [table(["방법","같은 구조의 OLS 대비","공동 OLS 대비","오차 감소 조건","7일 구간 전체 음수"],rows),"",
      "기존 대조군인 **PCA 요인에 회귀한 잔차**는 순차 OLS보다 RMSE가 0.74~0.90% 낮았다(약 0.072~0.090bp). "
      "다섯 조건의 고정 예측 7일 구간도 모두 음수였다. 다만 입력 공통 요인을 구성하는 방식의 차이이며 "
      "새 비선형 ML의 성과나 새로운 경제적 기여로 세지 않는다. 이 비교도 아래 탐색적 추론의 한계에 해당한다.","",
      "구간은 2,000회 달력 블록 재표집으로 계산했다. 적합된 예측에 조건부인 점별 탐색 구간이며, "
      "이번 후보 확대까지 포함한 사후 탐색·다중 비교·재학습 불확실성을 전부 반영하지 않는다. "
      "0을 포함하면 우위가 없음을 입증한 것도, 같음을 입증한 것도 아니다.","",
      "### 과거 검증으로 ML을 고르는 절차의 실제 성능","",
      "특정 모델의 평가 점수가 좋다는 관찰과, 평가 결과를 보지 않고 그 모델을 고를 수 있었는지는 다르다. "
      "아래는 각 구조에서 과거 검증으로 고른 절차의 같은 구조 OLS 대비 RMSE 차이(bp)와 7일 95% 구간이다.",""]
    rows=[]
    for s,label in zip(SCENARIOS,LABELS):
        row=[label]
        for method in ["seq_selected","seq_shrunk","joint_selected","joint_shrunk"]:
            base="sequential_ols" if method.startswith("seq_") else "joint_ols"
            a=ci.loc[(ci.scenario==s)&(ci.model==method)&(ci.baseline==base)&(ci.block_days==7)].iloc[0]
            row.append("%+.3f [%+.3f, %+.3f]"%(a.delta_rmse_bp,a.rmse_ci_low,a.rmse_ci_high))
        rows.append(row)
    notes += [table(["시각 가정","순차 ML 선택","순차 축소 선택","공동 ML 선택","공동 축소 선택"],rows),"",
      "### 보정 강도 λ의 선택","",
      "λ=0은 선형, λ=1은 비선형 보정 전체다. λ=0 선택을 ML 개선이라고 세지 않는다. "
      "선형 선택 가능성을 열어도 미래 평가 오차가 선형 이하로 보장되지는 않는다.",""]
    rows=[]
    for s,label in zip(SCENARIOS,LABELS):
        for method in ["seq_shrunk","joint_shrunk"]:
            a=eval_selected.loc[(eval_selected.scenario==s)&(eval_selected.method==method)].sort_values("month")
            rows.append([label,NAMES[method]]+["%s / λ=%g"%(r.chosen_family,r.weight) for r in a.itertuples()])
    notes += [table(["시각 가정","선택 절차","1월","2월","3월"],rows),"",
      "## 2. 동일 후속 분석: 하방·포지셔닝 계수","",
      "모든 13개 방법(세 대조군, 여섯 비선형 모형군, 네 선택 절차)에 같은 1·6·12시간·q10/q50/q90·통제변수·표본 규칙을 적용했다. "
      "2025년 8월~2026년 3월 원점을 사용했다. 아래는 기본 통제 M2·6시간·q10의 설명변수 1표준편차당 bp **점추정**이다. "
      "A3 후보의 생성 잔차 재적합 신뢰구간을 새로 계산한 표가 아니므로 부호만으로 유의성을 선언하지 않는다.",""]
    rows=[]
    for method in CONTROLS_FOR_REPORT()+methods:
        a=coef.loc[(coef.method==method)&(coef.scope=="core")&(coef["sample"]=="all")&
                   (coef.h==6)&(coef.model=="M2")&(coef.q==.1)]
        d=a.loc[a.term=="downside","effect_bp_per_sd"]
        l=a.loc[a.term=="account_ls","effect_bp_per_sd"]
        rows.append([NAMES[method],"%+.2f~%+.2f"%(d.min(),d.max()),"%+.2f~%+.2f"%(l.min(),l.max())])
    notes += [table(["방법","BTC 하방위험: 시각 가정별 범위","롱/숏 비율: 시각 가정별 범위"],rows),"",
      "거시 통제·정확한 1/6/12시간 공통 원점·분위수별 결과는 [전체 계수](results/coefficients.csv), "
      "전후반·스트레스 날짜 민감도는 [추가 표](results/subperiod_and_stress.csv)에 있다. "
      "공통 원점이 없거나 최소 표본 요건을 충족하지 못한 조건은 추정 불가로 저장했다([표본 수](results/sample_counts.csv)). "
      "분해 오차가 감소하더라도 잔차가 한국 고유 수요의 참값이 되거나 포지셔닝의 인과 경로가 입증되는 것은 아니다.","",
      "## 3. 계산 검증과 한계","",
      "- 모델·표준화는 매월 이전 자료로만 적합하고 원점과 미래 잔차에 같은 함수를 사용했다.",
      "- 기존 A1/A2/three_way를 보존했다. 선형·PCA 대조를 이전 결과와 직접 대조한다.",
      "- 모든 후보·검증 점수·선택·월별 평가·관측별 잔차를 보존했다. 수렴 경고를 숨기지 않았다."]
    warning_fits=[];selected_warning_fits=[];mlp_fits=0
    for s in SCENARIOS:
        fit=json.loads((OUT/s/"fits.json").read_text())
        used={(r.month,r.candidate_id) for r in selected.loc[selected.scenario==s].itertuples()}
        for r in fit:
            mlp_fits += int(r["candidate"]["family"].endswith("mlp"))
            if r["warnings"]:
                warning_fits.append((s,r["month"],r["candidate"]["candidate_id"]))
                if (r["month"],r["candidate"]["candidate_id"]) in used:
                    selected_warning_fits.append((s,r["month"],r["candidate"]["candidate_id"]))
    notes += ["- MLP 적합 %d개 중 경고가 있는 적합: %d개. 실제 출력에 사용된 후보·월 조합 중 경고: %d개. "
              "경고별 내용과 반복 수는 시간대별 `fits.json`에 있다. 일부 설정은 L-BFGS 반복 한도에 도달했다. "
              "고정된 반복 예산의 학습 절차를 평가한 것이며, 신경망의 충분히 최적화된 최선 성능을 확정한 것이 아니다."%
              (mlp_fits,len(warning_fits),len(selected_warning_fits)),
      "- FX/거시 시각 미확정, USDC/USD=1 가정, 짧고 이미 여러 번 분석한 역사자료라는 제약이 남는다.",
      "- A3는 잔차 분해와 동일 후속 점추정 비교다. 새 후보를 대상으로 미래 잔차 예측모형까지 재평가한 것으로 표시하지 않는다.","",
      "[실행 규칙](PROTOCOL_KO.md), `models.py`, `run_comparison.py`, `check_guards.py`, `verify_outputs.py`, "
      "[실행 해시](results/RUN_MANIFEST.json), [독립 출력 검증](VERIFICATION.json).",""]
    (HERE/"RESULTS_KO.md").write_text("\n".join(notes))
    print("A3 report written; warning fits:",len(warning_fits))


def CONTROLS_FOR_REPORT():
    return ["sequential_ols","joint_ols","pca"]


if __name__=="__main__":
    main()
