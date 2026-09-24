"""Compare fresh original-code runs with all six manuscript tables.

The DOCX accompanying the PDF supplies the printed table cells. Matching means
agreement at the manuscript's displayed precision, not statistical validity.
Historical CSVs are labelled separately and are never used as fresh estimates.
"""
import os
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".runtime/matplotlib"))

import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO / "research/reference/original_v4"
MANUSCRIPT = REPO / "research/reference/submitted/경영과학학술지_김수빈_V2_stablecoin"
REPORT = HERE / "comparison"
SOURCES = {
    "default_B300": HERE / "runs/default_B300/paper_outputs",
    "archive_main_B40": HERE / "runs/archive_main_B40/paper_outputs",
    "archive_supplement_B200": HERE / "runs/archive_supplement_B200/paper_outputs",
    "historical_saved_NOT_RERUN": ORIGINAL / "paper_outputs",
}


def extract_tables():
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(MANUSCRIPT.with_suffix(".docx")) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    tables = [
        [["".join(node.text or "" for node in cell.findall(".//w:t", ns))
          for cell in row.findall("w:tc", ns)] for row in table.findall("w:tr", ns)]
        for table in root.findall(".//w:tbl", ns)
    ]
    if len(tables) != 6 or [len(t) for t in tables] != [13, 3, 6, 4, 4, 6]:
        raise ValueError("Unexpected manuscript tables; review the extraction")
    return tables


def compact(s):
    return re.sub(r"\s+", "", str(s))


def pair(s):
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", compact(s))]


def cell(table, label, stat, printed, value, source, filename, column):
    raw = compact(printed)
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return dict(table=table, label=label, stat=stat, printed=raw, source=source,
                recomputed=float(value), matches_printed_precision=bool(
                    abs(float(value) - float(raw)) <= .500001 * 10 ** (-decimals)),
                file=filename, column=column)


def compare_tables(tables, df):
    checks = []
    cols = ["RESIDUAL_FINAL", "USDT_KP", "MKT_KP_EQ", "BTC_DOWNSIDE_24", "BTC_VOL_24",
            "ACCOUNT_LS", "TOPTRADER_ACCOUNT_LS", "OI", "FUNDING", "DXY_ret", "VIX", "USDKRW_ret"]
    units = [1e4, 100, 100, 1e4, 100, 1, 1, .001, 1e4, 100, 1, 100]
    stats = ["N", "mean", "std", "min", "q25", "median", "q75", "max", "skew", "excess_kurtosis"]
    descriptive = []
    for row, col, unit in zip(tables[0][1:], cols, units):
        s = df[col].dropna() * unit
        values = [len(s), s.mean(), s.std(), s.min(), s.quantile(.25), s.median(),
                  s.quantile(.75), s.max(), s.skew(), s.kurt()]
        descriptive.append(dict(variable=col, scale=unit, **dict(zip(stats, values))))
        for stat, raw, value in zip(stats, row[1:], values):
            checks.append(cell(1, col, stat, raw, value, "default_B300",
                               "baseline_dataset_with_residual_final.csv", col))
    pd.DataFrame(descriptive).to_csv(REPORT / "table1_recomputed.csv", index=False)

    cache = {}
    def add(table, label, stat, printed, filename, selector, column, scale=1):
        for source, folder in SOURCES.items():
            path = folder / filename
            if not path.exists():
                if source == "default_B300":
                    raise FileNotFoundError(path)
                continue  # Historical profiles deliberately run only their scripts.
            if path not in cache:
                cache[path] = pd.read_csv(path)
            data = cache[path]
            chosen = data.loc[selector(data), column]
            if len(chosen) != 1:
                raise ValueError(f"Expected one cell for {source}/{table}/{label}/{column}")
            checks.append(cell(table, label, stat, printed, float(chosen.iloc[0]) * scale,
                               source, filename, column))

    for row, regime in zip(tables[1][1:], ["Normal", "Downside"]):
        for stat, raw, column, scale in zip(["N", "mean", "std", "skew"], row[1:],
                                           ["n", "mean", "std", "skew"], [1, 1e4, 1e4, 1]):
            add(2, regime, stat, raw, "table1_residual_descriptive.csv",
                lambda d, r=regime: d.regime == r, column, scale)
    for row in tables[2][1:]:
        q = float(compact(row[0]))
        for stat, raw, column in [("beta", row[1], "beta_downside"), ("p", row[2], "p_boot")]:
            add(3, str(q), stat, raw, "table3_main_quantile_tail.csv", lambda d, q=q: d.q == q, column)
    for row, model in zip(tables[3][1:], ["M0_base+downside", "M1_+global_state", "M2_+ACCOUNT_LS"]):
        for raw, column in zip(row[1:], ["downside", "btc_vol", "account_ls"]):
            values = pair(raw)
            if not values:
                continue
            for stat, printed, col in [("beta", f"{values[0]:.3f}", column),
                                        ("p", f"{values[1]:.2f}", "p_" + column)]:
                add(4, model + ":" + column, stat, printed, "table7_decisive_robustness.csv",
                    lambda d, m=model: (d.q == .1) & (d.model == m), col)
    for row, definition in zip(tables[4][1:], ["EQ", "CAP", "PCA"]):
        for raw, q in zip(row[1:3], [.1, .5]):
            beta, p = pair(raw)
            for stat, value, column in [("beta", beta, "beta_downside"), ("p", p, "p")]:
                add(5, definition + f"_q{q}", stat, f"{value:.2f}", "robustness_kp_main_quantile.csv",
                    lambda d, k=definition, q=q: (d.definition == k) & (d.q == q), column)
        add(5, definition + "_M2", "p", row[3], "robustness_kp_mediation.csv",
            lambda d, k=definition: (d.definition == k) & (d.channel == "ACCOUNT_LS (decisive M2)"), "ch_p")
    for row, sample in zip(tables[5][1:], ["FULL", "DROP_TOP1_DAY", "DROP_TOP5_DAYS", "FIRST_HALF", "SECOND_HALF"]):
        beta, p = pair(row[2])
        for stat, value, column, decimals in [
            ("N", float(compact(row[1])), "n", 0), ("beta", beta, "q10_dn", 2),
            ("p", p, "q10_p", 2), ("leverage_p", pair(row[3])[-1], "M2_leverage_p", 2),
        ]:
            add(6, sample, stat, f"{value:.{decimals}f}", "exp2_subsample_stability.csv",
                lambda d, s=sample: d["sample"] == s, column)
    out = pd.DataFrame(checks)
    if len(out[out.source == "default_B300"]) != 185:
        raise AssertionError("Not all 185 manuscript numerical result cells were checked")
    out.to_csv(REPORT / "manuscript_cells.csv", index=False)
    out[~out.matches_printed_precision].to_csv(REPORT / "mismatches.csv", index=False)
    return out


def compare_archived_csvs():
    keys = {
        "table1_residual_descriptive.csv": ["regime"],
        "table2_stationarity.csv": ["test"],
        "table3_main_quantile_tail.csv": ["q"],
        "table4_quantile_local_projection.csv": ["h", "q"],
        "table5_stress_regime.csv": ["q"],
        "table6_demand_channel_mediation.csv": ["q", "channel"],
        "table7_decisive_robustness.csv": ["q", "model"],
        "robustness_kp_main_quantile.csv": ["definition", "q"],
        "robustness_kp_mediation.csv": ["definition", "channel"],
        "exp1_persistence_mediation.csv": ["mediator", "h", "q"],
        "exp2_subsample_stability.csv": ["sample"],
        "economic_magnitude.csv": ["지표"],
    }
    rows = []
    for source, folder in SOURCES.items():
        if source == "historical_saved_NOT_RERUN":
            continue
        for path in sorted(folder.glob("*.csv")):
            archived = ORIGINAL / "paper_outputs" / path.name
            if not archived.exists():
                continue
            old, new = pd.read_csv(archived), pd.read_csv(path)
            same_schema = list(old.columns) == list(new.columns) and old.shape == new.shape
            good, bad = 0, []
            if same_schema:
                # Some original tables sort by p-value, so align model identities
                # before comparing. A changed sort order is not a changed model.
                key = keys[path.name]
                if old.duplicated(key).any() or new.duplicated(key).any():
                    raise ValueError(f"Non-unique row identifiers: {path.name}")
                old = old.sort_values(key).reset_index(drop=True)
                new = new.sort_values(key).reset_index(drop=True)
                for column in old:
                    if pd.api.types.is_numeric_dtype(old[column]) and pd.api.types.is_numeric_dtype(new[column]):
                        ok = np.isclose(old[column], new[column], atol=1e-8, rtol=1e-8, equal_nan=True)
                    else:
                        ok = (old[column].fillna("").astype(str) == new[column].fillna("").astype(str)).values
                    good += int(ok.sum())
                    bad.extend(f"{column}[{i}]: {old[column].iloc[i]} -> {new[column].iloc[i]}"
                               for i in np.flatnonzero(~ok))
            rows.append(dict(source=source, file=path.name, same_schema=same_schema,
                             matching_cells=good, total_cells=old.size,
                             reason="; ".join(bad) if same_schema else "schema/shape differs"))
    out = pd.DataFrame(rows)
    out.to_csv(REPORT / "archived_csv_comparison.csv", index=False)
    return out


def write_report(cells, meta, df, manifest):
    summary = cells.groupby(["table", "source"]).matches_printed_precision.agg(["sum", "count"]).reset_index()
    summary.to_csv(REPORT / "table_summary.csv", index=False)
    default = cells[cells.source == "default_B300"]
    lines = [
        "# 국내 원고 원본 재현 결과", "",
        "ML 실험을 제외하고 보존된 전처리 데이터와 원본 코드 5개를 다시 실행한 결과다. "
        "재현 성공 여부와 분석의 타당성은 별개다. 이 단계에서 원본 계산은 수정하지 않았다.", "",
        "**2026-09-25 추가 확인:** 아래 기본값 실행의 불일치 12개 중 10개는 표별 회귀 반복·부트스트랩 횟수 차이로 재현 확인했다. "
        "나머지 2개는 원고 숫자의 실제 작성 경위가 확인되지 않았다. [원인 추적과 통제 실행](difference_audit/RESULTS_KO.md)을 참조한다. "
        "아래 기본값 결과는 그대로 보존한다.", "",
        f"- 입력: {len(df)}개 관측치, {df.index.min()} ~ {df.index.max()}, {df.index.normalize().nunique()}일.",
        f"- 잔차 재계산과 저장 잔차의 최대 절대 차이: {meta['rebuild_max_dev']:.3e} (프리미엄 비율 단위).",
        f"- 1단계 시장 요인 회귀 R²: {meta['step1_r2']:.10f}; 2단계 글로벌 괴리 회귀 R²: {meta['step2_r2']:.10f}.",
        f"- 원본 기본값 B=300 실행: 원고 표 1–6의 숫자 {len(default)}개 중 {int(default.matches_printed_precision.sum())}개가 원고의 표시 자릿수에서 일치.",
        "- 같은 값은 완전 동일한 원자료 재수집·전처리를 입증하지 않는다. 이번 재현의 시작점은 보존된 전처리 CSV다.",
        "- 기존 CSV와 대조하기 위한 B=40, B=200 실행도 각각 보관했다. 이는 2026-09-08 감사에서 확인한 설정이며, 유의성에 맞춰 새로 탐색한 값이 아니다.",
        "", "| 원고 표 | 원본 기본값 B=300 | 과거 설정으로 재실행 | 당시 저장 CSV (재실행 아님) |",
        "|---|---:|---:|---:|",
    ]
    def count(table, source):
        part = cells[(cells.table == table) & (cells.source == source)]
        return f"{int(part.matches_printed_precision.sum())}/{len(part)}" if len(part) else "—"
    for table in range(1, 7):
        source = "archive_main_B40" if table <= 4 else "archive_supplement_B200"
        historical = count(table, source)
        if historical != "—":
            historical += " (B=40)" if table <= 4 else " (B=200)"
        lines.append(f"| 표 {table} | {count(table, 'default_B300')} | {historical} | {count(table, 'historical_saved_NOT_RERUN')} |")
    lines += ["", "숫자는 일치 셀/대조 셀이다. 모형 이름·분위수·열 제목은 셀 수에서 제외했다. "
              "모든 프로필을 함께 공개하며, 표마다 가장 잘 맞는 설정을 골라 하나의 실행 결과처럼 제시하지 않는다.",
              "", "## 기본값 실행에서 원고와 다른 셀", "",
              "| 표 | 항목 | 통계량 | 원고 | 재실행 |", "|---|---|---|---:|---:|"]
    for row in default[~default.matches_printed_precision].itertuples():
        lines.append(f"| {row.table} | {row.label} | {row.stat} | {row.printed} | {row.recomputed:.9g} |")
    lines += [
        "", "## 일치하지 않는 부분의 해석", "",
        "1. 표 1의 BTC 하방위험 왜도는 원자료 계산값이 약 4.8945로, 두 자리 반올림은 4.89이다. 원고에는 4.90으로 적혀 있다.",
        "2. 표 3은 B=300 실행에서 일치하지만, 표 6은 B=200 실행에서 일치한다. 단일 부트스트랩 설정으로 원고 전체가 생성됐다고 볼 수 없다.",
        "3. 표 4의 M2 BTC 변동성 p값은 B=300에서 0.027(표시하면 0.03), 원고는 0.05이다.",
        "4. 표 5의 당시 저장 CSV는 원고와 일치한다. 기본 코드 설정으로는 CAP 계수 등을 재생성하지 못했으나, "
        "추가 진단에서 점추정 반복 60회·부트스트랩 내부 반복 35회·B=150이면 두 과거 CSV가 모두 재현됨을 확인했다. "
        "기본 코드의 점추정 반복은 70회, B는 300회다. [셀별 원인](difference_audit/twelve_cell_causes.csv)을 별도 보존했다.",
        "5. 원본 코드의 정상성 검정은 이번 환경에서 statsmodels를 사용해 실행된다. 당시 저장 CSV는 해당 패키지 미설치로 건너뛴 기록이므로 형식이 다르다.",
        "", "## 그림", "",
        "원본 main 코드로 4개 그림을 새로 생성했다. 원고 그림 1/2/3/4와 코드 파일 fig1/fig3/fig4/fig5의 번호가 다르다. "
        "원고 그림 2의 90% 신뢰구간과 그림 3·4의 오차막대는 보존된 main 코드에 생성 과정이 없다. "
        "따라서 이 시각 요소까지 동일하게 재현했다고 주장하지 않는다. 원고에 포함된 그림을 복사하여 새로 계산한 그림으로 대체하지 않았다.",
        "", "| 원고 | 새로 생성한 그림 | 한계 |", "|---|---|---|",
        "| 그림 1 | [시계열](runs/default_B300/paper_outputs/fig1_timeseries.png) | 스타일/픽셀 동일성은 보장하지 않음 |",
        "| 그림 2 | [기간별 분위수 반응](runs/default_B300/paper_outputs/fig3_localprojection.png) | 원고의 신뢰구간 재현 불가 |",
        "| 그림 3 | [수요·포지셔닝 변수 비교](runs/default_B300/paper_outputs/fig4_mediation.png) | 원고의 오차막대 재현 불가 |",
        "| 그림 4 | [M0/M1/M2](runs/default_B300/paper_outputs/fig5_decisive.png) | 원고의 오차막대 재현 불가 |",
        "", "## 보존한 계산과 다음 단계", "",
        "- 원본 2단계 회귀, 가격 환산식, 결측 제거 표본, 행 기준 시차, 분위수 IRLS 반복 횟수, p값 계산법을 유지했다.",
        "- main 코드가 계산하는 RESIDUAL_FINAL_REBUILD를 저장 잔차와 대조했다. 원본과 마찬가지로 후속 분석은 저장된 RESIDUAL_FINAL을 사용한다.",
        "- 원본의 `hours ahead` 표기, 레버리지 `mediates` 판정, 최소 관측 가격 변화에 근거한 tick 계산도 코드에 그대로 남아 있다. "
        "실제 시간 간격·인과적 매개·공식 호가단위의 타당성을 이번 재현이 확인한 것은 아니다.",
        "- 가격 표시 통화, 실제 시간 기준 시차, 수치해법 수렴, 부트스트랩 추론을 점검·수정하는 작업은 다음 개발 브랜치에서 한다. ML은 이 기준선 이후에 평가한다.",
        "", "## 증거 파일", "",
        "- [실행 환경·입력/코드/출력 SHA-256](run_manifest.json)",
        "- [원고에서 추출한 표](comparison/manuscript_tables.json)",
        "- [전체 셀 비교](comparison/manuscript_cells.csv) / [불일치](comparison/mismatches.csv)",
        "- [과거 CSV와 새 실행 비교](comparison/archived_csv_comparison.csv)",
        "- [잔차 재계산 수치](comparison/rebuild.json) / [표 1 재계산](comparison/table1_recomputed.csv)",
        "", "환경: Python " + manifest["python"].split()[0] + "; " + ", ".join(
            f"{k}={v}" for k, v in manifest["packages"].items()) + ".", "",
    ]
    (HERE / "RESULTS_KO.md").write_text("\n".join(lines))
    return summary


def main():
    REPORT.mkdir(exist_ok=True)
    manifest = json.loads((HERE / "run_manifest.json").read_text())
    if len(manifest["runs"]) != 9 or any(x["exit_code"] for x in manifest["runs"]):
        raise RuntimeError("All nine original-code runs must complete before comparison")
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed since execution: {name}")
    for name, expected in manifest["outputs_sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Output changed since execution: {name}")
    tables = extract_tables()
    (REPORT / "manuscript_tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=2) + "\n")
    (REPORT / "manuscript_sha256.json").write_text(json.dumps({
        str(MANUSCRIPT.with_suffix(ext).relative_to(REPO)):
        hashlib.sha256(MANUSCRIPT.with_suffix(ext).read_bytes()).hexdigest()
        for ext in [".pdf", ".docx"]}, indent=2) + "\n")
    script = HERE / "runs/default_B300/stablecoin_paper_pipeline.py"
    spec = importlib.util.spec_from_file_location("original_paper_rebuild", script)
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)
    df, meta = paper.load_and_build()
    if len(df) != 715 or meta["rebuild_max_dev"] >= 1e-12:
        raise AssertionError("Original stored residual was not reproduced")
    (REPORT / "rebuild.json").write_text(json.dumps(meta, indent=2) + "\n")
    df[["USDT_KP", "MKT_KP_EQ", "DEPEG_GLOBAL", "RESIDUAL_OLS", "RESIDUAL_FINAL",
        "RESIDUAL_FINAL_REBUILD"]].to_csv(REPORT / "residual_rebuild.csv")
    cells = compare_tables(tables, df)
    compare_archived_csvs()
    summary = write_report(cells, meta, df, manifest)
    print(summary.to_string(index=False))
    print(f"Rebuilt residual max error: {meta['rebuild_max_dev']:.3e}")
    print(f"Report: {HERE / 'RESULTS_KO.md'}")


if __name__ == "__main__":
    main()
