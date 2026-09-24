"""Map every PDF/default-run discrepancy to a checked numerical mechanism."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import skew

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
REPO = HERE.parents[2]
SOURCE = REPO / "research/reference/original_v4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coefficient_iterations():
    k = load_module(REPRO / "runs/default_B300/kp_robustness.py", "original_kp")
    df = pd.read_csv(k.DATA, parse_dates=["datetime_utc"], index_col="datetime_utc").sort_index()
    cap = pd.DataFrame({c: df[c + "_BINANCE_CLOSE"] * k.SUPPLY[c] for c in k.COINS})
    weights = cap.div(cap.sum(axis=1), axis=0)
    df["MKT_KP_CAP"] = (weights.values * df[k.KP].values).sum(axis=1)
    K = df[k.KP].dropna()
    Z = (K - K.mean()) / K.std()
    _, vec = np.linalg.eigh(np.corrcoef(Z.values.T))
    v = vec[:, -1]
    if v.sum() < 0:
        v = -v
    df.loc[K.index, "MKT_KP_PCA"] = Z.values @ v
    saved = pd.read_csv(SOURCE / "paper_outputs/robustness_kp_main_quantile.csv")
    current = pd.read_csv(REPRO / "runs/default_B300/paper_outputs/robustness_kp_main_quantile.csv")
    rows = []
    for definition, col in [("EQ", "MKT_KP_EQ"), ("CAP", "MKT_KP_CAP"), ("PCA", "MKT_KP_PCA")]:
        d = df.copy()
        r1 = k.ols_resid(d.USDT_KP.values, d[col].values)
        d["RF"] = k.ols_resid(r1, d.DEPEG_GLOBAL.values)
        d["RESIDUAL_FINAL_lag1"] = d.RF.shift(1)
        dd = d[["RF"] + k.BX].dropna()
        X, y = k.addc(dd[k.BX].values), dd.RF.values
        for q in [.1, .25, .5, .75, .9]:
            target = float(saved[(saved.definition == definition) & (saved.q == q)].beta_downside.iloc[0])
            row = dict(definition=definition, q=q, saved_coefficient=target)
            for it in [40, 60, 70, 80]:
                value = float(k.qr(X, y, q, it=it)[2])
                row[f"it{it}_raw"] = value
                row[f"it{it}_csv3dp"] = round(value, 3)
            observed = float(current[(current.definition == definition) & (current.q == q)].beta_downside.iloc[0])
            assert row["it70_csv3dp"] == observed
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "coefficient_iterations.csv", index=False)
    assert (out.it60_csv3dp == out.saved_coefficient).all()


def compare_historical_csvs():
    rows = []
    for run in sorted((HERE / "runs").glob("kp_*")):
        for name in ["robustness_kp_main_quantile.csv", "robustness_kp_mediation.csv"]:
            old = pd.read_csv(SOURCE / "paper_outputs" / name)
            new = pd.read_csv(run / "paper_outputs" / name).rename(columns={"result": "mediates"})
            assert old.shape == new.shape and list(old.columns) == list(new.columns)
            for col in old:
                if pd.api.types.is_numeric_dtype(old[col]):
                    ok = np.isclose(old[col], new[col], rtol=0, atol=1e-9, equal_nan=True)
                else:
                    ok = old[col].fillna("").astype(str).values == new[col].fillna("").astype(str).values
                for i, match in enumerate(ok):
                    rows.append(dict(case=run.name, file=name, row=i, column=col,
                                     saved=old[col].iloc[i], recomputed=new[col].iloc[i], matches=bool(match)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "historical_csv_cells.csv", index=False)
    summary = out.groupby(["case", "file"]).matches.agg(["sum", "count"])
    summary.to_csv(HERE / "historical_csv_summary.csv")
    chosen = out[out["case"] == "kp_B150_point60_boot35"]
    assert len(chosen) == 123 and chosen.matches.all()
    return summary


def compare_all_manuscript_cells():
    cmp = load_module(REPRO / "compare_manuscript.py", "manuscript_comparator")
    cmp.REPORT = HERE / "cell_comparison"
    cmp.REPORT.mkdir(exist_ok=True)
    cmp.SOURCES.update({name: HERE / "runs" / name / "paper_outputs"
                        for name in ["kp_B300_point60_boot35", "kp_B150_point60_boot35", "kp_B150_point60_boot30",
                                     "main_B300_point80_boot30", "main_B300_point80_boot35", "main_B300_table7_only"]})
    paper = load_module(REPRO / "runs/default_B300/stablecoin_paper_pipeline.py", "main_rebuild")
    df, meta = paper.load_and_build()
    tables = cmp.extract_tables()
    cells = cmp.compare_tables(tables, df)
    summary = cells.groupby(["table", "source"]).matches_printed_precision.agg(["sum", "count"])
    summary.to_csv(HERE / "manuscript_settings_summary.csv")
    chosen = cells[(cells.table == 5) & (cells.source == "kp_B150_point60_boot35")]
    assert len(chosen) == 15 and chosen.matches_printed_precision.all()
    out = []
    missing = cells[(cells.source == "default_B300") & ~cells.matches_printed_precision]
    assert len(missing) == 12
    for _, row in missing.iterrows():
        record = row.to_dict()
        if row.table == 5:
            source = "kp_B150_point60_boot35"
            reason = "Point fit iterations 70 vs 60" if row.stat == "beta" else "Bootstrap count 300 vs 150 (fit call sequence fixed)"
            confidence = "controlled_reproduction_confirmed"
        elif row.table == 6:
            source = "archive_supplement_B200"
            reason = "Bootstrap count 300 vs 200; original code unchanged"
            confidence = "controlled_reproduction_confirmed"
        else:
            source = ""
            reason = "Printed skew differs from both pandas and scipy; historical calculation/edit unavailable" if row.table == 1 else "Full-run p=0.027; isolated-table p=0.047 rounds to 0.05, but other p-values then disagree; historical invocation unavailable"
            confidence = "historical_origin_unresolved"
        record.update(cause=reason, evidence_status=confidence, reconstruction_source=source)
        if source:
            matching = cells[(cells.source == source) & (cells.table == row.table) &
                             (cells.label == row.label) & (cells.stat == row.stat)]
            assert len(matching) == 1 and matching.iloc[0].matches_printed_precision
            record["reconstructed"] = float(matching.iloc[0].recomputed)
        out.append(record)
    causes = pd.DataFrame(out)
    causes.to_csv(HERE / "twelve_cell_causes.csv", index=False)
    assert (causes.evidence_status == "controlled_reproduction_confirmed").sum() == 10
    s = df.BTC_DOWNSIDE_24
    diagnostic = dict(n=len(s), pandas_skew=float(s.skew()), scipy_adjusted_skew=float(skew(s, bias=False)),
                      scipy_unadjusted_skew=float(skew(s, bias=True)), printed_skew=4.90,
                      direct_round_2dp=round(float(s.skew()), 2), rebuild=meta)
    (HERE / "descriptive_check.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    return cells, causes


def document_lineage():
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    docs = [SOURCE / f"manuscript_경영과학_김수빈{suffix}.docx" for suffix in ["", "_v2", "_v3", "_v4"]]
    docs.append(REPO / "research/reference/submitted/경영과학학술지_김수빈_V2_stablecoin.docx")
    rows = []
    for path in docs:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        tables = [[["".join(t.text or "" for t in cell.findall(".//w:t", ns)) for cell in row.findall("w:tc", ns)]
                   for row in table.findall("w:tr", ns)] for table in root.findall(".//w:tbl", ns)]
        rows.append(dict(file=str(path.relative_to(REPO)), sha256=sha(path), tables=tables,
                         tracked_changes=len(root.findall(".//w:ins", ns)) + len(root.findall(".//w:del", ns))))
    (HERE / "document_lineage.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def main():
    manifest = json.loads((HERE / "controlled_manifest.json").read_text())
    assert len(manifest["cases"]) == 6 and all(r["exit_code"] == 0 for r in manifest["cases"])
    for name, expected in manifest["output_sha256"].items():
        assert sha(HERE / name) == expected
    coefficient_iterations()
    historical = compare_historical_csvs()
    cells, causes = compare_all_manuscript_cells()
    document_lineage()
    verification = dict(controlled_runs=6, main_baseline_unchanged=True,
                        default_mismatching_cells=12, controlled_explanations=10, historical_origin_unresolved=2,
                        reconstructed_table5_cells=15, reconstructed_table5_matches=15,
                        reconstructed_kp_archived_cells=123, reconstructed_kp_archived_matches=123,
                        original_table6_B200_matches=20, original_table6_cells=20)
    (HERE / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(historical.to_string())
    print(causes[["table", "label", "stat", "printed", "recomputed", "reconstructed", "evidence_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
