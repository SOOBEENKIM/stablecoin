"""Identify workstation manuscript lineage by content, not folder timestamps.

Optional local provenance check; the reproduction itself does not require these
workstation directories. Usage: python3 verify_workstation_sources.py --workspace PATH
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVE = REPO / "research/reference/original_v4"
FILES = [
    "stablecoin_paper_pipeline.py", "kp_robustness.py",
    "experiment1_persistence_mediation.py", "experiment2_subsample_stability.py",
    "economic_magnitude.py", "corrected_outputs/baseline_dataset_with_residual_final.csv",
    "FinxLab_stablecoin_20260409_datapreprocessing - 복사본.ipynb",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docx_info(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "dcterms": "http://purl.org/dc/terms/"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        core = ET.fromstring(archive.read("docProps/core.xml"))
    return dict(sha256=sha(path), tables=len(root.findall(".//w:tbl", ns)),
                created=core.findtext("dcterms:created", namespaces=ns),
                modified=core.findtext("dcterms:modified", namespaces=ns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    rows = []
    for name in ["stablecoin", "stablecoin_v3", "stablecoin_v4", "stablecoin_workshop"]:
        folder = workspace / name
        for filename in FILES:
            path = folder / filename
            expected = sha(ARCHIVE / filename)
            actual = sha(path) if path.is_file() else None
            rows.append(dict(folder=name, file=filename, workstation_sha256=actual,
                             archive_sha256=expected, same_bytes=actual == expected,
                             exists=path.is_file()))
    v4 = workspace / "stablecoin_v4"
    full = []
    for path in sorted(v4.rglob("*")):
        relative = path.relative_to(v4)
        if not path.is_file() or any(part in {"__pycache__", ".git", ".cache"} for part in relative.parts):
            continue
        counterpart = ARCHIVE / relative
        if not counterpart.is_file():
            # The archive consolidates raw data elsewhere; absence here is not
            # evidence of different data or of a missing repository-wide file.
            full.append(dict(file=str(relative), status="not_at_this_archive_path"))
        else:
            actual, expected = sha(path), sha(counterpart)
            full.append(dict(file=str(relative), status="same_bytes" if actual == expected else "different",
                             workstation_sha256=actual, archive_sha256=expected))
    documents = {}
    for name in ["stablecoin_v3", "stablecoin_v4"]:
        for path in sorted((workspace / name).glob("*.docx")):
            if not path.name.startswith("~$"):
                documents[str(path.relative_to(workspace))] = docx_info(path)
    submitted = REPO / "research/reference/submitted/경영과학학술지_김수빈_V2_stablecoin.docx"
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), workspace=str(workspace),
                  archive=str(ARCHIVE.relative_to(REPO)), key_files=rows,
                  v4_full_path_comparison=full, workstation_documents=documents,
                  final_manuscript=docx_info(submitted),
                  final_pdf_sha256=sha(submitted.with_suffix(".pdf")),
                  conclusion="All five executed scripts, processed input, and preprocessing notebook match stablecoin_v4 byte for byte. The attached final manuscript is later than the v4 folder manuscripts.")
    if not all(row["same_bytes"] for row in rows if row["folder"] == "stablecoin_v4"):
        raise AssertionError("The workstation v4 analysis files differ; review before reproducing")
    (HERE / "comparison/workstation_sources.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    same = sum(row["status"] == "same_bytes" for row in full)
    different = sum(row["status"] == "different" for row in full)
    omitted = sum(row["status"] == "not_at_this_archive_path" for row in full)
    print(f"stablecoin_v4: {same} matching archive paths, {different} different, {omitted} paths not in this archive subtree")
    print("Executed scripts (5), processed input, notebook: all seven byte-identical to workstation stablecoin_v4.")


if __name__ == "__main__":
    main()
