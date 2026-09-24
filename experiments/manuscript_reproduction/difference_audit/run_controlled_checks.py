"""Forensic parameter checks; these are NOT new preferred research estimates.

Keep input, factors, row lags, seed and call sequence fixed. Change only the
recorded iteration/bootstrap settings. Matching a historical output identifies
a numerical mechanism; it does not prove an unavailable historical invocation.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
REPO = HERE.parents[2]
SOURCE = REPO / "research/reference/original_v4"
INPUT = Path("corrected_outputs/baseline_dataset_with_residual_final.csv")
CASES = [
    # A successive controlled chain from the previously committed baseline.
    ("kp_B300_point60_boot35", "kp_robustness.py", 300, 60, 35, False),
    ("kp_B150_point60_boot35", "kp_robustness.py", 150, 60, 35, False),
    ("kp_B150_point60_boot30", "kp_robustness.py", 150, 60, 30, False),
    # Finite iteration limits present in the other original scripts, and an
    # isolated function invocation to check the documented RNG-order issue.
    ("main_B300_point80_boot30", "stablecoin_paper_pipeline.py", 300, 80, 30, False),
    ("main_B300_point80_boot35", "stablecoin_paper_pipeline.py", 300, 80, 35, False),
    ("main_B300_table7_only", "stablecoin_paper_pipeline.py", 300, 80, 40, True),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(case):
    name, script, B, point, bootstrap, isolated = case
    folder = HERE / "runs" / name
    (folder / INPUT.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / INPUT, folder / INPUT)
    code = (SOURCE / script).read_text()
    edits = []
    if script == "kp_robustness.py":
        edits = [("def qr(X, y, tau, it=70, eps=1e-7):", f"def qr(X, y, tau, it={point}, eps=1e-7):"),
                 ("bb = qr(Xf[rr], yv[rr], tau, it=35)", f"bb = qr(Xf[rr], yv[rr], tau, it={bootstrap})")]
    else:
        edits = [("b = quantile_fit(Xf[rr], yv[rr], tau, n_iter=40)",
                  f"b = quantile_fit(Xf[rr], yv[rr], tau, n_iter={bootstrap})")]
        if isolated:
            edits.append(('if __name__ == "__main__":\n    main()',
                          'if __name__ == "__main__":\n    df, meta = load_and_build()\n    table7_decisive(df)'))
    for old, new in edits:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    (folder / script).write_text(code)
    env = dict(os.environ, B_BOOT=str(B), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLCONFIGDIR=str(REPRO / ".runtime/matplotlib"))
    start = time.monotonic()
    with (folder / "run.log").open("w") as log:
        proc = subprocess.run([sys.executable, script], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = dict(case=name, B_BOOT=B, point_iterations=point, bootstrap_iterations=bootstrap,
                  seed=42, isolated_table7=isolated, exit_code=proc.returncode,
                  seconds=round(time.monotonic() - start, 3),
                  original_script_sha256=sha(SOURCE / script), input_sha256=sha(folder / INPUT),
                  executed_script_sha256=sha(folder / script), edits=edits)
    print(f"{name}: exit={proc.returncode}, {result['seconds']}s", flush=True)
    return result


def main():
    before = {name: sha(SOURCE / name) for name in ["kp_robustness.py", "stablecoin_paper_pipeline.py", str(INPUT)]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, case) for case in CASES]
        results = [f.result() for f in as_completed(futures)]
    assert before == {name: sha(SOURCE / name) for name in before}
    manifest = dict(purpose="Forensic reconstruction, not statistical model selection", python=sys.version,
                    packages={name: version(name) for name in ["numpy", "pandas", "scipy"]},
                    original_sources_unchanged=True, cases=sorted(results, key=lambda r: r["case"]),
                    output_sha256={str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / "runs").glob("*/paper_outputs/*.csv"))})
    (HERE / "controlled_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if any(r["exit_code"] for r in results):
        raise RuntimeError("A diagnostic run failed")


if __name__ == "__main__":
    main()
