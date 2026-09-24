"""Run byte-identical original scripts against the preserved processed dataset.

No fitting code, random-number sequence, input prices, or lag definitions are
changed here. Each script runs in its own process, as in the original workflow.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "research/reference/original_v4"
INPUT = Path("corrected_outputs/baseline_dataset_with_residual_final.csv")
SCRIPTS = [
    "stablecoin_paper_pipeline.py", "kp_robustness.py",
    "experiment1_persistence_mediation.py", "experiment2_subsample_stability.py",
    "economic_magnitude.py",
]
# These are documented historical settings, not settings chosen to optimize a
# coefficient or p-value. All profiles and their discrepancies are reported.
PROFILES = {
    "default_B300": (300, SCRIPTS),
    "archive_main_B40": (40, SCRIPTS[:1]),
    "archive_supplement_B200": (200, SCRIPTS[1:4]),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_script(profile, script, bootstrap):
    folder = HERE / "runs" / profile
    env = os.environ.copy()
    env.update(B_BOOT=str(bootstrap), PYTHONUNBUFFERED="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLCONFIGDIR=str(HERE / ".runtime/matplotlib"))
    log = folder / (Path(script).stem + ".log")
    start = time.monotonic()
    with log.open("w") as stream:
        result = subprocess.run([sys.executable, script], cwd=folder, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    elapsed = round(time.monotonic() - start, 3)
    print(f"{profile}/{script}: exit={result.returncode}, {elapsed}s", flush=True)
    return dict(profile=profile, script=script, B_BOOT=bootstrap, seed=42,
                exit_code=result.returncode, seconds=elapsed,
                log=str(log.relative_to(HERE)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    paths = [SOURCE / x for x in SCRIPTS] + [SOURCE / INPUT]
    before = {str(p.relative_to(REPO)): sha256(p) for p in paths}
    started = datetime.now(timezone.utc).isoformat()
    jobs = []
    for profile, (bootstrap, scripts) in PROFILES.items():
        folder = HERE / "runs" / profile
        (folder / INPUT.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / INPUT, folder / INPUT)
        for script in scripts:
            shutil.copy2(SOURCE / script, folder / script)
            assert sha256(folder / script) == sha256(SOURCE / script)
            jobs.append((profile, script, bootstrap))
        assert sha256(folder / INPUT) == sha256(SOURCE / INPUT)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_script, *job) for job in jobs]
        results = [future.result() for future in as_completed(futures)]
    after = {str(p.relative_to(REPO)): sha256(p) for p in paths}
    if before != after:
        raise RuntimeError("Original source/input changed during reproduction")
    manifest = dict(
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, platform=platform.platform(),
        packages={name: version(name) for name in
                  ["numpy", "pandas", "scipy", "statsmodels", "matplotlib"]},
        source_sha256=before, original_sources_unchanged=True,
        scope="Preserved processed 715-row input; upstream raw-data ETL is not rerun.",
        runs=sorted(results, key=lambda x: (x["profile"], x["script"])),
        outputs_sha256={str(p.relative_to(HERE)): sha256(p)
                        for p in sorted((HERE / "runs").glob("*/paper_outputs/*"))
                        if p.is_file()},
    )
    (HERE / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if any(r["exit_code"] != 0 for r in results):
        raise RuntimeError("An original script failed; inspect the recorded logs")
    if not args.run_only:
        subprocess.run([sys.executable, str(HERE / "compare_manuscript.py")], check=True)


if __name__ == "__main__":
    main()
