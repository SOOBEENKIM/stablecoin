"""Immutable execution lock, in addition to the existing immutable design lock."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN_DIR = HERE.parent / 'residual_factorial_design'
OUT = HERE / 'results'
LOCK = HERE / 'LOCK.json'
BRANCH = 'research/icaif2026-manuscript-development'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def environment():
    return dict(python=sys.version.split()[0], **{n: importlib.import_module(n).__version__
                for n in ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib']})


def check_hashes(entries):
    for name, digest in entries.items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise RuntimeError('Locked file changed: ' + name)


def design_hashes():
    d = json.loads((DESIGN_DIR / 'DESIGN_LOCK.json').read_text())
    hashes = {}
    for group in ['design_and_preflight_sha256', 'existing_dependency_sha256',
                  'input_sha256', 'archived_origin_source_sha256']:
        check_hashes(d[group])
        hashes.update(d[group])
    hashes[str((DESIGN_DIR / 'DESIGN_LOCK.json').relative_to(ROOT))] = sha(DESIGN_DIR / 'DESIGN_LOCK.json')
    if environment() != d['environment']:
        raise RuntimeError('Environment differs from design preflight')
    return hashes


def create():
    hashes = design_hashes()
    for p in sorted(HERE.glob('*.py')) + [HERE / 'EXECUTION_KO.md', HERE / 'PRE_RUN_TESTS.txt']:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    write_new(LOCK, dict(created_utc=now(), environment=environment(),
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        file_sha256=hashes, previously_seen_evaluation_period=True, independent_confirmation=False))
    print('Execution lock created; commit/push before historical forecasts.')


def verify_lock():
    d = json.loads(LOCK.read_text())
    check_hashes(d['file_sha256'])
    if environment() != d['environment']:
        raise RuntimeError('Execution environment changed')
    rel = str(LOCK.relative_to(ROOT))
    if subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT) != LOCK.read_bytes():
        raise RuntimeError('Execution lock is not committed')
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', rel], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'refs/remotes/origin/' + BRANCH], cwd=ROOT, check=True)
    return dict(lock_commit=commit, lock_sha256=sha(LOCK))


def verify_seal():
    stamp = verify_lock()
    seal = json.loads((OUT / 'PREDICTIONS_SEALED.json').read_text())
    if seal['lock_sha256'] != stamp['lock_sha256']:
        raise RuntimeError('Forecast seal belongs to another lock')
    check_hashes(seal['file_sha256'])
    return stamp, seal


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['create', 'verify'])
    args = p.parse_args()
    if args.action == 'create':
        create()
    else:
        print(json.dumps(verify_lock(), indent=2))
