"""Hash lock and immutable score-opening marker for a previously seen sample."""
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
LOCK = HERE/'LOCK.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return dict(python=sys.version.split()[0], **{n:importlib.import_module(n).__version__
        for n in ['numpy','pandas','scipy','sklearn','matplotlib']})


def write_new(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(content,f,ensure_ascii=False,indent=2);f.write('\n')


def check_hashes(entries, root=ROOT):
    for name,digest in entries.items():
        p=root/name
        if not p.is_file() or sha(p)!=digest:
            raise RuntimeError('Locked file changed: '+name)


def verify_lock():
    lock=json.loads(LOCK.read_text())
    check_hashes(lock['source_sha256']);check_hashes(lock['input_sha256'])
    if environment()!=lock['environment']:
        raise RuntimeError('Package environment differs from lock')
    rel=str(LOCK.relative_to(ROOT))
    if subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)!=LOCK.read_bytes():
        raise RuntimeError('Lock must be committed before execution')
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    for name,digest in lock['source_sha256'].items():
        content=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=digest:
            raise RuntimeError('Source is not committed: '+name)
    subprocess.run(['git','merge-base','--is-ancestor',commit,
        'refs/remotes/origin/research/icaif2026-manuscript-development'],cwd=ROOT,check=True)
    return dict(lock_commit=commit,lock_sha256=sha(LOCK))


def create_lock():
    sys.path.insert(0,str(HERE.parent/'residual_dynamics_ml'))
    from data import INPUT_PATHS
    files=list(HERE.glob('*.py'))+[HERE/'design.json',HERE/'PROTOCOL_KO.md']
    for folder in ['residual_dynamics_ml','residual_dynamics_adaptive','residual_economic_validation']:
        files.extend((HERE.parent/folder).glob('*.py'))
    write_new(LOCK,dict(created_utc=now(),environment=environment(),
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)},
        input_sha256={str(f.relative_to(ROOT)):sha(f) for f in INPUT_PATHS},
        design_uses_previously_seen_period=True,independent_confirmation=False))
    print('Lock created. Commit and push before fitting/evaluating.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','verify']);args=p.parse_args()
    if args.action=='create':create_lock()
    else:print(json.dumps(verify_lock(),indent=2))
