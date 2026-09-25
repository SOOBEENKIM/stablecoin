import json
import subprocess
from hs_core import HERE, ROOT, OLD, sha, now, write_new, verify_old_lock

LOCK = HERE / 'LOCK.json'
BRANCH = 'research/icaif2026-manuscript-development'


def create():
    verify_old_lock()
    paths = list(HERE.glob('*.py')) + [HERE/'PROTOCOL_KO.md', HERE/'PRE_RUN_TESTS.txt', OLD/'LOCK.json']
    paths += list((OLD/'results/candidates').glob('available_macro__EQ__*.csv.gz'))
    paths += [OLD/'results/predictions.csv.gz', OLD/'results/selections.json']
    write_new(LOCK, dict(created_utc=now(), previously_seen_period=True, independent_confirmation=False,
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))


def verify():
    verify_old_lock()
    d = json.loads(LOCK.read_text())
    for name, digest in d['file_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Changed locked file: '+name)
    rel = str(LOCK.relative_to(ROOT))
    if subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT) != LOCK.read_bytes():
        raise ValueError('Uncommitted lock')
    commit = subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True)
    return dict(lock_commit=commit,lock_sha256=sha(LOCK))


if __name__ == '__main__':
    import sys
    create() if sys.argv[1] == 'create' else print(verify())
