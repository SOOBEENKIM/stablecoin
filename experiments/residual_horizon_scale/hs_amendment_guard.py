import json
import subprocess
from hs_core import HERE, ROOT, OUT, sha, now, write_new
from hs_guard import verify, BRANCH

FILE=HERE/'VERIFIER_AMENDMENT_LOCK.json'


def create():
    verify()
    assert not (OUT/'SCORES_OPENED.json').exists()
    names=['hs_verify_v2.py','selection_check.py','test_selection_check.py','hs_amendment_guard.py',
           'VERIFIER_AMENDMENT_KO.md','VERIFIER_TESTS.txt']
    write_new(FILE,dict(created_utc=now(),scores_not_opened=True,
        predictions_sha256=sha(OUT/'predictions.csv.gz'),
        seal_sha256=sha(OUT/'PREDICTIONS_SEALED.json'),
        file_sha256={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in names}))


def verify_amendment():
    d=json.loads(FILE.read_text())
    for p,s in d['file_sha256'].items():assert sha(ROOT/p)==s,p
    assert sha(OUT/'predictions.csv.gz')==d['predictions_sha256']
    assert sha(OUT/'PREDICTIONS_SEALED.json')==d['seal_sha256']
    rel=str(FILE.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==FILE.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True)
    return dict(commit=commit,amendment_sha256=sha(FILE))


if __name__=='__main__':
    import sys
    create() if sys.argv[1]=='create' else print(verify_amendment())
