import json
import subprocess
import sys
from r12_core import HERE,ROOT,OUT,PARENT,sha,now,write_new,verify_parent,verify_amendment

LOCK=HERE/'LOCK.json'
BRANCH='research/icaif2026-manuscript-development'


def parent_check():
    verify_parent();verify_amendment()
    for name,s in json.loads((PARENT/'SEED_LOCK.json').read_text())['file_sha256'].items():assert sha(ROOT/name)==s,name


def create():
    parent_check()
    files=list(HERE.glob('*.py'))+[HERE/'PROTOCOL_KO.md',HERE/'PRE_RUN_TESTS.txt']
    files+=list((PARENT/'results/candidates').glob('12__*.csv.gz'))
    files+=[PARENT/n for n in ['SEED_LOCK.json','hs_seed_check.py','results/predictions.csv.gz','results/targets.csv.gz',
        'results/selections.json','results/seed_sensitivity/candidate_streams.csv.gz','results/seed_sensitivity/four_seed_mean.csv']]
    write_new(LOCK,dict(created_utc=now(),existing_data_only=True,independent_confirmation=False,
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))


def verify():
    parent_check()
    lock=json.loads(LOCK.read_text())
    for n,s in lock['file_sha256'].items():assert sha(ROOT/n)==s,n
    rel=str(LOCK.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==LOCK.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True)
    return dict(lock_commit=commit,lock_sha256=sha(LOCK))


def verify_seal():
    stamp=verify()
    seal=json.loads((OUT/'PREDICTIONS_SEALED.json').read_text())
    assert seal['lock_sha256']==stamp['lock_sha256']
    for n,s in seal['file_sha256'].items():assert sha(ROOT/n)==s,n
    return stamp,seal


if __name__=='__main__':create() if sys.argv[1]=='create' else print(verify())
