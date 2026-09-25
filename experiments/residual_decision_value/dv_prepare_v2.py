"""Pre-score amendment: compare only archived outer-evaluation target rows."""
import json
import subprocess
from pathlib import Path
import dv_prepare as original
from dv_core import HERE,ROOT,OUT,PARENT,r12,sha,now,write_new,verify_lock


def verify_amendment():
    verify_lock()
    path=HERE/'PREPARATION_AMENDMENT.json'
    lock=json.loads(path.read_text())
    for name,value in lock['file_sha256'].items():assert sha(ROOT/name)==value,name
    rel=str(path.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==path.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/research/icaif2026-manuscript-development'],cwd=ROOT,check=True)
    return dict(amendment_commit=commit,amendment_sha256=sha(path))


def main():
    stamp=verify_amendment()
    reader=original.read
    def outer_target_reader(path):
        d=reader(path)
        if Path(path)==PARENT/'results/targets.csv.gz':
            d=d[d.origin>=r12.hs.START].reset_index(drop=True)
        return d
    original.read=outer_target_reader
    original.main()
    write_new(OUT/'PREPARATION_AMENDMENT_APPLIED.json',dict(**stamp,applied_utc=now(),
        inputs_seal_sha256=sha(OUT/'INPUTS_SEALED.json'),forecast_and_sample_rules_unchanged=True))


if __name__=='__main__':main()
