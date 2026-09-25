"""Recover missing archived current-state metadata by an exact validated join."""
import json
import subprocess
import numpy as np
import pandas as pd
from dv_core import HERE,ROOT,OUT,PARENT,read,sha,now,write_new
from dv_verify_v2 import verify_verifier_amendment


def aligned_read(path):
    from pathlib import Path
    d=read(path)
    if Path(path) not in [OUT/'archived_forecasts.csv.gz',PARENT/'results/predictions.csv.gz']:return d
    ctx=read(OUT/'context.csv.gz')
    keys=['definition','origin','fold']
    assert not ctx.duplicated(keys).any()
    lookup=ctx.set_index(keys).reindex(pd.MultiIndex.from_frame(d[keys]))
    assert np.isfinite(lookup.e_now).all()
    np.testing.assert_allclose(d.target,lookup.target,atol=1e-11,rtol=0)
    missing=d.e_now.isna().to_numpy()
    np.testing.assert_allclose(d.e_now.to_numpy()[~missing],lookup.e_now.to_numpy()[~missing],atol=1e-11,rtol=0)
    d.loc[missing,'e_now']=lookup.e_now.to_numpy()[missing]
    assert np.isfinite(d.e_now).all()
    return d


def verify_alignment():
    previous=verify_verifier_amendment()
    path=HERE/'ALIGNMENT_AMENDMENT.json';lock=json.loads(path.read_text())
    for name,value in lock['file_sha256'].items():assert sha(ROOT/name)==value,name
    rel=str(path.relative_to(ROOT))
    assert subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)==path.read_bytes()
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/research/icaif2026-manuscript-development'],cwd=ROOT,check=True)
    return dict(**previous,alignment_commit=commit,alignment_sha256=sha(path))
