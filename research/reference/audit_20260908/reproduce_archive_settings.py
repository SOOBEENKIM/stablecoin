"""Test whether undocumented bootstrap counts explain archived CSV differences."""
from pathlib import Path
import shutil, subprocess, os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'stablecoin_v4'
cases=[('archival_B40',40,['stablecoin_paper_pipeline.py']),
       ('supplement_B200',200,['kp_robustness.py','experiment1_persistence_mediation.py','experiment2_subsample_stability.py'])]
rows=[]
for folder,B,scripts in cases:
    dest=ROOT/folder;(dest/'corrected_outputs').mkdir(parents=True,exist_ok=True)
    shutil.copy2(SRC/'corrected_outputs/baseline_dataset_with_residual_final.csv',dest/'corrected_outputs/baseline_dataset_with_residual_final.csv')
    for name in scripts:
        shutil.copy2(SRC/name,dest/name)
        env=dict(os.environ,B_BOOT=str(B),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/stablecoin_audit_mpl')
        with (dest/(name+'.log')).open('w') as log:
            subprocess.run(['python3',str(dest/name)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for f in (dest/'paper_outputs').glob('*.csv'):
        archived=pd.read_csv(SRC/'paper_outputs'/f.name);actual=pd.read_csv(f)
        if list(archived.columns)!=list(actual.columns) or archived.shape!=actual.shape:
            rows.append(dict(config=folder,file=f.name,matching_cells=0,total_cells=archived.size,reason='schema or shape mismatch'));continue
        good=0;bad=[]
        for c in archived:
            if pd.api.types.is_numeric_dtype(archived[c]) and pd.api.types.is_numeric_dtype(actual[c]):
                ok=np.isclose(archived[c],actual[c],atol=1e-8,rtol=1e-8,equal_nan=True)
            else:ok=(archived[c].fillna('').astype(str)==actual[c].fillna('').astype(str)).values
            good+=int(ok.sum())
            for i in np.flatnonzero(~ok):bad.append(f'{c}[{i}]: {archived[c].iloc[i]} -> {actual[c].iloc[i]}')
        rows.append(dict(config=folder,file=f.name,matching_cells=good,total_cells=archived.size,reason='; '.join(bad[:12])))
    pd.DataFrame(rows).to_csv(ROOT/'archive_settings_comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
