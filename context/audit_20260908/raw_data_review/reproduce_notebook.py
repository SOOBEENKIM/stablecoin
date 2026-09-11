"""Replay reviewed preprocessing cells in an isolated copy, preserving the old model."""
from pathlib import Path
import json, os, shutil, contextlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'stablecoin_v4/data/raw data'
RUN=HERE/'reproduction'
INPUTS=['USDKRW.csv','VIXY.csv','SPY US.csv','DXY.csv','XAUUSD.csv',
        'binance_1h_2025-06-01_2026-03-19.csv','upbit_1h_2025-06-01_2026-03-19.csv']
CELLS=list(range(2,12))+[26,27,35,50,51,52,54,55,56,58,59,61,64,66,67,73,74,77,227]

def compare(a,b,name):
    if list(a.columns)!=list(b.columns) or a.shape!=b.shape:
        return {'file':name,'shape_generated':list(a.shape),'shape_saved':list(b.shape),'same_schema':False}
    rows=[]
    for c in a:
        x,y=a[c],b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            ok=np.isclose(x.to_numpy(dtype=float),y.to_numpy(dtype=float),rtol=1e-9,atol=1e-12,equal_nan=True)
            err=(x-y).abs().max()
            rows.append({'file':name,'column':c,'matches':bool(ok.all()),'mismatches':int((~ok).sum()),'max_abs_error':float(err) if pd.notna(err) else None})
        else:
            ok=x.fillna('').astype(str).equals(y.fillna('').astype(str))
            rows.append({'file':name,'column':c,'matches':ok,'mismatches':0 if ok else None})
    return rows

def main():
    RUN.mkdir(exist_ok=True)
    for n in INPUTS:shutil.copy2(SOURCE/n,RUN/n)
    nb=json.loads((ROOT/'stablecoin_v4/FinxLab_stablecoin_20260409_datapreprocessing - 복사본.ipynb').read_text())
    scope={'pd':pd,'np':np}
    os.chdir(RUN)
    with (HERE/'reproduction.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        for i in CELLS:
            cell=nb['cells'][i]
            if cell['cell_type']!='code':raise ValueError('Unexpected cell type')
            print('REPLAY CELL',i,flush=True)
            exec(compile(''.join(cell['source']),'notebook_cell_'+str(i),'exec'),scope)
    comparisons=[]
    for p in sorted(RUN.glob('*.csv')):
        if p.name in INPUTS or not (SOURCE/p.name).exists():continue
        a,b=pd.read_csv(p),pd.read_csv(SOURCE/p.name)
        result=compare(a,b,p.name)
        comparisons.extend(result if isinstance(result,list) else [result])
    pd.DataFrame(comparisons).to_csv(HERE/'preprocessing_comparison.csv',index=False)
    generated=pd.read_csv(RUN/'baseline_dataset_with_residual_final.csv')
    archived=pd.read_csv(ROOT/'stablecoin_v4/corrected_outputs/baseline_dataset_with_residual_final.csv')
    common=[c for c in generated if c in archived]
    result=compare(generated[common],archived[common],'v4_baseline_common_columns')
    pd.DataFrame(result).to_csv(HERE/'baseline_comparison.csv',index=False)
    summary={'cells_replayed':CELLS,'input_files':INPUTS,'generated_baseline_shape':list(generated.shape),
             'v4_baseline_shape':list(archived.shape),'common_columns':len(common),
             'baseline_mismatches':[r for r in result if not r.get('matches',False)],
             'preprocessing_mismatches':[r for r in comparisons if not r.get('matches',False)],
             'note':'Reproduces the archived definitions, including known quote/time errors; not a corrected final analysis.'}
    (HERE/'reproduction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
