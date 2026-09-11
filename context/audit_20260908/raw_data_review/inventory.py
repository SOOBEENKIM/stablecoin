"""Inventory candidate research inputs without changing source files."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, re, zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'stablecoin_v4/data/raw data'
RAW={'USDKRW.csv':['datetime','price'],'VIXY.csv':['datetime','price'],
     'DXY.csv':['datetime','price'],'SPY US.csv':['datetime','open','high','low','close'],
     'XAUUSD.csv':['datetime','open','high','low','close']}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    return pd.read_csv(path,header=None,names=RAW[path.name]) if path.name in RAW else pd.read_csv(path,low_memory=False)

def main():
    files=sorted(p for p in SOURCE.iterdir() if p.is_file())
    manifest={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    inventory=[]
    for p in files:
        row={'file':p.name,'extension':p.suffix,**manifest[p.name]}
        try:
            if p.suffix=='.csv':
                d=read_csv(p)
                row.update(rows=len(d),columns=list(d.columns),column_count=len(d.columns),
                           missing_cells=int(d.isna().sum().sum()),duplicate_rows=int(d.duplicated().sum()))
                numeric=d.select_dtypes(include=np.number).to_numpy()
                row['infinite_cells']=int(np.isinf(numeric).sum())
                candidates=[c for c in d if re.search(r'^(datetime(?:_utc)?|date|snapped_at|time|open_time|candle_date_time_utc|timestamp|Unnamed: 0)$',str(c),re.I)]
                for c in candidates:
                    s=d[c]
                    if pd.api.types.is_numeric_dtype(s):
                        if len(s) and s.median()>1e11:
                            ts=pd.to_datetime(s,unit='ms',utc=True,errors='coerce')
                        else:continue
                    else:
                        ts=pd.to_datetime(s,format='mixed',utc=True,errors='coerce')
                    if ts.notna().mean()<.9:continue
                    valid=ts.dropna()
                    delta=valid.sort_values().diff().dt.total_seconds()/3600
                    row.update(time_column=c,start=str(valid.min()),end=str(valid.max()),
                               timezone_explicit=bool(s.astype(str).str.contains(r'(?:[+-]\d\d:\d\d|Z| UTC)$',regex=True).mean()>.9 or pd.api.types.is_numeric_dtype(s)),
                               invalid_timestamps=int(ts.isna().sum()),duplicate_timestamps=int(valid.duplicated().sum()),
                               intervals_hours={str(k):int(v) for k,v in delta.value_counts().head(5).items()},
                               rows_in_paper_period=int(valid.between(pd.Timestamp('2025-06-01 18:00Z'),pd.Timestamp('2026-03-19 16:00Z')).sum()))
                    break
                if len(d):row['sample']=[{str(k):str(v) for k,v in record.items()} for record in d.iloc[:2,:min(7,len(d.columns))].to_dict('records')]
            elif p.suffix=='.xlsx':
                with zipfile.ZipFile(p) as z:
                    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                    strings=[]
                    if 'xl/sharedStrings.xml' in z.namelist():
                        strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
                    sh=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
                    rows=sh.findall('s:sheetData/s:row',ns)
                    headers=[]
                    for c in rows[0]:
                        v=c.find('s:v',ns);value=v.text if v is not None else ''.join(c.itertext())
                        headers.append(strings[int(value)] if c.get('t')=='s' else value)
                    row.update(rows_including_header=len(rows),columns=headers)
            elif p.suffix=='.parquet':
                row['status']='retained; value inspection pending parquet reader'
            elif p.suffix=='.ipynb':
                nb=json.loads(p.read_text())
                row['cells']=len(nb['cells'])
                text='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
                row['csv_references']=sorted(set(re.findall(r'[\w .()-]+\.csv',text)))
        except Exception as exc:
            row['read_error']=str(exc)
        inventory.append(row)
    (HERE/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['columns','sample','csv_references','intervals_hours']} for r in inventory]).to_csv(HERE/'inventory.csv',index=False)
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    grouped=defaultdict(list)
    for name,item in manifest.items():grouped[item['sha256']].append(name)
    duplicates=[v for v in grouped.values() if len(v)>1]
    (HERE/'exact_duplicates.json').write_text(json.dumps(duplicates,ensure_ascii=False,indent=2))
    summary={'files':len(files),'extensions':dict(Counter(p.suffix for p in files)),
             'read_errors':[{'file':r['file'],'error':r['read_error']} for r in inventory if 'read_error' in r],
             'duplicate_groups':duplicates,'source_changed':[n for n,m in manifest.items() if sha(SOURCE/n)!=m['sha256']]}
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
