"""Read-only session-pattern evidence; inference is not export metadata."""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'data/raw data'
rows, summaries = [], {}
for name in ['USDKRW.csv','VIXY.csv','DXY.csv','SPY US.csv','XAUUSD.csv']:
    raw = pd.read_csv(RAW/name, header=None)
    assert raw.shape[1] in [2,5]
    t = pd.to_datetime(raw.iloc[:,0], errors='coerce')
    values = raw.iloc[:,1:].apply(pd.to_numeric, errors='coerce')
    good = t.notna() & values.notna().all(axis=1)
    d = pd.DataFrame({'timestamp':t[good], 'value':values.loc[good].iloc[:,-1]})
    d['flat_ohlc'] = values.loc[good].nunique(axis=1).eq(1) if raw.shape[1]==5 else False
    d = d.sort_values('timestamp')
    d['day'] = d.timestamp.dt.strftime('%Y-%m-%d')
    d['hm'] = d.timestamp.dt.strftime('%H:%M')
    daily = d.groupby('day').agg(first=('hm','first'),last=('hm','last'),n=('value','size'),flat_ohlc=('flat_ohlc','sum'))
    for day, row in daily.iterrows():
        rows.append({'file':name,'date':day,**row.to_dict()})
    summaries[name] = {'valid_rows':len(d),'days':len(daily),
        'time_counts':d.hm.value_counts().sort_index().to_dict(),
        'weekday_counts':d.timestamp.dt.day_name().value_counts().to_dict(),
        'flat_ohlc_by_time':d.groupby('hm').flat_ohlc.sum().to_dict()}
pd.DataFrame(rows).to_csv(HERE/'daily_session_patterns.csv',index=False)
(HERE/'session_time_evidence.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
print('Saved session patterns for five macro files; original timestamps unchanged.')
