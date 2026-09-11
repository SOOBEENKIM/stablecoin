"""Descriptive event availability, not a fitted predictor or a novelty test.

All thresholds use June-October 2025 only. Full clock grid is preserved.
No choice among definitions is made by forecast performance.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'extension_20260909/basis_definitions.csv'
COINS = ['BTC','ETH','XRP','SOL','DOGE']
TRAIN_END = pd.Timestamp('2025-11-01', tz='UTC')
TEST_START = pd.Timestamp('2026-01-01', tz='UTC')


def panel():
    z = pd.read_csv(SOURCE,index_col=0,parse_dates=True)
    z.index = pd.to_datetime(z.index,utc=True)
    grid = pd.date_range(z.index.min(),z.index.max(),freq='h')
    return z.reindex(grid)


def labels(z):
    thresholds = z.loc[z.index<TRAIN_END,COINS+['mean5']].quantile(.1)
    valid = z[COINS].notna().all(axis=1)
    low = z[COINS].lt(thresholds[COINS])
    definitions = {
        'btc_eth': low[['BTC','ETH']].all(axis=1),
        'three_of_five': low.sum(axis=1)>=3,
        'three_of_four_no_doge': low[['BTC','ETH','XRP','SOL']].sum(axis=1)>=3,
    }
    out = pd.DataFrame(index=z.index)
    for name,v in definitions.items():
        state = v.astype(float).where(valid)
        future = pd.concat([state.shift(-h) for h in range(1,7)],axis=1)
        complete = future.notna().all(axis=1)
        # At least three CONSECUTIVE hourly close observations in next six hours.
        runs = [(future.iloc[:,j:j+3].sum(axis=1)==3) for j in range(4)]
        out[name+'_state'] = state
        out[name+'_sustain6'] = pd.concat(runs,axis=1).any(axis=1).astype(float).where(complete)
        out[name+'_count6'] = future.sum(axis=1).where(complete)
    return out, thresholds


def episode_starts(state, washout=6):
    # A fresh episode requires six known, non-event hourly observations.
    prior = state.shift().rolling(washout,min_periods=washout)
    return state.eq(1)&prior.sum().eq(0)&prior.count().eq(washout)


def main():
    z = panel()
    ys,thresholds = labels(z)
    mean_tail = z.mean5.lt(thresholds.mean5).astype(float).where(z.mean5.notna())
    triggers = episode_starts(mean_tail)
    periods = [('train_Jun_Oct',z.index<TRAIN_END),
               ('validation_Nov_Dec',(z.index>=TRAIN_END)&(z.index<TEST_START)),
               ('evaluation_Jan_Mar',z.index>=TEST_START)]
    rows=[]
    for name in ['btc_eth','three_of_five','three_of_four_no_doge']:
        st = ys[name+'_state']; future = ys[name+'_sustain6']
        starts = episode_starts(st)
        for period,mask in periods:
            p=ys.loc[mask]
            eligible = mask & future.notna()
            fresh = eligible & triggers
            rows.append(dict(period=period,definition=name,
                observed_hours=int(p[name+'_state'].notna().sum()),
                current_consensus_hours=int(st.loc[mask].eq(1).sum()),
                current_consensus_days=int(st.loc[mask&st.eq(1)].index.normalize().nunique()),
                separated_consensus_onsets=int(starts.loc[mask].sum()),
                eligible_origins=int(eligible.sum()),
                positive_future_sustain_origins=int(future.loc[eligible].sum()),
                positive_future_days=int(future.loc[eligible&future.eq(1)].index.normalize().nunique()),
                fresh_mean_alarm_origins=int(fresh.sum()),
                fresh_mean_alarms_followed_by_sustain=int(future.loc[fresh].sum()),
                fresh_mean_alarm_followthrough=float(future.loc[fresh].mean())))
    pd.DataFrame(rows).to_csv(HERE/'event_feasibility.csv',index=False)
    ys.to_csv(HERE/'event_labels.csv.gz',compression='gzip')
    thresholds.to_csv(HERE/'training_thresholds.csv',header=['q10_train_bp'])
    (HERE/'FEASIBILITY_METADATA.json').write_text(json.dumps(dict(
        source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        clock_start=str(z.index.min()),clock_end=str(z.index.max()),grid_hours=len(z),
        valid_price_hours=int(z[COINS].notna().all(axis=1).sum()),
        last_label_time=str(ys.index[ys.btc_eth_sustain6.notna()][-1]),
        note='Descriptive feasibility only. Overlapping positive origins and separated onsets are not independent observations. No performance-based target selection.'
    ),indent=2))
    print(thresholds.round(3).to_string())
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__=='__main__':
    main()
