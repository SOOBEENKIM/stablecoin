"""Exploratory option-1 development diagnostics; no causal or AI claims.

The one-tick probe and event counts were inspected before this script was saved.
Historical options remain frozen. Quotes are moved in deterministic scenarios;
scenario exits are not identified false alarms or realizable trades.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
COINS=['BTC','ETH','XRP','SOL','DOGE']
TEST=pd.Timestamp('2026-01-01',tz='UTC')
TICKS={'BTC':1000.,'ETH':1000.,'XRP':1.,'SOL':100.,'DOGE':1.,'USDT':1.}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot():
    return {str(p.relative_to(ROOT)):sha(p)
            for name in ['extension_20260909','workshop2_20260909','workshop3_20260909']
            for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def read(name):
    d=pd.read_csv(ROOT/'data/raw data'/name,index_col=0)
    d.index=pd.to_datetime(d.index,utc=True)+pd.Timedelta(hours=1)
    assert d.index.is_unique and d.index.is_monotonic_increasing
    return d


def main():
    before=snapshot()
    inputs=[ROOT/'data/raw data'/f'{venue}_1h_2025-06-01_2026-03-19.csv'
            for venue in ['binance','upbit']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    d=read(inputs[0].name).join(read(inputs[1].name),how='outer')
    grid=pd.date_range(d.index.min(),d.index.max(),freq='h',tz='UTC')
    d=d.reindex(grid)
    log_r=pd.DataFrame({c:np.log(d[c+'_UPBIT_CLOSE']/d[c+'_BINANCE_CLOSE']) for c in COINS})
    b=log_r.rsub(np.log(d.USDT_UPBIT_CLOSE),axis=0)*1e4
    b['mean5']=b[COINS].mean(axis=1,skipna=False)
    b['median5']=b[COINS].median(axis=1,skipna=False)
    b['without_DOGE']=b[COINS[:-1]].mean(axis=1,skipna=False)
    old=pd.read_csv(ROOT/'extension_20260909/basis_definitions.csv',index_col=0)
    old.index=pd.to_datetime(old.index,utc=True)
    np.testing.assert_allclose(b.reindex(old.index)[old.columns.intersection(b.columns)],
        old[old.columns.intersection(b.columns)],atol=1e-7,rtol=1e-8,equal_nan=True)
    pre=b.loc[b.index<TEST]
    z=b.loc[(b.index>=TEST)&b[COINS].notna().all(axis=1)].copy()
    prices=d.reindex(z.index)
    # Verified post-reform price bands for this specific 2026 sample.
    bands={'BTC':(2e6,np.inf),'ETH':(2e6,np.inf),'XRP':(1000,5000),
           'SOL':(100000,500000),'DOGE':(100,1000),'USDT':(1000,5000)}
    for coin,(lo,hi) in bands.items():
        assert prices[coin+'_UPBIT_CLOSE'].between(lo,hi,inclusive='left').all()
    specs=[('q05',float(pre.mean5.quantile(.05)),'lower'),
           ('q10',float(pre.mean5.quantile(.1)),'lower'),
           ('minus5',-5.,'lower'),('minus10',-10.,'lower'),('minus20',-20.,'lower'),
           ('q90',float(pre.mean5.quantile(.9)),'upper'),
           ('q95',float(pre.mean5.quantile(.95)),'upper')]
    records=[];panel=z.copy()
    for name,cut,side in specs:
        event=z.mean5.lt(cut) if side=='lower' else z.mean5.gt(cut)
        for coin,tick in TICKS.items():
            k=prices[coin+'_UPBIT_CLOSE']
            # Favourable direction for EXITING the selected tail, others fixed.
            step_sign=1 if (coin=='USDT')==(side=='lower') else -1
            step=step_sign*tick
            movement=(10000 if coin=='USDT' else -2000)*np.log1p(step/k)
            counter=z.mean5+movement
            exits=event&(counter.ge(cut) if side=='lower' else counter.le(cut))
            # Independently re-evaluate the raw price formula for every scenario.
            changed_log_r=log_r.reindex(z.index).copy()
            changed_u=prices.USDT_UPBIT_CLOSE.copy()
            if coin=='USDT':
                changed_u+=step
            else:
                changed_log_r[coin]=np.log((k+step)/prices[coin+'_BINANCE_CLOSE'])
            direct=(np.log(changed_u)-changed_log_r.mean(axis=1,skipna=False))*1e4
            np.testing.assert_allclose(counter,direct,atol=1e-7,rtol=1e-8)
            periods=[('all',np.ones(len(z),dtype=bool))]
            periods += [(month,z.index.strftime('%Y-%m')==month)
                        for month in sorted(set(z.index.strftime('%Y-%m')))]
            for month,mask in periods:
                n=int(event[mask].sum())
                records.append(dict(threshold=name,threshold_bp=cut,side=side,coin=coin,
                    period=month,n_prices=int(mask.sum()),tail_points=n,
                    scenario_exits=int(exits[mask].sum()),
                    scenario_exit_share=float(exits[mask].sum()/n) if n else None,
                    median_scenario_movement_bp=float(abs(movement[mask]).median())))
            if name=='q10':
                panel['q10_exit_'+coin]=exits
                panel['q10_movement_'+coin]=movement
    pd.DataFrame(records).to_csv(HERE/'one_tick_scenarios.csv',index=False)
    panel.to_csv(HERE/'measurement_screening_panel.csv.gz',compression='gzip')

    # USDT cancels from each comparison: a_i = b_BTC - b_i.
    a=(log_r.sub(log_r.BTC,axis=0))*1e4
    np.testing.assert_allclose(a[COINS],b[COINS].rsub(b.BTC,axis=0),
                               atol=1e-7,rtol=1e-8,equal_nan=True)
    policy=[]
    pre_end=pd.Timestamp('2025-07-30',tz='UTC')
    post_start=pd.Timestamp('2025-08-02',tz='UTC')
    for window in [14,28,42]:
        for period,left,right in [('before',pre_end-pd.Timedelta(days=window),pre_end),
                                  ('after',post_start,post_start+pd.Timedelta(days=window))]:
            keep=(a.index>=left)&(a.index<right)&a[COINS].notna().all(axis=1)
            for coin in COINS[1:]:
                v=a.loc[keep,coin]
                policy.append(dict(window_days=window,period=period,coin=coin,n=len(v),
                    days=int(v.index.normalize().nunique()),mean_bp=float(v.mean()),
                    mean_abs_bp=float(v.abs().mean()),std_bp=float(v.std()),
                    q90_abs_bp=float(v.abs().quantile(.9))))
    pd.DataFrame(policy).to_csv(HERE/'policy_window_descriptive.csv',index=False)

    # Virtual regridding of the SAME pre-policy observed prices. This is a
    # measurement transformation, not simulated market behaviour after reform.
    regrid=[]
    for window in [14,28,42]:
        keep=(d.index>=pre_end-pd.Timedelta(days=window))&(d.index<pre_end)
        v=d.loc[keep].dropna()
        btc_ratio=v.BTC_UPBIT_CLOSE/v.BTC_BINANCE_CLOSE
        for coin in COINS[1:]:
            k=v[coin+'_UPBIT_CLOSE'];tick=TICKS[coin]
            lo,hi=bands[coin]
            assert k.between(lo,hi,inclusive='left').all()
            options={'observed':k,
                'nearest_half_up':np.floor(k/tick+.5+1e-9)*tick,
                'floor':np.floor(k/tick+1e-9)*tick,
                'ceiling':np.ceil(k/tick-1e-9)*tick}
            for rule,changed in options.items():
                gap=10000*np.log((changed/v[coin+'_BINANCE_CLOSE'])/btc_ratio)
                regrid.append(dict(window_days=window,coin=coin,rule=rule,n=len(v),
                    mean_abs_bp=float(gap.abs().mean()),std_bp=float(gap.std()),
                    mean_bp=float(gap.mean())))
    pd.DataFrame(regrid).to_csv(HERE/'virtual_regridding.csv',index=False)

    # Check sparse-event feasibility, not outcome differences between groups.
    refs=['mean5','BTC','ETH','XRP']
    cuts=pre[refs].quantile(.1)
    st=(b[refs]<cuts).astype(float).where(b[refs].notna())
    eligible=st[['mean5','BTC','ETH','XRP']].notna().all(axis=1)
    for h in range(1,7):
        eligible &= st.XRP.shift(-h).notna()
    onset=st.mean5.eq(1)&st.mean5.shift().rolling(6,min_periods=6).sum().eq(0)
    eligible &= (b.index>=TEST)&onset
    counts=[]
    for group,condition in [('both_BTC_ETH',st.BTC.eq(1)&st.ETH.eq(1)),
                            ('neither_BTC_ETH',st.BTC.eq(0)&st.ETH.eq(0)),
                            ('one_BTC_ETH',st.BTC.add(st.ETH).eq(1))]:
        ix=b.index[eligible&condition]
        counts.append(dict(group=group,onsets=len(ix),days=int(ix.normalize().nunique())))
    pd.DataFrame(counts).to_csv(HERE/'event_group_feasibility.csv',index=False)
    assert snapshot()==before
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    manifest=dict(status='exploratory descriptive screening; no new predictive model fitted',
        probe_inspected_before_script_saved=True,raw_sha256=hashes,
        preserved_artifact_count=len(before),preserved_options_unchanged=True,
        n_2026_complete_prices=len(z),q10_threshold_bp=float(cuts.mean5),
        tests=['raw-price and frozen-basis agreement','one-tick scenarios recalculated from raw prices',
               'USDT-free spread identity','post-policy price bands','old options and raw prices unchanged'],
        scenario_scope='single local quote changed by one tick; all other prices fixed; NOT false alarms or causal effects',
        policy_scope='14/28/42-day descriptive windows excluding transition; NOT a causal policy estimate',
        virtual_regridding_scope='deterministic transformations of previously observed pre-policy prices; no latent efficient price or behavioural response identified')
    (HERE/'SCREENING_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    (HERE/'PRESERVED_OPTIONS_SHA256.json').write_text(json.dumps(before,indent=2))
    print(pd.DataFrame(records).query("threshold=='q10' and period=='all'").round(5).to_string(index=False))
    print(pd.DataFrame(policy).query("window_days==28").round(5).to_string(index=False))
    print(pd.DataFrame(counts).to_string(index=False))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
