"""Prepare an observable three-reference forecasting problem, without model fitting.

The saved windows are actual hours, targets are exactly six hours ahead, and
reference definitions remain fixed. Counts describe dependent observations.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']
REFS = ['mean5', 'median5', 'BTC']
NOV = pd.Timestamp('2025-11-01', tz='UTC')
JAN = pd.Timestamp('2026-01-01', tz='UTC')
INPUTS = [ROOT/'data/raw data'/f'{venue}_1h_2025-06-01_2026-03-19.csv'
          for venue in ['binance', 'upbit']]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserved_files():
    files = []
    for name in ['extension_20260909', 'workshop2_20260909']:
        files.extend(p for p in (ROOT/name).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def read_hourly(path):
    d = pd.read_csv(path)
    d.index = pd.to_datetime(d.pop('datetime_utc'), utc=True) + pd.Timedelta(hours=1)
    assert d.index.is_unique and d.index.is_monotonic_increasing
    return d


def logpos(s):
    return np.log(s.where(s > 0))


def prepare():
    before = preserved_files()
    raw_hashes = {str(p.relative_to(ROOT)): digest(p) for p in INPUTS}
    d = read_hourly(INPUTS[0]).join(read_hourly(INPUTS[1]), how='outer')
    grid = pd.date_range(d.index.min(), d.index.max(), freq='h', tz='UTC')
    d = d.reindex(grid)
    log_r = pd.DataFrame({c: logpos(d[c+'_UPBIT_CLOSE']/d[c+'_BINANCE_CLOSE'])
                          for c in COINS})
    individual = log_r.rsub(logpos(d.USDT_UPBIT_CLOSE), axis=0)*1e4
    b = pd.DataFrame({'mean5': individual.mean(axis=1, skipna=False),
                      'median5': individual.median(axis=1, skipna=False),
                      'BTC': individual.BTC})
    old = pd.read_csv(ROOT/'extension_20260909/basis_definitions.csv', index_col=0)
    old.index = pd.to_datetime(old.index, utc=True)
    np.testing.assert_allclose(b.reindex(old.index)[REFS], old[REFS],
                               atol=1e-7, rtol=1e-8, equal_nan=True)
    x = pd.DataFrame(index=grid)
    x['basis'] = b.mean5
    x['change1'], x['change6'] = b.mean5.diff(), b.mean5.diff(6)
    x['mean24'], x['std24'] = b.mean5.rolling(24).mean(), b.mean5.rolling(24).std()
    ret = logpos(d.BTC_BINANCE_CLOSE).diff()
    x['btc_ret1'], x['btc_ret6'] = ret, ret.rolling(6).sum()
    x['btc_vol24'] = ret.rolling(24).std()
    x['btc_downside24'] = ret.clip(upper=0).pow(2).rolling(24).mean()
    x['hour_sin'] = np.sin(2*np.pi*grid.hour/24)
    x['hour_cos'] = np.cos(2*np.pi*grid.hour/24)
    x['weekend'] = (grid.dayofweek >= 5).astype(float)
    x['q10_7d'] = b.mean5.rolling('7D', min_periods=120).quantile(.1)
    x['q90_7d'] = b.mean5.rolling('7D', min_periods=120).quantile(.9)
    x['dispersion'] = log_r.std(axis=1, skipna=False)*1e4
    x['usdt_range'] = logpos(d.USDT_UPBIT_HIGH/d.USDT_UPBIT_LOW)
    vol = pd.DataFrame({c: logpos(d[c+'_UPBIT_VOLUME']/d[c+'_BINANCE_VOLUME'])
                        for c in COINS})
    x['venue_volume'] = vol.mean(axis=1, skipna=False)
    lv = logpos(d.USDT_UPBIT_VOLUME)
    x['usdt_volume_surprise'] = lv - lv.shift().rolling(24).mean()
    x['usdt_unchanged'] = d.USDT_UPBIT_CLOSE.eq(d.USDT_UPBIT_CLOSE.shift()).astype(float)
    for c in COINS:
        x['basis_'+c] = individual[c]
    x['basis_median5'] = b.median5
    # Arithmetic one-won scenarios, NOT historical minimum tick units.
    x['doge_one_won_bp'] = 2000*np.log1p(1/d.DOGE_UPBIT_CLOSE)
    x['usdt_one_won_bp'] = 10000*np.log1p(1/d.USDT_UPBIT_CLOSE)
    arrays, positions = [], []
    future = b.reindex(grid+pd.Timedelta(hours=6)).to_numpy()
    for i in range(23, len(grid)):
        a = x.iloc[i-23:i+1].to_numpy(dtype=float)
        if np.isfinite(a).all() and np.isfinite(future[i]).all():
            assert grid[i]-grid[i-23] == pd.Timedelta(hours=23)
            arrays.append(a)
            positions.append(i)
    X = np.stack(arrays)
    idx = np.asarray(positions)
    origin = grid[idx]
    target_time = origin+pd.Timedelta(hours=6)
    Y = future[idx]
    current = b.iloc[idx].to_numpy()
    # Diagnostics are defined on pre-November observed prices, not test quantiles.
    cuts = b.loc[b.index < NOV].quantile(.1)
    scales = b.loc[b.index < NOV].quantile(.75)-b.loc[b.index < NOV].quantile(.25)
    assert (scales > 0).all()
    pd.DataFrame({'q10_train': cuts, 'iqr_train_bp': scales}).to_csv(HERE/'reference_scales.csv')
    future_low = Y < cuts.to_numpy()
    masks = {'initial_training': target_time < NOV,
             'development': (origin >= NOV) & (target_time < JAN),
             'evaluation': origin >= JAN}
    counts = []
    for split, mask in masks.items():
        for j, ref in enumerate(REFS):
            counts.append(dict(split=split, reference=ref, n_origins=int(mask.sum()),
                origin_days=int(origin[mask].normalize().nunique()),
                future_lower_points=int(future_low[mask, j].sum()),
                future_lower_days=int(origin[mask & future_low[:, j]].normalize().nunique()),
                future_lower_rate=float(future_low[mask, j].mean()),
                first_origin=str(origin[mask].min()), last_origin=str(origin[mask].max())))
    pd.DataFrame(counts).to_csv(HERE/'feasibility_counts.csv', index=False)
    records = []
    disp = x.dispersion.to_numpy()[idx]
    disp_cut = float(x.loc[x.index < NOV, 'dispersion'].quantile(.75))
    label_disagree = future_low.any(axis=1) & ~future_low.all(axis=1)
    for split, mask in masks.items():
        for state, smask in [('all', np.ones(len(X), dtype=bool)),
                             ('high_current_dispersion', disp > disp_cut),
                             ('other_current_dispersion', disp <= disp_cut)]:
            keep = mask & smask
            records.append(dict(split=split, state=state, n=int(keep.sum()),
                future_label_disagreement_rate=float(label_disagree[keep].mean()),
                future_all_three_lower_rate=float(future_low[keep].all(axis=1).mean()),
                mean_future_reference_range_bp=float(np.ptp(Y[keep], axis=1).mean())))
    pd.DataFrame(records).to_csv(HERE/'target_disagreement.csv', index=False)
    f = pd.DataFrame(Y, columns=['target_'+r for r in REFS], index=origin)
    f.index.name = 'origin'
    f['target_time'] = target_time
    for j, ref in enumerate(REFS):
        f['current_'+ref] = current[:, j]
    f['current_dispersion'] = disp
    f.to_csv(HERE/'target_panel.csv.gz', compression='gzip')
    np.savez_compressed(HERE/'design_data.npz', X=X.astype(np.float32),
        y=Y.astype(np.float32), current=current.astype(np.float32),
        origin_ns=origin.asi8, target_ns=target_time.asi8,
        feature_names=np.asarray(x.columns), reference_names=np.asarray(REFS),
        reference_scale=scales.to_numpy())
    # Validate saved targets against an independent direct calculation at t+6.
    future_d = d.reindex(target_time)
    direct = np.column_stack([10000*np.log(
        future_d.USDT_UPBIT_CLOSE*future_d[c+'_BINANCE_CLOSE']/future_d[c+'_UPBIT_CLOSE'])
        for c in COINS])
    direct_y = np.column_stack([direct.mean(axis=1), np.median(direct, axis=1), direct[:, 0]])
    np.testing.assert_allclose(Y, direct_y, atol=1e-7, rtol=1e-8)
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    assert np.all(np.diff(grid.asi8) == pd.Timedelta(hours=1).value)
    assert np.all(target_time.asi8-origin.asi8 == pd.Timedelta(hours=6).value)
    after = preserved_files()
    assert before == after
    assert raw_hashes == {str(p.relative_to(ROOT)): digest(p) for p in INPUTS}
    meta = dict(status='design and data feasibility only; no new AI fitted',
        dataset_shape=list(X.shape), reference_names=REFS, feature_names=list(x.columns),
        actual_lookback_hours=24, target_horizon_hours=6,
        earliest_origin=str(origin.min()), latest_origin=str(origin.max()),
        n_complete_price_rows=int(np.isfinite(b).all(axis=1).sum()), n_grid=len(grid),
        partition_boundary_origins=int(len(X)-sum(int(m.sum()) for m in masks.values())),
        train_dispersion_q75=disp_cut, raw_sha256=raw_hashes,
        prepared_data_sha256=digest(HERE/'design_data.npz'),
        script_sha256=digest(Path(__file__)), preserved_artifacts=len(before),
        previous_options_unchanged=True, reused_evaluation_period=True,
        checks=['same basis as option1', 'exact six-hour raw-price targets',
                'complete 24 actual-hour histories', 'all inputs and targets finite',
                'original options and raw prices unchanged'])
    (HERE/'FEASIBILITY_MANIFEST.json').write_text(json.dumps(meta, indent=2))
    (HERE/'PRESERVED_OPTIONS_SHA256.json').write_text(json.dumps(before, indent=2))
    print(json.dumps(meta, indent=2))
    print(pd.DataFrame(counts).to_string(index=False))
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    prepare()
