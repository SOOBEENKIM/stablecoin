"""Read-only source checks and replay the archived derivative merge (no downloads)."""
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'stablecoin_v4/data/raw data'


def profile(name, headerless=False):
    data = pd.read_csv(RAW / name, header=None if headerless else 'infer')
    if headerless:
        data.columns = ['datetime', 'price'] if data.shape[1] == 2 else ['datetime', 'open', 'high', 'low', 'close']
    times = pd.to_datetime(data.iloc[:, 0], errors='coerce')
    valid = data.loc[times.notna()].copy()
    ts = pd.DatetimeIndex(times.dropna())
    prices = valid.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    result = dict(file=name, rows=len(data), valid_timestamp_rows=len(valid),
                  all_blank_rows=int(data.isna().all(axis=1).sum()),
                  invalid_nonblank_timestamp_rows=int((times.isna() & ~data.isna().all(axis=1)).sum()),
                  missing_cells_with_valid_timestamp=int(prices.isna().sum().sum()),
                  duplicate_valid_timestamps=int(ts.duplicated().sum()),
                  start=str(ts.min()), end=str(ts.max()), timezone=str(ts.tz),
                  observed_minutes=sorted(set(int(x) for x in ts.minute)),
                  observed_hours=sorted(set(int(x) for x in ts.hour)),
                  grid_hours=len(pd.date_range(ts.min(), ts.max(), freq='1h')),
                  numeric_infinite_cells=int(np.isinf(prices.to_numpy(dtype=float)).sum()))
    result['per_column_valid_counts'] = {str(k): int(v) for k, v in prices.notna().sum().items()}
    result['ohlc_checks'] = {}
    result['all_missing_timestamps'] = list(times.loc[prices.isna().all(axis=1).reindex(data.index, fill_value=False)].astype(str))
    groups = [''] if 'open' in prices else [c[:-5] for c in prices if c.endswith('_OPEN')]
    for prefix in groups:
        cs = ['open', 'high', 'low', 'close'] if not prefix else [prefix + '_' + s for s in ['OPEN','HIGH','LOW','CLOSE']]
        x = prices[cs].dropna()
        o,h,l,c = (x[s] for s in cs)
        violations = (h < x.max(axis=1)) | (l > x.min(axis=1)) | (l <= 0)
        volume = 'volume' if not prefix else prefix + '_VOLUME'
        result['ohlc_checks'][prefix or name] = dict(complete_bars=len(x), invalid_bars=int(violations.sum()),
                                                   flat_bars=int((x.max(axis=1)==x.min(axis=1)).sum()),
                                                   negative_volume=int((prices[volume]<0).sum()) if volume in prices else None)
    return result


def main():
    names = ['USDKRW.csv', 'VIXY.csv', 'SPY US.csv', 'DXY.csv', 'XAUUSD.csv',
             'binance_1h_2025-06-01_2026-03-19.csv', 'upbit_1h_2025-06-01_2026-03-19.csv']
    results = [profile(n, i < 5) for i,n in enumerate(names)]
    (HERE/'primary_quality.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results]).to_csv(HERE/'primary_quality.csv', index=False)

    vix = pd.read_csv(RAW/'VIXY.csv', header=None, names=['datetime','price']).dropna()
    vix['datetime'] = pd.to_datetime(vix['datetime'])
    last = vix.sort_values('datetime').groupby(vix['datetime'].dt.date).tail(1).copy()
    last['date'] = last.datetime.dt.normalize()
    official = pd.read_csv(HERE/'cboe_VIX_History.csv')
    official['date'] = pd.to_datetime(official['DATE'])
    match = last.merge(official[['date','CLOSE']], on='date', how='left', validate='one_to_one')
    match['absolute_error'] = (match.price-match.CLOSE).abs()
    match.to_csv(HERE/'vix_identity_comparison.csv',index=False)
    vix_identity = dict(days=len(match), official_missing=int(match.CLOSE.isna().sum()),
                        exact_matches=int((match.absolute_error==0).sum()), max_abs_error=float(match.absolute_error.max()),
                        start=str(match.date.min()), end=str(match.date.max()),
                        source='https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
                        limitation='Only daily last observations checked; no external intraday-price verification.')
    (HERE/'vix_identity_summary.json').write_text(json.dumps(vix_identity,indent=2))

    def read_panel(path):
        d = pd.read_csv(path)
        d['datetime_utc'] = pd.to_datetime(d['datetime_utc'], utc=True, format='mixed')
        return d.set_index('datetime_utc').sort_index()
    saved_dir = ROOT/'stablecoin_v4/corrected_outputs'
    df = read_panel(HERE/'reproduction/baseline_dataset_with_residual_final.csv')
    funding = read_panel(saved_dir/'binance_btcusdt_funding_rate.csv').reset_index()
    oi = read_panel(saved_dir/'binance_btcusdt_open_interest_hist.csv').reset_index()
    metrics = read_panel(saved_dir/'binance_btcusdt_data_vision_metrics.csv')
    metrics_hourly = metrics.resample('1h').last().dropna(how='all')
    saved_oi = oi.set_index('datetime_utc')
    oi_match = metrics_hourly.index.equals(saved_oi.index) and np.isclose(metrics_hourly, saved_oi, equal_nan=True,rtol=1e-9,atol=1e-12).all()

    # Execute only the existing pure merge block, stopping before diagnostic writes.
    script = (ROOT/'stablecoin_workshop/reference/legacy_v3/corrected_main_empirical_pipeline.py').read_text()
    tree = ast.parse(script)
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect_and_merge_derivatives_data')
    start = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='merged' for t in n.targets))
    block = []
    for n in fn.body[start:]:
        if isinstance(n,ast.Expr) or isinstance(n,ast.Return): break
        block.append(n)
    scope = dict(df=df, funding=funding, oi=oi, np=np, pd=pd)
    exec(compile(ast.Module(body=block,type_ignores=[]),'archived_derivatives_merge','exec'),scope)
    merged = scope['merged']
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ['COIN_VOLUME_RATIO_COLUMNS','USDT_VOLUME_RATIO_COLUMN']:
                    scope[target.id] = ast.literal_eval(node.value)
    demand = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='add_demand_channel_proxies')
    demand.returns = None
    for arg in demand.args.args: arg.annotation = None
    exec(compile(ast.Module(body=[demand],type_ignores=[]),'archived_demand_features','exec'),scope)
    merged,_ = scope['add_demand_channel_proxies'](merged)
    saved = read_panel(saved_dir/'baseline_dataset_with_residual_final.csv')
    checks = []
    for c in saved:
        if c not in merged:
            checks.append(dict(column=c, matches=False, reason='not regenerated'))
            continue
        x,y = merged[c],saved[c]
        eq = np.isclose(x,y,rtol=1e-9,atol=1e-12,equal_nan=True)
        checks.append(dict(column=c,matches=bool(eq.all()),mismatches=int((~eq).sum()),max_abs_error=float((x-y).abs().max())))
    pd.DataFrame(checks).to_csv(HERE/'full_baseline_comparison.csv',index=False)
    summary = dict(oi_resample_matches=bool(oi_match), generated_rows=len(merged),saved_rows=len(saved),
                   same_index=merged.index.equals(saved.index),generated_columns_including_timestamp=len(merged.columns)+1,
                   saved_columns_including_timestamp=len(saved.columns)+1,
                   saved_columns_not_generated=sorted(set(saved)-set(merged)),
                   mismatching_columns=[r for r in checks if not r['matches']])
    (HERE/'full_baseline_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(primary=results,vix=vix_identity,full_baseline=summary),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
