"""Availability of a proposed price-basis benchmark; no forecasting/model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'stablecoin_v4/data/raw data'
COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']


def main():
    b = pd.read_csv(RAW/'binance_1h_2025-06-01_2026-03-19.csv',index_col=0,parse_dates=True)
    u = pd.read_csv(RAW/'upbit_1h_2025-06-01_2026-03-19.csv',index_col=0,parse_dates=True)
    assert b.index.equals(u.index)
    assert b.index.equals(pd.date_range(b.index.min(),b.index.max(),freq='1h'))
    legs = pd.DataFrame({c:np.log(u[c+'_UPBIT_CLOSE'])-np.log(b[c+'_BINANCE_CLOSE']) for c in COINS})
    basis = (np.log(u['USDT_UPBIT_CLOSE'])-legs.mean(axis=1,skipna=False))*1e4
    metrics = pd.read_csv(ROOT/'stablecoin_v4/corrected_outputs/binance_btcusdt_open_interest_hist.csv',index_col=0,parse_dates=True)
    result = {
        'date':'2026-09-08',
        'measure':'Candidate direct-versus-synthetic KRW/USDT log price basis, equal weights',
        'formula':'10000 * (log(Upbit_USDT_KRW) - mean_i(log(Upbit_coin_i_KRW) - log(Binance_coin_i_USDT)))',
        'coins':COINS,
        'grid_rows':len(basis),
        'valid_basis_rows':int(basis.notna().sum()),
        'days_with_basis':int(basis.dropna().index.normalize().nunique()),
        'start':str(basis.index.min()),'end':str(basis.index.max()),
        'origin_future_nonmissing_pairs':{h:int((basis.notna() & basis.shift(-h).notna()).sum()) for h in [1,3,6,12]},
        'positioning_timestamp_overlap':int((basis.notna() & metrics.ACCOUNT_LS.reindex(basis.index).notna()).sum()),
        'forecast_models_fitted':False,
        'limitations':[
            'This proposed measure is different from the submitted paper OLS residual; it is not a corrected or expanded version of that residual.',
            'Hourly closing prices are not synchronized executable quotes. No order-book, transaction-cost, or executable arbitrage claim.',
            'No predictive performance, novelty, causal mechanism, or acceptance probability established.',
            'Counts are before forecasting lags, release-time alignment, chronological splits and purging.'
        ]
    }
    (HERE/'icaif_feasibility_diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
