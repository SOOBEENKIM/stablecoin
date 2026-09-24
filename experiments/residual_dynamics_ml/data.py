"""Clock-aligned inputs and training-only versions of the manuscript residual.

No archived residual/lag columns are used. All future columns returned by
make_design are outcomes for evaluation, never model features.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'research/data/raw data'
DERIV = ROOT / 'research/corrected_outputs'
COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']
SUPPLY = np.array([19.9e6, 120.5e6, 59.5e9, 590e6, 149e9])
SCENARIOS = {
    'NY_delay1': ('America/New_York', 1),
    'NY_delay0': ('America/New_York', 0),
    'FX_UTC_delay1': ('UTC', 1),
    'FX_KST_delay1': ('Asia/Seoul', 1),
}
BASE = ['e_now', 'e_change1', 'downside24', 'btc_vol24', 'btc_ret1',
        'dxy_ret1', 'fx_ret1', 'vix_ret1', 'local_volume_surprise']
POSITION = ['account_log', 'funding_bp', 'oi_ret1', 'oi_surprise']
INPUT_PATHS = [RAW / x for x in [
    'binance_1h_2025-06-01_2026-03-19.csv',
    'upbit_1h_2025-06-01_2026-03-19.csv', 'USDKRW.csv', 'DXY.csv', 'VIXY.csv']]
INPUT_PATHS += [DERIV / x for x in [
    'binance_btcusdt_data_vision_metrics.csv', 'binance_btcusdt_funding_rate.csv']]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_panel(path):
    d = pd.read_csv(path)
    d['datetime_utc'] = pd.to_datetime(d.datetime_utc, utc=True, format='mixed')
    d = d.set_index('datetime_utc').sort_index()
    if d.index.has_duplicates:
        raise ValueError('Duplicate timestamp: ' + str(path))
    return d


def past_observation(s, grid, tolerance):
    s = s.dropna().sort_index()
    if s.index.has_duplicates:
        raise ValueError('Duplicate source timestamp')
    m = pd.merge_asof(pd.DataFrame({'t': grid}),
        pd.DataFrame({'source_time': s.index, 'value': s.to_numpy()}),
        left_on='t', right_on='source_time', direction='backward',
        tolerance=pd.Timedelta(tolerance))
    age = (m.t - m.source_time).dt.total_seconds().to_numpy() / 60
    assert np.all(age[np.isfinite(age)] >= 0)
    return pd.Series(m.value.to_numpy(), index=grid), pd.Series(age, index=grid)


def macro(name, zone, delay):
    d = pd.read_csv(RAW / name, header=None, names=['time', 'value'])
    d.time = pd.to_datetime(d.time, errors='coerce')
    d.value = pd.to_numeric(d.value, errors='coerce')
    d = d.dropna()
    idx = pd.DatetimeIndex(d.time).tz_localize(zone, ambiguous='raise',
        nonexistent='raise').tz_convert('UTC') + pd.Timedelta(hours=delay)
    return pd.Series(d.value.to_numpy(), index=idx)


def build_panel(scenario):
    fxzone, delay = SCENARIOS[scenario]
    d = read_panel(INPUT_PATHS[0]).join(read_panel(INPUT_PATHS[1]))
    d.index += pd.Timedelta(hours=1)
    if not (d.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError('Candle grid is not exactly hourly')
    p = pd.DataFrame(index=d.index)
    for col, file, zone in [('fx', 'USDKRW.csv', fxzone),
                           ('dxy', 'DXY.csv', 'America/New_York'),
                           ('vix', 'VIXY.csv', 'America/New_York')]:
        p[col], p[col + '_age_min'] = past_observation(
            macro(file, zone, delay), p.index, '59min59s')
        p[col + '_ret1'] = np.log(p[col]).diff()
    p['q'] = d.USDT_BINANCE_CLOSE
    p['local_usdt'] = d.USDT_UPBIT_CLOSE
    p['y'] = p.local_usdt * p.q / p.fx - 1
    p['g'] = (p.q - 1).abs()
    caps = []
    for i, coin in enumerate(COINS):
        p['kp_' + coin] = d[coin + '_UPBIT_CLOSE'] * p.q / (
            p.fx * d[coin + '_BINANCE_CLOSE']) - 1
        caps.append(d[coin + '_BINANCE_CLOSE'].to_numpy() * SUPPLY[i])
    caps = np.column_stack(caps)
    k = p[['kp_' + c for c in COINS]]
    p['m_EQ'] = k.mean(axis=1, skipna=False)
    p['m_CAP'] = np.sum(k.to_numpy() * caps / caps.sum(axis=1)[:, None], axis=1)
    ret = np.log(d.BTC_BINANCE_CLOSE).diff()
    p['btc_ret1'] = ret * 1e4
    p['btc_vol24'] = ret.rolling(24, min_periods=24).std(ddof=1) * 1e4
    p['downside24'] = ret.clip(upper=0).pow(2).rolling(24, min_periods=24).mean() * 1e8
    lv = np.log(d.USDT_UPBIT_VOLUME.where(d.USDT_UPBIT_VOLUME > 0))
    p['local_volume_surprise'] = lv - lv.shift(1).rolling(24, min_periods=24).mean()
    p['local_log_volume'] = lv
    for c in ['dxy_ret1', 'fx_ret1', 'vix_ret1']:
        p[c] *= 1e4
    m = read_panel(INPUT_PATHS[-2])
    m.index += pd.Timedelta(minutes=5)
    for src, dst in [('ACCOUNT_LS', 'account'), ('OI', 'oi')]:
        p[dst], p[dst + '_age_min'] = past_observation(m[src], p.index, '10min')
    p['account_log'] = np.log(p.account.where(p.account > 0))
    loi = np.log(p.oi.where(p.oi > 0))
    p['oi_ret1'] = loi.diff() * 1e4
    p['oi_surprise'] = loi - loi.shift(1).rolling(24, min_periods=24).mean()
    f = read_panel(INPUT_PATHS[-1])
    p['funding_bp'], p['funding_age_min'] = past_observation(f.FUNDING * 1e4, p.index, '8h')
    p = p.replace([np.inf, -np.inf], np.nan)
    p.index.name = 'origin'
    return p


class Residualizer:
    def __init__(self, definition='EQ', projection='sequential'):
        self.definition, self.projection = definition, projection

    def fit(self, p, cutoff):
        cols = ['y', 'g'] + ['kp_' + c for c in COINS]
        fit = p.loc[p.index < cutoff].dropna(subset=cols)
        if len(fit) < 50:
            raise ValueError('Insufficient first-stage training rows')
        self.fit_end = fit.index.max()
        self.fit_n = len(fit)
        if self.definition == 'PCA':
            k = fit[['kp_' + c for c in COINS]].to_numpy()
            self.k_mean = k.mean(axis=0)
            self.k_std = k.std(axis=0, ddof=1)
            z = (k - self.k_mean) / self.k_std
            _, s, v = np.linalg.svd(z, full_matrices=False)
            self.loading = v[0] * (1 if v[0].sum() >= 0 else -1)
            self.explained = float(s[0] ** 2 / np.square(s).sum())
        m = self.market(fit)
        y, g = fit.y.to_numpy(), fit.g.to_numpy()
        if self.projection == 'joint':
            b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), m, g]), y, rcond=None)[0]
            self.intercept, self.beta_market, self.beta_g = b
        else:
            b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), m]), y, rcond=None)[0]
            u = y - b[0] - b[1] * m
            c = np.linalg.lstsq(np.column_stack([np.ones(len(y)), g]), u, rcond=None)[0]
            self.intercept, self.beta_market, self.beta_g = b[0] + c[0], b[1], c[1]
        return self

    def market(self, p):
        if self.definition == 'PCA':
            return ((p[['kp_' + c for c in COINS]].to_numpy() - self.k_mean)
                    / self.k_std) @ self.loading
        return p['m_' + self.definition].to_numpy()

    def transform(self, p):
        m = self.market(p)
        e = (p.y.to_numpy() - self.intercept - self.beta_market * m
             - self.beta_g * p.g.to_numpy()) * 1e4
        return pd.Series(e, index=p.index), pd.Series(m, index=p.index)

    def metadata(self):
        out = dict(definition=self.definition, projection=self.projection,
            fit_end=self.fit_end.isoformat(), fit_n=self.fit_n,
            intercept=float(self.intercept), beta_market=float(self.beta_market),
            beta_g=float(self.beta_g))
        if self.definition == 'PCA':
            out.update(pca_explained=self.explained, pca_mean=self.k_mean.tolist(),
                       pca_std=self.k_std.tolist(), pca_loading=self.loading.tolist())
        return out


def lead(s, hours):
    """Exact clock lookup; missing target observations remain missing."""
    return pd.Series(s.reindex(s.index + pd.Timedelta(hours=hours)).to_numpy(), index=s.index)


def make_design(p, r, hours):
    e, m = r.transform(p)
    d = p[[c for c in BASE if not c.startswith('e_')] + POSITION].copy()
    d['e_now'] = e
    d['e_change1'] = e - lead(e, -1)
    d['target'] = lead(e, hours)
    d['target_time'] = d.index + pd.Timedelta(hours=hours)
    d['delta_e'] = d.target - e
    d['delta_market_bp'] = -r.beta_market * (lead(m, hours) - m) * 1e4
    d['delta_g_bp'] = -r.beta_g * (lead(p.g, hours) - p.g) * 1e4
    # Exact additive attribution of a multiplicative price ratio, using its
    # logarithmic mean. This is a price identity, not causal mediation.
    ratio = 1 + p.y
    ratio_next = lead(ratio, hours)
    dr = ratio_next - ratio
    dlr = np.log(ratio_next) - np.log(ratio)
    factor = np.divide(dr.to_numpy(), dlr.to_numpy(), out=ratio.to_numpy().copy(),
                       where=np.abs(dlr.to_numpy()) > 1e-12)
    for col, name, sign in [('local_usdt', 'delta_local_bp', 1),
                            ('q', 'delta_quote_bp', 1), ('fx', 'delta_fx_bp', -1)]:
        d[name] = sign * factor * (np.log(lead(p[col], hours)) - np.log(p[col])) * 1e4
    d['delta_log_volume'] = lead(p.local_log_volume, hours) - p.local_log_volume
    # Common complete-case mask across all models and position ablations.
    d = d.dropna(subset=BASE + POSITION + ['target'])
    parts = ['delta_local_bp', 'delta_quote_bp', 'delta_fx_bp', 'delta_market_bp', 'delta_g_bp']
    if not np.allclose(d[parts].sum(axis=1), d.delta_e, rtol=1e-8, atol=1e-8):
        raise AssertionError('Residual price attribution does not add up')
    return d


def feature_columns(info):
    return BASE + ([] if info == 'base' else ['account_log'] if info == 'account' else POSITION)
