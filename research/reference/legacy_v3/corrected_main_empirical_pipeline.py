"""
Corrected main empirical pipeline for the USDT Korean premium residual study.

This script intentionally does not recollect Binance/Upbit data. It reuses the
existing preprocessed CSV files, identifies RESIDUAL_FINAL consistently, and
runs the main OLS, quantile, regime-comparison, and local-projection tests.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import time
from typing import Any
import io
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR


warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "corrected_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

INPUT_FILE = DATA_DIR / "baseline_dataset_with_micro.csv"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_DATA_VISION_BASE = "https://data.binance.vision"
DERIVATIVE_SYMBOL = "BTCUSDT"


# ── Bootstrap configuration ──────────────────────────────────────────────────
# The baseline sample has ~4 consecutive observations per trading day (UTC 9-12),
# separated by ~21-hour gaps.  Moving-block bootstrap with block_size=4 therefore
# resamples whole trading-day blocks, preserving within-day serial structure.
# Increase N_BOOTSTRAP to ≥ 1000 for final publication-quality results.
N_BOOTSTRAP: int = 500
QR_BLOCK_SIZE: int = 4
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "USDT_KP",
    "MKT_KP_EQ",
    "USDT_BINANCE_CLOSE",
    "BTC_DOWNSIDE_24",
    "DXY_ret",
    "USDKRW_ret",
    "VIX",
]

COIN_VOLUME_RATIO_COLUMNS = [
    "BTC_VOL_RATIO_LOG",
    "ETH_VOL_RATIO_LOG",
    "XRP_VOL_RATIO_LOG",
    "SOL_VOL_RATIO_LOG",
    "DOGE_VOL_RATIO_LOG",
]

USDT_VOLUME_RATIO_COLUMN = "USDT_VOL_RATIO_LOG"


def load_master_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_FILE}\n"
            "Run the original data-preprocessing/collection cells first, or put "
            "baseline_dataset_with_micro.csv in the parent VisualCode folder."
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["datetime_utc"],
        index_col="datetime_utc",
    )
    df.index = pd.to_datetime(df.index, utc=True)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {INPUT_FILE.name}: {missing}")

    return df.dropna(subset=REQUIRED_COLUMNS).copy()


def _get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 20):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_binance_funding_rate(
    symbol: str,
    start_ms: int,
    end_ms: int,
    sleep_seconds: float = 0.20,
) -> pd.DataFrame:
    """Fetch Binance USD-M funding rate history.

    Endpoint: GET /fapi/v1/fundingRate
    Binance funding is usually every 8 hours, so it is later aligned to hourly
    observations with a bounded forward fill.
    """
    rows: list[dict[str, Any]] = []
    cursor = start_ms

    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = _get_json(f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate", params=params)
        if not data:
            break

        rows.extend(data)
        last_time = int(data[-1]["fundingTime"])
        next_cursor = last_time + 1
        if next_cursor <= cursor or len(data) < 1000:
            break
        cursor = next_cursor
        time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=["datetime_utc", "FUNDING"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["fundingTime"])
    df["datetime_utc"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["FUNDING"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["datetime_utc", "FUNDING"]].sort_values("datetime_utc")


def fetch_binance_open_interest_hist(
    symbol: str,
    start_ms: int,
    end_ms: int,
    period: str = "1h",
    sleep_seconds: float = 0.20,
) -> pd.DataFrame:
    """Fetch Binance USD-M open interest history.

    Endpoint: GET /futures/data/openInterestHist
    Some Binance futures-data endpoints can restrict historical depth. If the
    requested sample is unavailable, the function returns an empty DataFrame and
    the main script continues with funding-only derivative proxies.
    """
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    max_window_ms = 20 * 24 * 60 * 60 * 1000

    while cursor <= end_ms:
        request_end_ms = min(cursor + max_window_ms, end_ms)
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": cursor,
            "endTime": request_end_ms,
            "limit": 500,
        }
        data = _get_json(f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist", params=params)
        if not data:
            cursor = request_end_ms + 60 * 60 * 1000
            continue

        rows.extend(data)
        last_raw = data[-1].get("timestamp", data[-1].get("time"))
        if last_raw is None:
            break
        last_time = int(last_raw)
        next_cursor = last_time + 60 * 60 * 1000
        if next_cursor <= cursor:
            break
        if len(data) < 500 and request_end_ms < end_ms:
            cursor = request_end_ms + 60 * 60 * 1000
        else:
            cursor = next_cursor
        time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=["datetime_utc", "OI", "OI_VALUE"])

    df = pd.DataFrame(rows).drop_duplicates()
    ts_col = "timestamp" if "timestamp" in df.columns else "time"
    df["datetime_utc"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    oi_col = "sumOpenInterest" if "sumOpenInterest" in df.columns else None
    oi_value_col = "sumOpenInterestValue" if "sumOpenInterestValue" in df.columns else None

    if oi_col is not None:
        df["OI"] = pd.to_numeric(df[oi_col], errors="coerce")
    else:
        df["OI"] = np.nan

    if oi_value_col is not None:
        df["OI_VALUE"] = pd.to_numeric(df[oi_value_col], errors="coerce")
    else:
        df["OI_VALUE"] = np.nan

    return df[["datetime_utc", "OI", "OI_VALUE"]].sort_values("datetime_utc")


def fetch_binance_vision_daily_metrics(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """Fetch historical futures metrics from Binance Data Vision archive.

    The REST openInterestHist endpoint often rejects older start/end windows.
    Binance Data Vision publishes historical daily metrics ZIP files with
    5-minute open-interest snapshots, which are more suitable for backfilling
    old research samples.
    """
    frames: list[pd.DataFrame] = []
    dates = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D", tz="UTC")

    for day in dates:
        day_str = day.strftime("%Y-%m-%d")
        url = (
            f"{BINANCE_DATA_VISION_BASE}/data/futures/um/daily/metrics/"
            f"{symbol}/{symbol}-metrics-{day_str}.zip"
        )
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                names = zf.namelist()
                if not names:
                    continue
                with zf.open(names[0]) as f:
                    daily = pd.read_csv(f)
            frames.append(daily)
            time.sleep(sleep_seconds)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(
            columns=[
                "datetime_utc",
                "OI",
                "OI_VALUE",
                "TOPTRADER_ACCOUNT_LS",
                "TOPTRADER_POSITION_LS",
                "ACCOUNT_LS",
                "TAKER_LONG_SHORT_VOL_RATIO",
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df["datetime_utc"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"])
    df = df[(df["datetime_utc"] >= start) & (df["datetime_utc"] <= end)]

    rename = {
        "sum_open_interest": "OI",
        "sum_open_interest_value": "OI_VALUE",
        "count_toptrader_long_short_ratio": "TOPTRADER_ACCOUNT_LS",
        "sum_toptrader_long_short_ratio": "TOPTRADER_POSITION_LS",
        "count_long_short_ratio": "ACCOUNT_LS",
        "sum_taker_long_short_vol_ratio": "TAKER_LONG_SHORT_VOL_RATIO",
    }
    df = df.rename(columns=rename)
    keep = ["datetime_utc", *rename.values()]

    for col in rename.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    return df[keep].sort_values("datetime_utc")


def collect_and_merge_derivatives_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect funding/OI and merge them into the empirical hourly panel."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    start = df.index.min()
    end = df.index.max()
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    diagnostics: dict[str, Any] = {
        "symbol": DERIVATIVE_SYMBOL,
        "sample_start": str(start),
        "sample_end": str(end),
        "funding_rows": 0,
        "oi_rows": 0,
        "errors": [],
    }

    funding_path = OUTPUT_DIR / "binance_btcusdt_funding_rate.csv"
    oi_path = OUTPUT_DIR / "binance_btcusdt_open_interest_hist.csv"
    metrics_path = OUTPUT_DIR / "binance_btcusdt_data_vision_metrics.csv"

    try:
        funding = fetch_binance_funding_rate(DERIVATIVE_SYMBOL, start_ms, end_ms)
        funding.to_csv(funding_path, index=False)
        diagnostics["funding_rows"] = len(funding)
    except Exception as exc:
        funding = pd.DataFrame(columns=["datetime_utc", "FUNDING"])
        diagnostics["errors"].append(f"funding: {exc}")

    try:
        oi = fetch_binance_open_interest_hist(DERIVATIVE_SYMBOL, start_ms, end_ms)
        oi.to_csv(oi_path, index=False)
        diagnostics["oi_rows"] = len(oi)
    except Exception as exc:
        oi = pd.DataFrame(columns=["datetime_utc", "OI", "OI_VALUE"])
        diagnostics["errors"].append(f"open_interest: {exc}")

    if oi.empty:
        try:
            metrics = fetch_binance_vision_daily_metrics(DERIVATIVE_SYMBOL, start, end)
            metrics.to_csv(metrics_path, index=False)
            diagnostics["data_vision_metric_rows"] = len(metrics)
            if not metrics.empty:
                metrics = metrics.set_index("datetime_utc").sort_index()
                hourly = metrics.resample("1h").last().dropna(how="all").reset_index()
                oi = hourly
                oi.to_csv(oi_path, index=False)
                diagnostics["oi_rows"] = len(oi)
                diagnostics["oi_source"] = "binance_data_vision_daily_metrics"
        except Exception as exc:
            diagnostics["errors"].append(f"data_vision_metrics: {exc}")

    merged = df.copy()

    if not funding.empty:
        funding = funding.set_index("datetime_utc").sort_index()
        funding_hourly = (
            funding.reindex(funding.index.union(merged.index))
            .sort_index()
            .ffill(limit=8)
            .reindex(merged.index)
        )
        merged["FUNDING"] = funding_hourly["FUNDING"]
        merged["FUNDING_lag1"] = merged["FUNDING"].shift(1)

    if not oi.empty:
        oi = oi.set_index("datetime_utc").sort_index()
        oi_hourly = oi.reindex(oi.index.union(merged.index)).sort_index().ffill(limit=1).reindex(merged.index)
        merged["OI"] = oi_hourly["OI"]
        merged["OI_VALUE"] = oi_hourly["OI_VALUE"]
        for col in [
            "TOPTRADER_ACCOUNT_LS",
            "TOPTRADER_POSITION_LS",
            "ACCOUNT_LS",
            "TAKER_LONG_SHORT_VOL_RATIO",
        ]:
            if col in oi_hourly.columns:
                merged[col] = oi_hourly[col]
        merged["OI_RET"] = np.log(merged["OI"]).diff()
        merged["OI_VALUE_RET"] = np.log(merged["OI_VALUE"]).diff()
        merged["OI_lag1"] = merged["OI"].shift(1)
        merged["OI_VALUE_lag1"] = merged["OI_VALUE"].shift(1)
        merged["OI_RET_lag1"] = merged["OI_RET"].shift(1)
        merged["OI_VALUE_RET_lag1"] = merged["OI_VALUE_RET"].shift(1)
        for col in [
            "TOPTRADER_ACCOUNT_LS",
            "TOPTRADER_POSITION_LS",
            "ACCOUNT_LS",
            "TAKER_LONG_SHORT_VOL_RATIO",
        ]:
            if col in merged.columns:
                merged[f"{col}_lag1"] = merged[col].shift(1)

    pd.Series(diagnostics).to_json(
        OUTPUT_DIR / "derivatives_collection_diagnostics.json",
        force_ascii=False,
        indent=2,
    )
    return merged, diagnostics


def identify_final_residual(df: pd.DataFrame) -> tuple[pd.DataFrame, object, object]:
    """Create RESIDUAL_FINAL using the corrected two-step identification."""
    df = df.copy()

    # Step 1: remove market-common Korean premium.
    step1 = df[["USDT_KP", "MKT_KP_EQ"]].dropna().copy()
    model_market = sm.OLS(
        step1["USDT_KP"],
        sm.add_constant(step1["MKT_KP_EQ"]),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    df["RESIDUAL_OLS"] = np.nan
    df.loc[step1.index, "RESIDUAL_OLS"] = model_market.resid

    # Step 2: remove global stablecoin depeg proxy.
    # In the original collection block, USDT_BINANCE_CLOSE is Binance USDCUSDT.
    # abs(USDCUSDT - 1) is therefore used as a stablecoin-pair depeg proxy.
    df["DEPEG_GLOBAL"] = (df["USDT_BINANCE_CLOSE"] - 1.0).abs()

    step2 = df[["RESIDUAL_OLS", "DEPEG_GLOBAL"]].dropna().copy()
    model_depeg = sm.OLS(
        step2["RESIDUAL_OLS"],
        sm.add_constant(step2["DEPEG_GLOBAL"]),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    df["RESIDUAL_FINAL"] = np.nan
    df.loc[step2.index, "RESIDUAL_FINAL"] = model_depeg.resid

    return df, model_market, model_depeg


def add_main_identification_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["RESIDUAL_FINAL_lag1"] = df["RESIDUAL_FINAL"].shift(1)
    df["BTC_DOWNSIDE_24_lag1"] = df["BTC_DOWNSIDE_24"].shift(1)
    df["DXY_ret_lag1"] = df["DXY_ret"].shift(1)
    df["USDKRW_ret_lag1"] = df["USDKRW_ret"].shift(1)

    vix_threshold = df["VIX"].quantile(0.80)
    df["HIGH_STRESS"] = (df["VIX"] >= vix_threshold).astype(int)
    df["HIGH_STRESS_lag1"] = df["HIGH_STRESS"].shift(1)

    return df


def add_demand_channel_proxies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add proxies that help compare trading demand vs dollar-access/hedge demand.

    Interpretation:
    - COIN_VOL_RATIO_LOG_MEAN approximates broad crypto trading intermediation
      pressure across non-stablecoin assets.
    - USDT_VOL_RATIO_LOG captures stablecoin-specific relative volume pressure.
    - USDT_RELATIVE_VOLUME_PRESSURE isolates USDT-specific volume pressure from
      broad coin-market volume pressure.
    - VOL_RATIO_DISPERSION captures cross-coin liquidity fragmentation.

    Funding/OI are not created here because no reliable local historical file is
    present in the project. If such columns are later merged into the input CSV,
    they will be automatically included by run_demand_channel_tests().
    """
    df = df.copy()
    available_coin_cols = [c for c in COIN_VOLUME_RATIO_COLUMNS if c in df.columns]

    if available_coin_cols:
        df["COIN_VOL_RATIO_LOG_MEAN"] = df[available_coin_cols].mean(axis=1)
        df["VOL_RATIO_DISPERSION"] = df[available_coin_cols].std(axis=1)

    if USDT_VOLUME_RATIO_COLUMN in df.columns and "COIN_VOL_RATIO_LOG_MEAN" in df.columns:
        df["USDT_RELATIVE_VOLUME_PRESSURE"] = (
            df[USDT_VOLUME_RATIO_COLUMN] - df["COIN_VOL_RATIO_LOG_MEAN"]
        )

    demand_candidates = [
        USDT_VOLUME_RATIO_COLUMN,
        "COIN_VOL_RATIO_LOG_MEAN",
        "USDT_RELATIVE_VOLUME_PRESSURE",
        "VOL_RATIO_DISPERSION",
        "FUNDING_lag1",
        "OI_lag1",
        "OI_VALUE_lag1",
        "OI_RET_lag1",
        "OI_VALUE_RET_lag1",
        "TOPTRADER_ACCOUNT_LS_lag1",
        "TOPTRADER_POSITION_LS_lag1",
        "ACCOUNT_LS_lag1",
        "TAKER_LONG_SHORT_VOL_RATIO_lag1",
        "LIQUIDATION_lag1",
    ]
    available_demand_cols = [c for c in demand_candidates if c in df.columns]

    return df, available_demand_cols


def get_empirical_sample(df: pd.DataFrame) -> pd.DataFrame:
    main_cols = [
        "RESIDUAL_FINAL",
        "RESIDUAL_FINAL_lag1",
        "BTC_DOWNSIDE_24_lag1",
        "DXY_ret_lag1",
        "USDKRW_ret_lag1",
        "HIGH_STRESS_lag1",
    ]
    return df.dropna(subset=main_cols).copy()


def _block_bootstrap_qr_pvalues(
    formula: str,
    data: pd.DataFrame,
    q: float,
    coef_names: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    block_size: int = QR_BLOCK_SIZE,
    seed: int = 42,
) -> dict[str, Any]:
    """Moving-block bootstrap SE and p-values for quantile regression.

    Standard QR inference assumes iid errors.  RESIDUAL_FINAL has very strong
    serial correlation (Ljung-Box lag-1 p < 10^-33), so standard p-values are
    anti-conservative.  Moving-block bootstrap with block_size=4 (one trading
    day) resamples whole-day blocks, preserving within-day dynamics.

    Returns dict with:
        bse       – bootstrap standard errors, aligned with coef_names
        pvalues   – two-tailed p-values via normal approximation (coef / bse)
        tvalues   – t-statistics (coef / bootstrap SE)
        n_success – number of successful bootstrap replications
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    data_reset = data.reset_index(drop=True)

    orig_model = smf.quantreg(formula, data_reset).fit(q=q, max_iter=5000)
    orig_coefs = orig_model.params[coef_names].values.copy()

    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size

    if max_start < 1:
        k = len(coef_names)
        return {
            "bse": np.full(k, np.nan),
            "pvalues": np.full(k, np.nan),
            "tvalues": np.full(k, np.nan),
            "n_success": 0,
        }

    boot_params: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        idx = np.concatenate(
            [np.arange(s, min(s + block_size, n)) for s in starts]
        )[:n]
        bd = data_reset.iloc[idx].reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.quantreg(formula, bd).fit(q=q, max_iter=2000)
            coef_vals = m.params[coef_names].values
            if np.all(np.isfinite(coef_vals)):
                boot_params.append(coef_vals)
        except Exception:
            continue

    n_success = len(boot_params)
    if n_success < 50:
        k = len(coef_names)
        return {
            "bse": np.full(k, np.nan),
            "pvalues": np.full(k, np.nan),
            "tvalues": np.full(k, np.nan),
            "n_success": n_success,
        }

    boot_arr = np.array(boot_params)
    bse = boot_arr.std(axis=0, ddof=1)
    tvals = np.where(bse > 1e-15, orig_coefs / bse, np.nan)
    pvals = 2.0 * stats.norm.sf(np.abs(tvals))

    return {"bse": bse, "pvalues": pvals, "tvalues": tvals, "n_success": n_success}


def run_main_ols(df_emp: pd.DataFrame):
    formula = (
        "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 "
        "+ BTC_DOWNSIDE_24_lag1 + DXY_ret_lag1 + USDKRW_ret_lag1"
    )
    return smf.ols(formula, data=df_emp).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5},
    )


def run_quantile_tail_tests(df_emp: pd.DataFrame) -> pd.DataFrame:
    formula = (
        "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 "
        "+ BTC_DOWNSIDE_24_lag1 + DXY_ret_lag1 + USDKRW_ret_lag1"
    )
    _boot_coefs = ["BTC_DOWNSIDE_24_lag1", "DXY_ret_lag1", "USDKRW_ret_lag1"]

    rows = []
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        model = smf.quantreg(formula, df_emp).fit(q=q, max_iter=5000)
        boot = _block_bootstrap_qr_pvalues(formula, df_emp, q, _boot_coefs)
        rows.append(
            {
                "q": q,
                "beta_downside_lag1": model.params["BTC_DOWNSIDE_24_lag1"],
                "p_downside_lag1": model.pvalues["BTC_DOWNSIDE_24_lag1"],
                "p_downside_lag1_boot": boot["pvalues"][0],
                "bse_downside_lag1_boot": boot["bse"][0],
                "beta_dxy_lag1": model.params["DXY_ret_lag1"],
                "p_dxy_lag1": model.pvalues["DXY_ret_lag1"],
                "p_dxy_lag1_boot": boot["pvalues"][1],
                "beta_fx_lag1": model.params["USDKRW_ret_lag1"],
                "p_fx_lag1": model.pvalues["USDKRW_ret_lag1"],
                "p_fx_lag1_boot": boot["pvalues"][2],
                "n": int(model.nobs),
                "boot_n_success": boot["n_success"],
            }
        )

    return pd.DataFrame(rows)


def run_stress_regime_comparison(df_emp: pd.DataFrame) -> pd.DataFrame:
    formula = (
        "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 "
        "+ BTC_DOWNSIDE_24_lag1 + HIGH_STRESS_lag1 "
        "+ BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1 "
        "+ DXY_ret_lag1 + USDKRW_ret_lag1"
    )

    interaction = "BTC_DOWNSIDE_24_lag1:HIGH_STRESS_lag1"
    _boot_coefs = ["BTC_DOWNSIDE_24_lag1", interaction]
    rows = []

    for q in [0.10, 0.50, 0.90]:
        model = smf.quantreg(formula, df_emp).fit(q=q, max_iter=5000)
        boot = _block_bootstrap_qr_pvalues(formula, df_emp, q, _boot_coefs)
        rows.append(
            {
                "q": q,
                "beta_downside_normal": model.params["BTC_DOWNSIDE_24_lag1"],
                "beta_downside_stress_total": (
                    model.params["BTC_DOWNSIDE_24_lag1"]
                    + model.params[interaction]
                ),
                "p_downside_stress_interaction": model.pvalues[interaction],
                "p_downside_stress_interaction_boot": boot["pvalues"][1],
                "bse_interaction_boot": boot["bse"][1],
                "n": int(model.nobs),
                "boot_n_success": boot["n_success"],
            }
        )

    return pd.DataFrame(rows)


def run_quantile_local_projection(df_emp: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for horizon in [1, 2, 3, 6, 12]:
        temp = df_emp.copy()
        lead_col = f"RESIDUAL_FINAL_lead_{horizon}"
        temp[lead_col] = temp["RESIDUAL_FINAL"].shift(-horizon)
        temp = temp.dropna(subset=[lead_col])

        # Standard LP (Jordà 2005): control with current-period state variable (t),
        # not the one-period lag (t-1). BTC_DOWNSIDE_24_lag1 is deliberately at t-1
        # as the predetermined impulse variable.
        formula = (
            f"{lead_col} ~ RESIDUAL_FINAL "
            "+ BTC_DOWNSIDE_24_lag1 + DXY_ret_lag1 + USDKRW_ret_lag1"
        )
        _boot_coefs = ["BTC_DOWNSIDE_24_lag1"]

        for q in [0.10, 0.50, 0.90]:
            model = smf.quantreg(formula, temp).fit(q=q, max_iter=5000)
            boot = _block_bootstrap_qr_pvalues(formula, temp, q, _boot_coefs)
            rows.append(
                {
                    "h": horizon,
                    "q": q,
                    "beta_downside_lag1": model.params["BTC_DOWNSIDE_24_lag1"],
                    "p_downside_lag1": model.pvalues["BTC_DOWNSIDE_24_lag1"],
                    "p_downside_lag1_boot": boot["pvalues"][0],
                    "bse_downside_lag1_boot": boot["bse"][0],
                    "n": int(model.nobs),
                    "boot_n_success": boot["n_success"],
                }
            )

    return pd.DataFrame(rows)


def save_regression_tables(
    model_main_ols,
    quantile_results: pd.DataFrame,
    regime_results: pd.DataFrame,
    local_projection_results: pd.DataFrame,
) -> None:
    table_dir = OUTPUT_DIR / "tables"
    table_dir.mkdir(exist_ok=True)

    ols_table = pd.DataFrame(
        {
            "coef": model_main_ols.params,
            "std_err": model_main_ols.bse,
            "z_or_t": model_main_ols.tvalues,
            "pvalue": model_main_ols.pvalues,
        }
    )
    ols_table.to_csv(table_dir / "main_hac_ols_table.csv")

    with (table_dir / "main_hac_ols_summary.txt").open("w", encoding="utf-8") as f:
        f.write(model_main_ols.summary().as_text())

    quantile_results.to_csv(table_dir / "quantile_tail_table.csv", index=False)
    regime_results.to_csv(table_dir / "stress_regime_comparison_table.csv", index=False)
    local_projection_results.to_csv(table_dir / "quantile_local_projection_table.csv", index=False)


def save_depeg_event_table(df_main: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "USDT_BINANCE_CLOSE",
        "DEPEG_GLOBAL",
        "USDT_KP",
        "MKT_KP_EQ",
        "RESIDUAL_OLS",
        "RESIDUAL_FINAL",
    ]
    existing = [c for c in cols if c in df_main.columns]
    top = df_main[existing].dropna().sort_values("DEPEG_GLOBAL", ascending=False).head(20)
    top.to_csv(OUTPUT_DIR / "top_depeg_proxy_events.csv")
    return top


def make_figures(
    df_main: pd.DataFrame,
    df_emp: pd.DataFrame,
    quantile_results: pd.DataFrame,
    regime_results: pd.DataFrame,
    local_projection_results: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Identification layers.
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(df_main.index, df_main["USDT_KP"], label="USDT_KP", linewidth=1.0)
    axes[0].plot(df_main.index, df_main["MKT_KP_EQ"], label="MKT_KP_EQ", linewidth=1.0)
    axes[0].set_title("USDT premium and market-common kimchi premium")
    axes[0].legend()
    axes[1].plot(df_main.index, df_main["RESIDUAL_OLS"], color="tab:purple", linewidth=1.0)
    axes[1].set_title("Residual after removing market-common premium")
    axes[2].plot(df_main.index, df_main["RESIDUAL_FINAL"], color="tab:green", linewidth=1.0)
    axes[2].set_title("Final residual after removing depeg proxy")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig01_residual_identification_layers.png", dpi=200)
    plt.close(fig)

    # Figure 2: The orange/blue gap the user asked about.
    gap_df = df_emp[["RESIDUAL_FINAL", "BTC_DOWNSIDE_24_lag1"]].dropna().copy()
    gap_df["RESIDUAL_Z"] = (
        gap_df["RESIDUAL_FINAL"] - gap_df["RESIDUAL_FINAL"].mean()
    ) / gap_df["RESIDUAL_FINAL"].std()
    gap_df["DOWNSIDE_Z"] = (
        gap_df["BTC_DOWNSIDE_24_lag1"] - gap_df["BTC_DOWNSIDE_24_lag1"].mean()
    ) / gap_df["BTC_DOWNSIDE_24_lag1"].std()
    gap_df["DOWNSIDE_MINUS_RESIDUAL_Z"] = gap_df["DOWNSIDE_Z"] - gap_df["RESIDUAL_Z"]
    gap_df["RESIDUAL_MINUS_DOWNSIDE_Z"] = gap_df["RESIDUAL_Z"] - gap_df["DOWNSIDE_Z"]
    gap_df.to_csv(OUTPUT_DIR / "standardized_downside_residual_gap.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(gap_df.index, gap_df["DOWNSIDE_Z"], color="tab:orange", label="BTC_DOWNSIDE_24_lag1 z")
    axes[0].plot(gap_df.index, gap_df["RESIDUAL_Z"], color="tab:blue", label="RESIDUAL_FINAL z")
    axes[0].set_title("Standardized downside stress vs final residual")
    axes[0].legend()
    axes[1].plot(
        gap_df.index,
        gap_df["DOWNSIDE_MINUS_RESIDUAL_Z"],
        color="tab:red",
        label="Downside z - Residual z",
    )
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_title("Gap: moments when downside rises while residual falls")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig02_downside_residual_gap.png", dpi=200)
    plt.close(fig)

    # Figure 3: Scatter and fitted line.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        df_emp["BTC_DOWNSIDE_24_lag1"],
        df_emp["RESIDUAL_FINAL"],
        alpha=0.35,
        s=18,
    )
    fit = np.polyfit(df_emp["BTC_DOWNSIDE_24_lag1"], df_emp["RESIDUAL_FINAL"], 1)
    xs = np.linspace(df_emp["BTC_DOWNSIDE_24_lag1"].min(), df_emp["BTC_DOWNSIDE_24_lag1"].max(), 100)
    ax.plot(xs, fit[0] * xs + fit[1], color="black", linewidth=1.2)
    ax.set_title("Residual response to lagged downside stress")
    ax.set_xlabel("BTC_DOWNSIDE_24_lag1")
    ax.set_ylabel("RESIDUAL_FINAL")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig03_scatter_downside_residual.png", dpi=200)
    plt.close(fig)

    # Figure 4: Quantile coefficient plot.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        quantile_results["q"],
        quantile_results["beta_downside_lag1"],
        marker="o",
        color="tab:blue",
        label="BTC downside coefficient",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Quantile regression: downside coefficient by residual quantile")
    ax.set_xlabel("Residual quantile")
    ax.set_ylabel("Coefficient on BTC_DOWNSIDE_24_lag1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig04_quantile_downside_coefficients.png", dpi=200)
    plt.close(fig)

    # Figure 5: Stress regime comparison.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        regime_results["q"],
        regime_results["beta_downside_normal"],
        marker="o",
        label="Normal VIX regime",
    )
    ax.plot(
        regime_results["q"],
        regime_results["beta_downside_stress_total"],
        marker="o",
        label="High VIX regime",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Downside effect: tail position vs high-stress regime")
    ax.set_xlabel("Residual quantile")
    ax.set_ylabel("Coefficient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig05_stress_regime_comparison.png", dpi=200)
    plt.close(fig)

    # Figure 6: Quantile local projection.
    fig, ax = plt.subplots(figsize=(8, 5))
    for q, subset in local_projection_results.groupby("q"):
        ax.plot(subset["h"], subset["beta_downside_lag1"], marker="o", label=f"q={q:g}")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Quantile local projection of downside shock")
    ax.set_xlabel("Horizon, hours")
    ax.set_ylabel("Coefficient on BTC_DOWNSIDE_24_lag1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig06_quantile_local_projection.png", dpi=200)
    plt.close(fig)

    # Figure 7: Depeg proxy magnitude.
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(df_main.index, df_main["USDT_BINANCE_CLOSE"], linewidth=1.0)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Binance stablecoin-pair price used for depeg proxy")
    axes[1].plot(df_main.index, df_main["DEPEG_GLOBAL"], color="tab:red", linewidth=1.0)
    axes[1].set_title("DEPEG_GLOBAL = abs(USDT_BINANCE_CLOSE - 1)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig07_depeg_proxy_magnitude.png", dpi=200)
    plt.close(fig)

    # Figure 8: Simple VAR IRF as an auxiliary dynamic visualization.
    # VAR requires level variables; BTC_DOWNSIDE_24 (not its lag) enters the system
    # so that the VAR's own lag structure captures the delayed dynamics correctly.
    var_df = df_emp[["BTC_DOWNSIDE_24", "RESIDUAL_FINAL"]].dropna().copy()
    var_df = (var_df - var_df.mean()) / var_df.std()
    try:
        var_res = VAR(var_df).fit(maxlags=1)
        irf = var_res.irf(12)
        fig = irf.plot(orth=False, impulse="BTC_DOWNSIDE_24", response="RESIDUAL_FINAL")
        fig.set_size_inches(7, 5)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig08_var_irf_downside_to_residual.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        (FIGURE_DIR / "fig08_var_irf_error.txt").write_text(str(exc), encoding="utf-8")


def run_demand_channel_tests(
    df_main: pd.DataFrame,
    available_demand_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run additional tests for trading-demand vs dollar/hedge-demand channels."""
    base_controls = [
        "RESIDUAL_FINAL_lag1",
        "BTC_DOWNSIDE_24_lag1",
        "DXY_ret_lag1",
        "USDKRW_ret_lag1",
    ]

    usable_cols = ["RESIDUAL_FINAL", *base_controls, *available_demand_cols]
    df_demand = df_main.dropna(subset=usable_cols).copy()

    if df_demand.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    rows = []

    # Baseline model: downside + dollar controls only.
    formula_base = (
        "RESIDUAL_FINAL ~ RESIDUAL_FINAL_lag1 "
        "+ BTC_DOWNSIDE_24_lag1 + DXY_ret_lag1 + USDKRW_ret_lag1"
    )
    model_base = smf.ols(formula_base, data=df_demand).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5},
    )
    rows.append(
        {
            "spec": "baseline_downside_dollar_controls",
            "n": int(model_base.nobs),
            "r2": model_base.rsquared,
            "aic": model_base.aic,
            "btc_downside_coef": model_base.params.get("BTC_DOWNSIDE_24_lag1", np.nan),
            "btc_downside_pvalue": model_base.pvalues.get("BTC_DOWNSIDE_24_lag1", np.nan),
            "added_vars": "",
        }
    )

    # Full available demand-channel model.
    if available_demand_cols:
        formula_full = formula_base + " + " + " + ".join(available_demand_cols)
        model_full = smf.ols(formula_full, data=df_demand).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 5},
        )
        rows.append(
            {
                "spec": "full_available_demand_channels",
                "n": int(model_full.nobs),
                "r2": model_full.rsquared,
                "aic": model_full.aic,
                "btc_downside_coef": model_full.params.get("BTC_DOWNSIDE_24_lag1", np.nan),
                "btc_downside_pvalue": model_full.pvalues.get("BTC_DOWNSIDE_24_lag1", np.nan),
                "added_vars": ", ".join(available_demand_cols),
            }
        )

    # Curated demand-channel model: one representative per correlated group.
    # The full model suffers from multicollinearity (OI_lag1/OI_VALUE_lag1 highly
    # correlated; positioning variables overlap).  Selecting one representative per
    # group gives a parsimonious but channel-spanning specification where BTC downside
    # significance can be evaluated without inflated VIF.
    _CURATED_GROUPS: list[tuple[str, list[str]]] = [
        ("usdt_vol",    ["USDT_VOL_RATIO_LOG"]),
        ("coin_vol",    ["COIN_VOL_RATIO_LOG_MEAN"]),
        ("vol_disp",    ["VOL_RATIO_DISPERSION"]),
        ("funding",     ["FUNDING_lag1"]),
        ("oi_level",    ["OI_lag1", "OI_VALUE_lag1"]),
        ("positioning", ["TOPTRADER_POSITION_LS_lag1", "ACCOUNT_LS_lag1",
                         "TOPTRADER_ACCOUNT_LS_lag1"]),
    ]
    curated_cols: list[str] = []
    for _, candidates in _CURATED_GROUPS:
        for cand in candidates:
            if cand in available_demand_cols:
                curated_cols.append(cand)
                break

    if len(curated_cols) >= 2:
        formula_curated = formula_base + " + " + " + ".join(curated_cols)
        model_curated = smf.ols(formula_curated, data=df_demand).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 5},
        )
        rows.append(
            {
                "spec": "curated_demand_channels",
                "n": int(model_curated.nobs),
                "r2": model_curated.rsquared,
                "aic": model_curated.aic,
                "btc_downside_coef": model_curated.params.get("BTC_DOWNSIDE_24_lag1", np.nan),
                "btc_downside_pvalue": model_curated.pvalues.get("BTC_DOWNSIDE_24_lag1", np.nan),
                "added_vars": ", ".join(curated_cols),
            }
        )

    # One-channel-at-a-time tests.
    for channel in available_demand_cols:
        formula = formula_base + f" + {channel}"
        model = smf.ols(formula, data=df_demand).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 5},
        )
        rows.append(
            {
                "spec": f"add_{channel}",
                "n": int(model.nobs),
                "r2": model.rsquared,
                "aic": model.aic,
                "btc_downside_coef": model.params.get("BTC_DOWNSIDE_24_lag1", np.nan),
                "btc_downside_pvalue": model.pvalues.get("BTC_DOWNSIDE_24_lag1", np.nan),
                "added_vars": channel,
                "channel_coef": model.params.get(channel, np.nan),
                "channel_pvalue": model.pvalues.get(channel, np.nan),
            }
        )

    comparison = pd.DataFrame(rows)

    # Quantile version for the strongest available trading/liquidity proxy set.
    quantile_rows = []
    if available_demand_cols:
        formula_q = formula_base + " + " + " + ".join(available_demand_cols)
    else:
        formula_q = formula_base

    # Bootstrap only for BTC_DOWNSIDE_24_lag1 (key economic question: does it
    # survive channel controls at different quantiles?).
    _boot_coefs_demand = ["BTC_DOWNSIDE_24_lag1"]

    for q in [0.10, 0.50, 0.90]:
        model = smf.quantreg(formula_q, df_demand).fit(q=q, max_iter=5000)
        boot = _block_bootstrap_qr_pvalues(formula_q, df_demand, q, _boot_coefs_demand)
        row = {
            "q": q,
            "n": int(model.nobs),
            "beta_downside_lag1": model.params.get("BTC_DOWNSIDE_24_lag1", np.nan),
            "p_downside_lag1": model.pvalues.get("BTC_DOWNSIDE_24_lag1", np.nan),
            "p_downside_lag1_boot": boot["pvalues"][0],
            "bse_downside_lag1_boot": boot["bse"][0],
            "beta_dxy_lag1": model.params.get("DXY_ret_lag1", np.nan),
            "p_dxy_lag1": model.pvalues.get("DXY_ret_lag1", np.nan),
            "beta_fx_lag1": model.params.get("USDKRW_ret_lag1", np.nan),
            "p_fx_lag1": model.pvalues.get("USDKRW_ret_lag1", np.nan),
            "boot_n_success": boot["n_success"],
        }
        for channel in available_demand_cols:
            row[f"beta_{channel}"] = model.params.get(channel, np.nan)
            row[f"p_{channel}"] = model.pvalues.get(channel, np.nan)
        quantile_rows.append(row)

    quantile = pd.DataFrame(quantile_rows)

    # Compact interpretation table: lower values mean weaker evidence.
    interpretation_rows = []
    for channel in available_demand_cols:
        one = comparison.loc[comparison["spec"] == f"add_{channel}"]
        if one.empty:
            continue
        interpretation_rows.append(
            {
                "channel": channel,
                "channel_pvalue_ols": one["channel_pvalue"].iloc[0],
                "downside_pvalue_after_channel": one["btc_downside_pvalue"].iloc[0],
                "channel_significant_5pct": one["channel_pvalue"].iloc[0] < 0.05,
                "downside_survives_5pct": one["btc_downside_pvalue"].iloc[0] < 0.05,
            }
        )
    interpretation = pd.DataFrame(interpretation_rows)

    return comparison, quantile, interpretation


def save_outputs(
    df_main: pd.DataFrame,
    quantile_results: pd.DataFrame,
    regime_results: pd.DataFrame,
    local_projection_results: pd.DataFrame,
    demand_comparison: pd.DataFrame,
    demand_quantile: pd.DataFrame,
    demand_interpretation: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df_main.index.name = "datetime_utc"
    df_main.to_csv(OUTPUT_DIR / "baseline_dataset_with_residual_final.csv")
    quantile_results.to_csv(OUTPUT_DIR / "corrected_quantile_tail_results.csv", index=False)
    regime_results.to_csv(OUTPUT_DIR / "corrected_regime_tail_compare.csv", index=False)
    local_projection_results.to_csv(
        OUTPUT_DIR / "corrected_quantile_local_projection.csv",
        index=False,
    )
    demand_comparison.to_csv(OUTPUT_DIR / "demand_channel_ols_comparison.csv", index=False)
    demand_quantile.to_csv(OUTPUT_DIR / "demand_channel_quantile_results.csv", index=False)
    demand_interpretation.to_csv(
        OUTPUT_DIR / "demand_channel_interpretation_flags.csv",
        index=False,
    )


def main() -> None:
    df_raw = load_master_dataset()
    df_main, model_market, model_depeg = identify_final_residual(df_raw)
    df_main = add_main_identification_variables(df_main)
    df_main, derivatives_diagnostics = collect_and_merge_derivatives_data(df_main)
    df_main, available_demand_cols = add_demand_channel_proxies(df_main)
    df_emp = get_empirical_sample(df_main)

    model_main_ols = run_main_ols(df_emp)
    quantile_results = run_quantile_tail_tests(df_emp)
    regime_results = run_stress_regime_comparison(df_emp)
    local_projection_results = run_quantile_local_projection(df_emp)
    demand_comparison, demand_quantile, demand_interpretation = run_demand_channel_tests(
        df_main,
        available_demand_cols,
    )

    save_outputs(
        df_main,
        quantile_results,
        regime_results,
        local_projection_results,
        demand_comparison,
        demand_quantile,
        demand_interpretation,
    )
    save_regression_tables(
        model_main_ols,
        quantile_results,
        regime_results,
        local_projection_results,
    )
    top_depeg_events = save_depeg_event_table(df_main)
    make_figures(df_main, df_emp, quantile_results, regime_results, local_projection_results)

    print("\n=== Corrected Main Empirical Pipeline ===")
    print(f"input file : {INPUT_FILE}")
    print(f"output dir : {OUTPUT_DIR}")
    print(f"sample     : {df_emp.index.min()} ~ {df_emp.index.max()}")
    print(f"n          : {len(df_emp)}")

    print("\n[Step 1] USDT_KP ~ MKT_KP_EQ")
    print(f"R-squared  : {model_market.rsquared:.6f}")
    print(model_market.params.to_string())

    print("\n[Step 2] RESIDUAL_OLS ~ DEPEG_GLOBAL")
    print(f"R-squared  : {model_depeg.rsquared:.6f}")
    print(model_depeg.params.to_string())

    print("\n[Main HAC OLS]")
    ols_table = pd.DataFrame(
        {
            "coef": model_main_ols.params,
            "pvalue": model_main_ols.pvalues,
        }
    )
    print(ols_table.round(6).to_string())

    print("\n[Quantile Tail Results]")
    print("  Columns *_boot = block-bootstrap (block=4) p-values / SEs, corrected for serial correlation.")
    print(quantile_results.round(6).to_string(index=False))

    print("\n[Stress Regime Comparison]")
    print("  p_*_boot = block-bootstrap p-values for interaction term.")
    print(regime_results.round(6).to_string(index=False))

    print("\n[Quantile Local Projection]")
    print("  p_*_boot = block-bootstrap p-values, corrected for serial correlation.")
    print(local_projection_results.round(6).to_string(index=False))

    print("\n[Demand Channel Variables Available]")
    if available_demand_cols:
        for col in available_demand_cols:
            print(f"- {col}")
    else:
        print("- None")
    missing_optional = [
        c
        for c in [
            "FUNDING_lag1",
            "OI_lag1",
            "OI_VALUE_lag1",
            "OI_RET_lag1",
            "OI_VALUE_RET_lag1",
            "TOPTRADER_ACCOUNT_LS_lag1",
            "TOPTRADER_POSITION_LS_lag1",
            "ACCOUNT_LS_lag1",
            "TAKER_LONG_SHORT_VOL_RATIO_lag1",
            "LIQUIDATION_lag1",
        ]
        if c not in df_main.columns
    ]
    if missing_optional:
        print("\nOptional funding/OI/liquidation columns not found:")
        for col in missing_optional:
            print(f"- {col}")

    print("\n[Derivatives Collection Diagnostics]")
    for key, value in derivatives_diagnostics.items():
        print(f"{key}: {value}")

    print("\n[Demand Channel OLS Comparison]")
    if demand_comparison.empty:
        print("No usable demand-channel sample.")
    else:
        print(demand_comparison.round(6).to_string(index=False))

    print("\n[Demand Channel Quantile Results]")
    if demand_quantile.empty:
        print("No usable demand-channel quantile results.")
    else:
        print(demand_quantile.round(6).to_string(index=False))

    print("\n[Demand Channel Interpretation Flags]")
    if demand_interpretation.empty:
        print("No demand-channel flags available.")
    else:
        print(demand_interpretation.round(6).to_string(index=False))

    print("\n[Top Depeg Proxy Events]")
    print(top_depeg_events.round(6).to_string())

    print("\nSaved files:")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"- {path}")
    for path in sorted((OUTPUT_DIR / "tables").glob("*")):
        print(f"- {path}")
    for path in sorted(FIGURE_DIR.glob("*")):
        print(f"- {path}")


if __name__ == "__main__":
    main()
