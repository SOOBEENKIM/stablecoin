"""Chronological conditional-fit pilot, NOT a future-price forecasting test.

Read archived raw files without changing them. Rebuild conditional clock panels,
fit all nuisance models on the past, and retain all specified clock scenarios.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / "research"
RAW = ARCHIVE / "data/raw data"
OUT = HERE / "results"
COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
SCENARIOS = ["clock_UTC_quote", "clock_NY_quote", "clock_FXUTC_USNY_quote",
             "clock_FXKST_USNY_quote", "clock_NY_macro_end1h_quote"]
MONTHS = ["2026-01", "2026-02", "2026-03"]
MODELS = ["sequential_ols", "joint_ols", "lightgbm", "linear_plus_lightgbm"]
LFS_PATHS = {"option1_factorial_20260910/evaluated_predictions.npz",
             "option1_model_followup_20260910/evaluated_predictions.npz"}
PARAMS = dict(objective="regression", n_estimators=100, learning_rate=0.03,
              max_depth=3, num_leaves=7, min_child_samples=100, reg_lambda=1.0,
              random_state=20260924, n_jobs=1, verbosity=-1,
              deterministic=True, force_col_wise=True)
SPEC = importlib.util.spec_from_file_location(
    "preserved_reanalysis", ARCHIVE / "reanalysis_20260909/recalculate.py")
LEGACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEGACY)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_archive():
    manifest = json.loads((ROOT / "provenance/archive_manifest.json").read_text())
    verified = 0
    pointers = []
    for name, entry in manifest["files"].items():
        path = ARCHIVE / name
        if name in LFS_PATHS and path.stat().st_size < 1024:
            pointer = path.read_text()
            assert pointer.startswith("version https://git-lfs.github.com/spec/v1\n")
            assert "oid sha256:" + entry["sha256"] in pointer
            assert "size " + str(entry["bytes"]) in pointer
            pointers.append(name)
            continue
        assert path.stat().st_size == entry["bytes"], name
        assert sha(path) == entry["sha256"], name
        verified += 1
    context = json.loads((ROOT / "provenance/context_manifest.json").read_text())
    for name, entry in context.items():
        assert sha(ROOT / name) == entry["sha256"], name
    return dict(verified_archive_files=verified, verified_context_files=len(context),
                validated_lfs_pointers=pointers,
                lfs_payloads_used=False,
                note="LFS pointer metadata checked; absent payloads are not claimed verified")


def build_panel(name):
    fx_zone, us_zone, delay, corrected = LEGACY.SCENARIOS[name]
    assert corrected
    b = LEGACY.read_panel(RAW / "binance_1h_2025-06-01_2026-03-19.csv")
    u = LEGACY.read_panel(RAW / "upbit_1h_2025-06-01_2026-03-19.csv")
    assert b.index.equals(u.index)
    assert b.index.is_unique and u.index.is_unique
    d = b.join(u)
    d.index += pd.Timedelta(hours=1)
    assert (d.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    age_notes = {}
    for col, file, zone in [("USDKRW", "USDKRW.csv", fx_zone),
                            ("DXY", "DXY.csv", us_zone),
                            ("VIXY", "VIXY.csv", us_zone)]:
        source = LEGACY.macro(file, zone, delay)
        values, age = LEGACY.past_observation(source, d.index, "59min59s")
        d[col] = values
        age_notes[col] = dict(observations=int(values.notna().sum()),
                              max_age_minutes=float(age.max()))
    metrics = LEGACY.read_panel(ARCHIVE / "corrected_outputs/binance_btcusdt_data_vision_metrics.csv")
    ls = metrics.ACCOUNT_LS.copy()
    ls.index += pd.Timedelta(minutes=5)
    d["account_ls"], age = LEGACY.past_observation(ls, d.index, "10min")
    age_notes["account_ls"] = dict(observations=int(d.account_ls.notna().sum()),
                                    max_age_minutes=float(age.max()), delay_minutes=5)
    q, fx = d.USDT_BINANCE_CLOSE, d.USDKRW
    assert (q.dropna() > 0).all() and (fx.dropna() > 0).all()
    # q is USDCUSDT, hence the USD price uses the explicit USDC/USD=1 assumption.
    y = d.USDT_UPBIT_CLOSE * q / fx - 1
    kp = pd.DataFrame({c: d[c + "_UPBIT_CLOSE"] * q /
                       (fx * d[c + "_BINANCE_CLOSE"]) - 1 for c in COINS})
    m = kp.mean(axis=1, skipna=False)
    g = (q - 1).abs()
    ret = np.log(d.BTC_BINANCE_CLOSE).diff()
    p = pd.DataFrame(dict(y=y, m=m, g=g,
        downside=ret.clip(upper=0).pow(2).rolling(24, min_periods=24).mean(),
        btc_vol=ret.rolling(24, min_periods=24).std(ddof=1), btc_ret=ret,
        dxy_ret=np.log(d.DXY).diff(), fx_ret=np.log(fx).diff(),
        vixy_ret=np.log(d.VIXY).diff(), account_ls=d.account_ls), index=d.index)
    assert "e" not in p.columns  # Do not import whole-sample fitted residuals.
    prior = LEGACY.read_panel(ARCHIVE / "reanalysis_20260909" / (name + "_panel.csv"))
    assert p.index.equals(prior.index)
    for col in p:
        old_col = "vix_ret" if col == "vixy_ret" else col
        np.testing.assert_allclose(p[col], prior[old_col], atol=1e-12, rtol=1e-10,
                                   equal_nan=True, err_msg=name + ":" + col)
    p[["y", "m", "g"]] *= 10000
    meta = dict(scenario=name, timezone_confirmed=False, fx_zone_assumption=fx_zone,
                us_zone_assumption=us_zone, macro_delay_hours=delay,
                source_ages=age_notes, usdc_usd_assumption=1.0,
                g_definition="absolute USDCUSDT deviation, bp; not direct USDT/USD peg",
                panel_reproduces_archived_nonresidual_columns=True)
    return p, meta


def available_pairs(p, columns, h):
    origins = np.isfinite(p[columns]).all(axis=1).to_numpy()
    future = p.index.get_indexer(p.index + pd.Timedelta(hours=h))
    target_ok = np.isfinite(p[["y", "m", "g"]]).all(axis=1).to_numpy()
    paired = (future >= 0) & origins
    paired[future >= 0] &= target_ok[future[future >= 0]]
    idx = p.index[paired]
    return dict(n_pairs=len(idx), days=int(idx.normalize().nunique()),
                evaluation_pairs=int((idx >= pd.Timestamp("2026-01-01", tz="UTC")).sum()))


def predict_models(train, test):
    x = train[["m", "g"]].to_numpy()
    xt = test[["m", "g"]].to_numpy()
    y = train.y.to_numpy()
    z = np.column_stack([np.ones(len(x)), x])
    zt = np.column_stack([np.ones(len(xt)), xt])
    coef = np.linalg.lstsq(z, y, rcond=None)[0]
    joint_train, joint_test = z @ coef, zt @ coef
    m = np.column_stack([np.ones(len(x)), x[:, 0]])
    mt = np.column_stack([np.ones(len(xt)), xt[:, 0]])
    first = np.linalg.lstsq(m, y, rcond=None)[0]
    g = np.column_stack([np.ones(len(x)), x[:, 1]])
    gt = np.column_stack([np.ones(len(xt)), xt[:, 1]])
    second = np.linalg.lstsq(g, y - m @ first, rcond=None)[0]
    plain = lgb.LGBMRegressor(**PARAMS).fit(x, y)
    correction = lgb.LGBMRegressor(**PARAMS).fit(x, y - joint_train)
    predictions = dict(sequential_ols=mt @ first + gt @ second,
                       joint_ols=joint_test, lightgbm=plain.predict(xt),
                       linear_plus_lightgbm=joint_test + correction.predict(xt))
    # Independent coefficient/projection check on the actual training sample.
    assert np.linalg.matrix_rank(z) == z.shape[1]
    np.testing.assert_allclose(z.T @ (y - joint_train), 0, atol=1e-5, rtol=0)
    assert all(np.isfinite(v).all() for v in predictions.values())
    return predictions, dict(joint_ols=coef.tolist(), first_stage=first.tolist(),
                             second_stage=second.tolist(), train_rows=len(train))


def metrics(frame):
    e = frame.residual_bp.to_numpy()
    s = pd.Series(e, index=pd.DatetimeIndex(frame.time))
    lag = s.reindex(s.index - pd.Timedelta(hours=1))
    lag.index = s.index
    paired = pd.concat([s.rename("now"), lag.rename("lag")], axis=1).dropna()
    q = np.quantile(e, [.1, .5, .9])
    return dict(n=len(e), days=int(s.index.normalize().nunique()),
                mae_bp=float(np.abs(e).mean()), rmse_bp=float(np.sqrt(np.mean(e ** 2))),
                residual_mean_bp=float(e.mean()), residual_q10_bp=float(q[0]),
                residual_q50_bp=float(q[1]), residual_q90_bp=float(q[2]),
                residual_lag1h_corr=float(paired.now.corr(paired.lag)),
                exact_1h_pairs=len(paired))


def main():
    before = audit_archive()
    OUT.mkdir(parents=True, exist_ok=True)
    availability, metadata, results, fits = [], [], [], []
    for name in SCENARIOS:
        panel, meta = build_panel(name)
        metadata.append(meta)
        valid = np.isfinite(panel[["y", "m", "g"]]).all(axis=1)
        base = ["y", "m", "g", "downside", "btc_vol", "btc_ret", "account_ls"]
        for h in [1, 6, 12]:
            for scope, cols in [("crypto_and_positioning", base),
                                ("plus_all_macro", base + ["fx_ret", "dxy_ret", "vixy_ret"])]:
                availability.append(dict(scenario=name, horizon_hours=h, scope=scope,
                    panel_hours=len(panel), conditional_fit_rows=int(valid.sum()),
                    **available_pairs(panel, cols, h)))
        for month in MONTHS:
            start = pd.Timestamp(month + "-01", tz="UTC")
            end = start + pd.offsets.MonthBegin(1)
            train = panel.loc[valid & (panel.index < start)]
            test = panel.loc[valid & (panel.index >= start) & (panel.index < end)]
            assert len(train) >= 500 and len(test) >= 20
            assert train.index.max() < start <= test.index.min()
            predicted, coefficients = predict_models(train, test)
            fits.append(dict(scenario=name, evaluation_month=month,
                cutoff=str(start), train_last=str(train.index.max()),
                train_first=str(train.index.min()), test_rows=len(test), **coefficients))
            for model in MODELS:
                results.append(pd.DataFrame(dict(time=test.index, scenario=name,
                    evaluation_month=month, fit_cutoff=start, model=model,
                    observed_bp=test.y.to_numpy(), fitted_bp=predicted[model],
                    residual_bp=test.y.to_numpy() - predicted[model])))
        print("Completed conditional-fit pilot: " + name, flush=True)
    predictions = pd.concat(results, ignore_index=True)
    scores = []
    for keys, group in predictions.groupby(["scenario", "model", "evaluation_month"], sort=False):
        scores.append(dict(zip(["scenario", "model", "period"], keys), **metrics(group)))
    for keys, group in predictions.groupby(["scenario", "model"], sort=False):
        scores.append(dict(zip(["scenario", "model"], keys), period="all", **metrics(group)))
    score = pd.DataFrame(scores)
    assert len(predictions.groupby(["scenario", "evaluation_month", "time"]).size().unique()) == 1
    assert predictions.groupby(["scenario", "evaluation_month", "time"]).size().iloc[0] == len(MODELS)
    predictions.to_csv(OUT / "conditional_fit_predictions.csv.gz", index=False,
                       compression={"method": "gzip", "mtime": 0})
    score.to_csv(OUT / "conditional_fit_scores.csv", index=False)
    pd.DataFrame(availability).to_csv(OUT / "data_availability.csv", index=False)
    (OUT / "clock_scenarios.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / "fold_fits.json").write_text(json.dumps(fits, indent=2) + "\n")
    after = audit_archive()
    assert before == after
    inputs = [RAW / name for name in ["binance_1h_2025-06-01_2026-03-19.csv",
        "upbit_1h_2025-06-01_2026-03-19.csv", "USDKRW.csv", "DXY.csv", "VIXY.csv"]]
    inputs += [ARCHIVE / "corrected_outputs/binance_btcusdt_data_vision_metrics.csv",
               ARCHIVE / "reanalysis_20260909/recalculate.py", HERE / "PROTOCOL_KO.md", Path(__file__).resolve()]
    manifest = dict(stage="A conditional fit only; stage B not run", publication_ready=False,
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        protocol_sha256=sha(HERE / "PROTOCOL_KO.md"),
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        model_parameters=PARAMS, months=MONTHS, scenarios=SCENARIOS,
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, lightgbm=lgb.__version__),
        archive_check=after,
        checks=["raw quote formula and rebuilt panel match archived corrected columns",
                "completed candles, backward macro joins, delayed positioning availability",
                "whole-sample archived residual never used",
                "each fit ends before evaluation month; no evaluation outcomes used in fitting",
                "joint OLS training normal equations checked",
                "all models evaluated on identical observations within each clock scenario",
                "residual autocorrelation uses exact one-hour pairs, not adjacent surviving rows",
                "original archive and context bytes preserved"],
        limitations=["FX and macro timezone assumptions unconfirmed",
                     "USDC/USD=1 assumption; global control is a relative pair proxy",
                     "contemporaneous common factors are observed at evaluation time",
                     "conditional-fit scores are not future forecasting scores",
                     "historical period previously explored; not independent confirmation",
                     "no channel significance or model superiority test in stage A"])
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(score.loc[score.period == "all", ["scenario", "model", "n", "mae_bp", "rmse_bp"]].to_string(index=False))
    print("Original archive preserved. Stage B remains unrun.", flush=True)


if __name__ == "__main__":
    main()
