"""Within-definition, exact-six-hour forecasts with positioning ablations."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pipeline as study

BASE = ["e_origin", "m", "g", "downside", "btc_vol", "btc_ret"]
EXTRA = ["account_ls", "downside_x_ls"]


def loss(y, prediction, q):
    e = np.asarray(y) - np.asarray(prediction)
    return np.maximum(q * e, (q - 1) * e)


def grid(algorithm):
    if algorithm == "linear_qr":
        return [dict(alpha=.01), dict(alpha=.1)]
    return [dict(n_estimators=n, min_child_samples=leaf) for n in [100, 300] for leaf in [50, 100]]


def fit(train, columns, q, algorithm, params):
    if algorithm == "linear_qr":
        model = make_pipeline(StandardScaler(), QuantileRegressor(quantile=q, solver="highs", **params))
    else:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, num_leaves=7, max_depth=3,
            learning_rate=.03, reg_lambda=1., n_jobs=1, random_state=20260924, verbosity=-1,
            deterministic=True, force_col_wise=True, **params)
    model.fit(train[columns].to_numpy(), train.target.to_numpy())
    return model


def training_before(data, cutoff):
    train = data.loc[(data.time < cutoff) & (data.target_time < cutoff)]
    assert len(train) >= 100
    assert train.target_time.max() < cutoff
    return train


def run(frames):
    records, selections, validation = [], [], []
    data = frames.loc[frames.h == 6].copy()
    data["downside_x_ls"] = data.downside * data.account_ls
    data = data.dropna(subset=BASE + EXTRA + ["target"])
    for method in study.METHODS:
        d = data.loc[data.method == method].sort_values("time")
        cache = {}
        for month in study.pilot.MONTHS:
            start = pd.Timestamp(month + "-01", tz="UTC")
            end = start + pd.offsets.MonthBegin(1)
            inner = [(start - pd.DateOffset(months=k)).strftime("%Y-%m") for k in [2, 1]]
            test = d.loc[(d.time >= start) & (d.time < end)]
            train = training_before(d, start)
            assert len(test) > 20
            for info, columns in [("without_positioning", BASE), ("with_positioning", BASE + EXTRA)]:
                for algorithm in ["linear_qr", "quantile_lightgbm"]:
                    for q in study.QUANTILES:
                        scores = []
                        for ci, params in enumerate(grid(algorithm)):
                            ls, n = 0., 0
                            for vm in inner:
                                key = (vm, info, algorithm, q, ci)
                                if key not in cache:
                                    vstart = pd.Timestamp(vm + "-01", tz="UTC")
                                    vend = vstart + pd.offsets.MonthBegin(1)
                                    vt = training_before(d, vstart)
                                    # Labels straddling the current outer cutoff cannot tune it.
                                    ve = d.loc[(d.time >= vstart) & (d.time < vend) & (d.target_time < start)]
                                    # Last validation month changes cutoff only in the final six hours;
                                    # caching keys include the outer cutoff below if any such rows exist.
                                    all_ve = d.loc[(d.time >= vstart) & (d.time < vend)]
                                    fitted = fit(vt, columns, q, algorithm, params)
                                    preds = fitted.predict(all_ve[columns].to_numpy())
                                    cache[key] = (all_ve.copy(), preds, vt.target_time.max(), len(vt))
                                ve_all, preds_all, last_target, train_n = cache[key]
                                allowed = (ve_all.target_time < start).to_numpy()
                                values = loss(ve_all.target.to_numpy()[allowed], preds_all[allowed], q)
                                ls += values.sum()
                                n += len(values)
                                validation.append(dict(method=method, evaluation_month=month,
                                    validation_month=vm, info=info, algorithm=algorithm, q=q,
                                    candidate=ci, params=json.dumps(params), n=len(values),
                                    loss_sum=float(values.sum()), train_target_last=str(last_target), train_n=train_n,
                                    validation_target_last=str(ve_all.loc[allowed, "target_time"].max())))
                            scores.append((ls / n, ci, params, n))
                        best_loss, ci, params, validation_n = min(scores, key=lambda x: (x[0], x[1]))
                        fitted = fit(train, columns, q, algorithm, params)
                        pred = fitted.predict(test[columns].to_numpy())
                        assert np.isfinite(pred).all()
                        selections.append(dict(method=method, month=month, info=info, algorithm=algorithm,
                            q=q, candidate=ci, params=params, validation_loss=best_loss,
                            validation_n=validation_n, train_n=len(train), test_n=len(test),
                            train_target_last=str(train.target_time.max()), cutoff=str(start)))
                        records.append(pd.DataFrame(dict(time=test.time.to_numpy(),
                            target_time=test.target_time.to_numpy(), month=month, method=method,
                            info=info, algorithm=algorithm, q=q, target=test.target.to_numpy(), prediction=pred,
                            pinball=loss(test.target.to_numpy(), pred, q))))
    predictions = pd.concat(records, ignore_index=True)
    assert (predictions.groupby(["method", "time", "q"]).size() == 4).all()
    assert predictions.groupby(["method", "time", "q"]).target.nunique().eq(1).all()
    scores = []
    keys = ["method", "info", "algorithm", "q"]
    for key, d in predictions.groupby(keys, sort=False):
        for period, f in [("all", d)] + list(d.groupby("month", sort=False)):
            scores.append(dict(zip(keys, key), period=period, n=len(f), days=int(pd.to_datetime(f.time, utc=True).dt.normalize().nunique()),
                               pinball=float(f.pinball.mean()), below_rate=float((f.target < f.prediction).mean())))
    crossing = []
    for key, d in predictions.groupby(["method", "info", "algorithm"], sort=False):
        w = d.pivot(index="time", columns="q", values="prediction")
        crossing.append(dict(zip(["method", "info", "algorithm"], key), n=len(w),
                             crossing_rate=float(((w[.1] > w[.5]) | (w[.5] > w[.9])).mean())))
    return predictions, pd.DataFrame(scores), selections, pd.DataFrame(validation), pd.DataFrame(crossing)


def comparisons(predictions):
    records = []
    for (method, q), group in predictions.groupby(["method", "q"], sort=False):
        errors = group.pivot(index="time", columns=["algorithm", "info"], values="pinball").sort_index()
        errors.index = pd.DatetimeIndex(errors.index)
        assert errors.notna().all().all()
        pairs = []
        for algorithm in ["linear_qr", "quantile_lightgbm"]:
            pairs.append(("add_positioning", algorithm, (algorithm, "with_positioning"), (algorithm, "without_positioning")))
        for info in ["without_positioning", "with_positioning"]:
            pairs.append(("ML_minus_linear", info, ("quantile_lightgbm", info), ("linear_qr", info)))
        for block in [3, 7]:
            calendar, weights = study.calendar_counts(errors.index, block, np.random.default_rng(20260924 + block), 2000)
            counts = errors.iloc[:, 0].groupby(errors.index.normalize()).size().reindex(calendar, fill_value=0).to_numpy()
            daily = errors.groupby(errors.index.normalize()).sum().reindex(calendar, fill_value=0)
            sums = (weights @ daily.to_numpy()) / (weights @ counts)[:, None]
            for contrast, conditioning, a, b in pairs:
                ai, bi = daily.columns.get_loc(a), daily.columns.get_loc(b)
                draws = sums[:, ai] - sums[:, bi]
                delta = float(errors[a].mean() - errors[b].mean())
                records.append(dict(method=method, q=q, contrast=contrast, conditioning=conditioning,
                    block_days=block, delta_pinball_bp=delta, relative_change_pct=100 * delta / errors[b].mean(),
                    ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975)),
                    n=len(errors), repetitions=2000))
    return pd.DataFrame(records)
