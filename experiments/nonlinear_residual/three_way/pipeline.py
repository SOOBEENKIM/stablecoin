"""Past-fitted residual definitions and identical exact-clock follow-up samples."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "expanded"))
import run_expanded as expanded
pilot = expanded.pilot
ROOT = pilot.ROOT
OUT = HERE / "results"
MONTHS = pd.period_range("2025-08", "2026-03", freq="M").astype(str).tolist()
METHODS = ["linear", "pca", "ml", "joint_linear"]
HORIZONS = [1, 6, 12]
QUANTILES = [.1, .5, .9]
KCOLS = ["kp_" + coin for coin in pilot.COINS]
STATES = ["downside", "btc_vol", "btc_ret", "account_ls", "dxy_ret", "fx_ret", "vixy_ret"]
SPECS = {
    "core": {"M0": ["e_origin", "downside"],
             "M1": ["e_origin", "downside", "btc_vol", "btc_ret"],
             "M2": ["e_origin", "downside", "btc_vol", "btc_ret", "account_ls"]},
    "full_macro": {"M0": ["e_origin", "downside", "dxy_ret", "fx_ret"],
                   "M1": ["e_origin", "downside", "dxy_ret", "fx_ret", "btc_vol", "btc_ret", "vixy_ret"],
                   "M2": ["e_origin", "downside", "dxy_ret", "fx_ret", "btc_vol", "btc_ret", "vixy_ret", "account_ls"]}}
TERMS = ["e_origin", "downside", "btc_vol", "account_ls"]


def build_panel(scenario):
    p, meta = pilot.build_panel(scenario)
    b = pilot.LEGACY.read_panel(pilot.RAW / "binance_1h_2025-06-01_2026-03-19.csv")
    u = pilot.LEGACY.read_panel(pilot.RAW / "upbit_1h_2025-06-01_2026-03-19.csv")
    b.index += pd.Timedelta(hours=1)
    u.index += pd.Timedelta(hours=1)
    fx_zone, _, delay, _ = pilot.LEGACY.SCENARIOS[scenario]
    fx, _ = pilot.LEGACY.past_observation(pilot.LEGACY.macro("USDKRW.csv", fx_zone, delay),
                                        p.index, "59min59s")
    for coin in pilot.COINS:
        p["kp_" + coin] = 10000 * (u[coin + "_UPBIT_CLOSE"] * b.USDT_BINANCE_CLOSE /
                                   (fx * b[coin + "_BINANCE_CLOSE"]) - 1)
    np.testing.assert_allclose(p[KCOLS].mean(axis=1, skipna=False), p.m,
                               rtol=1e-11, atol=1e-9, equal_nan=True)
    return p, meta


class PCASequential:
    def fit(self, train, cutoff):
        self.train = train.loc[train.index < cutoff]
        assert len(self.train) >= 100 and self.train.index.max() < cutoff
        k = self.train[KCOLS]
        self.mean, self.sd = k.mean().to_numpy(), k.std(ddof=1).to_numpy()
        z = (k.to_numpy() - self.mean) / self.sd
        ev, vec = np.linalg.eigh(np.cov(z, rowvar=False))
        self.loading = vec[:, -1]
        if self.loading.sum() < 0:
            self.loading *= -1
        self.share = float(ev[-1] / ev.sum())
        pc = z @ self.loading
        x = np.column_stack([np.ones(len(k)), pc])
        self.first = np.linalg.lstsq(x, self.train.y, rcond=None)[0]
        g = np.column_stack([np.ones(len(k)), self.train.g])
        self.second = np.linalg.lstsq(g, self.train.y - x @ self.first, rcond=None)[0]
        return self

    def predict(self, frame):
        pc = ((frame[KCOLS].to_numpy() - self.mean) / self.sd) @ self.loading
        return (np.column_stack([np.ones(len(frame)), pc]) @ self.first +
                np.column_stack([np.ones(len(frame)), frame.g]) @ self.second)


def select_ml(panel):
    data = panel.dropna(subset=["y", "m", "g"] + KCOLS)
    grid = [c for c in expanded.candidates() if c["family"] in expanded.NONLINEAR]
    assert len(grid) == 31
    cache, choices, validation = {}, {}, []
    for month in MONTHS:
        start = pd.Timestamp(month + "-01", tz="UTC")
        inner = [(start - pd.DateOffset(months=k)).strftime("%Y-%m") for k in [3, 2, 1]]
        inner = [m for m in inner if m >= "2025-07"]
        for m in inner:
            if m not in cache:
                cache[m] = expanded.monthly_validation(data, m, grid)
                validation.extend(cache[m])
        v = pd.DataFrame([r for m in inner for r in cache[m]])
        scores = v.groupby("candidate_id").agg(sse=("sse", "sum"), n=("n", "sum"))
        scores["mse"] = scores.sse / scores.n
        winner = scores.reset_index().sort_values(["mse", "candidate_id"]).iloc[0]
        candidate = next(c for c in grid if c["candidate_id"] == winner.candidate_id)
        choices[month] = dict(candidate=candidate, validation_months=inner,
                              validation_mse=float(winner.mse), validation_n=int(winner.n))
        print("Residual selection %s: %s" % (month, candidate["family"]), flush=True)
    return choices, validation


def residual_frames(panel, choices, day_weights=None):
    data = panel.dropna(subset=["y", "m", "g"] + KCOLS)
    frames, fits = [], []
    for month in MONTHS:
        start = pd.Timestamp(month + "-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        train = data.loc[data.index < start]
        if day_weights is not None:
            freq = day_weights.reindex(train.index.normalize()).to_numpy(dtype=int)
            train = train.iloc[np.repeat(np.arange(len(train)), freq)]
        test = data.loc[(data.index >= start) & (data.index < end + pd.Timedelta(hours=max(HORIZONS)))]
        origins = test.index[test.index < end]
        models = {
            "linear": expanded.ConditionalModel(dict(family="sequential_ols", params={})).fit(train, start),
            "pca": PCASequential().fit(train, start),
            "ml": expanded.ConditionalModel(choices[month]["candidate"]).fit(train, start),
            "joint_linear": expanded.ConditionalModel(dict(family="joint_ols", params={})).fit(train, start)}
        for method, model in models.items():
            e = pd.Series(test.y.to_numpy() - model.predict(test), index=test.index)
            log = dict(month=month, method=method, cutoff=str(start), train_n=len(train),
                       train_last=str(train.index.max()), train_first=str(train.index.min()))
            if method == "pca":
                log.update(pca_share=model.share, pca_loadings=model.loading.tolist(),
                           pca_mean=model.mean.tolist(), pca_sd=model.sd.tolist())
            if method == "ml":
                log.update(choices[month])
            fits.append(log)
            for h in HORIZONS:
                times = origins + pd.Timedelta(hours=h)
                frame = panel.loc[origins, STATES + ["m", "g"]].copy()
                frame["time"] = origins
                frame["target_time"] = times
                frame["month"] = month
                frame["fit_cutoff"] = start
                frame["method"], frame["h"] = method, h
                frame["e_origin"] = e.reindex(origins).to_numpy()
                frame["target"] = e.reindex(times).to_numpy()
                frame["y_change"] = panel.y.reindex(times).to_numpy() - panel.y.reindex(origins).to_numpy()
                frame["m_change"] = panel.m.reindex(times).to_numpy() - panel.m.reindex(origins).to_numpy()
                frame = frame.dropna(subset=["e_origin", "target"])
                assert ((frame.target_time - frame.time).dt.total_seconds() == h * 3600).all()
                assert (frame.fit_cutoff <= frame.time).all()
                frames.append(frame.reset_index(drop=True))
    result = pd.concat(frames, ignore_index=True)
    assert (result.groupby(["h", "time"]).size() == len(METHODS)).all()
    return result, fits


def enough(frame, columns):
    return (len(frame) >= max(100, 10 * (len(columns) + 1)) and
            np.all(frame[columns].std(ddof=1).to_numpy() > 1e-16) and frame.target.std() > 1e-12)


def fit_quantile(frame, columns, q, weights=None):
    x = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy()])
    return pilot.LEGACY.quantile_fit(x, frame.target.to_numpy(), q, weights)


def common_samples(frames):
    samples, counts = {}, []
    for scope, models in SPECS.items():
        valid = frames.dropna(subset=["target"] + models["M2"])
        horizons = {}
        for h in HORIZONS:
            x = valid.loc[valid.h == h]
            sets = [set(x.loc[x.method == method, "time"]) for method in METHODS]
            assert all(s == sets[0] for s in sets)
            horizons[h] = sets[0]
        common = set.intersection(*horizons.values())
        for h in HORIZONS:
            for sample, times in [("all", horizons[h]), ("matched", common)]:
                for method in METHODS:
                    f = valid.loc[(valid.h == h) & (valid.method == method) & valid.time.isin(times)].sort_values("time")
                    samples[(scope, sample, h, method)] = f.reset_index(drop=True)
                    counts.append(dict(scope=scope, sample=sample, h=h, method=method, n=len(f),
                                       days=int(f.time.dt.normalize().nunique())))
    return samples, counts


def point_estimates(frames, panel):
    samples, counts = common_samples(frames)
    records, ci_cases, sensitivity = [], [], []
    for (scope, sample, h, method), frame in samples.items():
        for model_name, columns in SPECS[scope].items():
            if not enough(frame, columns):
                records.append(dict(scope=scope, sample=sample, h=h, method=method,
                                    model=model_name, n=len(frame), status="insufficient"))
                continue
            for q in QUANTILES:
                beta = fit_quantile(frame, columns, q)
                sds = frame[columns].std(ddof=1).to_numpy()
                for j, term in enumerate(columns, 1):
                    records.append(dict(scope=scope, sample=sample, h=h, method=method,
                        model=model_name, q=q, term=term, n=len(frame), status="estimated",
                        coef_raw=float(beta[j]), effect_bp_per_sd=float(beta[j] * sds[j - 1])))
                include = (model_name == "M2" and ((sample == "all" and h == 6 and q in [.1, .5]) or
                                                   (sample == "matched" and q == .1)))
                if include:
                    ci_cases.append(dict(scope=scope, sample=sample, h=h, method=method, q=q,
                        columns=columns, times=frame.time.tolist(), sds=sds, point=beta))
    stress = panel.loc[panel.index >= pd.Timestamp("2025-08-01", tz="UTC"), "downside"]
    ranked = stress.groupby(stress.index.normalize()).mean().sort_values(ascending=False, kind="mergesort")
    for scope in SPECS:
        cols = SPECS[scope]["M2"]
        for method in METHODS:
            full = samples[(scope, "all", 6, method)]
            cut = pd.Timestamp("2025-12-01", tz="UTC")
            variants = {"Aug_Nov": full.loc[full.time < cut], "Dec_Mar": full.loc[full.time >= cut]}
            for k in [1, 5]:
                bad = ranked.index[:k]
                variants["exclude_top%d_downside_days" % k] = full.loc[
                    ~full.time.dt.normalize().isin(bad) & ~full.target_time.dt.normalize().isin(bad)]
            for name, frame in variants.items():
                if not enough(frame, cols):
                    sensitivity.append(dict(scope=scope, method=method, variant=name, n=len(frame), status="insufficient"))
                    continue
                beta = fit_quantile(frame, cols, .1)
                for term in TERMS:
                    j = cols.index(term) + 1
                    sensitivity.append(dict(scope=scope, method=method, variant=name, n=len(frame),
                        status="estimated", term=term, coef_raw=float(beta[j]),
                        effect_bp_per_sd=float(beta[j] * full[term].std(ddof=1))))
    return records, ci_cases, counts, sensitivity


def calendar_counts(index, block, rng, repetitions=1):
    calendar = pd.date_range(index.min().normalize().replace(day=1), index.max().normalize(), freq="D")
    months = np.asarray(calendar.strftime("%Y-%m"))
    weights = np.zeros((repetitions, len(calendar)), dtype=int)
    for month in np.unique(months):
        positions = np.flatnonzero(months == month)
        starts = rng.integers(0, len(positions), size=(repetitions, int(np.ceil(len(positions) / block))))
        draws = ((starts[:, :, None] + np.arange(block)) % len(positions)).reshape(repetitions, -1)
        draws = positions[draws[:, :len(positions)]]
        for row, draw in enumerate(draws):
            weights[row] += np.bincount(draw, minlength=len(calendar))
    return calendar, weights
