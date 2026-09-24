"""A2 exploratory chronological model comparison; no future-price forecast."""
from pathlib import Path
import json
import platform
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_pilot as pilot

ROOT = pilot.ROOT
OUT = HERE / "results"
SEED = 20260924
FAMILIES = ["sequential_ols", "joint_ols", "rolling_ols", "quadratic_ridge",
            "linear_plus_spline", "linear_plus_svr", "linear_plus_rf",
            "lightgbm", "linear_plus_lightgbm"]
NONLINEAR = FAMILIES[3:]


def candidates():
    rows = []
    def add(family, **params):
        rows.append(dict(candidate_id="c%02d" % len(rows), family=family, params=params))
    add("sequential_ols")
    add("joint_ols")
    for days in [30, 60, 120]:
        add("rolling_ols", days=days)
    for alpha in [.1, 10., 1000.]:
        add("quadratic_ridge", alpha=alpha)
    for knots in [3, 5]:
        for alpha in [1., 100.]:
            add("linear_plus_spline", knots=knots, alpha=alpha)
    for c in [1., 10.]:
        for gamma in [.1, 1.]:
            add("linear_plus_svr", C=c, gamma=gamma)
    for depth in [3, None]:
        for leaf in [25, 100]:
            add("linear_plus_rf", max_depth=depth, min_samples_leaf=leaf)
    for family in ["lightgbm", "linear_plus_lightgbm"]:
        for trees in [100, 300]:
            for leaves in [7, 15]:
                for leaf in [30, 100]:
                    add(family, n_estimators=trees, num_leaves=leaves,
                        min_child_samples=leaf, max_depth=3 if leaves == 7 else 4)
    assert len(rows) == 36
    return rows


class ConditionalModel:
    def __init__(self, candidate):
        self.family = candidate["family"]
        self.params = candidate["params"]

    def fit(self, data, cutoff):
        train = data.loc[data.index < cutoff]
        if self.family == "rolling_ols":
            train = train.loc[train.index >= cutoff - pd.Timedelta(days=self.params["days"])]
        assert len(train) >= 100 and train.index.max() < cutoff
        x, y = train[["m", "g"]].to_numpy(), train.y.to_numpy()
        self.train = train
        z = np.column_stack([np.ones(len(x)), x])
        self.coef = np.linalg.lstsq(z, y, rcond=None)[0]
        np.testing.assert_allclose(z.T @ (y - z @ self.coef), 0, atol=1e-5, rtol=0)
        if self.family == "sequential_ols":
            m = np.column_stack([np.ones(len(x)), x[:, 0]])
            g = np.column_stack([np.ones(len(x)), x[:, 1]])
            self.first = np.linalg.lstsq(m, y, rcond=None)[0]
            self.second = np.linalg.lstsq(g, y - m @ self.first, rcond=None)[0]
        elif self.family == "quadratic_ridge":
            self.learner = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                        StandardScaler(), Ridge(alpha=self.params["alpha"]))
        elif self.family == "linear_plus_spline":
            self.learner = make_pipeline(SplineTransformer(n_knots=self.params["knots"], degree=3,
                knots="uniform", extrapolation="linear", include_bias=False),
                StandardScaler(), Ridge(alpha=self.params["alpha"]))
        elif self.family == "linear_plus_svr":
            self.learner = make_pipeline(StandardScaler(), SVR(kernel="rbf", epsilon=.5,
                                                               **self.params))
        elif self.family == "linear_plus_rf":
            self.learner = RandomForestRegressor(n_estimators=150, random_state=SEED,
                                                 n_jobs=1, **self.params)
        elif self.family in ["lightgbm", "linear_plus_lightgbm"]:
            self.learner = lgb.LGBMRegressor(objective="regression", learning_rate=.03,
                reg_lambda=1., random_state=SEED, n_jobs=1, verbosity=-1,
                deterministic=True, force_col_wise=True, **self.params)
        if hasattr(self, "learner"):
            target = y - z @ self.coef if self.family.startswith("linear_plus_") else y
            self.learner.fit(x, target)
        self.train_rmse = float(np.sqrt(np.mean((y - self.predict(train)) ** 2)))
        return self

    def predict(self, data):
        x = data[["m", "g"]].to_numpy()
        z = np.column_stack([np.ones(len(x)), x])
        if self.family in ["joint_ols", "rolling_ols"]:
            result = z @ self.coef
        elif self.family == "sequential_ols":
            result = np.column_stack([np.ones(len(x)), x[:, 0]]) @ self.first
            result += np.column_stack([np.ones(len(x)), x[:, 1]]) @ self.second
        elif self.family.startswith("linear_plus_"):
            result = z @ self.coef + self.learner.predict(x)
        else:
            result = self.learner.predict(x)
        assert np.isfinite(result).all()
        return result


def monthly_validation(data, month, grid):
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    test = data.loc[(data.index >= start) & (data.index < end)]
    assert len(test) >= 20
    records = []
    for candidate in grid:
        model = ConditionalModel(candidate).fit(data, start)
        error = test.y.to_numpy() - model.predict(test)
        records.append(dict(candidate_id=candidate["candidate_id"], family=candidate["family"],
            validation_month=month, n=len(test), sse=float(error @ error),
            sae=float(np.abs(error).sum()), train_last=str(model.train.index.max()),
            train_first=str(model.train.index.min()), train_n=len(model.train),
            train_rmse_bp=model.train_rmse))
    return records


def bootstrap_comparisons(predictions, repetitions=2000):
    """Fixed-prediction paired calendar blocks; exploratory pointwise intervals."""
    rows = []
    for scenario, data in predictions.groupby("scenario", sort=False):
        errors = data.pivot(index="time", columns="model", values="residual_bp").sort_index()
        assert errors.notna().all().all()
        errors.index = pd.DatetimeIndex(errors.index)
        names = list(errors.columns)
        squared, absolute = errors ** 2, errors.abs()
        calendar = pd.date_range(errors.index.min().normalize().replace(day=1),
                                 errors.index.max().normalize(), freq="D")
        count = errors.iloc[:, 0].groupby(errors.index.normalize()).size().reindex(calendar, fill_value=0)
        sq = squared.groupby(squared.index.normalize()).sum().reindex(calendar, fill_value=0).to_numpy()
        ab = absolute.groupby(absolute.index.normalize()).sum().reindex(calendar, fill_value=0).to_numpy()
        months = np.array(calendar.strftime("%Y-%m"))
        for block in [3, 7]:
            rng = np.random.default_rng(SEED + block)
            weights = np.zeros((repetitions, len(calendar)), dtype=int)
            for month in np.unique(months):
                positions = np.flatnonzero(months == month)
                starts = rng.integers(0, len(positions), size=(repetitions, int(np.ceil(len(positions) / block))))
                draws = ((starts[:, :, None] + np.arange(block)) % len(positions)).reshape(repetitions, -1)
                draws = positions[draws[:, :len(positions)]]
                for i, draw in enumerate(draws):
                    weights[i] += np.bincount(draw, minlength=len(calendar))
            total = weights @ count.to_numpy()
            assert (total > 0).all()
            brmse = np.sqrt((weights @ sq) / total[:, None])
            bmae = (weights @ ab) / total[:, None]
            for baseline in ["joint_ols", "rolling_ols"]:
                bi = names.index(baseline)
                for model in names:
                    if model == baseline:
                        continue
                    mi = names.index(model)
                    dr, da = brmse[:, mi] - brmse[:, bi], bmae[:, mi] - bmae[:, bi]
                    rows.append(dict(scenario=scenario, model=model, baseline=baseline,
                        block_calendar_days=block, repetitions=repetitions,
                        delta_rmse_bp=float(np.sqrt(squared[model].mean()) - np.sqrt(squared[baseline].mean())),
                        delta_mae_bp=float(absolute[model].mean() - absolute[baseline].mean()),
                        rmse_ci_low=float(np.quantile(dr, .025)), rmse_ci_high=float(np.quantile(dr, .975)),
                        mae_ci_low=float(np.quantile(da, .025)), mae_ci_high=float(np.quantile(da, .975))))
    return pd.DataFrame(rows)


def main():
    before = pilot.audit_archive()
    old_manifest = json.loads((HERE.parent / "results/RUN_MANIFEST.json").read_text())
    for name, digest in old_manifest["input_sha256"].items():
        assert pilot.sha(ROOT / name) == digest, name
    OUT.mkdir(parents=True, exist_ok=True)
    grid = candidates()
    grid_by_id = {c["candidate_id"]: c for c in grid}
    validation, selection, predictions, fit_logs, bins, metadata = [], [], [], [], [], []
    fit_count = 0
    for scenario in pilot.SCENARIOS:
        panel, meta = pilot.build_panel(scenario)
        metadata.append(meta)
        data = panel[["y", "m", "g"]].replace([np.inf, -np.inf], np.nan).dropna()
        cache = {}
        for month in pilot.MONTHS:
            start = pd.Timestamp(month + "-01", tz="UTC")
            end = start + pd.offsets.MonthBegin(1)
            inner_months = [(start - pd.DateOffset(months=k)).strftime("%Y-%m") for k in [3, 2, 1]]
            for inner in inner_months:
                if inner not in cache:
                    cache[inner] = monthly_validation(data, inner, grid)
                    fit_count += len(grid)
                    print("Validated %s %s (%d candidates)" % (scenario, inner, len(grid)), flush=True)
            rows = pd.DataFrame([r for inner in inner_months for r in cache[inner]])
            assert (pd.to_datetime(rows.train_last, utc=True) < start).all()
            table = rows.groupby(["candidate_id", "family"], sort=False).agg(sse=("sse", "sum"),
                                                                                 n=("n", "sum")).reset_index()
            table["validation_mse"] = table.sse / table.n
            chosen = {}
            for family in FAMILIES:
                ranked = table.loc[table.family == family].sort_values(["validation_mse", "candidate_id"])
                chosen[family] = ranked.iloc[0].candidate_id
            overall = table.sort_values(["validation_mse", "candidate_id"]).iloc[0]
            chosen["past_selected"] = overall.candidate_id
            test = data.loc[(data.index >= start) & (data.index < end)]
            # Nothing below may change chosen after observing test errors.
            local = {}
            for family in FAMILIES:
                candidate = grid_by_id[chosen[family]]
                model = ConditionalModel(candidate).fit(data, start)
                fit_count += 1
                fitted = model.predict(test)
                frame = pd.DataFrame(dict(time=test.index, scenario=scenario, evaluation_month=month,
                    fit_cutoff=start, model=family, candidate_id=chosen[family],
                    observed_bp=test.y.to_numpy(), fitted_bp=fitted, residual_bp=test.y.to_numpy() - fitted))
                local[family] = frame
                predictions.append(frame)
                train = model.train
                outside = ((test[["m", "g"]] < train[["m", "g"]].min()) |
                           (test[["m", "g"]] > train[["m", "g"]].max())).any(axis=1)
                fit_logs.append(dict(scenario=scenario, evaluation_month=month, model=family,
                    candidate_id=chosen[family], train_first=str(train.index.min()),
                    train_last=str(train.index.max()), train_n=len(train), test_n=len(test),
                    cutoff=str(start), train_rmse_bp=model.train_rmse,
                    test_rmse_bp=float(np.sqrt(np.mean(frame.residual_bp ** 2))),
                    outside_training_range_share=float(outside.mean()),
                    joint_ols_coefficients=model.coef.tolist()))
                for feature in ["m", "g"]:
                    cuts = np.unique(np.quantile(train[feature], [.2, .4, .6, .8]))
                    labels = np.searchsorted(cuts, test[feature].to_numpy(), side="right")
                    for label in np.unique(labels):
                        mask = labels == label
                        bins.append(dict(scenario=scenario, evaluation_month=month, model=family,
                            feature=feature, bin_index=int(label), n=int(mask.sum()),
                            feature_min=float(test[feature].to_numpy()[mask].min()),
                            feature_max=float(test[feature].to_numpy()[mask].max()),
                            residual_mean_bp=float(frame.residual_bp.to_numpy()[mask].mean()),
                            cuts=json.dumps(cuts.tolist())))
            selected_family = grid_by_id[chosen["past_selected"]]["family"]
            selected = local[selected_family].copy()
            selected["model"] = "past_selected"
            predictions.append(selected)
            for model, cid in chosen.items():
                score = table.loc[table.candidate_id == cid].iloc[0]
                selection.append(dict(scenario=scenario, evaluation_month=month, model=model,
                    selected_family=grid_by_id[cid]["family"], candidate_id=cid,
                    validation_mse=float(score.validation_mse), validation_n=int(score.n),
                    validation_months=inner_months, params=grid_by_id[cid]["params"]))
            print("Evaluated %s %s; past-selected=%s" % (scenario, month, selected_family), flush=True)
        for records in cache.values():
            validation += [dict(scenario=scenario, **r) for r in records]
    pred = pd.concat(predictions, ignore_index=True)
    assert not pred.duplicated(["scenario", "model", "time"]).any()
    assert (pred.groupby(["scenario", "time"]).size() == len(FAMILIES) + 1).all()
    old = pd.read_csv(HERE.parent / "results/conditional_fit_predictions.csv.gz")
    for model in ["joint_ols", "sequential_ols"]:
        new = pred.loc[pred.model == model].sort_values(["scenario", "time"])
        previous = old.loc[old.model == model].sort_values(["scenario", "time"])
        assert len(new) == len(previous)
        assert np.array_equal(new.scenario.to_numpy(), previous.scenario.to_numpy())
        assert np.array_equal(pd.to_datetime(new.time, utc=True).to_numpy(),
                              pd.to_datetime(previous.time, utc=True).to_numpy())
        np.testing.assert_allclose(new.fitted_bp, previous.fitted_bp, rtol=1e-11, atol=1e-9)
    scores = []
    for keys, group in pred.groupby(["scenario", "model", "evaluation_month"], sort=False):
        scores.append(dict(zip(["scenario", "model", "period"], keys), **pilot.metrics(group)))
    for keys, group in pred.groupby(["scenario", "model"], sort=False):
        scores.append(dict(zip(["scenario", "model"], keys), period="all", **pilot.metrics(group)))
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    pred.to_csv(OUT / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(validation).to_csv(OUT / "inner_validation.csv", index=False)
    pd.DataFrame(fit_logs).to_csv(OUT / "fits.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / "residual_bins.csv", index=False)
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    bootstrap_comparisons(pred).to_csv(OUT / "paired_block_intervals.csv", index=False)
    after = pilot.audit_archive()
    assert before == after
    inputs = dict(old_manifest["input_sha256"])
    for path in [HERE / "PROTOCOL_KO.md", Path(__file__).resolve()]:
        inputs[str(path.relative_to(ROOT))] = pilot.sha(path)
    output_hashes = {p.name: pilot.sha(p) for p in OUT.iterdir() if p.name != "RUN_MANIFEST.json"}
    manifest = dict(stage="A2 exploratory conditional fit; B not run", publication_ready=False,
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        protocol_sha256=pilot.sha(HERE / "PROTOCOL_KO.md"), input_sha256=inputs,
        output_sha256=output_hashes, candidates=grid, unique_fits=fit_count,
        scenarios=metadata, months=pilot.MONTHS, archive_check=after,
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, lightgbm=lgb.__version__),
        checks=["A1 input hashes unchanged", "archive bytes unchanged",
                "A1 sequential and joint predictions reproduced on identical timestamps",
                "inner selection and outer fit strictly earlier than respective evaluation months",
                "identical evaluation observations across model families", "finite predictions",
                "joint linear training normal equations", "all candidates and all clock assumptions retained"],
        limitations=["A2 designed after A1; historical evaluation previously explored",
                     "FX and macro timezone assumptions remain unconfirmed",
                     "relative USDC/USDT proxy assumes USDC/USD=1",
                     "same-time factors: conditional fit, not future prediction or causal inference",
                     "intervals are exploratory, pointwise, conditional on fitted predictions",
                     "no generated-residual or model-refitting uncertainty in loss bootstrap"])
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed %d fits. Original archive preserved." % fit_count, flush=True)
    print(pd.DataFrame(scores).query("period == 'all'")[["scenario", "model", "rmse_bp", "mae_bp"]].to_string(index=False))


if __name__ == "__main__":
    main()
