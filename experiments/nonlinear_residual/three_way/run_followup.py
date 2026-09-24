"""Run identical inference and prediction experiments for three residual definitions."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import sklearn
import lightgbm
import pipeline as study
import forecast

CACHE = study.ROOT / ".cache/three_way"
B = 199
BLOCKS = [3, 7]


def prepare(scenario):
    started = time.monotonic()
    directory = study.OUT / scenario
    directory.mkdir(parents=True, exist_ok=True)
    panel, metadata = study.build_panel(scenario)
    choices, validation = study.select_ml(panel)
    frames, fits = study.residual_frames(panel, choices)
    points, cases, counts, sensitivity = study.point_estimates(frames, panel)
    frames.to_csv(directory / "residual_pairs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(points).to_csv(directory / "coefficients.csv", index=False)
    pd.DataFrame(counts).to_csv(directory / "sample_counts.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(directory / "subperiod_and_stress.csv", index=False)
    pd.DataFrame(validation).to_csv(directory / "residual_validation.csv", index=False)
    (directory / "residual_fits.json").write_text(json.dumps(fits, indent=2) + "\n")
    (directory / "residual_selection.json").write_text(json.dumps(choices, indent=2) + "\n")
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / (scenario + ".pkl")).open("wb") as stream:
        pickle.dump(dict(panel=panel, choices=choices, cases=cases), stream, protocol=4)
    print("Point estimates ready: %s; %d bootstrap cases" % (scenario, len(cases)), flush=True)
    predictions, scores, selected, valid, crossing = forecast.run(frames)
    predictions.to_csv(directory / "forecast_predictions.csv.gz", index=False,
                       compression={"method": "gzip", "mtime": 0})
    scores.to_csv(directory / "forecast_scores.csv", index=False)
    valid.to_csv(directory / "forecast_validation.csv", index=False)
    crossing.to_csv(directory / "forecast_crossing.csv", index=False)
    (directory / "forecast_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    forecast.comparisons(predictions).to_csv(directory / "forecast_comparisons.csv", index=False)
    print("Prepared %s in %.1fs" % (scenario, time.monotonic() - started), flush=True)
    return metadata


def bootstrap(scenario, block):
    with (CACHE / (scenario + ".pkl")).open("rb") as stream:
        payload = pickle.load(stream)
    panel, choices, cases = payload["panel"], payload["choices"], payload["cases"]
    rng = np.random.default_rng(20260924 + block)
    calendar, all_weights = study.calendar_counts(panel.index, block, rng, B)
    records, failures = [], []
    started = time.monotonic()
    for rep, frequencies in enumerate(all_weights):
        weights = pd.Series(frequencies, index=calendar)
        try:
            frames, _ = study.residual_frames(panel, choices, weights)
            groups = {(method, h): d.set_index("time") for (method, h), d in frames.groupby(["method", "h"])}
        except Exception as exc:
            failures.append(dict(rep=rep, stage="residual_refit", error=repr(exc)))
            continue
        for case in cases:
            try:
                frame = groups[(case["method"], case["h"])].loc[case["times"]]
                w = weights.reindex(pd.DatetimeIndex(case["times"]).normalize()).to_numpy()
                beta = study.fit_quantile(frame, case["columns"], case["q"], w)
                for term in study.TERMS:
                    if term not in case["columns"]:
                        continue
                    j = case["columns"].index(term) + 1
                    records.append(dict(rep=rep, scope=case["scope"], sample=case["sample"],
                        h=case["h"], method=case["method"], q=case["q"], term=term,
                        coef_raw=float(beta[j]), effect_bp_per_sd=float(beta[j] * case["sds"][j - 1])))
            except Exception as exc:
                failures.append(dict(rep=rep, stage="quantile", scope=case["scope"], sample=case["sample"],
                    h=case["h"], method=case["method"], q=case["q"], error=repr(exc)))
        if (rep + 1) % 20 == 0:
            print("Bootstrap %s d%d %d/%d (%.0fs)" %
                  (scenario, block, rep + 1, B, time.monotonic() - started), flush=True)
    directory = study.OUT / scenario
    draws = pd.DataFrame(records)
    keys = ["scope", "sample", "h", "method", "q", "term"]
    successes = draws.groupby(keys).rep.nunique()
    assert len(successes) == sum(sum(t in c["columns"] for t in study.TERMS) for c in cases)
    assert successes.min() >= .95 * B, (scenario, block, successes.min(), failures[:3])
    draws.to_csv(directory / ("bootstrap_draws_d%d.csv.gz" % block), index=False,
                 compression={"method": "gzip", "mtime": 0})
    (directory / ("bootstrap_failures_d%d.json" % block)).write_text(json.dumps(failures, indent=2) + "\n")
    return dict(scenario=scenario, block_days=block, replicates=B, min_success=int(successes.min()),
                failures=len(failures), seconds=time.monotonic() - started)


def summarize_intervals():
    intervals, contrasts = [], []
    keys = ["scope", "sample", "h", "method", "q", "term"]
    for scenario in study.pilot.SCENARIOS:
        directory = study.OUT / scenario
        point = pd.read_csv(directory / "coefficients.csv")
        point = point.loc[(point.status == "estimated") & (point.model == "M2")].set_index(keys)
        for block in BLOCKS:
            draws = pd.read_csv(directory / ("bootstrap_draws_d%d.csv.gz" % block))
            for key, group in draws.groupby(keys, sort=False):
                for metric in ["coef_raw", "effect_bp_per_sd"]:
                    value = float(point.loc[key, metric])
                    intervals.append(dict(scenario=scenario, block_days=block, **dict(zip(keys, key)),
                        metric=metric, point=value, ci_low=float(group[metric].quantile(.025)),
                        ci_high=float(group[metric].quantile(.975)), successful=len(group)))
            comparisons = []
            for key in point.index:
                scope, sample, h, method, q, term = key
                if method in ["pca", "ml", "joint_linear"]:
                    other = (scope, sample, h, "linear", q, term)
                    comparisons.append(("method_minus_linear", key, other))
                if method == "ml":
                    comparisons.append(("ML_minus_joint_linear", key, (scope, sample, h, "joint_linear", q, term)))
                if sample == "all" and h == 6 and q == .1:
                    comparisons.append(("q10_minus_q50", key, (scope, sample, h, method, .5, term)))
                if sample == "matched" and h in [6, 12] and q == .1:
                    comparisons.append(("h%d_minus_h1" % h, key, (scope, sample, 1, method, q, term)))
            grouped = {key: group.set_index("rep") for key, group in draws.groupby(keys, sort=False)}
            for kind, a, b in comparisons:
                if a not in grouped or b not in grouped:
                    continue
                both = grouped[a][["coef_raw", "effect_bp_per_sd"]].join(
                    grouped[b][["coef_raw", "effect_bp_per_sd"]], lsuffix="_a", rsuffix="_b", how="inner")
                assert len(both) >= .95 * B
                for metric in ["coef_raw", "effect_bp_per_sd"]:
                    delta = both[metric + "_a"] - both[metric + "_b"]
                    contrasts.append(dict(scenario=scenario, block_days=block, contrast=kind,
                        **dict(zip(keys, a)), comparator_method=b[3], comparator_h=b[2], comparator_q=b[4],
                        metric=metric, point=float(point.loc[a, metric] - point.loc[b, metric]),
                        ci_low=float(delta.quantile(.025)), ci_high=float(delta.quantile(.975)), successful=len(delta)))
    pd.DataFrame(intervals).to_csv(study.OUT / "coefficient_intervals.csv", index=False)
    pd.DataFrame(contrasts).to_csv(study.OUT / "paired_coefficient_contrasts.csv", index=False)


def main():
    started = time.monotonic()
    before = study.pilot.audit_archive()
    initial = json.loads((study.HERE.parent / "results/RUN_MANIFEST.json").read_text())
    inputs = dict(initial["input_sha256"])
    for name, digest in inputs.items():
        assert study.pilot.sha(study.ROOT / name) == digest, name
    for path in list(study.HERE.glob("*.py")) + [study.HERE / "PROTOCOL_KO.md",
                                               study.HERE.parent / "expanded/run_expanded.py"]:
        inputs[str(path.relative_to(study.ROOT))] = study.pilot.sha(path)
    study.OUT.mkdir(parents=True, exist_ok=True)
    metadata = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(prepare, scenario): scenario for scenario in study.pilot.SCENARIOS}
        for job in as_completed(jobs):
            metadata.append(job.result())
    progress = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(bootstrap, scenario, block): (scenario, block)
                for scenario in study.pilot.SCENARIOS for block in BLOCKS}
        for job in as_completed(jobs):
            progress.append(job.result())
            print("Bootstrap job complete: " + str(jobs[job]), flush=True)
    summarize_intervals()
    for filename in ["coefficients.csv", "sample_counts.csv", "subperiod_and_stress.csv",
                     "forecast_scores.csv", "forecast_comparisons.csv", "forecast_crossing.csv"]:
        parts = []
        for scenario in study.pilot.SCENARIOS:
            data = pd.read_csv(study.OUT / scenario / filename)
            data.insert(0, "scenario", scenario)
            parts.append(data)
        pd.concat(parts, ignore_index=True).to_csv(study.OUT / filename, index=False)
    after = study.pilot.audit_archive()
    assert before == after
    for name, digest in inputs.items():
        assert study.pilot.sha(study.ROOT / name) == digest, name
    outputs = {str(p.relative_to(study.OUT)): study.pilot.sha(p) for p in study.OUT.rglob("*")
               if p.is_file() and p.name != "RUN_MANIFEST.json"}
    manifest = dict(stage="three-way downstream associations, horizon checks and forecasting completed",
        publication_ready=False, timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=study.ROOT, text=True).strip(),
        input_sha256=inputs, output_sha256=outputs, scenarios=metadata,
        primary_methods=study.METHODS[:3], projection_control=study.METHODS[3],
        origin_months=study.MONTHS, horizons=study.HORIZONS, quantiles=study.QUANTILES,
        bootstrap=progress, elapsed_seconds=time.monotonic() - started, archive_check=after,
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      sklearn=sklearn.__version__, lightgbm=lightgbm.__version__),
        checks=["raw premiums reproduce A1 common factor", "PCA and nuisance regressions use past only",
                "ML selection uses only earlier validation months", "same-origin function defines present and future residual",
                "exact elapsed-hour outcomes", "identical samples across residual methods and nested regressions",
                "matched-origin horizon comparisons", "past target availability for forecasting",
                "same definition-specific targets for prediction algorithms and positioning ablations",
                "PCA and nuisance models refit in paired calendar bootstrap", "original archive unchanged"],
        limitations=["exploratory extension after A1/A2; history already examined",
                     "FX timezone and USDC/USD assumptions unresolved",
                     "main residual comparisons mix factor definition and functional form; joint OLS supplied as control",
                     "bootstrap fixes selected ML hyperparameters; intervals are pointwise and exploratory",
                     "forecast loss bootstrap conditions on saved fits",
                     "event exclusions hold residual generation fixed", "no causal or structural-channel identification"])
    (study.OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed all three-way follow-ups in %.0fs" % (time.monotonic() - started), flush=True)


if __name__ == "__main__":
    main()
