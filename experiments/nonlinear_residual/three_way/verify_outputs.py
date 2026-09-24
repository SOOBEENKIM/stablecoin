"""Independently verify saved samples, prior-result parity and temporal selections."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"


def main():
    a1 = pd.read_csv(HERE.parent / "results/conditional_fit_predictions.csv.gz")
    a2 = pd.read_csv(HERE.parent / "expanded/results/predictions.csv.gz")
    for data in [a1, a2]:
        data["time"] = pd.to_datetime(data.time, utc=True)
    selection_count, scenario_count = 0, 0
    for directory in sorted(OUT.iterdir()):
        if not directory.is_dir():
            continue
        scenario_count += 1
        r = pd.read_csv(directory / "residual_pairs.csv.gz", parse_dates=["time", "target_time", "fit_cutoff"])
        assert ((r.target_time - r.time).dt.total_seconds() == r.h * 3600).all()
        assert r.groupby(["time", "h"]).size().eq(4).all()
        assert not r.duplicated(["method", "time", "h"]).any()
        for method, old_name in [("linear", "sequential_ols"), ("joint_linear", "joint_ols")]:
            x = r[(r.method == method) & (r.h == 1) & (r.time >= pd.Timestamp("2026-01-01", tz="UTC"))]
            y = a1[(a1.scenario == directory.name) & (a1.model == old_name)]
            z = x.merge(y, on="time", validate="one_to_one")
            assert len(z) == len(x)
            np.testing.assert_allclose(z.e_origin, z.residual_bp, atol=1e-8, rtol=1e-10)
        validation = pd.read_csv(directory / "residual_validation.csv")
        choices = json.loads((directory / "residual_selection.json").read_text())
        for month, choice in choices.items():
            a = validation[validation.validation_month.isin(choice["validation_months"])]
            s = a.groupby("candidate_id").agg(sse=("sse", "sum"), n=("n", "sum"))
            s["mse"] = s.sse / s.n
            winner = s.reset_index().sort_values(["mse", "candidate_id"]).iloc[0]
            assert winner.candidate_id == choice["candidate"]["candidate_id"]
            assert max(choice["validation_months"]) < month
            if month >= "2026-01":
                x = r[(r.method == "ml") & (r.h == 1) & (r.month == month)]
                y = a2[(a2.scenario == directory.name) & (a2.model == choice["candidate"]["family"]) &
                       (a2.evaluation_month == month)]
                z = x.merge(y, on="time", validate="one_to_one")
                assert len(z) == len(x)
                np.testing.assert_allclose(z.e_origin, z.residual_bp, atol=1e-8, rtol=1e-10)
        f = pd.read_csv(directory / "forecast_predictions.csv.gz", parse_dates=["time", "target_time"])
        assert ((f.target_time - f.time).dt.total_seconds() == 6 * 3600).all()
        assert f.groupby(["method", "time", "q"]).target.nunique().eq(1).all()
        assert f.groupby(["method", "time", "q"]).size().eq(4).all()
        residual = r[r.h == 6][["method", "time", "target"]]
        z = f.merge(residual, on=["method", "time"], suffixes=("_forecast", "_residual"), validate="many_to_one")
        assert len(z) == len(f)
        np.testing.assert_allclose(z.target_forecast, z.target_residual, atol=1e-9)
        error = f.target - f.prediction
        np.testing.assert_allclose(np.maximum(f.q * error, (f.q - 1) * error), f.pinball, atol=1e-10)
        scores = pd.read_csv(directory / "forecast_scores.csv")
        for row in scores.itertuples(index=False):
            d = f[(f.method == row.method) & (f["info"] == row.info) &
                  (f.algorithm == row.algorithm) & (f.q == row.q)]
            if row.period != "all":
                d = d[d.month == row.period]
            assert len(d) == row.n
            assert np.isclose(d.pinball.mean(), row.pinball)
        valid = pd.read_csv(directory / "forecast_validation.csv")
        for choice in json.loads((directory / "forecast_selection.json").read_text()):
            cutoff = pd.Timestamp(choice["cutoff"])
            assert pd.Timestamp(choice["train_target_last"]) < cutoff
            d = valid[(valid.method == choice["method"]) & (valid.evaluation_month == choice["month"]) &
                      (valid["info"] == choice["info"]) & (valid.algorithm == choice["algorithm"]) & (valid.q == choice["q"])]
            assert len(d) > 0
            assert (pd.to_datetime(d.validation_target_last, utc=True) < cutoff).all()
            assert (pd.to_datetime(d.train_target_last, utc=True) <
                    pd.to_datetime(d.validation_month + "-01", utc=True)).all()
            s = d.groupby("candidate").agg(total=("loss_sum", "sum"), n=("n", "sum"))
            s["loss"] = s.total / s.n
            winner = s.reset_index().sort_values(["loss", "candidate"]).iloc[0]
            assert winner.candidate == choice["candidate"]
            selection_count += 1
    assert scenario_count == 5 and selection_count == 720
    manifest_path = OUT / "RUN_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["input_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        for name, digest in manifest["output_sha256"].items():
            assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
        assert all(job["min_success"] >= .95 * job["replicates"] for job in manifest["bootstrap"])
    print("PASS: five scenarios, exact clocks, same samples, A1/A2 parity, fixed targets/losses, 720 past-only forecast selections.")
    print("Run-manifest hashes checked:", manifest_path.exists())


if __name__ == "__main__":
    main()
