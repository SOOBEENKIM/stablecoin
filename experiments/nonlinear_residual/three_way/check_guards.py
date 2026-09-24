"""Leakage, PCA, clock-link, month-boundary and label-availability guards."""
import numpy as np
import pandas as pd
import pipeline as study
import forecast


def main():
    rng = np.random.default_rng(194)
    idx = pd.date_range("2025-01-01", periods=24 * 70, freq="h", tz="UTC")
    common = rng.normal(size=len(idx))
    p = pd.DataFrame({name: common + .1 * rng.normal(size=len(idx)) for name in study.KCOLS}, index=idx)
    p["m"] = p[study.KCOLS].mean(axis=1)
    p["g"] = rng.normal(size=len(idx))
    p["y"] = .9 * p.m + .2 * p.g + rng.normal(scale=.1, size=len(idx))
    for col in study.STATES:
        p[col] = rng.normal(size=len(idx))
    cutoff = pd.Timestamp("2025-02-01", tz="UTC")
    changed = p.copy()
    changed.loc[changed.index >= cutoff, study.KCOLS + ["y", "g"]] = 1e8
    a, b = study.PCASequential().fit(p, cutoff), study.PCASequential().fit(changed, cutoff)
    np.testing.assert_allclose(a.predict(p), b.predict(p), rtol=0, atol=0)
    assert a.share > .95
    old_months = study.MONTHS
    study.MONTHS = ["2025-02"]
    choice = {"2025-02": dict(candidate=dict(family="quadratic_ridge", params=dict(alpha=.1)),
                               validation_months=["2025-01"], validation_mse=0., validation_n=1)}
    missing = pd.Timestamp("2025-02-02 02:00", tz="UTC")
    sparse = p.drop(index=missing)
    frames, _ = study.residual_frames(sparse, choice)
    assert not (frames.target_time == missing).any()
    assert ((frames.target_time - frames.time).dt.total_seconds() == frames.h * 3600).all()
    row = frames.loc[(frames.method == "linear") & (frames.h == 6) &
                     (frames.time == pd.Timestamp("2025-02-28 20:00", tz="UTC"))].iloc[0]
    fitted = study.expanded.ConditionalModel(dict(family="sequential_ols", params={})).fit(sparse, cutoff)
    target = row.target_time
    expected = sparse.loc[target, "y"] - fitted.predict(sparse.loc[[target]])[0]
    assert abs(row.target - expected) < 1e-12
    d = frames.loc[(frames.method == "linear") & (frames.h == 6)]
    cut = pd.Timestamp("2025-02-20", tz="UTC")
    train = forecast.training_before(d, cut)
    assert train.target_time.max() < cut
    assert ((d.time < cut) & (d.target_time >= cut)).any()
    for scope in study.SPECS:
        samples, _ = study.common_samples(frames)
        for h in study.HORIZONS:
            times = [samples[(scope, "all", h, method)].time.tolist() for method in study.METHODS]
            assert all(t == times[0] for t in times)
    study.MONTHS = old_months
    print("PASS: past-only PCA, sparse clock links, fixed function across month boundary, common samples, forecast label purge.")


if __name__ == "__main__":
    main()
