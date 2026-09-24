"""Behavior checks for past-only fitting and sequential shrinkage algebra."""
import numpy as np
import pandas as pd
from models import Model, candidates, FAMILIES, followup


def main():
    rng = np.random.default_rng(641)
    x = rng.uniform(-2, 2, (850, 2))
    y = 4*x[:, 0] + 2*x[:, 0]**2 + .7*x[:, 1] + rng.normal(0, .1, 850)
    data = pd.DataFrame(dict(y=y, m=x[:, 0], g=x[:, 1]),
        index=pd.date_range("2025-01-01", periods=len(y), freq="h", tz="UTC"))
    for i, col in enumerate(followup.KCOLS):
        data[col] = x[:, 0] + rng.normal(0, .1, len(y))
    cutoff = data.index[650]
    altered = data.copy()
    altered.loc[altered.index >= cutoff] = 1e8
    test = data.loc[data.index >= cutoff]
    seen, errors = set(), {}
    for c in candidates():
        if c["family"] in seen:
            continue
        seen.add(c["family"])
        a, b = Model(c).fit(data, cutoff), Model(c).fit(altered, cutoff)
        np.testing.assert_allclose(a.predict(test), b.predict(test), atol=1e-10, rtol=0)
        errors[c["family"]] = float(np.mean((test.y-a.predict(test))**2))
    assert errors["seq_spline"] < .2*errors["sequential_ols"]
    assert errors["joint_tensor"] < .2*errors["joint_ols"]
    c = next(c for c in candidates() if c["family"] == "seq_spline")
    a = Model(c).fit(data, cutoff)
    base = Model(candidates()[0]).fit(data, cutoff)
    train = data.loc[data.index < cutoff]
    z = np.column_stack([np.ones(len(train)), train.m])
    zt = np.column_stack([np.ones(len(test)), test.m])
    g = np.column_stack([np.ones(len(train)), train.g])
    gt = np.column_stack([np.ones(len(test)), test.g])
    first = np.linalg.lstsq(z, train.y, rcond=None)[0]
    for weight in [0., .1, .25, .5, 1.]:
        nonlinear_train = a.model.predict(train[["m"]].to_numpy())
        nonlinear_test = a.model.predict(test[["m"]].to_numpy())
        ftrain = z@first + weight*(nonlinear_train-z@first)
        ftest = zt@first + weight*(nonlinear_test-zt@first)
        second = np.linalg.lstsq(g, train.y-ftrain, rcond=None)[0]
        direct = ftest + gt@second
        blend = base.predict(test)+weight*(a.predict(test)-base.predict(test))
        np.testing.assert_allclose(direct, blend, atol=1e-10, rtol=0)
    print("PASS: all nine families past-only; known curvature recovered; sequential shrinkage identity.")


if __name__ == "__main__":
    main()
