"""Behavioral checks: no future fit inputs; known nonlinear signal recoverable."""
import numpy as np
import pandas as pd
import run_expanded as experiment


def main():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(1100, 2))
    y = (2 * x[:, 0] + 3 * x[:, 1] ** 2 + 1.5 * x[:, 0] * x[:, 1]
         + rng.normal(scale=.15, size=1100))
    data = pd.DataFrame(dict(m=x[:, 0], g=x[:, 1], y=y),
                        index=pd.date_range("2025-01-01", periods=1100, freq="h", tz="UTC"))
    cutoff = data.index[900]
    altered = data.copy()
    altered.loc[altered.index >= cutoff, ["y", "m", "g"]] = 1e8
    errors = {}
    for candidate in experiment.candidates():
        family = candidate["family"]
        if family in errors:
            continue
        original = experiment.ConditionalModel(candidate).fit(data, cutoff)
        changed = experiment.ConditionalModel(candidate).fit(altered, cutoff)
        test = data.loc[data.index >= cutoff]
        np.testing.assert_allclose(original.predict(test), changed.predict(test), rtol=0, atol=1e-10)
        errors[family] = float(np.mean((test.y - original.predict(test)) ** 2))
    assert len(errors) == 9
    assert errors["quadratic_ridge"] < errors["joint_ols"] * .1
    print("PASS: all nine families ignore future fit inputs; known nonlinear signal recovered.")


if __name__ == "__main__":
    main()
