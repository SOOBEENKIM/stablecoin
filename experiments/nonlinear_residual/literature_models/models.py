"""A3 conditional-mean estimators, including manuscript-order extensions."""
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "three_way"))
import pipeline as followup

SEED = 20260924
FAMILIES = ["seq_xgb", "seq_mlp", "seq_spline", "joint_xgb", "joint_mlp", "joint_tensor"]
CONTROLS = ["sequential_ols", "joint_ols", "pca"]
STRATEGIES = ["seq_selected", "seq_shrunk", "joint_selected", "joint_shrunk"]
METHODS = CONTROLS + FAMILIES + STRATEGIES


def candidates():
    rows = []
    def add(family, **params):
        rows.append(dict(candidate_id="a%02d" % len(rows), family=family, params=params))
    for family in CONTROLS:
        add(family)
    for family in FAMILIES:
        if family.endswith("xgb"):
            for depth in [1, 2, 3]:
                for trees in [100, 300]:
                    add(family, max_depth=depth, n_estimators=trees)
        elif family.endswith("mlp"):
            for width in [8, 16]:
                for alpha in [1., 100.]:
                    add(family, width=width, alpha=alpha)
        else:
            for knots in [3, 5]:
                for alpha in [1., 100.]:
                    add(family, knots=knots, alpha=alpha)
    assert len(rows) == 31
    return rows


class Correction:
    """OLS anchor plus a learner fitted to training residuals, no future inputs."""
    def __init__(self, kind, params):
        self.kind, self.params, self.warnings = kind, params, []

    def basis(self, x):
        b = [t.transform(x[:, i:i+1]) for i, t in enumerate(self.transforms)]
        if len(b) == 2:
            b.append((b[0][:, :, None] * b[1][:, None, :]).reshape(len(x), -1))
        return np.concatenate(b, axis=1)

    def fit(self, x, y):
        z = np.column_stack([np.ones(len(x)), x])
        self.coef = np.linalg.lstsq(z, y, rcond=None)[0]
        target = y - z @ self.coef
        if self.kind == "xgb":
            self.learner = XGBRegressor(objective="reg:squarederror", learning_rate=.03,
                min_child_weight=30, reg_lambda=10., subsample=1., colsample_bytree=1.,
                tree_method="hist", n_jobs=1, random_state=SEED, **self.params)
            self.learner.fit(x, target)
        elif self.kind == "mlp":
            self.scaler = StandardScaler().fit(x)
            xx = self.scaler.transform(x)
            self.target_mean, self.target_sd = float(target.mean()), float(target.std())
            self.target_sd = max(self.target_sd, 1e-10)
            yy = (target - self.target_mean) / self.target_sd
            self.networks = []
            for seed in [SEED, SEED + 1, SEED + 2]:
                model = MLPRegressor(hidden_layer_sizes=(self.params["width"],), activation="tanh",
                    alpha=self.params["alpha"], solver="lbfgs", max_iter=500, max_fun=20000,
                    tol=1e-6, random_state=seed, early_stopping=False)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(xx, yy)
                self.warnings.extend(str(w.message) for w in caught)
                self.networks.append(model)
        else:
            self.transforms = [SplineTransformer(n_knots=self.params["knots"], degree=3,
                knots="uniform", extrapolation="linear", include_bias=False).fit(x[:, i:i+1])
                for i in range(x.shape[1])]
            b = self.basis(x)
            self.scaler = StandardScaler().fit(b)
            self.learner = Ridge(alpha=self.params["alpha"]).fit(self.scaler.transform(b), target)
        return self

    def predict(self, x):
        linear = np.column_stack([np.ones(len(x)), x]) @ self.coef
        if self.kind == "xgb":
            correction = self.learner.predict(x)
        elif self.kind == "mlp":
            correction = np.mean([m.predict(self.scaler.transform(x)) for m in self.networks], axis=0)
            correction = self.target_mean + self.target_sd * correction
        else:
            correction = self.learner.predict(self.scaler.transform(self.basis(x)))
        return linear + correction


class Model:
    def __init__(self, candidate):
        self.candidate, self.family = candidate, candidate["family"]

    def fit(self, data, cutoff):
        self.train = data.loc[data.index < cutoff]
        assert len(self.train) >= 100 and self.train.index.max() < cutoff
        self.warnings = []
        if self.family in CONTROLS:
            if self.family == "pca":
                self.model = followup.PCASequential().fit(self.train, cutoff)
            else:
                self.model = followup.expanded.ConditionalModel(dict(family=self.family, params={})).fit(self.train, cutoff)
        else:
            self.features = ["m"] if self.family.startswith("seq_") else ["m", "g"]
            x, y = self.train[self.features].to_numpy(), self.train.y.to_numpy()
            kind = "xgb" if self.family.endswith("xgb") else "mlp" if self.family.endswith("mlp") else "spline"
            self.model = Correction(kind, self.candidate["params"]).fit(x, y)
            self.warnings = self.model.warnings
            if self.family.startswith("seq_"):
                g = np.column_stack([np.ones(len(self.train)), self.train.g])
                self.second = np.linalg.lstsq(g, y - self.model.predict(x), rcond=None)[0]
        self.train_rmse = float(np.sqrt(np.mean((self.train.y - self.predict(self.train)) ** 2)))
        return self

    def predict(self, frame):
        if self.family in CONTROLS:
            result = self.model.predict(frame)
        else:
            result = self.model.predict(frame[self.features].to_numpy())
            if self.family.startswith("seq_"):
                result += np.column_stack([np.ones(len(frame)), frame.g]) @ self.second
        assert np.isfinite(result).all()
        return result
