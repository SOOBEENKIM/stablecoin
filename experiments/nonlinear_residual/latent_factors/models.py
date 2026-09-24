"""A4: reference-only nonlinear factors and a state-dependent factor loading.

No USDT price, future observation or positioning variable enters the encoder.
"""
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "three_way"))
import pipeline as study

SEED = 20260925
SEEDS = [SEED, SEED + 1, SEED + 2]
CONTROLS = ["sequential_ols", "pca_1", "pca_2", "conditional_linear"]
FAMILIES = ["ae_relu_1", "ae_relu_2", "ae_tanh_1", "ae_tanh_2", "conditional_neural"]
STRATEGIES = ["pca_selected", "ae_selected", "ae_reconstruction_selected"]
METHODS = CONTROLS + FAMILIES + STRATEGIES


class ReconstructionMLP(MLPRegressor):
    """Consistent multioutput objective for the pinned sklearn 1.3.2 runtime.

    Its squared_loss averages over outputs whereas _backprop sums their
    gradients. Use 0.5 * mean(row squared norm) + the existing L2 penalty.
    This override is covered by a numerical gradient check, including L2.
    """
    def _init_coef(self, fan_in, fan_out, dtype):
        weights, biases = super()._init_coef(fan_in, fan_out, dtype)
        if self.activation == "relu":
            biases = np.ones_like(biases)
        return weights, biases

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        _, cg, ig = super()._backprop(X, y, activations, deltas, coef_grads, intercept_grads)
        loss = .5*np.mean(np.sum((activations[-1]-y)**2, axis=1))
        loss += .5*self.alpha*sum(np.sum(w*w) for w in self.coefs_)/len(X)
        return float(loss), cg, ig


def candidates():
    grid = []
    def add(family, **params):
        grid.append(dict(candidate_id="f%02d" % len(grid), family=family, params=params))
    for name in CONTROLS:
        add(name)
    for activation in ["relu", "tanh"]:
        for rank in [1, 2]:
            for alpha in [1., 100.]:
                add("ae_%s_%d" % (activation, rank), activation=activation, rank=rank, alpha=alpha)
    for width in [4, 8]:
        for penalty in [.001, .1]:
            add("conditional_neural", width=width, penalty=penalty)
    return grid


def activate(x, kind):
    return np.maximum(x, 0) if kind == "relu" else np.tanh(x)


def latent(network, x):
    # 5 -> 8 -> k -> 8 -> 5: return the bottleneck, not the reconstruction.
    h = activate(x @ network.coefs_[0] + network.intercepts_[0], network.activation)
    return activate(h @ network.coefs_[1] + network.intercepts_[1], network.activation)


def conditional_value_gradient(theta, z, c, y, width, penalty):
    """Analytic MSE gradient for a + z * [b0+b1*c+v'tanh(u*c+d)]."""
    a, b0, b1 = theta[:3]
    u, d, v = np.split(theta[3:], 3)
    h = np.tanh(c[:, None] * u + d)
    fitted = a + z * (b0 + b1*c + h @ v)
    error = fitted-y
    deriv = 2*error/len(y)
    grad = np.empty_like(theta)
    grad[:3] = [deriv.sum(), deriv @ z, deriv @ (z*c)]
    temp = (deriv*z)[:, None]*(1-h*h)*v
    grad[3:3+width] = (temp*c[:, None]).sum(axis=0)
    grad[3+width:3+2*width] = temp.sum(axis=0)
    grad[3+2*width:] = h.T @ (deriv*z)
    # Only the nonlinear departure is penalized; the nested linear model is free.
    loss = np.mean(error**2) + penalty*np.mean(theta[3:]**2)
    grad[3:] += 2*penalty*theta[3:]/(3*width)
    return float(loss), grad


class Model:
    def __init__(self, candidate):
        self.candidate = candidate
        self.family = candidate["family"]
        self.params = candidate["params"]

    def fit(self, data, cutoff):
        self.train = data.loc[data.index < cutoff]
        assert len(self.train) >= 100 and self.train.index.max() < cutoff
        t = self.train
        self.log = dict(warnings=[], members=[])
        # Canonical layout also makes nonlinear optimization invariant to pandas
        # block consolidation when an otherwise identical frame is copied.
        raw = np.ascontiguousarray(t[study.KCOLS].to_numpy(), dtype=np.float64)
        self.mean = raw.mean(axis=0)
        self.sd = raw.std(axis=0, ddof=1)
        assert (self.sd > 0).all()
        x = self.standardize(t)
        ev, vec = np.linalg.eigh(np.cov(x, rowvar=False))
        self.loading = vec[:, ::-1].copy()
        for j in range(5):
            if self.loading[:, j].sum() < 0:
                self.loading[:, j] *= -1
        self.log["pca_variance_shares"] = (ev[::-1]/ev.sum()).tolist()
        y = t.y.to_numpy()
        if self.family == "sequential_ols":
            self.first = np.linalg.lstsq(np.column_stack([np.ones(len(t)), t.m]), y, rcond=None)[0]
        elif self.family.startswith("pca_"):
            self.rank = int(self.family[-1])
            self.first = np.linalg.lstsq(np.column_stack([np.ones(len(t)), x @ self.loading[:, :self.rank]]), y, rcond=None)[0]
        elif self.family.startswith("ae_"):
            self.rank = self.params["rank"]
            self.members = []
            for seed in SEEDS:
                network = ReconstructionMLP(hidden_layer_sizes=(8, self.rank, 8),
                    activation=self.params["activation"], alpha=self.params["alpha"],
                    solver="lbfgs", max_iter=1500, max_fun=30000, tol=1e-6,
                    early_stopping=False, random_state=seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    network.fit(x, x)
                warn = [str(w.message) for w in caught]
                self.log["warnings"].extend(warn)
                f = latent(network, x)
                center, scale = f.mean(axis=0), f.std(axis=0, ddof=1)
                scale = np.maximum(scale, 1e-10)
                z = (f-center)/scale
                first = np.linalg.lstsq(np.column_stack([np.ones(len(t)), z]), y, rcond=None)[0]
                self.members.append((network, center, scale, first))
                self.log["members"].append(dict(seed=seed, iterations=int(network.n_iter_),
                    objective=float(network.loss_), reconstruction_mse=float(np.mean((x-network.predict(x))**2)),
                    latent_sd=f.std(axis=0, ddof=1).tolist(), warnings=warn))
        elif self.family.startswith("conditional_"):
            z, c = self.condition(t, fitting=True)
            self.ymean, self.ysd = y.mean(), max(y.std(ddof=1), 1e-10)
            yy = (y-self.ymean)/self.ysd
            base = np.linalg.lstsq(np.column_stack([np.ones(len(t)), z, z*c]), yy, rcond=None)[0]
            self.thetas = []
            if self.family == "conditional_linear":
                self.thetas.append(base)
            else:
                width, penalty = self.params["width"], self.params["penalty"]
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    initial = np.r_[base, rng.normal(0, .3, width), rng.normal(0, .1, width), np.zeros(width)]
                    result = minimize(conditional_value_gradient, initial,
                        args=(z, c, yy, width, penalty), jac=True, method="L-BFGS-B",
                        options=dict(maxiter=1500, maxfun=30000, ftol=1e-12, gtol=1e-6))
                    assert np.isfinite(result.fun) and np.isfinite(result.x).all()
                    self.thetas.append(result.x)
                    message = str(result.message)
                    self.log["members"].append(dict(seed=seed, iterations=int(result.nit),
                        objective=float(result.fun), success=bool(result.success), message=message,
                        max_gradient=float(np.abs(result.jac).max())))
                    if not result.success:
                        self.log["warnings"].append(message)
        else:
            raise ValueError(self.family)
        first = self.predict_first(t)
        g = np.column_stack([np.ones(len(t)), t.g])
        self.second = np.linalg.lstsq(g, y-first, rcond=None)[0]
        self.log["train_rmse_bp"] = float(np.sqrt(np.mean((y-self.predict(t))**2)))
        return self

    def standardize(self, frame):
        return (np.ascontiguousarray(frame[study.KCOLS].to_numpy(), dtype=np.float64)-self.mean)/self.sd

    def condition(self, frame, fitting=False):
        z = self.standardize(frame) @ self.loading[:, 0]
        # Dispersion of the five reference premiums, observed contemporaneously.
        # No downside, long/short or USDT variable is included here.
        raw = np.ascontiguousarray(frame[study.KCOLS].to_numpy(), dtype=np.float64)
        c = np.log1p(raw.std(axis=1, ddof=1))
        if fitting:
            self.zmean, self.zsd = z.mean(), max(z.std(ddof=1), 1e-10)
            self.cmean, self.csd = c.mean(), max(c.std(ddof=1), 1e-10)
        return (z-self.zmean)/self.zsd, (c-self.cmean)/self.csd

    def predict_first(self, frame):
        if self.family == "sequential_ols":
            return np.column_stack([np.ones(len(frame)), frame.m]) @ self.first
        if self.family.startswith("pca_"):
            f = self.standardize(frame) @ self.loading[:, :self.rank]
            return np.column_stack([np.ones(len(frame)), f]) @ self.first
        if self.family.startswith("ae_"):
            x = self.standardize(frame)
            return np.mean([np.column_stack([np.ones(len(x)), (latent(net, x)-center)/scale]) @ coef
                            for net, center, scale, coef in self.members], axis=0)
        z, c = self.condition(frame)
        pred = []
        for theta in self.thetas:
            a, b0, b1 = theta[:3]
            beta = b0 + b1*c
            if len(theta) > 3:
                u, d, v = np.split(theta[3:], 3)
                beta = beta + np.tanh(c[:, None]*u+d) @ v
            pred.append(a+z*beta)
        return self.ymean+self.ysd*np.mean(pred, axis=0)

    def predict(self, frame):
        pred = self.predict_first(frame) + np.column_stack([np.ones(len(frame)), frame.g]) @ self.second
        assert np.isfinite(pred).all()
        return pred

    def reconstruction_error(self, frame):
        if self.family == "sequential_ols":
            return np.full(len(frame), np.nan)  # EQ regression has no reference decoder.
        x = self.standardize(frame)
        if self.family.startswith("ae_"):
            fitted = np.mean([net.predict(x) for net, _, _, _ in self.members], axis=0)
        else:
            rank = int(self.family[-1]) if self.family.startswith("pca_") else 1
            fitted = x @ self.loading[:, :rank] @ self.loading[:, :rank].T
        return np.mean((x-fitted)**2, axis=1)

    def checkpoint(self):
        state = dict(family=self.family, candidate=self.candidate, mean=self.mean.tolist(),
            sd=self.sd.tolist(), loading=self.loading.tolist(), second=self.second.tolist(), log=self.log)
        if self.family.startswith("ae_"):
            state["members"] = [dict(weights=[w.tolist() for w in net.coefs_],
                biases=[v.tolist() for v in net.intercepts_], activation=net.activation,
                center=center.tolist(), scale=scale.tolist(), first=coef.tolist())
                for net, center, scale, coef in self.members]
        elif self.family.startswith("conditional_"):
            state.update(thetas=[t.tolist() for t in self.thetas],
                scales={k:float(getattr(self,k)) for k in ["zmean","zsd","cmean","csd","ymean","ysd"]})
        else:
            state["first"] = self.first.tolist()
        return state
