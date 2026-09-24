"""Bounded, explicitly post-result regularization/basis diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data import build_panel, feature_columns, sha256
from models import ForecastModel, pinball
from run import FOLDS, prepare_splits
from analyze import day_weights, boot_mean, interval

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/spline_diagnostic'


class ScaledSpline(ForecastModel):
    def prepare(self, x, fit=False):
        z = super().prepare(x, fit)
        if fit:
            self.basis_scaler = StandardScaler().fit(z)
        return self.basis_scaler.transform(z)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        exploratory_post_result=True, protocol_sha256=sha256(HERE / 'SPLINE_DIAGNOSTIC_KO.md'),
        source_sha256=sha256(Path(__file__)),
        initial_predictions_sha256=sha256(HERE / 'results/predictions.csv.gz'),
        candidates={'weak_penalty': [0, .001, .01], 'scaled_basis': [0, .001, .005, .01, .05]}
    ), indent=2), encoding='utf-8')
    p = build_panel('NY_delay1')
    rows, choices, diagnostics = [], [], []
    for definition in ['EQ', 'CAP', 'PCA']:
        for cutoff in FOLDS:
            ti, va, tr, te, _ = prepare_splits(p, definition, 'sequential', 1, cutoff)
            for info in ['base', 'full']:
                cols = feature_columns(info)
                for variant, cls, alphas in [
                    ('weak_penalty', ForecastModel, [0, .001, .01]),
                    ('scaled_basis', ScaledSpline, [0, .001, .005, .01, .05])]:
                    candidates = []
                    for a in alphas:
                        model = cls('spline', .1, {'alpha': a}).fit(ti[cols], ti.target)
                        loss = pinball(va.target, model.predict(va[cols]), .1).mean()
                        candidates.append((loss, a))
                        choices.append(dict(definition=definition, fold=cutoff.strftime('%Y-%m'),
                            info=info, variant=variant, alpha=a, validation_loss_bp=loss))
                    _, a = min(candidates)
                    model = cls('spline', .1, {'alpha': a}).fit(tr[cols], tr.target)
                    pred = model.predict(te[cols])
                    d = te[['target', 'target_time']].copy()
                    d['pred'], d['loss'] = pred, pinball(te.target, pred, .1)
                    d['below'] = (te.target.to_numpy() < pred).astype(int)
                    for k, v in dict(definition=definition, fold=cutoff.strftime('%Y-%m'),
                                    info=info, variant=variant, alpha=a).items():
                        d[k] = v
                    rows.append(d.reset_index())
                    diagnostics.append(dict(definition=definition, fold=cutoff.strftime('%Y-%m'),
                        info=info, variant=variant, alpha=a,
                        nonzero_coefficients=int((np.abs(model.model.coef_) > 1e-10).sum()),
                        prediction_sd_bp=float(np.std(pred))))
            print('DONE', definition, cutoff.strftime('%Y-%m'), flush=True)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(OUT / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(choices).to_csv(OUT / 'candidates.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / 'fit_diagnostics.csv', index=False)
    original = pd.read_csv(HERE / 'results/predictions.csv.gz')
    original.origin = pd.to_datetime(original.origin, utc=True)
    original = original[(original.scenario == 'NY_delay1') & (original.projection == 'sequential') &
                        (original.h == 1) & (original.q == .1)]
    metrics, comparisons = [], []
    for (definition, variant, info), g in d.groupby(['definition', 'variant', 'info']):
        g = g.sort_values('origin').reset_index(drop=True)
        w = day_weights(g, 5)
        metrics.append(dict(definition=definition, variant=variant, info=info,
            n=len(g), loss_bp=g.loss.mean(), below_rate=g.below.mean()))
        refs = [('linear', original[(original.definition == definition) &
                    (original.model == 'linear') & (original['info'] == info)])]
        if info == 'full':
            refs.append(('same_variant_base', d[(d.definition == definition) &
                (d.variant == variant) & (d['info'] == 'base')]))
        for ref, b in refs:
            b = b.sort_values('origin').reset_index(drop=True)
            assert g.origin.equals(b.origin)
            np.testing.assert_allclose(g.target, b.target, atol=1e-10)
            diff = g.loss.to_numpy() - b.loss.to_numpy()
            boot = boot_mean(w, diff)
            lo, hi = interval(-100 * boot / boot_mean(w, b.loss))
            comparisons.append(dict(definition=definition, variant=variant, info=info,
                reference=ref, improvement_pct=100 * (1 - g.loss.mean() / b.loss.mean()),
                improvement_lo=lo, improvement_hi=hi, exploratory=True))
    pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv', index=False)
    c = pd.DataFrame(comparisons)
    c.to_csv(OUT / 'comparisons.csv', index=False)
    print(c.to_string(index=False))


if __name__ == '__main__':
    main()
