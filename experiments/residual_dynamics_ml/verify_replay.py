"""Independently replay selected forecasts and verify the original source hashes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data import ROOT, build_panel, feature_columns, sha256
from models import ForecastModel
from run import prepare_splits
from diagnose_spline import ScaledSpline

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert sha256(HERE / name) == digest, name
    assert sha256(HERE / 'PROTOCOL_KO.md') == manifest['protocol_sha256']
    for name, digest in manifest['inputs'].items():
        assert sha256(ROOT / name) == digest, name
    assert json.loads((OUT / 'failures.json').read_text()) == []
    assert json.loads((OUT / 'warnings.json').read_text()) == []
    p = build_panel('NY_delay1')
    pred = pd.read_csv(OUT / 'predictions.csv.gz')
    pred.origin = pd.to_datetime(pred.origin, utc=True)
    diag = pd.read_csv(OUT / 'spline_diagnostic/predictions.csv.gz')
    diag.origin = pd.to_datetime(diag.origin, utc=True)
    cutoff = pd.Timestamp('2026-02-01', tz='UTC')
    differences = []
    for definition in ['EQ', 'CAP', 'PCA']:
        _, _, tr, te, _ = prepare_splits(p, definition, 'sequential', 1, cutoff)
        cols = feature_columns('full')
        for kind in ['linear', 'interaction', 'spline', 'boosting']:
            g = pred[(pred.scenario == 'NY_delay1') & (pred.definition == definition) &
                (pred.projection == 'sequential') & (pred.h == 1) & (pred.q == .1) &
                (pred.fold == '2026-02') & (pred.model == kind) & (pred['info'] == 'full')].sort_values('origin')
            assert list(g.origin) == list(te.index)
            params = json.loads(g.params.iloc[0])
            m = ForecastModel(kind, .1, params).fit(tr[cols], tr.target)
            replay = m.predict(te[cols])
            np.testing.assert_allclose(g.pred, replay, rtol=0, atol=1e-9)
            differences.append(dict(definition=definition, model=kind,
                max_abs_error_bp=float(np.abs(g.pred.to_numpy() - replay).max())))
        for variant, cls in [('weak_penalty', ForecastModel), ('scaled_basis', ScaledSpline)]:
            g = diag[(diag.definition == definition) & (diag.fold == '2026-02') &
                (diag['info'] == 'full') & (diag.variant == variant)].sort_values('origin')
            assert list(g.origin) == list(te.index)
            m = cls('spline', .1, {'alpha': float(g.alpha.iloc[0])}).fit(tr[cols], tr.target)
            replay = m.predict(te[cols])
            np.testing.assert_allclose(g.pred, replay, rtol=0, atol=1e-9)
            differences.append(dict(definition=definition, model=variant,
                max_abs_error_bp=float(np.abs(g.pred.to_numpy() - replay).max())))
    report = dict(original_sources_unchanged=True, original_protocol_unchanged=True,
        input_hashes_verified=True, failed_fits=0, solver_warnings=0,
        replay_month='2026-02', replay_model_definition_pairs=len(differences),
        replay=differences, script_sha256=sha256(Path(__file__)))
    (OUT / 'REPLAY_VERIFICATION.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
