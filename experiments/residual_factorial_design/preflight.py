"""Check data, calendar, and train-only regime availability; fit NO forecaster.

The existing monthly linear residualizer is used to define rows and count
threshold-regime training observations. No new forecast or loss is generated.
"""
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from mechanics import (DESIGN, HERE, calibrate_candidate, candidate_specs,
                       feature_sets, with_calendar)

ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'residual_nested_validation'))
import nested as legacy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = HERE / 'preflight'
    if out.exists():
        raise RuntimeError('Refusing to overwrite an existing preflight')
    # Read only sample identifiers from the archived forecasts, never scores.
    old_file = HERE.parent / 'residual_nested_validation/results/predictions.csv.gz'
    old = pd.read_csv(old_file, usecols=['mode', 'definition', 'origin', 'target_time'])
    old.origin = pd.to_datetime(old.origin, utc=True)
    old.target_time = pd.to_datetime(old.target_time, utc=True)
    start = pd.Timestamp(DESIGN['evaluation_start'])
    end = pd.Timestamp(DESIGN['evaluation_end_exclusive'])
    folds = pd.date_range(DESIGN['candidate_stream_start'], end - pd.offsets.MonthBegin(1), freq='MS')
    rows, schedules, calibration_rows = [], [], []
    for mode, definition in DESIGN['cases']:
        panel = legacy.a.panel(mode)
        stream = []
        for cutoff in folds:
            train, test, residualizer = legacy.outer_sample(panel, mode, definition, cutoff)
            train, test = with_calendar(train), with_calendar(test)
            features = feature_sets(mode)
            assert np.isfinite(train[features['F']].to_numpy()).all()
            assert np.isfinite(test[features['F']].to_numpy()).all()
            assert train.target_time.max() < cutoff
            assert residualizer.fit_end < cutoff
            assert ((test.target_time - test.index) == pd.Timedelta(hours=1)).all()
            inner = test[test.target_time < cutoff + pd.offsets.MonthBegin(1)]
            for window in DESIGN['windows']:
                fit = legacy.a.window(train, cutoff, window)
                for band in [[.2, .8], [.35, .65]]:
                    cuts = np.quantile(fit.e_now, band)
                    counts = np.bincount(np.searchsorted(cuts, fit.e_now, side='right'), minlength=3)
                    assert counts.min() >= DESIGN['threshold_minimum_regime_training_rows']
                    rows.append(dict(mode=mode, definition=definition, month=cutoff.strftime('%Y-%m'),
                        window=window, band=str(band), train_n=len(fit), forecast_n=len(test),
                        inner_eligible_n=len(inner), smallest_threshold_regime=int(counts.min()),
                        residual_fit_end=residualizer.fit_end.isoformat(),
                        train_last_label=fit.target_time.max().isoformat(),
                        features_R=len(features['R']), features_B=len(features['B']), features_F=len(features['F'])))
            stream.append(pd.DataFrame(dict(origin=test.index, target_time=test.target_time.to_numpy(),
                                            target=0., pred_raw=0.)))
            schedules.append(dict(mode=mode, definition=definition, month=cutoff.strftime('%Y-%m'),
                                  train_n=len(train), forecast_n=len(test), inner_eligible_n=len(inner)))
        stream = pd.concat(stream, ignore_index=True)
        # Zero-error synthetic stream on real timestamps: counts/availability only.
        cal = calibrate_candidate(stream)
        scored = cal[cal.origin >= pd.Timestamp(DESIGN['inner_scoring_start'])]
        assert scored.calibration_n.min() >= DESIGN['calibration']['minimum']
        new_sample = stream[(stream.origin >= start) & (stream.target_time < end)][['origin', 'target_time']]
        old_sample = old[(old['mode'] == mode) & (old.definition == definition) &
                         (old.origin >= start) & (old.target_time < end)][['origin', 'target_time']].drop_duplicates()
        pd.testing.assert_frame_equal(new_sample.sort_values('origin').reset_index(drop=True),
                                      old_sample.sort_values('origin').reset_index(drop=True))
        assert len(new_sample) == DESIGN['expected_outer_n'][mode]
        calibration_rows.append(dict(mode=mode, definition=definition,
            first_inner_origin=scored.origin.min().isoformat(),
            minimum_available_calibration_errors=int(scored.calibration_n.min()),
            outer_n=len(new_sample), archived_outer_origins_match=True))
    specs = candidate_specs()
    # Grid parity is checked without fitting old/new models.
    for kind in DESIGN['families']:
        old_grid = (legacy.threshold_candidates() if kind == 'threshold' else legacy.a.candidates(kind))
        actual = [(s['window'], s['form'], s['params']) for s in specs if s['kind'] == kind]
        assert actual == [(str(w), f, p) for w, f, p in old_grid]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        forecast_models_fitted=0, performance_scores_computed=0,
        residualizers_fitted_for_row_and_regime_checks=len(schedules),
        candidate_count=len(specs), calibration_availability=calibration_rows,
        minimum_threshold_regime_rows=min(r['smallest_threshold_regime'] for r in rows),
        environment=dict(python=sys.version.split()[0], **{n: importlib.import_module(n).__version__
                         for n in ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib']}),
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in legacy.a.INPUT_PATHS},
        archived_origin_source_sha256={str(old_file.relative_to(ROOT)): sha(old_file)},
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob('*.py'))},
        design_sha256=sha(HERE / 'design.json'))
    out.mkdir()
    pd.DataFrame(rows).to_csv(out / 'regime_and_feature_counts.csv', index=False)
    pd.DataFrame(schedules).to_csv(out / 'monthly_availability.csv', index=False)
    (out / 'PREFLIGHT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('sha256')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
