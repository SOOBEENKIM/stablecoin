# Relative price-grid sensitivity: completed option-1 extension

Read [Korean findings and conclusion](/home/ssd-990/soobeenkim/stablecoin_workshop/option1_resolution_20260910/RESULTS_KO.md).

All planned new experiments are complete: multi-reference classification sensitivity, equal relative price shocks, minimum integer grid moves, calendar/month and extreme-day influence, event-window robustness, common-control asset comparisons, and frozen-forecast evaluation scenarios. A full per-observation reporting panel is included. Existing 205 research artifacts are preserved.

The evidence supports a measurement/evaluation paper on unequal relative price grids. It does not establish false-alarm rates, a unique fair-price benchmark, a causal policy effect or AI superiority. The period has already been explored; the protocol is a within-project specification freeze, not independent preregistration. See `SUPPLEMENT_SCOPE_KO.md` for explicitly subsequent diagnostics.

Python environment: Python 3.8.10, numpy 1.23.5, pandas 2.0.3, scipy 1.10.1, statsmodels 0.14.1, matplotlib 3.7.5. Scripts import the preserved `option1_core_20260909/common.py` numerical helpers without modifying them. Exact input/output checksums are saved.

From this directory, reproduce in order (outputs here will be overwritten; preserve a copy for an exact historical run):

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-mpl
python3 measurement.py
python3 policy_robustness.py
python3 prediction_evaluation.py
python3 supplement.py
python3 risk_report.py
python3 verify.py
python3 figures.py
```

`measurement_panel.npz` stores raw valid prices, individual basis values and all one-coin perturbations. `minimum_tick_arrays.npz` uses 0 for originally non-risk observations, 1–100 for exact grid steps, 101 for more than 100 steps or no influence. Half-tick intervals are continuous sensitivity boxes, not executable quotes or latent-efficient-price confidence intervals.

The primary measurement family is exactly four differences in the all-observation flippable fraction at -10 bp, with paired calendar resampling and Holm correction. Policy pairs also have a shared-X supplement. Forecast models are not retrained; both receive the same perturbed target. Per-observation loss envelopes include all piecewise-linear breakpoints rather than subtracting separate models' best/worst values at different targets. The 24-test forecast adjustment is an explicitly secondary post-result summary.

`risk_reporting_panel.csv.gz` is a diagnostic artifact, not a trading rule. Raw classifications are retained; scenario-sensitive rows are not deleted or called errors. `VERIFICATION.json` records meaningful numerical/timing/preservation checks. `ARTIFACT_SHA256.json` records completed outputs except itself.

No manuscript has been overwritten or submitted. Workshop 3 remains paused.
