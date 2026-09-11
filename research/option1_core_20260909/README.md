# Option 1 core checks, 2026-09-09

Completed pretrend, controlled event-window, and same-sample economic coefficient comparisons.

Read [Korean findings](/home/ssd-990/soobeenkim/stablecoin_workshop/option1_core_20260909/RESULTS_KO.md). The conditional DOGE-versus-ETH gap increase survives the primary controls and price-level sensitivity. None of the 24 primary economic coefficient contrasts passes joint Holm correction. Neither a causal leverage effect nor new AI forecast superiority is established.

The initial protocol was frozen before fitting these regressions, after prior exploratory work on the same historical sample. This is not independent preregistration or an untouched confirmatory period. The price-level supplement was specified after inspecting the primary results and is explicitly secondary.

Reproduce in this directory, with Python 3.8.10, numpy 1.23.5, pandas 2.0.3, scipy 1.10.1, statsmodels 0.14.1, and matplotlib (exact runtime versions in `ENVIRONMENT.json`). Commands overwrite outputs in this core directory; copy it first to preserve a particular run. They do not modify the raw data or the previous option directories.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-mpl
python3 prepare_data.py
python3 policy.py
python3 economics.py
python3 supplement.py
python3 verify.py
python3 figures.py
```

Economic QR uses exact, unpenalized HiGHS linear programming, with independent primal/dual checks. Two single-thread workers perform paired calendar-block resampling. The point models, raw bootstrap arrays, integer resampling weights and rank failures are all retained. All primary economic contrasts use the same within-horizon origins, features and weights across five reference constructions.

Bootstrap percentile intervals and centered approximate p-values are separately reported; they are not mathematical inversions of each other. Holm correction covers exactly 24 prespecified primary economic differences. Individual intervals are not simultaneous confidence intervals. Bootstrap failures are retained as NaN rather than replaced.

`VERIFICATION.json` contains numerical, timing, calendar covariance and preservation checks. `PRESERVED_OPTIONS_SHA256.json` covers 150 prior research artifacts. `ARTIFACT_SHA256.json` records the completed core directory files other than itself. Figures have PNG previews and vector PDF exports.

No manuscript was overwritten or submitted. Workshop 3 remains paused.
