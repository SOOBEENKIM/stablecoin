# Residual conditional calibration

See `PROTOCOL_KO.md` for the frozen scope, selection clocks and interpretation limits.

Run from the repository root with Python 3.8.10 and existing packages:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/residual_conditional_calibration/cc_prepare.py
python3 experiments/residual_conditional_calibration/test_cc.py
python3 experiments/residual_conditional_calibration/cc_core.py lock
# Commit/push the lock and source before running new forecasts.
python3 experiments/residual_conditional_calibration/cc_run.py --workers 12
python3 experiments/residual_conditional_calibration/cc_verify.py
python3 experiments/residual_conditional_calibration/cc_evaluate.py
```

Seals are exclusive-write and intentionally prevent overwriting finalized experiments. Caches in `.runs` carry source-lock and file hashes. Regeneration must use a separate output location with a documented run identity; do not delete published seals. Files ending in `RESULTS_KO.md` and a report renderer, if added after evaluation, are presentation only.
