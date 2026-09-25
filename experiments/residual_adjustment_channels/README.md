# Residual adjustment channels

New extension preserving the original common-factor regression residual. Tests how relative premium/discount states relate to actual domestic USDT and coin-implied price paths, with past-only random-forest nuisance control and linear sensitivity.

Read [the frozen protocol](PROTOCOL_KO.md) and [reference boundaries](REFERENCES.md). The existing December–March periods have been reused; this is exploratory development, not independent confirmation. Later `RESULTS_KO.md` will report the full results.

From the repository root, Python3.8.10/numpy1.23.5/pandas2.0.3/sklearn1.3.2:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/residual_adjustment_channels/ac_prepare.py
python3 experiments/residual_adjustment_channels/test_ac.py
python3 experiments/residual_adjustment_channels/ac_core.py lock
# Commit/push the lock and its files before the next command.
python3 experiments/residual_adjustment_channels/ac_run.py --workers 12
python3 experiments/residual_adjustment_channels/ac_verify.py
python3 experiments/residual_adjustment_channels/ac_evaluate.py
```

Exclusive output creation protects the archived run. Use a separate checkout/destination for a full reproduction, preserving the published evidence. Re-running verification does not select a model or change conclusions, but its exclusive output file must be handled in that reproduction checkout.
