# 2026-09-09 literature and empirical reassessment

Start with [RESEARCH_REASSESSMENT_KO.md](/home/ssd-990/soobeenkim/stablecoin_workshop/reanalysis_20260909/RESEARCH_REASSESSMENT_KO.md).

This is a conditional sensitivity study, not a publication-ready replacement for the submitted manuscript. Bloomberg timezone and bar availability metadata could not be recovered. Four alternative timezone assignments are examined; none is identified as the true assignment. A fifth timing sensitivity adds an hour of macro publication delay, for point estimates only. The archived and quote-only scenarios retain irregular row lags deliberately to isolate the quote correction.

The six primary bootstrap scenarios each use 499 circular moving-calendar-block resamples, with a block length of five days. Both original sequential OLS residual stages are refitted inside each resample. All model/lead pairs are constructed on real dates before resampling. Quantile regressions use frequency weights and minimize check loss with scipy HiGHS. Percentile intervals are descriptive sensitivity intervals; the sign-tail statistic in raw CSV files is not a calibrated null-imposed p-value. Intervals do not adjust for searching across scenarios, horizons or models. Time-series stationarity and block-length assumptions remain limitations.

`comparison_intervals.csv` combines the six main interval files (162 rows). `matched_origin_horizons.csv` uses identical predictor dates at all four horizons and reports effects in basis points per predictor standard deviation. Two timezone assignments have no common origins at all four horizons under the specified freshness rule; this is a missing-data limitation, not a zero effect.

Additional sensitivity checks: under the all-New-York assumption, the h=1 analysis was repeated with one-day blocks and with joint rather than sequential OLS residuals. Neither restores a 95% interval excluding zero for ACCOUNT_LS.

Run from `/home/ssd-990/soobeenkim`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 stablecoin_workshop/reanalysis_20260909/recalculate.py check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 stablecoin_workshop/reanalysis_20260909/recalculate.py points
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 stablecoin_workshop/reanalysis_20260909/recalculate.py matched
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 stablecoin_workshop/reanalysis_20260909/recalculate.py bootstrap --scenario clock_NY_quote --B 499 --block-days 5 --horizons 1 6 12
```

Repeat the last command with archived, quote_only, clock_UTC_quote, clock_FXUTC_USNY_quote, and clock_FXKST_USNY_quote for the other primary comparisons. The seed is fixed at 20260909. Original inputs are read only; outputs are written to this directory. The numerical self-checks test quote units, sparse-clock links, backward availability, weighted QR versus literal resampling, and archived/quote-only point reproduction. Input hashes were verified unchanged after calculations. ML forecasting, final manuscript revisions, all original CAP/PCA/outlier/subperiod analyses, joint cross-quantile/horizon tests, and a full publication inference protocol have not been completed by this diagnostic.
