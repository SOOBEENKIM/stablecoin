# Stablecoin workshop extension

Read [RESEARCH_FINDINGS_KO.md](RESEARCH_FINDINGS_KO.md) for the current findings and limits. [WORKSHOP_DRAFT.md](WORKSHOP_DRAFT.md) is the editable English paper; [WORKSHOP_DRAFT.pdf](WORKSHOP_DRAFT.pdf) is its review rendering. The PDF is a review draft, not a claim of compliance with a final conference template or completed author review.

The study evaluates reference-sensitive Korean USDT relative-price targets, chronological quantile forecasts, information ablations and delayed-feedback calibration. It does not show that the tested boosting model or offshore signals outperform strong adaptive baselines. It does identify material reference-asset sensitivity and improved risk-frequency calibration.

Run from this directory, with Python 3.8 and numpy 1.23.5, pandas 2.0.3, scipy 1.10.1, scikit-learn 1.3.2, LightGBM 4.5.0. Matplotlib, Pillow and ReportLab are required only for figures/PDF.

```bash
export MPLCONFIGDIR=/tmp/stablecoin-mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 timing_checks.py
python3 measurement_diagnostics.py
python3 run_extension.py --workers 3
python3 analyze_extension.py
python3 supplementary_analysis.py
python3 grid_sensitivity.py
python3 build_pdf.py
```

These commands overwrite generated files in this extension directory; they do not overwrite source inputs or the earlier pilot. `PROTOCOL_KO.md` and `RUN_MANIFEST.json` record the frozen main specification and input/code hashes. `timing_checks.py` contains three meaningful information-timing checks, including invariance to outcomes not yet observed.

Primary results: `all_scores.csv`, `paired_inference.csv`, and each cell's `*_monthly.csv`, `*_states.csv`, `*_training.csv`, `*_metadata.json`, and `*_forecasts.csv.gz`. Five cells are reported without choosing the best target or horizon. Main bootstrap comparisons use 1,999 five-calendar-day blocks; alternative block lengths, six nonoverlapping origin phases and event-influence deletions are recorded. The supplementary inference uses the same predictions and is not a new fitted specification.

Measurement and figure data: `basis_definitions.csv`, `basis_distribution.csv`, `extreme_reference_comparison.csv`, `tail_definition_overlap.csv`, `price_grid_monthly.csv`, `tick_sensitivity_monthly.csv`, and `*_figure_data.csv`. The one-tick results are deterministic sensitivity calculations, not identified causal policy effects. Full predicted-observation and source-data counts differ; see the paper before mixing them.

The same overall period was inspected in a prior pilot. This is exploratory chronological evidence, not independent external validation. Review the Korean memo before treating this as a submission package.
