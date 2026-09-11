> GitHub 탐색용 사본입니다. [체크섬이 보존된 원본](../../../research/option1_tick_learning_20260910/README.md)의 내용과 과거 시점 표기를 유지하고 문서 링크만 변환했습니다. 실행은 [현재 재현 안내](../../REPRODUCING.md)를 따릅니다.

# Tick-aware learning experiment, 2026-09-10

Completed option-1 AI extension. [Korean findings and decision](RESULTS_KO.md).

Tick-aware TCN reduced one-tick input sensitivity by 10.83% versus clean TCN, but observed-input loss increased 0.26% and primary worst-scenario loss increased 0.11%. None of the four locked joint success comparisons passed. This is a partial sensitivity result, not established predictive superiority. Option 3 remains paused.

## Main artifacts

| Files | Contents |
|---|---|
| PROTOCOL_KO.md; PROTOCOL_LOCK.json | Frozen before this experiment's training/outcomes |
| DATA_MANIFEST.json; data.npz; sample_counts.csv | Hashed raw inputs, sequences, fixed target, chronological splits |
| train.py; TRAIN_SETTINGS.json; TRAIN_COMPLETION.json | Neural training and actual runtime |
| checkpoints/; neural_training.csv; training_curves.csv | All 72 saved outer fits and inner epoch-selection traces |
| baselines.py; BASELINE_COMPLETION.json | Direct penalized linear QR and LightGBM |
| neural_predictions.npz; baseline_predictions.npz | All December–March predictions, including calibration warmup |
| evaluated_predictions.npz; calibration_offsets.npz | All 52 raw/calibrated model-stage predictions, January–March evaluation |
| all_scores.csv; monthly_scores.csv; seed_diagnostics.csv | Complete scores; no best-seed selection |
| PRIMARY_DECISION.csv; primary_bootstrap.npz; inference_weights.npz | Four locked comparisons, all draws and weights |
| legacy_common_origin_comparison.csv | Earlier models evaluated on the same 1,814 origins, clean only |
| diagnostics.py; DIAGNOSTIC_SCOPE.json | Post-result descriptive decomposition; no new primary tests |
| scenario_diagnostics.csv; worst_scenario_drivers.csv; secondary_decomposition.csv | All scenarios and exploratory sensitivity intervals |
| model_comparison.png/pdf; monthly_comparison.png/pdf | Standalone scientific figures |
| verify.py; verify_checkpoints.py; VERIFICATION.json; CHECKPOINT_VERIFICATION.json | Independent accounting and saved-model replay |
| PRESERVED_OPTIONS_SHA256.json | 260 older files preserved |
| ARTIFACT_MANIFEST.json; FINAL_STATUS.json | Final inventory and decision |

## Runtime and reproduction

Working directory: `/home/ssd-990/soobeenkim/stablecoin_workshop/option1_tick_learning_20260910`.

Host Python 3.8.10: numpy 1.23.5, pandas 2.0.3, scipy 1.10.1, scikit-learn 1.3.2, statsmodels 0.14.1, LightGBM 4.5.0, matplotlib 3.7.5. CPU training used the existing local image `pi-deeponet-reproduction:py311`, ID `sha256:3d32f82477e107e1b101af118b314c66055550400f46bffb073c0f9dc61d6d7a`, with torch 2.9.1+cpu, numpy 2.3.0, pandas 2.3.0. No network or GPU was used inside the training container.

The data preparation also checks the previous `extension_20260909/basis_definitions.csv`. Analysis imports calendar block utilities from `option1_core_20260909/common.py` and reads earlier forecasts from `extension_20260909/mean5_h6_forecasts.csv.gz`. Preserve this study's sibling directories to reproduce its older-result comparisons. Raw input paths/hashes are in DATA_MANIFEST.json.

Run in this order. Re-running overwrites this directory's generated experimental outputs; archive it first if retaining byte-identical artifacts is required.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-mpl
python3 prepare.py
docker run --rm --pull=never --network none --cpus 2 --memory 4g --user 1003:1003 -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop/option1_tick_learning_20260910:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/train.py
python3 baselines.py
python3 analyze.py
docker run --rm --pull=never --network none --cpus 2 --memory 4g --user 1003:1003 -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop/option1_tick_learning_20260910:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/verify_checkpoints.py
python3 verify.py
python3 diagnostics.py
python3 figures.py
```

Bootstrap inference conditions on saved monthly forecasts; it does not retrain neural nets inside each bootstrap sample. Confidence intervals are exploratory because this historical test period has been used before. Raw and calibrated results, all seeds and months remain accessible. Rolling residual calibration does not assert a distribution-free coverage guarantee under temporal dependence.

## Sources

- [Upbit official KRW tick change, July 31, 2025](https://docs.upbit.com/kr/changelog/krw_tick_unit_change_250731): authoritative post-change grid; grandfathered orders could remain. All prices actually used here pass grid checks.
- [Bai, Kolter and Koltun (2018), TCN](https://arxiv.org/abs/1803.01271): architecture background; no claim of a new neural architecture.
- [Robey et al. (2022), Probabilistically Robust Learning](https://proceedings.mlr.press/v162/robey22a.html): robust-learning context; this code uses augmentation with empirical pinball loss, not their exact algorithm or minimax optimization.

Sources revisited 2026-09-10. No claim of first use in the literature is based on this experiment alone.
