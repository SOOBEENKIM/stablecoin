> GitHub 탐색용 사본입니다. [체크섬이 보존된 원본](../../../research/option1_model_followup_20260910/README.md)의 내용과 과거 시점 표기를 유지하고 문서 링크만 변환했습니다. 실행은 [현재 재현 안내](../../REPRODUCING.md)를 따릅니다.

# Option 1: architecture and hard-scenario follow-up

Completed and independently verified. [Korean results](RESULTS_KO.md). GRU improves over the earlier TCN. Hard-tick TCN reduces observed-input loss by 1.63%, one-hour worst-scenario loss by 4.46%, and sensitivity by 78.08% versus the earlier tick-augmentation TCN. Worst-loss differences survive Holm12/16 correction, but the locked 10% worst-loss reduction target is not met. All 96 new checkpoints were replayed. This is a positive exploratory result, not an independent confirmatory study or a full joint-gate success. Option 3 remains paused.

The three architectures receive identical 24-hour × 30-feature inputs and have similar parameter counts (TCN 5,923; GRU 5,673; MLP 5,795). GRU and MLP use clean, tick, and magnitude-matched shuffled augmentation. TCN additionally uses hard_tick and hard_shuffled. The hard procedure selects the highest-loss candidate in evaluation mode, using historical training labels, then optimizes a 50/50 mixture of clean and selected losses in training mode. It does not select scenarios using future test labels during training or deployment.

The candidate family in hard learning contains clean plus 10 latest-hour one-tick changes. Candidate losses average q10/q50/q90; the primary evaluation metric averages q10/q90. This distinction and the change from 1/6-hour random augmentation to 1-hour hard selection are explicit in PROTOCOL_KO.md. The comparison changes the learning procedure; it is not a pure causal estimate of one objective coefficient.

## Artifacts

- PROTOCOL_KO.md / PROTOCOL_LOCK.json: frozen before new training.
- PRESERVED_OPTIONS_SHA256.json: 381 previous artifacts remain identical.
- data.npz: byte-identical copy of the previous validated dataset.
- preflight.py / PREFLIGHT.json: parameter counts, equal candidate budgets, independent hardest-case selection and gradient checks.
- train.py / TRAIN_SETTINGS.json / TRAIN_COMPLETION.json: training implementation and actual runtime status.
- neural_training.csv / training_curves.csv / checkpoints/: 96 outer fits, inner selection and outer refit traces, all saved weights.
- neural_predictions.npz: all 3-seed predictions for the new 8 procedures, 4 months, 31 input scenarios.
- analyze.py / all_scores.csv / monthly_scores.csv / seed_diagnostics.csv: complete new and previous comparisons, no best-seed selection.
- evaluated_predictions.npz / calibration_offsets.npz: all raw/calibrated model stages and strict-past calibration.
- PRIMARY_DECISION.csv / primary_bootstrap.npz / inference_weights.npz: 12 locked comparisons, same day-block draws; Holm12 and supplementary Holm16.
- verify_checkpoints.py / CHECKPOINT_VERIFICATION.json / checkpoint_replay.csv: all 96 checkpoints replayed; GRU and TCN causal-prefix checks.
- verify.py / VERIFICATION.json: chronological splits, identical targets, hard-selection accounting, independent scores/calibration/decision checks.
- model_comparison.png/pdf and monthly_comparison.png/pdf: standalone scientific figures.
- decompose.py / descriptive_decomposition.csv / DECOMPOSITION_SCOPE.json: post-result accounting of stress excess loss; no new primary metric or test.
- RESULTS_KO.md / FINAL_STATUS.json / ARTIFACT_MANIFEST.json: final interpretation and checksummed inventory.

## Runtime and reproduction

Host: Python3.8.10, numpy1.23.5, pandas2.0.3, scipy1.10.1, statsmodels0.14.1, matplotlib3.7.5. Neural runtime: existing local Docker image pi-deeponet-reproduction:py311, PyTorch2.9.1+cpu / numpy2.3.0 / pandas2.3.0. Image ID sha256:3d32f82477e107e1b101af118b314c66055550400f46bffb073c0f9dc61d6d7a. CPU only, no container network.

Working directory: `/home/ssd-990/soobeenkim/stablecoin_workshop/option1_model_followup_20260910`.

Preserve sibling directories: analysis imports calendar/calibration utilities from the previous experiments and reconstructs their original double-precision metrics from saved forecasts and offsets. Those 52 original model-stage metrics must match to 1e-12 before comparisons proceed. The previous 8 primary models remain in every score table.

Archive current generated outputs before re-running if byte-identical retention is needed. Commands overwrite this follow-up's generated files only.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-mpl
docker run --rm --pull=never --network none --cpus 2 --memory 4g --user 1003:1003 -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop:/research:ro -v /home/ssd-990/soobeenkim/stablecoin_workshop/option1_model_followup_20260910:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/preflight.py
docker run --rm --pull=never --network none --cpus 2 --memory 4g --user 1003:1003 -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop:/research:ro -v /home/ssd-990/soobeenkim/stablecoin_workshop/option1_model_followup_20260910:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/train.py
python3 analyze.py
docker run --rm --pull=never --network none --cpus 2 --memory 4g --user 1003:1003 -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop:/research:ro -v /home/ssd-990/soobeenkim/stablecoin_workshop/option1_model_followup_20260910:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/verify_checkpoints.py
python3 verify.py
python3 figures.py
python3 decompose.py
```

No bootstrap refitting is performed; inference conditions on stored monthly fits. Months and seeds are fully disclosed. Holm16 covers this follow-up's 12 comparisons and the immediately preceding 4, not the full history of adaptive experiments on this dataset.

## Method references

- [Chung et al. (2014), GRU comparison](https://arxiv.org/abs/1412.3555): gated recurrent architecture background, not evidence of superiority in this market.
- [Bai et al. (2018), TCN](https://arxiv.org/abs/1803.01271): the earlier architecture's background.
- [Madry et al. (ICLR 2018), robust optimization](https://arxiv.org/abs/1706.06083): conceptual basis for difficult-perturbation training. The current finite price-scenario selection is not their continuous PGD algorithm or a reproduction of their guarantees.

Sources checked 2026-09-10. The experiment does not establish a new architecture, first use, or a cause for earlier predictive failures.
