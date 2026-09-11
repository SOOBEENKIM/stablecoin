# Option 1 controlled factorial experiment

Completed and independently verified: 96 monthly fits. This study isolates candidate placement and candidate-loss aggregation in TCN and GRU. Each architecture has tick/shuffled × mean/max cells, three fixed seeds, four monthly training windows. The entire set of 11 price candidates, clean-loss weight, initialization, batch order, shared dropout stream and number of optimizer updates are matched within each architecture/seed/month. Data and the future target are unchanged. See [the Korean findings](/home/ssd-990/soobeenkim/stablecoin_workshop/option1_factorial_20260910/RESULTS_KO.md) and FINAL_STATUS.json.

Financial placement and max-loss training show complementary effects in both architectures (interaction Holm19 p=.0095, Holm35 p=.0175). Within financial candidates, max versus mean reduces worst one-tick tail loss by 2.71% in TCN and 2.77% in GRU; clean loss changes by +0.0016% and +0.0904%, respectively. Both pass the original clean-accuracy noninferiority checks against their archived clean architecture and QR, but all four joint utility checks fail the unchanged 10% stress-loss reduction requirement. No independent predictive-accuracy superiority or true-price recovery is established.

The protocol is prospective to these new fits but retrospective to several previous experiments on the same historical evaluation period. It is exploratory, not an independent confirmation. Option 3 remains paused.

## Implementation distinctions

All candidates are processed in one grouped forward pass. Dropout masks are shared across the 11 candidates belonging to the same original example. The two objectives are `0.5 L(clean) + 0.5 mean(L(candidates))` and `0.5 L(clean) + 0.5 max(L(candidates))`, where each L averages q10/q50/q90 pinball loss in the original quantile-head order. Inference sorts output quantiles and scores the q10/q90 average as before.

This is a controlled new implementation, distinct from the earlier hard model's eval-mode candidate selection followed by train-mode clean/selected fitting. The earlier result is preserved as a reference, not advertised as a numerical reproduction of the new max cell.

Epoch counts are taken from the saved prior clean models' past-only validation, fixed before new outcomes and identical across the four cells. Conditions are not separately early-stopped. Therefore the factorial comparison tests objective effects at a fixed past-selected budget, not each method's individually tuned optimum.

## Files

- PROTOCOL_KO.md / PROTOCOL_LOCK.json: frozen experiment and exact hypotheses.
- contrasts.json: 19 prespecified contrasts, including the signed interactions.
- anchor_epochs.csv: 24 past clean-validation epoch choices and source hashes.
- data.npz: byte-identical validated data copy.
- PRESERVED_OPTIONS_SHA256.json: 518 prior artifacts protected.
- train.py / preflight.py / PREFLIGHT.json: grouped dropout, exact loss/gradient checks and paired-training controls.
- TRAIN_SETTINGS.json / TRAIN_COMPLETION.json / neural_training.csv / training_curves.csv: settings, 96 fits, objective components and candidate argmax counts.
- checkpoints/ / neural_predictions.npz: every trained model and all 31 scenario forecasts.
- analyze.py / all_scores.csv / monthly_scores.csv / evaluated_predictions.npz: new and archived model stages; archived results reproduced to 1e-12.
- calibration_offsets.npz / calibration_checks.csv: strictly past calibration for every new seed and ensemble.
- factorial_contrasts.csv / secondary_contrasts.csv / seed_contrasts.csv / monthly_contrasts.csv: primary effects, clean/stress-excess/sensitivity decomposition, all months/seeds.
- contrast_bootstrap.npz / inference_weights.npz: all paired day-block draws and weights.
- UTILITY_DECISION.csv: four utility comparisons under the unchanged practical success thresholds.
- verify.py / verify_checkpoints.py / VERIFICATION.json / CHECKPOINT_VERIFICATION.json: independent objective accounting, chronology, same-budget controls, 96 saved-model replays and dropout checks.
- factorial_interactions.png/pdf and factorial_effects.png/pdf: standalone figures.
- FINAL_STATUS.json / ARTIFACT_MANIFEST.json: completed study verdict and file inventory.

## Reproduction

Working directory: `/home/ssd-990/soobeenkim/stablecoin_workshop/option1_factorial_20260910`.

Host Python3.8.10, numpy1.23.5, pandas2.0.3, scipy1.10.1, statsmodels0.14.1, matplotlib3.7.5. CPU Docker runtime: existing `pi-deeponet-reproduction:py311`, PyTorch2.9.1+cpu, numpy2.3.0, pandas2.3.0. No network or GPU is used in the container. Preserve sibling study directories because validated data, epoch anchors, archived predictions/calibration and calendar-block helpers are referenced.

Archive generated outputs before re-running if byte-identical preservation is required. These commands overwrite this experiment's generated outputs only.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/stablecoin-mpl
docker run --rm --pull=never --network none --cpus 2 --memory 4g --user 1003:1003 -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop:/research:ro -v /home/ssd-990/soobeenkim/stablecoin_workshop/option1_factorial_20260910:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/preflight.py
docker run --rm --pull=never --network none --cpus 2 --memory 4g --user 1003:1003 -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop:/research:ro -v /home/ssd-990/soobeenkim/stablecoin_workshop/option1_factorial_20260910:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/train.py
python3 analyze.py
docker run --rm --pull=never --network none --cpus 2 --memory 4g --user 1003:1003 -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop:/research:ro -v /home/ssd-990/soobeenkim/stablecoin_workshop/option1_factorial_20260910:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/verify_checkpoints.py
python3 verify.py
python3 figures.py
python3 training_diagnostics.py
python3 finalize.py
```

Inference uses 1,999 common circular calendar-block bootstrap draws of 5 days. It conditions on stored monthly fits rather than refitting the models in every bootstrap sample. Holm19 covers the present primary comparisons; Holm35 adds the immediately preceding 16 comparisons but not the complete adaptive research history. Secondary intervals are individual exploratory intervals.

## Sources

- [Madry et al., robust optimization](https://arxiv.org/abs/1706.06083): conceptual background, not a claim to reproduce continuous PGD or global guarantees.
- [TCN](https://arxiv.org/abs/1803.01271) and [GRU](https://arxiv.org/abs/1412.3555): standard architectures.
- [PyTorch Dropout 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Dropout.html): standard random-mask behavior; this experiment implements candidate-shared masks explicitly.

The study makes no novelty, true-price recovery, realized trading-profit or real-world attack-frequency claim from these controlled input scenarios.
