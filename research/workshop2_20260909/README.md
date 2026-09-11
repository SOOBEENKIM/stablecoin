# 워크숍 2안

[연구 방향·최신 문헌·실험 보고서](/home/ssd-990/soobeenkim/stablecoin_workshop/workshop2_20260909/WORKSHOP2_DESIGN_AND_EVIDENCE_KO.md)를 먼저 읽는다.

첫 발생 경보 파일럿에서 LightGBM 우위는 확인되지 않았다. 후속 GRU·TCN·MLP 비교에서 TCN은 LightGBM보다 좋은 점추정을 보였지만, 현재 특징 로지스틱 대비 추가 가치는 입증하지 못했다. 3 seed 평균·월별 과거 학습·강한 비교군을 사용한 제한된 탐색이다. 이전부터 본 시장 기간을 사용하므로 독립적인 외부 검증이 아니다.

## 결과 파일

- `PILOT_PROTOCOL_KO.md`, `SEQUENCE_PROTOCOL_KO.md`: 각각 결과 계산 전 고정한 사양. 후자는 첫 파일럿 이후 추가한 비교임을 명시한다.
- `event_feasibility.csv`, `onset_feasibility.csv`, `training_thresholds.csv`: 사건 수와 임계값. origin 수는 독립 사건 수가 아니다.
- `pilot_scores.csv`, `pilot_paired_inference.csv`, `onset_forecasts.csv.gz`: 첫 경보 파일럿의 전체 결과.
- `sequence_scores.csv`, `sequence_inference.csv`, `sequence_monthly.csv`: 같은 표본에서 신경망·비교군·개별 seed 결과.
- `sequence_all_forecasts.csv.gz`: 전체 예측·실현 라벨·관측 시각.
- `sequence_training.csv`, `sequence_training_curves.csv`: seed·월별 적합과 내부 검증 학습 곡선.
- `sequence_data.npz`: 실제 연속 24시간 × 33변수, 6,218개 적격 origin. 과거 위험 상태 시간도 포함한다.
- `model_comparison.png`, `model_comparison.pdf`: 비교 그림. 원수치는 `figure_scores.csv`, `figure_tcn_comparisons.csv`. 그림의 구간은 개별 95% 구간이고 Holm p는 추론 CSV를 따른다.
- `references.csv`: 18개 출처와 접근 범위.
- `FINAL_VERIFICATION.json`: 원자료·이전 결과 보존과 6개 검사 기록.

## 재현

호스트 Python 3.8 / numpy 1.23.5 / pandas 2.0.3 / sklearn 1.3.2 / LightGBM 4.5.0. 이 디렉터리에서 아래 파일들을 순서대로 실행한다.

```bash
python3 feasibility.py
python3 onset_counts.py
python3 run_pilot.py
python3 analyze_pilot.py
python3 prepare_sequences.py
```

신경망은 이미 설치된 Docker 이미지에서 CPU로 실행한다. 기존 실행 컨테이너나 작업을 변경하지 않았다.

```bash
docker run --rm --network none --cpus 2 --memory 4g --user 1003:1003 -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -v /home/ssd-990/soobeenkim/stablecoin_workshop/workshop2_20260909:/work -w /work --entrypoint python pi-deeponet-reproduction:py311 /work/train_sequences.py
```

이미지 ID: `sha256:3d32f82477e107e1b101af118b314c66055550400f46bffb073c0f9dc61d6d7a`. PyTorch 2.9.1+cpu / numpy 2.3.0. 런타임·해시는 `SEQUENCE_RUN_MANIFEST.json`에 기록한다. gzip·npz 파일 해시는 파일 메타데이터에 따라 달라질 수 있으므로 재현 비교는 내부 수치도 확인한다.

학습 후 호스트에서 `analyze_sequences.py`, `plot_results.py`를 실행한다. 검사: 호스트의 `timing_checks.py`, PyTorch 이미지의 `sequence_checks.py`. plotting/import에는 `MPLCONFIGDIR=/tmp/stablecoin-mpl`을 지정한다. 보존된 1안은 `extension_20260909/`에 있다.
