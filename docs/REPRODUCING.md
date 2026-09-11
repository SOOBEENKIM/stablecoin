# 재현 방법

이 저장소는 역사적 연구 산출물과 실행 도구를 함께 제공한다. `research/`의 파일은 당시 체크섬을 유지한다. 재현 도구는 매 실행마다 `.runs/<단계-시각>/research/`에 복사하고 그 안에서 실행한다. 실험별로 **당시 보존된 선행 입력**을 사용하므로, 앞 단계의 새 출력으로 뒤 단계의 과거 체크섬을 무심코 바꾸지 않는다.

## 1. 자료 내려받기와 무결성

Git LFS를 설치한 환경에서 저장소 루트에 진입한다.

```bash
git lfs pull
python3 scripts/verify_archive.py
```

917개 연구 파일, 최신 단계의 141개 manifest 항목, 이전 단계의 보존 항목 518개, 추가 감사 자료를 검증한다. 두 큰 파일이 작은 LFS 포인터로만 내려와 있으면 검사가 실패하므로 `git lfs pull`을 먼저 완료한다.

## 2. 분석 환경

원래 수치 분석 환경은 Python **3.8.10**, NumPy 1.23.5, pandas 2.0.3, SciPy 1.10.1, statsmodels 0.14.1, scikit-learn 1.3.2, LightGBM 4.5.0이다. Linux에서 Python 3.8 환경을 사용한다. 현재 최신 Python에 이 과거 패키지 조합을 그대로 설치하는 방식은 재현 환경이 아니다.

```bash
python3.8 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r environments/requirements-analysis.txt
```

Pillow·ReportLab은 초기 원고 PDF 생성용이다. 분석과 그림 재현의 설치 버전을 함께 기록했다. 원래 서버의 패키지 환경과 새 신경망 Docker 환경의 실행 기록은 [패키지 검증](../provenance/PACKAGE_VERIFICATION.json)을 따른다.

## 3. 원자료 전처리와 저장 결과 확인

```bash
python scripts/reproduce.py prepare-data
python scripts/reproduce.py core-checks
python scripts/reproduce.py measurement-checks
python scripts/reproduce.py latest-analysis
```

- `prepare-data`: 두 거래소 원자료로 24시간×30개 특징과 실제 6시간 뒤 목표를 다시 만들고 **모든 저장 배열이 기존 배열과 정확히 같은지** 검사한다.
- `core-checks`: 사전 추세·통제 회귀·경제적 계수 단계의 수치·추론 검증을 실행한다.
- `measurement-checks`: 측정·호가·고정 예측 평가 단계의 검증을 실행한다.
- `latest-analysis`: 저장된 예측에서 최신 19개 대비·상호작용·성공 기준을 다시 계산하고 검증·그림을 생성한다. 모델을 다시 학습하는 명령은 아니다.

표준 출력에 생성된 `.runs/` 경로가 표시되고, 스크립트별 전체 로그와 `RUN.json`이 저장된다. 실행이 실패하면 해당 로그의 마지막 부분과 경로를 표시한다. 각 실행은 자료 사본을 만들므로 디스크 공간은 실행 수에 따라 증가한다.

## 4. CPU 신경망 환경과 체크포인트 재현

```bash
docker build -f environments/Dockerfile.neural -t stablecoin-neural:2.9.1 .
python scripts/reproduce.py latest-checkpoints
```

Dockerfile은 Python 3.11.13의 기본 이미지 digest와 패키지를 고정하고 PyTorch 2.9.1+cpu, NumPy 2.3.0, pandas 2.3.0을 설치한다. 모델 실행 시 네트워크를 끄고 CPU 2개·메모리 4GB·단일 수치 연산 스레드를 사용한다. GPU가 필요하지 않다.

`latest-checkpoints`는 96개 저장 모델 각각 3개 시점 × 31개 시나리오를 재실행하고 시간 방향·공유 dropout 검사와 전체 수치 검증을 수행한다. 체크포인트별 표본 재현이며 모든 시점 재실행은 아니다. Linux의 Docker 실행 권한이 필요하다.

## 5. 실험을 실제로 다시 수행하기

사용 가능한 전체 명령을 확인한다.

```bash
python scripts/reproduce.py list
```

| 명령의 단계 이름 | 수행 범위 |
|---|---|
| `reanalysis-check` | 가격·시계 자료 점검 |
| `reanalysis-points` | 시간대 가정별 점추정 재계산 |
| `pilot` | 최초 분위수 예측 실험 |
| `extension` | 초기 1안 확장의 다섯 목표·시계 셀, 분석과 진단 |
| `development` | 초기 호가 측정 실험 |
| `core` | 자료 준비·통제 회귀·경제적 계수·보조 분석·검증·그림 |
| `resolution` | 측정·정책 대조·평가 민감도·전체 위험 보고·검증·그림 |
| `prepare-data` | 실제 시계와 원자료에서 AI 입력·목표 생성 및 배열 대조 |
| `tick-baselines` | 첫 AI 실험의 선형 QR·LightGBM 기준 모형 |
| `tick-learning` | 첫 AI 실험의 신경망 학습·분석·모델 재현·검증 |
| `model-followup` | GRU·MLP 및 hard TCN 후속 학습·분석·검증 |
| `latest-full` | 최신 TCN·GRU 96개 통제 적합·분석·모델 재현·검증·그림 |

예를 들어 최신 통제 실험을 다시 학습하려면 다음을 실행한다.

```bash
python scripts/reproduce.py latest-full
```

신경망 Docker와 분석용 Python 환경 둘 다 사용한다. 각 단계는 저장된 선행 결과를 이용한 역사적 실험 재현이다. 여러 단계를 새로 수행한 출력으로 일괄 연결하는 새로운 연구 실행과는 구별해야 한다. 이 저장소는 각 단계의 독립 재현을 제공한다.

이전 가격·시간대 분석의 bootstrap은 조합별 인수가 있으므로 [당시 재계산 README](research/reanalysis_20260909/README.md)와 저장 설정을 따른다. 초기 원고 PDF 재생성은 [extension README](research/extension_20260909/README.md)의 명령을 따른다. 이런 장시간 추정과 PDF 생성까지 이번 포장 작업에서 전부 재실행한 것은 아니다.

## 6. 과거 문서와 실행 경로

`docs/research/`는 그림과 파일 링크를 GitHub 상대 경로로 바꾼 보고서 사본이다. `research/` 원본의 절대 경로나 당시 사용한 Docker 이미지 이름은 역사적 기록으로 남아 있다. 현재 실행은 이 안내와 `scripts/reproduce.py`를 우선한다.

과거 `finalize.py`는 원래 서버의 문서 경로까지 검사하는 기록용 스크립트다. 이동한 저장소에서는 각 단계의 수치 `verify.py`와 `scripts/verify_archive.py`를 사용한다. 경로를 바꾸려고 과거 프로토콜·학습 코드의 해시를 수정하지 않았다.

원래 국내 원고용 노트북, 감사·정리 코드에는 원래 폴더 배치와 수동 Bloomberg 내보내기에 의존하는 부분이 있다. 자동 원자료 재수집을 보장하지 않는다. 확보한 원자료에서 출발하는 최신 전처리와 연구 단계의 실행 경로를 제공한다.
