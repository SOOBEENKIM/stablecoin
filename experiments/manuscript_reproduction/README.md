# 국내 원고 재현

대상 원고: **시장공통·글로벌 요인을 제거한 한국 USDT 잔차의 하단 꼬리 반응과 레버리지 포지셔닝 채널**.

현재 브랜치에 추가했던 `experiments/nonlinear_residual/`의 ML 실험을 제거하고, 원본 분석의 재현을 기준선으로 삼는다. 이전 ML 실험의 마지막 커밋은 `b051620e86bfac07efc90d6b7a257c0ba9d86d1d`이며 Git 이력에 남아 있다. `main`에서 물려받은 과거 연구 아카이브는 원고 재현과 구분한다.

결과는 [RESULTS_KO.md](RESULTS_KO.md)에 있다. **잔차와 핵심 분위수 회귀표는 재현되지만, 원고 전체의 모든 수치·그림이 동일하게 재현된 상태는 아니다.**

## 실행

저장소 루트에서 Python 3.8.10 및 [기록한 패키지 버전](requirements.txt)을 사용한다.

```bash
python3 experiments/manuscript_reproduction/run_reproduction.py
```

원본 코드와 전처리 CSV를 `runs/`에 바이트 그대로 복사한 후, 독립 프로세스로 실행하고 원고 표와 자동 대조한다. CPU 동시 실행 수는 `--jobs 3`으로 지정할 수 있다. 복사본과 실행 캐시는 Git에서 제외하며, 새로 생성한 CSV·그림·로그·비교표·해시는 저장한다.

원자료 수집·전처리 노트북을 처음부터 실행하는 패키지가 아니다. 보존된 **715개 관측치의 전처리 CSV**에서 잔차를 다시 계산하고 후속 분석을 재실행하는 범위다.

## 실행 설정

| 실행 폴더 | 설정 | 실행 코드 |
|---|---|---|
| `runs/default_B300` | 원본 기본값 B=300, seed=42 | main, EQ/CAP/PCA, 지속성·채널, 표본 안정성, 경제적 크기 |
| `runs/archive_main_B40` | 과거 CSV 재현 설정 B=40, seed=42 | main |
| `runs/archive_supplement_B200` | 과거 보충 CSV 재현 설정 B=200, seed=42 | EQ/CAP/PCA, 지속성·채널, 표본 안정성 |

모형·함수 호출 순서와 난수 소비 순서를 유지한다. 기존 CSV를 새로운 실행의 출력으로 복사하지 않는다. B=40·200 설정의 근거는 [이전 감사 기록](../../research/reference/audit_20260908/REVIEW_KO.md)이며 모든 비교 결과를 공개한다.

## 파일

- [원본 PDF](../../research/reference/submitted/경영과학학술지_김수빈_V2_stablecoin.pdf), [동일 원고 DOCX](../../research/reference/submitted/경영과학학술지_김수빈_V2_stablecoin.docx)
- [수정하지 않은 원본 코드·전처리 데이터](../../research/reference/original_v4/)
- [새 실행 결과](runs/default_B300/paper_outputs/), [실행 환경과 해시](run_manifest.json)
- [원고 표 전체 대조](comparison/manuscript_cells.csv), [불일치](comparison/mismatches.csv)

## 재현 이후

이 결과를 기준점으로 보존하고 별도 개발 브랜치에서 수정한다. 우선 원고와 코드 차이, 가격 환산, 실제 시간 기준 시차, 수치해법·추론을 점검한다. 그 뒤 연구 질문에 도움이 되는 ML의 역할을 정하고 같은 표본·같은 후속 분석으로 비교한다. 기존 연구의 목표는 한국 USDT 잔차의 하방 반응·지속성과 포지셔닝 변수의 관계를 검증하는 것이다.
