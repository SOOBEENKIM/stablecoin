# 워크스테이션 원본 대조

2026-09-24에 `/home/ssd-990/soobeenkim/`의 `stablecoin`, `stablecoin_v3`, `stablecoin_v4`, `stablecoin_workshop`을 직접 읽어 GitHub 보존본과 대조했다. 폴더 수정일만으로 최종본을 판단하지 않고 파일 내용의 SHA-256을 비교했다.

**이번 재현에 사용한 원본은 워크스테이션 `stablecoin_v4`와 같은 코드·데이터다.** 실행한 분석 코드 5개, 전처리 CSV, 전처리 노트북 모두 바이트 단위로 일치한다.

| 워크스테이션 폴더 | 이번 실행과의 관계 |
|---|---|
| `stablecoin` | 전처리 CSV·노트북은 같지만 main 분석 코드는 다르며 보충 코드들이 없음 |
| `stablecoin_v3` | main·EQ/CAP/PCA·지속성·표본 분할 코드 4개 및 전처리 CSV·노트북이 같음. 경제적 크기 코드 없음 |
| `stablecoin_v4` | 분석 코드 5개 및 전처리 CSV·노트북 모두 같음 |
| `stablecoin_workshop` | 루트에 남겨 둔 국내 원고용 코드 5개 및 전처리 CSV·노트북은 v4와 같음. 이후 워크숍 실험은 별도 하위 폴더에 있음 |

`stablecoin_v4`와 `research/reference/original_v4`의 같은 상대 경로에 존재하는 42개 파일은 모두 같고, 다른 파일은 없다. 원자료 폴더는 저장소에서 별도 경로로 정리되어 있으므로 이 42개 확인을 전체 원자료의 동일성 검증으로 확대하지 않는다.

첨부 최종 PDF는 GitHub의 `research/reference/submitted/경영과학학술지_김수빈_V2_stablecoin.pdf`와 SHA-256이 같다. 다만 **최종 원고와 v4 폴더 안의 초안은 동일 문서가 아니다.** 최종 원고 DOCX는 6월 26일 수정본으로 표가 6개이고, v4의 마지막 이름을 가진 `manuscript_경영과학_김수빈_v4.docx`에는 표가 5개다. 따라서 v4 코드가 첨부 최종 원고의 모든 셀·그림을 만들었다고 단정하지 않는다.

결론적으로 이번에 잘못된 워크숍 ML 코드를 실행한 것은 아니다. 국내 원고 계열의 v4 코드로 재실행한 것이 맞지만, 그 코드와 나중에 편집된 최종 원고 사이에는 [별도로 기록한 불일치](RESULTS_KO.md)가 남아 있다.

2026-09-25 추가 추적에서는 표 4의 p=0.05가 이미 v4 폴더의 초안에 있었음을 확인했다. 따라서 원고 편집일 차이가 수치 불일치의 원인이라고 해석해서는 안 된다. 표 5·6의 주요 차이는 [회귀 반복·부트스트랩 설정 차이](difference_audit/RESULTS_KO.md)로 재현 확인했다.

확인 결과: [파일별 SHA-256 및 문서 메타데이터](comparison/workstation_sources.json).

같은 워크스테이션에서 확인을 다시 실행하려면 다음 명령을 사용한다. 일반 GitHub 이용자는 이 로컬 확인 없이 원고 재현 실행기를 사용할 수 있다.

```bash
python3 experiments/manuscript_reproduction/verify_workstation_sources.py --workspace /home/ssd-990/soobeenkim
```
