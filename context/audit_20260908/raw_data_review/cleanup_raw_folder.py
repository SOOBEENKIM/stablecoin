"""Explicitly scoped cleanup: classify by inspected content, archive, verify, unlink.

Default only writes the reviewable classification. --apply performs the cleanup.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT/'stablecoin_v4/data/raw data'
BACKUP = HERE/'removed_files_backup.tar.gz'
PRIMARY = set(json.loads((HERE/'reproduction_summary.json').read_text())['input_files'])
REPRODUCED = set()
with (HERE/'preprocessing_comparison.csv').open() as f:
    REPRODUCED.update(r['file'] for r in csv.DictReader(f))

COMMODITY_RESULTS = {
    'b_extended_grid_search_results_0.csv', 'bm_extended_grid_search_results_0.csv',
    'bmom_lstm_noise_experiment_results.csv', 'bmom_noise_robustness_comparison.csv',
    'bmom_noise_robustness_results.csv', 'perf_monthly_lstm_bmom.csv',
}
OPTION_RESULTS = {
    'Exhibit14_market_SV_lambda0.csv', 'Exhibit15_market_SV_lambda3.csv',
    'Exhibit16_empirical_lambda3.csv', 'RealPath_results.csv',
    'options_2023-08-17.csv', 'prices_realpath.csv', 'cheb_fit.csv',
    'delta_actions_all.csv', 'delta_by_t_summary.csv',
    'robust_rl_buyer24_0pct.csv', 'robust_rl_buyer24_3pct.csv',
}
UNCERTAIN = {'predicted_ranking.csv','rolling_path_mdd.csv','perf_monthly_lstm.csv'}
DUPLICATE = 'panel_full_checked.parquet'
CANONICAL = 'panel_upbit_binance_fx_kp_full_1h.parquet'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):h.update(b)
    return h.hexdigest()


def classify(entry):
    name = entry['file']
    cols = entry.get('columns', [])
    if name in PRIMARY:
        return 'keep_primary', '기존 전처리와 715행 분석자료 재현에 사용한 원자료'
    if name in REPRODUCED:
        return 'keep_reproduced_intermediate', '원 노트북에서 재생성한 값과 일치하는 전처리 중간자료'
    if name == DUPLICATE:
        return 'remove_exact_duplicate', 'SHA-256이 '+CANONICAL+'와 동일; 해당 파일 보존'
    if any(c in cols for c in ['CL_ret','CL1_ret','CL','CL1']) and any(c in cols for c in ['NG_ret','NG1_ret','NG','NG1']):
        return 'remove_other_research', 'CL/NG 등 원자재 선물 가격·모멘텀·예측/백테스트 열 확인; 이 논문 입력과 무관'
    if name in COMMODITY_RESULTS:
        return 'remove_other_research', 'Basis-Momentum/LSTM 학습·성과 요약 결과; 파일명·열·표본을 확인, 현 논문 입력 아님'
    if name == 'Final_ARIMA_mom_single.ipynb':
        return 'remove_other_research', '노트북 코드에서 CL/NG 등 19종 원자재의 ARIMA/VAR 모멘텀 연구 확인'
    if name in OPTION_RESULTS:
        return 'remove_other_research', '주식 옵션·행사경계·델타·강화학습 헤지 결과; 열과 표본 확인, 현 논문 입력 아님'
    if name in UNCERTAIN:
        return 'keep_uncertain', '현 논문 입력은 아니나 일반적인 파일명·열만으로 연구 소속 확정 어려워 보존'
    if entry['extension']=='.parquet':
        return 'keep_related_reference', 'footer의 열 메타데이터로 과거 김프/환율 패널 확인; 전체 값 검증 미실시, 보존'
    return 'keep_related_reference', '이전 기간·다른 거래소/ETF·스테이블코인 관련 원자료 또는 분석 결과; 현 제출본 입력으로 승인하지 않고 참고용 보존'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    inventory = json.loads((HERE/'inventory.json').read_text())
    manifest = json.loads((HERE/'source_manifest.json').read_text())
    rows = []
    for e in inventory:
        category,reason = classify(e)
        rows.append(dict(file=e['file'],category=category,reason=reason,sha256=e['sha256'],bytes=e['bytes']))
    with (HERE/'file_decisions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'file_decisions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    selected=[r for r in rows if r['category'].startswith('remove_')]
    summary=dict(original_files=len(rows),categories=dict(Counter(r['category'] for r in rows)),
                 selected_for_removal=len(selected),selected_bytes=sum(r['bytes'] for r in selected))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not args.apply:
        for r in selected:print(r['file'],'|',r['reason'])
        return

    # Refuse to operate on changed, newly uploaded, nested, or symlinked inputs.
    assert {p.name for p in RAW.iterdir()} == set(manifest), 'Source file set changed since inventory'
    for r in rows:
        p=RAW/r['file']
        assert p.is_file() and not p.is_symlink() and p.parent==RAW
        assert sha(p)==r['sha256'], 'Input changed: '+p.name
    assert sha(RAW/DUPLICATE)==sha(RAW/CANONICAL)
    assert not BACKUP.exists(), 'Refusing to overwrite an existing recovery archive'
    with tarfile.open(BACKUP,'x:gz') as archive:
        for r in selected:archive.add(RAW/r['file'],arcname=r['file'],recursive=False)
    with tarfile.open(BACKUP,'r:gz') as archive:
        assert set(archive.getnames())=={r['file'] for r in selected}
        for r in selected:
            restored=archive.extractfile(r['file'])
            h=hashlib.sha256()
            for b in iter(lambda:restored.read(1024*1024),b''):h.update(b)
            assert h.hexdigest()==r['sha256'],'Archive mismatch: '+r['file']
    summary.update(backup=str(BACKUP),backup_sha256=sha(BACKUP),backup_verified=True,removed_files=[])
    journal=HERE/'cleanup_result.json'
    journal.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    for r in selected:
        p=RAW/r['file']
        assert sha(p)==r['sha256'], 'Changed immediately before deletion: '+p.name
        p.unlink()
        summary['removed_files'].append(r)
        journal.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    kept=[r for r in rows if not r['category'].startswith('remove_')]
    assert {p.name for p in RAW.iterdir()}=={r['file'] for r in kept}
    for r in kept:assert sha(RAW/r['file'])==r['sha256']
    summary.update(remaining_files=len(kept),remaining_bytes=sum(r['bytes'] for r in kept),
                   retained_hashes_verified=True,completed=True)
    journal.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Cleanup completed:',len(selected),'removed,',len(kept),'retained; archive verified.')


if __name__=='__main__':main()
