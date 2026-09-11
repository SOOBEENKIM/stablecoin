"""Check final reporting, preserve prior work and inventory the completed study."""
from shared import *
from datetime import datetime, timezone
import re


def main():
    assert_frozen()
    checks={name:json.loads((HERE/name).read_text()) for name in [
        'TRAIN_COMPLETION.json','ANALYSIS_COMPLETION.json','PREFLIGHT.json',
        'VERIFICATION.json','CHECKPOINT_VERIFICATION.json']}
    assert all(v['status'] in ['complete','passed'] for v in checks.values())
    assert checks['TRAIN_COMPLETION.json']['fits']==96
    scores=pd.read_csv(HERE/'all_scores.csv').set_index('model')
    tests=pd.read_csv(HERE/'factorial_contrasts.csv')
    tests=tests[tests.stage=='cal'].set_index('name')
    utility=pd.read_csv(HERE/'UTILITY_DECISION.csv')
    report=(HERE/'RESULTS_KO.md').read_text()
    checked=0
    for family in ['tcn','gru']:
        for placement in ['tick','shuffled']:
            for aggregation in ['mean','max']:
                row=scores.loc[f'{family}_{placement}_{aggregation}_ensemble_cal']
                for col in ['clean_tail','worst_tail_h1','sensitivity_h1']:
                    assert f'{row[col]:.6f}' in report,(family,placement,aggregation,col)
                    checked+=1
        for term in ['training_main','financial_main','interaction']:
            row=tests.loc[family+'_'+term]
            for col in ['difference','low95','high95']:
                assert f'{row[col]:.6f}'.replace('-','−') in report,(family,term,col)
                checked+=1
    assert not utility.all_gates_pass.any()
    assert not utility.worst_reduction_10pct.any()
    assert utility[['clean_noninferior_2pct','sensitivity_reduction_10pct',
        'width_increase_le5pct','coverage_error_increase_le1pp']].all().all()
    for file in [HERE/'RESULTS_KO.md',HERE/'README.md',HERE.parent/'README_WORKSHOP.md']:
        for path in re.findall(r'\]\((/[^)]+)\)',file.read_text()):
            assert Path(path).exists(),(file,path)
    status=dict(
        status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        study='option1_factorial_20260910',fits=96,evaluation_origins=1814,observed_days=77,
        protocol_scope='Locked before these fits, exploratory after prior results on the same period',
        primary_contrasts=19,secondary_combined_holm_family=35,
        shared_controls=['candidate arrays within placement','initialization','epoch count',
            'original minibatches','candidate-shared dropout RNG trajectory','optimizer update count'],
        primary_interactions={family:tests.loc[family+'_interaction',
            ['difference','low95','high95','p_centered','p_holm_19','p_holm_35']].astype(float).to_dict()
            for family in ['tcn','gru']},
        utility=json.loads(utility.to_json(orient='records')),
        supported_claim='Financial candidate placement complements maximum candidate-loss training in both TCN and GRU under the controlled finite-scenario evaluation.',
        limits=['All four utility checks miss the unchanged 10% total stress-loss reduction target.',
            'No established superiority in original-input predictive accuracy against QR.',
            'The earlier hard TCN clean-accuracy improvement is not numerically reproduced by this changed controlled implementation.',
            'Repeated historical period; bootstrap conditional on stored fits; no independent confirmation.',
            'No real-price recovery, realized profit, market-causal effect, novelty or publication-acceptance claim.'],
        verification='passed',reported_numeric_values_checked=checked,local_report_links_checked=True,
        previous_artifacts_preserved=len(json.loads((HERE/'PRESERVED_OPTIONS_SHA256.json').read_text())),
        original_paper_unchanged=True,option3='paused',
        report_sha256=sha(HERE/'RESULTS_KO.md'),
        verification_file='VERIFICATION.json',checkpoint_verification_file='CHECKPOINT_VERIFICATION.json')
    (HERE/'FINAL_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='ARTIFACT_MANIFEST.json'
        and '__pycache__' not in p.parts}
    manifest=dict(study=HERE.name,file_count=len(files),excludes=['ARTIFACT_MANIFEST.json','__pycache__'],files=files)
    (HERE/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for name,entry in files.items():assert sha(HERE/name)==entry['sha256'],name
    assert_frozen()
    print(json.dumps(dict(status='complete',verification='passed',reported_values=checked,
        previous_files_preserved=status['previous_artifacts_preserved'],files_in_manifest=len(files),
        utility_joint_pass=0),indent=2))


if __name__=='__main__':main()
