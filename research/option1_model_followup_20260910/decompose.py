"""Post-result descriptive accounting; does not modify primary thresholds or tests."""
from shared import *

def main():
    s=pd.read_csv(HERE/'all_scores.csv').set_index('model');a=s.loc['tcn_hard_tick_ensemble_cal'];rows=[]
    for ref in ['tcn_clean_ensemble_cal','tcn_tick_ensemble_cal','tcn_hard_shuffled_ensemble_cal','exact_qr_cal']:
        b=s.loc[ref];r=dict(reference=ref)
        for c in ['clean_tail','worst_tail_h1','sensitivity_h1','worst_tail_h6','worst_tail_h24']:r[c+'_change_pct']=100*(a[c]/b[c]-1)
        r['proposal_stress_excess']=a.worst_tail_h1-a.clean_tail;r['reference_stress_excess']=b.worst_tail_h1-b.clean_tail
        r['excess_reduction_pct']=100*(1-r['proposal_stress_excess']/r['reference_stress_excess']);rows.append(r)
    pd.DataFrame(rows).to_csv(HERE/'descriptive_decomposition.csv',index=False)
    (HERE/'DECOMPOSITION_SCOPE.json').write_text(json.dumps(dict(scope='Post-result descriptive decomposition; no additional confirmatory test and no change to primary thresholds',code_sha256=sha(Path(__file__))),indent=2))

if __name__=='__main__':main()
