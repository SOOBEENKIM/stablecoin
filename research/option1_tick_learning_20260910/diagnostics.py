"""Descriptive diagnostics after the locked primary test; no model reselection."""
from shared import *
from analyze import quantities,Resample

def main():
    d=np.load(HERE/'evaluated_predictions.npz');dates=pd.to_datetime(d['origin_ns'],utc=True);y=d['target']
    scores=pd.read_csv(HERE/'all_scores.csv');names=scores.loc[scores.primary_model&scores.model.str.endswith('_cal'),'model'].tolist()
    rows=[];drivers=[];rs=Resample(dates);q={}
    for name in names:
        p=d[name].astype(float);loss=pinball(y[:,None],p);tail=loss[:,:,[0,2]].mean(2);q[name]=quantities(y,p)
        for j,scenario in enumerate(d['scenario_names']):
            sens=np.maximum(abs(p[:,j,0]-p[:,0,0]),abs(p[:,j,2]-p[:,0,2]))
            rows.append(dict(model=name,scenario=scenario,tail_loss=tail[:,j].mean(),q10_loss=loss[:,j,0].mean(),
                sensitivity_bp=sens.mean(),signed_q10_change=(p[:,j,0]-p[:,0,0]).mean(),signed_q90_change=(p[:,j,2]-p[:,0,2]).mean()))
        excess=tail[:,:11].max(1)-tail[:,0]
        for j,c in enumerate(COINS):
            coin_excess=tail[:,1+2*j:3+2*j].max(1)-tail[:,0]
            drivers.append(dict(model=name,coin=c,share_tied_for_largest_positive_loss_increase=np.mean((excess>1e-6)&(coin_excess>=excess-1e-6))))
    pd.DataFrame(rows).to_csv(HERE/'scenario_diagnostics.csv',index=False)
    pd.DataFrame(drivers).to_csv(HERE/'worst_scenario_drivers.csv',index=False)
    comparisons=[];a=q['tcn_tick_ensemble_cal']
    for ref in ['tcn_clean_ensemble_cal','tcn_shuffled_ensemble_cal','linear_tick_ensemble_cal','exact_qr_cal']:
        b=q[ref];ratio=rs.means(a['sensitivity_h1'])/rs.means(b['sensitivity_h1'])
        lo,hi=np.quantile(1-ratio,[.025,.975])
        comparisons.append(dict(reference=ref,sensitivity_reduction=1-a['sensitivity_h1'].mean()/b['sensitivity_h1'].mean(),
            reduction_low95=lo,reduction_high95=hi,proposal_stress_excess=(a['worst_tail_h1']-a['clean_tail']).mean(),
            reference_stress_excess=(b['worst_tail_h1']-b['clean_tail']).mean()))
    pd.DataFrame(comparisons).to_csv(HERE/'secondary_decomposition.csv',index=False)
    (HERE/'DIAGNOSTIC_SCOPE.json').write_text(json.dumps(dict(status='complete',scope='Descriptive diagnostics after primary results; no new confirmatory tests or model tuning',
        all_primary_models=8,scenarios_each=31,intervals='Unadjusted exploratory 95% intervals on four sensitivity ratios',code_sha256=sha(Path(__file__))),indent=2))
    print(pd.DataFrame(comparisons).round(6).to_string(index=False),flush=True)

if __name__=='__main__':main()
