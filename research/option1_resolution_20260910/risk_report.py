from shared import *

def main():
    a=np.load(HERE/'measurement_panel.npz');dates=pd.to_datetime(a['dates'],utc=True);cut=-10.
    b=a['b'];low=a['any_single_low'];high=a['any_single_high']
    out=pd.DataFrame(index=dates)
    for j,ref in enumerate(REFS):
        out[ref+'_bp']=b[:,j]
        out[ref+'_lower10bp']=b[:,j]<cut
        out[ref+'_any_single_low_bp']=low[:,j];out[ref+'_any_single_high_bp']=high[:,j]
        out[ref+'_state']=np.where(high[:,j]<cut,'risk_persists_within_scenario',np.where(low[:,j]>=cut,'nonrisk_persists_within_scenario','classification_sensitive_to_scenario'))
    steps=np.load(HERE/'minimum_tick_arrays.npz')['minus10_DOGE'][:,0]
    out['mean5_DOGE_exit_steps_up_to_100']=steps
    out['number_of_references_below_minus10']=(b<cut).sum(axis=1)
    out.to_csv(HERE/'risk_reporting_panel.csv.gz',compression='gzip')
    selection=[]
    for state in out.mean5_state.unique():
        part=out.loc[out.mean5_state==state].copy()
        selection.extend([part.index[0],part.mean5_bp.idxmin(),part.mean5_bp.idxmax()])
    out.loc[sorted(set(selection))].to_csv(HERE/'risk_reporting_examples.csv')
    print('Risk report states',out.mean5_state.value_counts().to_dict())

if __name__=='__main__':main()
