from shared import *
from statsmodels.stats.multitest import multipletests

def main():
    d=pd.read_csv(CORE/'policy_daily.csv',index_col=0);d.index=pd.to_datetime(d.index,utc=True)
    keep=((d.index>=pd.Timestamp('2025-07-02',tz='UTC'))&(d.index<pd.Timestamp('2025-07-30',tz='UTC')))|((d.index>=pd.Timestamp('2025-08-02',tz='UTC'))&(d.index<pd.Timestamp('2025-08-30',tz='UTC')))
    d=d.loc[keep];weights=np.load(CORE/'policy_weights.npz')['policy_w28']
    z=pd.DataFrame({'const':1.,'post':(d.index>=pd.Timestamp('2025-08-02',tz='UTC')).astype(float)},index=d.index)
    for c in [prefix+coin for coin in ['DOGE','ETH','XRP','SOL'] for prefix in ['ret_','abs_ret_','volume_']]+['btc_rv']:
        z[c]=(d[c]-d[c].mean())/d[c].std(ddof=1)
    z['weekend']=d.weekend
    pairs=[('DOGE','ETH'),('ETH','XRP'),('SOL','ETH')]
    y=np.column_stack([(d['abs_a_'+a]-d['abs_a_'+b]).to_numpy() for a,b in pairs]);x=z.to_numpy()
    assert np.linalg.matrix_rank(x)==x.shape[1]
    co=ols(x,y);draws=np.full((1999,3),np.nan);errors=[]
    for i,w in enumerate(weights):
        try:draws[i]=ols(x,y,w)[1]
        except Exception as e:errors.append(dict(rep=i,error=str(e)))
    assert len(errors)<.05*len(draws)
    rows=[]
    for j,(a,b) in enumerate(pairs):
        bb,se,p,cov=calendar_hac(x,y[:,j],d.index)
        rows.append(dict(pair=a+'_'+b,n=len(x),p=x.shape[1],condition_number=np.linalg.cond(x),estimate=co[1,j],hac_se=se[1],**interval(co[1,j],draws[:,j])))
    contrasts=[]
    for j in [1,2]:
        est=co[1,0]-co[1,j];v=draws[:,0]-draws[:,j]
        # A common-X contrast must equal the coefficient for the outcome difference.
        np.testing.assert_allclose(ols(x,y[:,0]-y[:,j]),co[:,0]-co[:,j],atol=1e-8)
        contrasts.append(dict(comparison='DOGE_ETH_minus_'+'_'.join(pairs[j]),estimate=est,**interval(est,v)))
    dif=pd.DataFrame(contrasts);dif['p_holm_2']=multipletests(dif.p_centered,method='holm')[1]
    pd.DataFrame(rows).to_csv(HERE/'policy_common_controls.csv',index=False);dif.to_csv(HERE/'policy_common_control_contrasts.csv',index=False)
    np.savez_compressed(HERE/'policy_common_controls.npz',x=x,y=y,dates=d.index.asi8,weights=weights,coefficients=co,bootstrap_post=draws,feature_names=np.asarray(z.columns,dtype=str))
    g=pd.read_csv(HERE/'prediction_scenario_gaps.csv');s=pd.read_csv(HERE/'prediction_current_states.csv')
    r=s.query('group=="fragile_minus_stable"')
    f=pd.concat([g[['reference','scenario','metric','p_centered']],r[['reference','group','metric','p_centered']].rename(columns={'group':'scenario'})],ignore_index=True)
    assert len(f)==24
    f['p_holm_24_secondary']=multipletests(f.p_centered,method='holm')[1]
    f.to_csv(HERE/'prediction_secondary_multiplicity.csv',index=False)
    (HERE/'SUPPLEMENT_COMPLETION.json').write_text(json.dumps(dict(status='complete',code_sha256=sha(Path(__file__)),scope_sha256=sha(HERE/'SUPPLEMENT_SCOPE_KO.md'),
        common_X_columns=x.shape[1],bootstrap_failures=errors,preserved_files=check_preserved()),indent=2))
    print(pd.DataFrame(rows).round(5).to_string(index=False));print(dif.round(5).to_string(index=False))

if __name__=='__main__':main()
