"""Matched exposure comparisons and exact same-origin price attribution."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from economic import HERE, ADAPTIVE, a
from state_design import EXPOSURES, MATCH_COLS, covariates, match_month
from analyze import day_weights, boot_mean, interval

OUT = HERE / 'results'
PARTS = ['delta_local_bp','delta_quote_bp','delta_fx_bp','delta_market_bp','delta_g_bp']


def balance_covariates(d):
    x = covariates(d)
    for c in a.columns('available_macro','base'):
        if c not in ['e_now','e_change1','btc_vol24','downside24']:
            x[c] = d[c]
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = a.panel('available_macro')
    pred = pd.read_csv(ADAPTIVE/'results/predictions.csv.gz')
    pred.origin = pd.to_datetime(pred.origin,utc=True)
    pred = pred[(pred['mode']=='available_macro')&(pred.model=='qrf')&(pred['info']=='history')]
    pair_rows, selected_rows, all_rows, audits, balance_rows = [],[],[],[],[]
    for definition in ['EQ','CAP','PCA']:
        for cutoff in a.FOLDS[1:]:
            fold = cutoff.strftime('%Y-%m')
            _,_,train,_,_,r = a.splits(p,'available_macro',definition,cutoff)
            frames = {}
            for h in [1,6,12]:
                d = a.frame(p,r,'available_macro',h)
                d = d[(d.index>=cutoff)&(d.index<cutoff+pd.offsets.MonthBegin(1))].copy()
                d['local_change_krw'] = (a.lead(p.local_usdt,h)-p.local_usdt).reindex(d.index)
                d['local_logreturn_bp'] = (np.log(a.lead(p.local_usdt,h)/p.local_usdt)*1e4).reindex(d.index)
                d['tail_event'] = (d.target < train.target.quantile(.1)).astype(int)
                d['negative_future'] = (d.target < 0).astype(int)
                if h == 1:
                    forecast = pred[(pred.definition==definition)&(pred.fold==fold)].set_index('origin').pred_calibrated
                    d['predicted_q10_h1'] = forecast.reindex(d.index)
                    assert d.predicted_q10_h1.notna().all()
                frames[h] = d
            common = frames[1].index.intersection(frames[6].index).intersection(frames[12].index)
            for cohort, ix, horizons in [('one_hour',frames[1].index,[1]),('common_path',common,[1,6,12])]:
                sample = frames[1].loc[ix]
                for exposure,column in EXPOSURES.items():
                    key = dict(definition=definition,fold=fold,cohort=cohort,exposure=exposure)
                    pairs,z,scaler,counts = match_month(train,sample,exposure)
                    audits.append(dict(**key,**counts,sample_n=len(sample),
                        fit_end=r.fit_end.isoformat(),train_last_label=train.target_time.max().isoformat(),
                        threshold_q10_bp=train.target.quantile(.1)))
                    hi = sample.index[sample[column] > counts['high_cut']]
                    lo = sample.index[sample[column] < counts['low_cut']]
                    bal_scale = StandardScaler().fit(balance_covariates(train))
                    bz = pd.DataFrame(bal_scale.transform(balance_covariates(sample)),index=sample.index,
                                      columns=balance_covariates(sample).columns)
                    for stage,hi_ix,lo_ix in [('before',hi,lo),
                        ('after',pd.DatetimeIndex(pairs.high_origin) if len(pairs) else hi[:0],
                                 pd.DatetimeIndex(pairs.low_origin) if len(pairs) else lo[:0])]:
                        for c in bz.columns:
                            for arm,ids in [('high',hi_ix),('low',lo_ix)]:
                                balance_rows.append(dict(**key,stage=stage,variable=c,arm=arm,n=len(ids),
                                    mean_training_z=bz.loc[ids,c].mean() if len(ids) else np.nan))
                    if len(pairs):
                        for c,v in key.items():
                            pairs[c] = v
                        pairs['pair_id'] = [definition+'_'+fold+'_'+cohort+'_'+exposure+'_'+str(i) for i in range(len(pairs))]
                        pair_rows.append(pairs)
                    for h in horizons:
                        d = frames[h].loc[ix].copy()
                        d['arm'] = np.where(d[column] > counts['high_cut'],'high',
                                            np.where(d[column] < counts['low_cut'],'low','middle'))
                        for c,v in key.items():
                            d[c] = v
                        d['h'] = h
                        all_rows.append(d.reset_index())
                        if len(pairs):
                            for arm in ['high','low']:
                                ids = pd.DatetimeIndex(pairs[arm+'_origin'])
                                s = d.loc[ids].copy()
                                s.index.name = 'origin'
                                s['arm'],s['pair_id'] = arm,pairs.pair_id.to_numpy()
                                selected_rows.append(s.reset_index())
    pairs = pd.concat(pair_rows,ignore_index=True)
    matched = pd.concat(selected_rows,ignore_index=True)
    all_data = pd.concat(all_rows,ignore_index=True)
    balance = pd.DataFrame(balance_rows)
    # Aggregate standardized covariate means with actual group sizes.
    balance['weighted_z'] = balance.mean_training_z.fillna(0)*balance.n
    bg = balance.groupby(['definition','cohort','exposure','stage','variable','arm']).agg(n=('n','sum'),total=('weighted_z','sum')).reset_index()
    bg['mean_training_z'] = bg.total/bg.n.replace(0,np.nan)
    wide = bg.pivot(index=['definition','cohort','exposure','stage','variable'],columns='arm',values='mean_training_z').reset_index()
    wide['standardized_difference'] = wide.high-wide.low
    estimates, coverage = [],[]
    outcomes = ['e_now','target','delta_e','tail_event','negative_future','local_change_krw','local_logreturn_bp']+PARTS
    for (definition,cohort,exposure), g in matched.groupby(['definition','cohort','exposure']):
        first = g[g.h==1]
        b = wide[(wide.definition==definition)&(wide.cohort==cohort)&(wide.exposure==exposure)&(wide.stage=='after')]
        max_match_smd = b[b.variable.isin(MATCH_COLS)].standardized_difference.abs().max()
        max_all_smd = b.standardized_difference.abs().max()
        days = first.origin.dt.normalize().nunique()
        n_pairs = first.pair_id.nunique()
        status = 'adequate' if n_pairs>=20 and days>=10 and max_match_smd<=.1 else 'limited'
        coverage.append(dict(definition=definition,cohort=cohort,exposure=exposure,pairs=n_pairs,
            days=days,max_match_smd=max_match_smd,max_all_base_smd=max_all_smd,status=status))
        for h, s in g.groupby('h'):
            s = s.sort_values('origin').reset_index(drop=True)
            high = (s.arm=='high').to_numpy().astype(float)
            low = 1-high
            ws = {length:day_weights(s,length) for length in [1,5,10]}
            for outcome in outcomes+(['predicted_q10_h1'] if h==1 else []):
                y = s[outcome].to_numpy()
                mh,ml = y[high==1].mean(),y[low==1].mean()
                for length,w in ws.items():
                    wh,wl = w*high[None,:],w*low[None,:]
                    draws = boot_mean(wh,y)-boot_mean(wl,y)
                    lo_ci,hi_ci = interval(draws)
                    estimates.append(dict(definition=definition,cohort=cohort,exposure=exposure,h=h,
                        outcome=outcome,block_days=length,pairs=n_pairs,days=days,status=status,
                        high_mean=mh,low_mean=ml,difference=mh-ml,ci_low=lo_ci,ci_high=hi_ci))
    # Preserve zero-match cases explicitly rather than silently omitting them.
    covered = {(r['definition'],r['cohort'],r['exposure']) for r in coverage}
    for definition in ['EQ','CAP','PCA']:
        for cohort in ['one_hour','common_path']:
            for exposure in EXPOSURES:
                if (definition,cohort,exposure) not in covered:
                    coverage.append(dict(definition=definition,cohort=cohort,exposure=exposure,pairs=0,days=0,
                        max_match_smd=np.nan,max_all_base_smd=np.nan,status='limited'))
    pairs.to_csv(OUT/'matched_pairs.csv',index=False)
    matched.to_csv(OUT/'matched_observations.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    all_data.to_csv(OUT/'eligible_paths.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(audits).to_csv(OUT/'matching_audit.csv',index=False)
    balance.to_csv(OUT/'balance_by_month.csv',index=False)
    wide.to_csv(OUT/'balance.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'matching_coverage.csv',index=False)
    result = pd.DataFrame(estimates)
    result.to_csv(OUT/'matched_contrasts.csv',index=False)
    err = np.max(np.abs(matched[PARTS].sum(axis=1)-matched.delta_e))
    assert err < 1e-8
    for _,s in matched[matched.cohort=='common_path'].groupby(['definition','exposure','pair_id','arm']):
        assert set(s.h)=={1,6,12}
        assert s.origin.nunique()==1 and s.e_now.max()-s.e_now.min()<1e-9
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),constrained_layout=True)
    for ax,exposure in zip(axes,EXPOSURES):
        d=result[(result.definition=='EQ')&(result.cohort=='common_path')&(result.exposure==exposure)&
            (result.outcome=='delta_e')&(result.block_days==5)].sort_values('h')
        ax.axhline(0,color='black',lw=.8)
        if len(d):
            ax.errorbar(d.h,d.difference,yerr=[d.difference-d.ci_low,d.ci_high-d.difference],marker='o',capsize=3)
            ax.set_title(exposure+': '+str(d.pairs.iloc[0])+' pairs / '+d.status.iloc[0])
        ax.set_xlabel('Actual hours after same origins');ax.set_xticks([1,6,12])
    axes[0].set_ylabel('High minus low exposure: residual change (bp)')
    fig.savefig(OUT/'matched_adjustment.png',dpi=180);plt.close(fig)
    (OUT/'path_verification.json').write_text(json.dumps(dict(source_sha256=a.sha256(HERE/'analyze_paths.py'),
        matching_source_sha256=a.sha256(HERE/'state_design.py'),
        protocol_sha256=a.sha256(HERE/'PROTOCOL_KO.md'),price_identity_max_error_bp=float(err),
        same_origin_paths_verified=True,matched_rows=len(matched),pairs=len(pairs)),indent=2))
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__=='__main__':
    main()
