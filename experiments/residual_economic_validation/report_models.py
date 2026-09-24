"""All predefined forecast comparisons, including unfavorable results."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from economic import HERE, ADAPTIVE, a, CASES, GROUPS
from analyze import day_weights, boot_mean, interval, holm

OUT = HERE / 'results'


def load_predictions():
    new = pd.read_csv(OUT / 'predictions.csv.gz')
    old = pd.read_csv(ADAPTIVE / 'results/predictions.csv.gz')
    old = old[old.model.isin(['linear','qrf']) & old['info'].isin(['base','history'])].copy()
    old = old[[tuple(x) in CASES for x in old[['mode','definition']].to_numpy()]]
    old['seed'], old['strategy'] = 20260925, 'previous'
    seed = pd.read_csv(ADAPTIVE / 'results/seed_predictions.csv.gz')
    seed = seed[seed['info'].isin(['base','history'])].copy()
    eq = old[(old['mode'] == 'available_macro') & (old.definition == 'EQ') & (old.model == 'qrf')]
    keys = ['origin','info']
    meta_cols = ['e_now','delta_e','downside24','btc_vol24','account_log']
    seed = seed.merge(eq[keys + meta_cols], on=keys, how='left', validate='many_to_one')
    seed['mode'], seed['definition'], seed['model'], seed['strategy'] = 'available_macro', 'EQ', 'qrf', 'previous'
    old = old[~((old['mode'] == 'available_macro') & (old.definition == 'EQ') & (old.model == 'qrf'))]
    d = pd.concat([new, old, seed], ignore_index=True)
    for c in ['origin','target_time','latest_calibration_label']:
        d[c] = pd.to_datetime(d[c], utc=True)
    d = d[d.evaluation].copy()
    for v in ['raw','calibrated']:
        d['loss_' + v] = a.pinball(d.target, d['pred_' + v], .1)
        d['below_' + v] = (d.target < d['pred_' + v]).astype(int)
    assert not d.duplicated(['mode','definition','model','info','strategy','seed','origin']).any()
    assert np.isfinite(d[['target','e_now','pred_raw','pred_calibrated']]).all().all()
    return d


def compare(x, y, weights, key):
    assert x.origin.equals(y.origin)
    np.testing.assert_allclose(x.target, y.target, atol=1e-10, rtol=0)
    delta = x.loss.to_numpy() - y.loss.to_numpy()
    draws = boot_mean(weights, delta)
    improvement = -100 * draws / boot_mean(weights, y.loss)
    low, high = interval(improvement)
    value = delta.mean()
    p = (1 + np.count_nonzero(np.abs(draws-value) >= abs(value))) / (len(draws)+1)
    return dict(**key, n=len(x), difference_bp=value,
        improvement_pct=100*(1-x.loss.mean()/y.loss.mean()), improvement_lo=low, improvement_hi=high,
        p_centered_boot=p, winning_months=int((x.groupby('fold').loss.mean() < y.groupby('fold').loss.mean()).sum()))


def main():
    pred = load_predictions()
    metrics, monthly, comparisons, by_seed, state_rows = [], [], [], [], []
    thresholds = pd.read_csv(ADAPTIVE / 'results/state_thresholds.csv')
    for (mode, definition), sub in pred.groupby(['mode','definition']):
        maps, seeds = {}, {}
        for (model, info, strategy), stream in sub.groupby(['model','info','strategy']):
            for version in ['raw','calibrated']:
                s = stream.copy()
                s['loss'], s['below'], s['pred'] = s['loss_'+version], s['below_'+version], s['pred_'+version]
                g = s.groupby('origin', as_index=False).agg(target=('target','first'), fold=('fold','first'),
                    loss=('loss','mean'), below=('below','mean'), pred=('pred','mean'), seed_n=('seed','nunique'),
                    e_now=('e_now','first'), downside24=('downside24','first'), account_log=('account_log','first'))
                assert g.seed_n.nunique() == 1
                np.testing.assert_allclose(s.groupby('origin').target.max(), s.groupby('origin').target.min(), atol=1e-10)
                key = dict(mode=mode, definition=definition, model=model, info=info, strategy=strategy, calibration=version)
                metrics.append(dict(**key, n=len(g), seeds=int(g.seed_n.iloc[0]),
                    loss_bp=g.loss.mean(), below_rate=g.below.mean()))
                for fold, f in g.groupby('fold'):
                    monthly.append(dict(**key, fold=fold, n=len(f), loss_bp=f.loss.mean(), below_rate=f.below.mean()))
                maps[model,info,strategy,version] = g
                seeds[model,info,strategy,version] = s
        ref = maps['qrf','history','previous','calibrated']
        ws = {k: day_weights(ref,k) for k in [1,5,10]}
        pairs = [
            (('qrf','history','previous'),('threshold','history','retuned'),'qrf_vs_threshold'),
            (('qrf','base','previous'),('threshold','base','retuned'),'qrf_base_vs_threshold_base'),
            (('qrf','history','previous'),('linear','history','previous'),'qrf_vs_linear'),
            (('threshold','history','retuned'),('linear','history','previous'),'threshold_vs_linear'),
            (('threshold','history','retuned'),('threshold','base','retuned'),'threshold_position'),
            (('qrf','history','previous'),('qrf','base','previous'),'all_position')]
        pairs += [(('qrf','history','previous'),('qrf','without_'+group,'retuned'),'contribution_'+group) for group in GROUPS]
        if mode == 'available_macro' and definition == 'EQ':
            pairs += [(('qrf','history','previous'),('qrf','without_'+group,'fixed_history'),'fixed_contribution_'+group) for group in GROUPS]
        for ak, bk, label in pairs:
            for version in ['raw','calibrated']:
                x,y = maps[ak+(version,)],maps[bk+(version,)]
                assert x.origin.equals(ref.origin)
                key = dict(mode=mode,definition=definition,comparison=label,calibration=version)
                for length in [1,5,10]:
                    comparisons.append(compare(x,y,ws[length],dict(**key,block_days=length)))
                sx,sy = seeds[ak+(version,)],seeds[bk+(version,)]
                for seed in sorted(sx.seed.unique()):
                    a1 = sx[sx.seed == seed].sort_values('origin')
                    b1 = sy[sy.seed == (seed if seed in sy.seed.values else sy.seed.iloc[0])].sort_values('origin')
                    assert list(a1.origin) == list(b1.origin)
                    by_seed.append(dict(**key,seed=int(seed),n=len(a1),
                        improvement_pct=100*(1-a1.loss.mean()/b1.loss.mean())))
                if label.startswith('contribution_'):
                    th = thresholds[(thresholds['mode']==mode)&(thresholds.definition==definition)]
                    state = x.merge(th[['fold','downside_p75','account_log_p75']],on='fold',validate='many_to_one').sort_values('origin')
                    state['delta_loss'] = x.loss.to_numpy()-y.loss.to_numpy()
                    state['reference_loss'] = y.loss.to_numpy()
                    state['high_downside'] = (state.downside24 > state.downside_p75).astype(int)
                    state['high_account'] = (state.account_log > state.account_log_p75).astype(int)
                    for (hd,ha),g in state.groupby(['high_downside','high_account']):
                        state_rows.append(dict(**key,high_downside=hd,high_account=ha,n=len(g),
                            difference_bp=g.delta_loss.mean(),improvement_pct=-100*g.delta_loss.mean()/g.reference_loss.mean()))
    metrics, comparisons = pd.DataFrame(metrics), pd.DataFrame(comparisons)
    primary = comparisons[(comparisons['mode']=='available_macro')&(comparisons.definition=='EQ')&
        (comparisons.calibration=='calibrated')&(comparisons.block_days==5)&
        comparisons.comparison.isin(['qrf_vs_threshold']+['contribution_'+g for g in GROUPS])].copy()
    assert len(primary)==4
    primary['p_holm_exploratory'] = holm(primary.p_centered_boot)
    metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    comparisons.to_csv(OUT/'comparisons.csv',index=False)
    primary.to_csv(OUT/'primary_tests.csv',index=False)
    pd.DataFrame(by_seed).to_csv(OUT/'seed_comparisons.csv',index=False)
    pd.DataFrame(state_rows).to_csv(OUT/'state_ablation.csv',index=False)
    # Concise figure of the stronger benchmark and group ablations.
    g = metrics[(metrics['mode']=='available_macro')&(metrics.definition=='EQ')&
        (metrics.calibration=='calibrated')&(metrics.strategy!='fixed_history')]
    labels,values = [],[]
    for model,info in [('linear','history'),('threshold','history'),('qrf','history')]+[('qrf','without_'+x) for x in GROUPS]:
        r = g[(g.model==model)&(g['info']==info)].iloc[0]
        labels.append(model+' / '+info);values.append(r.loss_bp)
    fig,ax = plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.barh(labels,values);ax.invert_yaxis();ax.set_xlabel('q10 pinball loss (bp), lower is better')
    ax.set_title('EQ / available macro / calibrated; QRF scores averaged over 4 seeds')
    fig.savefig(OUT/'benchmark_ablation.png',dpi=180);plt.close(fig)
    (OUT/'analysis_manifest.json').write_text(json.dumps(dict(source_sha256=a.sha256(HERE/'report_models.py'),
        score_averaging='per-origin mean of seed losses; no sample multiplication',
        primary_comparisons=len(primary)),indent=2))
    print(primary.to_string(index=False))


if __name__ == '__main__':
    main()
