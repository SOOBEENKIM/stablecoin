"""Post-result arithmetic audit: historical-mean skill versus zero residual."""
from common import *


def main():
    rows,intervals,examples=[],[],[]
    for scenario in study.pilot.SCENARIOS:
        p=pd.read_csv(OUT/scenario/'probe_predictions.csv.gz',parse_dates=['time'])
        for key,g in p.groupby(['method','scope','algorithm']):
            fields=dict(zip(['method','scope','algorithm'],key),scenario=scenario)
            for period,f in [('all',g)]+list(g.groupby('month')):
                zero=float(np.mean(f.target**2))
                rows.append(dict(fields,period=period,n=len(f),zero_mse=zero,probe_mse=f.loss.mean(),
                    past_mean_mse=f.baseline_loss.mean(),skill_vs_zero=1-f.loss.mean()/zero,
                    skill_vs_past_mean=1-f.loss.mean()/f.baseline_loss.mean()))
            b=g[['time','target']].copy();b['loss']=b.target**2
            intervals.extend([dict(fields,**r) for r in paired(g,b)])
        pred=pd.read_csv(OUT/scenario/'predictions.csv.gz',parse_dates=['time'])
        selected=pred.loc[pred.evaluation_month.isin(MONTHS)]
        date=selected.time.min()
        examples.append(selected.loc[selected.time==date,['time','model','observed_bp','fitted_bp','residual_bp']].assign(scenario=scenario))
    save(pd.DataFrame(rows),OUT/'probe_zero_baseline_audit.csv')
    save(pd.DataFrame(intervals),OUT/'probe_zero_baseline_intervals.csv')
    save(pd.concat(examples),OUT/'rmse_examples.csv')
    c=pd.read_csv(OUT/'forecast_intervals.csv')
    # Identical regularized QR solutions can differ by machine rounding only.
    tol=1e-8
    c['interpretation']=np.where(c.delta.abs()<=tol,'numerical_tie',
        np.where(c.ci_high < -tol,'lower_loss_interval',
        np.where(c.ci_low > tol,'higher_loss_interval','interval_includes_zero')))
    save(c,OUT/'forecast_contrast_audit.csv')


if __name__=='__main__':main()
