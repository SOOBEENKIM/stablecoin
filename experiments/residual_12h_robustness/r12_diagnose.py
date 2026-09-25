"""Accounting and state diagnostics only; no fitting, tuning, or new forecasts."""
import json
import numpy as np
import pandas as pd
from r12_core import OUT,old,hs,DEFS,VARIANTS,FAMILIES,SEEDS,read,save,file_key,sha,now,write_new
from r12_guard import verify_seal


def origin_losses(g):
    d=g.copy()
    d['raw_loss']=old.loss(d.target,d.pred_raw);d['cal_loss']=old.loss(d.target,d.pred_calibrated)
    d['below_raw']=(d.target<d.pred_raw).astype(float);d['below_cal']=(d.target<d.pred_calibrated).astype(float)
    return d.groupby('origin',as_index=False).agg(fold=('fold','first'),target=('target','first'),
        raw_loss=('raw_loss','mean'),cal_loss=('cal_loss','mean'),below_raw=('below_raw','mean'),
        below_cal=('below_cal','mean'),correction=('correction','mean'),seed_n=('seed','nunique'))


def main():
    stamp,_=verify_seal();assert (OUT/'EVALUATION_COMPLETE.json').exists()
    pred=read(OUT/'predictions.csv.gz');pred=pred[(pred.information=='F')&pred.strategy.isin(['ml_selected','linear','threshold'])]
    attribution=[];calibration=[];daily=[];leaveout=[];history=[]
    for definition in DEFS:
        for variant in VARIANTS:
            context=dict(definition=definition,variant=variant)
            case=pred[(pred.definition==definition)&(pred.variant==variant)]
            cells={s:origin_losses(g) for s,g in case.groupby('strategy')}
            for s,d in cells.items():
                for fold,g in d.groupby('fold'):
                    calibration.append(dict(**context,strategy=s,fold=fold,n=len(g),seeds=int(g.seed_n.iloc[0]),
                        raw_loss_bp=g.raw_loss.mean(),calibrated_loss_bp=g.cal_loss.mean(),
                        calibration_gain_bp=g.raw_loss.mean()-g.cal_loss.mean(),
                        below_rate_raw=g.below_raw.mean(),below_rate_calibrated=g.below_cal.mean(),mean_correction_bp=g.correction.mean()))
            for refname in ['linear','threshold']:
                ref,new=cells[refname],cells['ml_selected']
                assert list(ref.origin)==list(new.origin)
                for fold in sorted(ref.fold.unique()):
                    r=ref[ref.fold==fold].reset_index(drop=True);n=new[new.fold==fold].reset_index(drop=True)
                    raw_gap=float((r.raw_loss-n.raw_loss).mean());cal_gap=float((r.cal_loss-n.cal_loss).mean())
                    ref_shift=float((r.cal_loss-r.raw_loss).mean());ml_shift=float((n.cal_loss-n.raw_loss).mean())
                    assert abs((cal_gap-raw_gap)-(ref_shift-ml_shift))<1e-12
                    attribution.append(dict(**context,reference=refname,fold=fold,n=len(r),
                        raw_advantage_bp=raw_gap,calibrated_advantage_bp=cal_gap,
                        change_in_advantage_bp=cal_gap-raw_gap,reference_cal_minus_raw_bp=ref_shift,ml_cal_minus_raw_bp=ml_shift))
                    if fold=='2026-03':
                        days=pd.DatetimeIndex(r.origin).normalize()
                        for day in sorted(days.unique()):
                            keep=days!=day;on=~keep
                            daily.append(dict(**context,reference=refname,day=day.isoformat(),n=int(on.sum()),
                                raw_advantage_sum_bp=float((r.raw_loss-n.raw_loss)[on].sum()),
                                calibrated_advantage_sum_bp=float((r.cal_loss-n.cal_loss)[on].sum())))
                            leaveout.append(dict(**context,reference=refname,excluded_day=day.isoformat(),remaining_n=int(keep.sum()),
                                raw_gain_pct=100*(1-n.raw_loss[keep].mean()/r.raw_loss[keep].mean()),
                                calibrated_gain_pct=100*(1-n.cal_loss[keep].mean()/r.cal_loss[keep].mean())))
            # Inspect each selected model's own calibration history, including extra seeds.
            first=pd.concat([read(OUT/'candidates'/(file_key(definition,variant,'F',k)+'.csv.gz')) for k in FAMILIES],ignore_index=True)
            streams={SEEDS[0]:first}
            for seed in SEEDS[1:]:streams[seed]=read(OUT/'extra_candidates'/(file_key(definition,variant,'F',seed)+'.csv.gz'))
            grouped={(seed,int(cid)):g.sort_values('origin').reset_index(drop=True) for seed,s in streams.items() for cid,g in s.groupby('candidate')}
            for row in case.itertuples():
                s=grouped[int(row.seed),int(row.candidate)]
                labels=pd.DatetimeIndex(s.target_time)
                stop=labels.searchsorted(row.origin,side='left')
                start=max(labels.searchsorted(row.origin-pd.Timedelta(days=90),side='left'),stop-60)
                hist=s.iloc[start:stop]
                assert len(hist)==row.calibration_n and len(hist)>=30 and hist.target_time.max()<row.origin
                errors=(hist.target-hist.pred_raw).to_numpy()
                expected=np.sort(errors)[int(np.ceil(.1*len(errors)))-1]
                assert abs(expected-row.correction)<1e-12
                history.append(dict(**context,strategy=row.strategy,seed=int(row.seed),candidate=int(row.candidate),fold=row.fold,
                    origin=row.origin,target_time=row.target_time,history_n=len(hist),
                    earliest_label=hist.target_time.min(),latest_label=hist.target_time.max(),
                    oldest_age_hours=(row.origin-hist.target_time.min()).total_seconds()/3600,
                    latest_age_hours=(row.origin-hist.target_time.max()).total_seconds()/3600,
                    history_label_days=hist.target_time.dt.normalize().nunique(),
                    previous_origin_month_fraction=float((hist.fold!=row.fold).mean()),
                    history_error_q10=expected,correction_bp=row.correction,
                    raw_error_bp=row.target-row.pred_raw,calibrated_error_bp=row.target-row.pred_calibrated))
    h=pd.DataFrame(history)
    keys=['definition','variant','strategy','seed','fold']
    monthly_history=h.groupby(keys,as_index=False).agg(n=('origin','size'),
        mean_correction_bp=('correction_bp','mean'),correction_min_bp=('correction_bp','min'),correction_max_bp=('correction_bp','max'),
        mean_oldest_age_hours=('oldest_age_hours','mean'),mean_latest_age_hours=('latest_age_hours','mean'),
        mean_history_label_days=('history_label_days','mean'),mean_previous_origin_month_fraction=('previous_origin_month_fraction','mean'),
        realized_month_raw_error_q10=('raw_error_bp',lambda x:float(np.quantile(x,.1,method='inverted_cdf'))))
    for name,d in [('calibration_effects',pd.DataFrame(calibration)),('gap_attribution',pd.DataFrame(attribution)),
        ('march_daily',pd.DataFrame(daily)),('march_leave_one_day_out',pd.DataFrame(leaveout)),
        ('calibration_history',h),('calibration_history_monthly',monthly_history)]:
        save(d,OUT/(name+('.csv.gz' if name=='calibration_history' else '.csv')))
    states=pd.DataFrame(json.loads((OUT/'state_distributions.json').read_text()))
    save(states.drop(columns='residualizer'),OUT/'state_distributions.csv')
    write_new(OUT/'DIAGNOSTICS_COMPLETE.json',dict(**stamp,completed_utc=now(),history_rows=len(h),
        exact_gap_accounting_verified=True,own_candidate_history_verified=True,
        economic_causality_identified=False,no_changed_calibration_forecast=True))
    print(pd.DataFrame(attribution).query("fold == '2026-03'").to_string(index=False))


if __name__=='__main__':main()
