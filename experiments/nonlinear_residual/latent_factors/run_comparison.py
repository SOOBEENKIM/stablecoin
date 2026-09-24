"""Run the frozen A4 latent-factor experiment, preserving earlier stages."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import sklearn
import scipy
from models import Model, candidates, CONTROLS, FAMILIES, METHODS, SEED, study

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
GRID = candidates()
BY_ID = {c["candidate_id"]: c for c in GRID}
CONTROL_IDS = {c["family"]: c["candidate_id"] for c in GRID if c["family"] in CONTROLS}
FIT_MONTHS = pd.period_range("2025-07", "2026-03", freq="M").astype(str).tolist()
PANELS = {}


def save_csv(frame, path):
    frame.to_csv(path, index=False, compression={"method":"gzip", "mtime":0} if str(path).endswith(".gz") else None)


def fit_month(scenario, month):
    started = time.monotonic()
    panel = PANELS[scenario]
    data = panel.dropna(subset=["y","m","g"]+study.KCOLS)
    cutoff = pd.Timestamp(month+"-01", tz="UTC")
    end = cutoff+pd.offsets.MonthBegin(1)
    test = data.loc[(data.index>=cutoff)&(data.index<end+pd.Timedelta(hours=12))]
    values = test[["y","m","g"]].copy()
    fits = []
    for c in GRID:
        model = Model(c).fit(data, cutoff)
        cid = c["candidate_id"]
        values[cid] = model.predict(test)
        values[cid+"_recon"] = model.reconstruction_error(test)
        record = dict(month=month, candidate=c, cutoff=str(cutoff),
            train_first=str(model.train.index.min()), train_last=str(model.train.index.max()),
            train_n=len(model.train), test_n=int((test.index<end).sum()), **model.log)
        fits.append(record)
        if c["family"].startswith("ae_"):
            from models import latent
            x = model.standardize(test)
            for j,(net,center,scale,coef) in enumerate(model.members):
                values[cid+"_member%d"%j] = (np.column_stack([np.ones(len(x)),(latent(net,x)-center)/scale])@coef+
                    np.column_stack([np.ones(len(x)),test.g])@model.second)
        path = OUT/scenario/"checkpoints"/(month+"_"+cid+".json")
        path.write_text(json.dumps(model.checkpoint(), separators=(",",":"))+"\n")
    frame = values.reset_index().rename(columns={values.index.name or "index":"time"})
    frame["fit_month"] = month
    save_csv(frame, OUT/scenario/(month+"_candidates.csv.gz"))
    (OUT/scenario/(month+"_fits.json")).write_text(json.dumps(fits, indent=2)+"\n")
    print("A4 fitted %s %s: 16 candidates, %.1fs"%(scenario,month,time.monotonic()-started), flush=True)


def eligible(method):
    if method in CONTROLS+FAMILIES:
        return [c["candidate_id"] for c in GRID if c["family"]==method]
    if method=="pca_selected":
        return [CONTROL_IDS["pca_1"],CONTROL_IDS["pca_2"]]
    return [c["candidate_id"] for c in GRID if c["family"].startswith("ae_")]


def analyze_scenario(scenario):
    out = OUT/scenario
    panel = PANELS[scenario]
    data = panel.dropna(subset=["y","m","g"]+study.KCOLS)
    cache = {m:pd.read_csv(out/(m+"_candidates.csv.gz"), parse_dates=["time"]).set_index("time") for m in FIT_MONTHS}
    predictions, pairs, selection, scores, bins, seed_spread = [], [], [], [], [], []
    for month in study.MONTHS:
        start = pd.Timestamp(month+"-01", tz="UTC")
        end = start+pd.offsets.MonthBegin(1)
        months = [(start-pd.DateOffset(months=k)).strftime("%Y-%m") for k in [3,2,1]]
        months = [m for m in months if m >= "2025-07"]
        valid = pd.concat([cache[m].loc[cache[m].index<pd.Timestamp(m+"-01",tz="UTC")+pd.offsets.MonthBegin(1)] for m in months])
        assert valid.index.max()<start
        values = cache[month]
        origins = values.index[values.index<end]
        train = data.loc[data.index<start]
        for method in METHODS:
            metric = "reference_reconstruction_mse" if method=="ae_reconstruction_selected" else "usdt_mse"
            ranked = []
            for cid in eligible(method):
                loss = float(valid[cid+"_recon"].mean()) if metric=="reference_reconstruction_mse" else float(np.mean((valid.y-valid[cid])**2))
                ranked.append((loss,cid))
                scores.append(dict(month=month,method=method,candidate_id=cid,criterion=metric,validation_loss=loss))
            loss,cid = min(ranked)
            selection.append(dict(month=month,method=method,candidate_id=cid,chosen_family=BY_ID[cid]["family"],
                criterion=metric,validation_loss=loss,validation_months=months,validation_n=len(valid),
                validation_first=str(valid.index.min()),validation_last=str(valid.index.max())))
            e = values.y-values[cid]
            predictions.append(pd.DataFrame(dict(time=origins,scenario=scenario,evaluation_month=month,
                model=method,candidate_id=cid,observed_bp=values.y.reindex(origins),
                fitted_bp=values[cid].reindex(origins),residual_bp=e.reindex(origins),
                reference_reconstruction_mse=values[cid+"_recon"].reindex(origins))).reset_index(drop=True))
            if BY_ID[cid]["family"].startswith("ae_"):
                members=values.loc[origins,[cid+"_member%d"%j for j in range(3)]].to_numpy()
                seed_spread.append(dict(month=month,method=method,candidate_id=cid,
                    member_prediction_sd_bp=float(np.mean(members.std(axis=1,ddof=1))),
                    member_rmses_bp=json.dumps(np.sqrt(np.mean((values.y.reindex(origins).to_numpy()[:,None]-members)**2,axis=0)).tolist())))
            for feature in ["m","g"]:
                cuts=np.unique(np.quantile(train[feature],[.2,.4,.6,.8]))
                labels=np.searchsorted(cuts,values.loc[origins,feature],side="right")
                for label in np.unique(labels):
                    mask=labels==label
                    bins.append(dict(month=month,method=method,feature=feature,bin=int(label),n=int(mask.sum()),
                        residual_mean_bp=float(e.reindex(origins).to_numpy()[mask].mean()),cuts=json.dumps(cuts.tolist())))
            for h in study.HORIZONS:
                targets=origins+pd.Timedelta(hours=h)
                f=panel.loc[origins,study.STATES+["m","g"]].copy()
                f["time"],f["target_time"],f["month"],f["fit_cutoff"] = origins,targets,month,start
                f["method"],f["h"] = method,h
                f["e_origin"],f["target"] = e.reindex(origins).to_numpy(),e.reindex(targets).to_numpy()
                f["y_change"]=panel.y.reindex(targets).to_numpy()-panel.y.reindex(origins).to_numpy()
                f["m_change"]=panel.m.reindex(targets).to_numpy()-panel.m.reindex(origins).to_numpy()
                pairs.append(f.dropna(subset=["e_origin","target"]).reset_index(drop=True))
    pred,frames=pd.concat(predictions,ignore_index=True),pd.concat(pairs,ignore_index=True)
    assert (pred.groupby("time").size()==len(METHODS)).all()
    assert (frames.groupby(["time","h"]).size()==len(METHODS)).all()
    study.METHODS=METHODS
    coef,_,counts,sensitivity=study.point_estimates(frames,panel)
    save_csv(pred,out/"predictions.csv.gz")
    save_csv(frames,out/"residual_pairs.csv.gz")
    for name,rows in [("coefficients",coef),("sample_counts",counts),("subperiod_and_stress",sensitivity),
                      ("validation_scores",scores),("residual_bins",bins),("seed_spread",seed_spread)]:
        save_csv(pd.DataFrame(rows),out/(name+".csv"))
    (out/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    print("A4 same downstream analysis complete: "+scenario,flush=True)


def intervals(pred):
    records=[]
    for scenario,group in pred.groupby("scenario",sort=False):
        errors=group.pivot(index="time",columns="model",values="residual_bp").sort_index()
        errors.index=pd.DatetimeIndex(errors.index)
        assert errors.notna().all().all()
        names=list(errors.columns)
        for block in [3,7]:
            cal,w=study.calendar_counts(errors.index,block,np.random.default_rng(SEED+block),2000)
            n=errors.iloc[:,0].groupby(errors.index.normalize()).size().reindex(cal,fill_value=0).to_numpy()
            sq=errors.pow(2).groupby(errors.index.normalize()).sum().reindex(cal,fill_value=0).to_numpy()
            ab=errors.abs().groupby(errors.index.normalize()).sum().reindex(cal,fill_value=0).to_numpy()
            rmse=np.sqrt((w@sq)/(w@n)[:,None]);mae=(w@ab)/(w@n)[:,None]
            for baseline in ["sequential_ols","pca_1","pca_2","pca_selected","conditional_linear"]:
                bi=names.index(baseline)
                for method in names:
                    if method==baseline:continue
                    mi=names.index(method)
                    dr=rmse[:,mi]-rmse[:,bi];da=mae[:,mi]-mae[:,bi]
                    records.append(dict(scenario=scenario,model=method,baseline=baseline,block_days=block,
                        delta_rmse_bp=float(np.sqrt(errors[method].pow(2).mean())-np.sqrt(errors[baseline].pow(2).mean())),
                        relative_rmse_pct=float(100*(np.sqrt(errors[method].pow(2).mean()/errors[baseline].pow(2).mean())-1)),
                        rmse_ci_low=float(np.quantile(dr,.025)),rmse_ci_high=float(np.quantile(dr,.975)),
                        delta_mae_bp=float(errors[method].abs().mean()-errors[baseline].abs().mean()),
                        mae_ci_low=float(np.quantile(da,.025)),mae_ci_high=float(np.quantile(da,.975))))
    return pd.DataFrame(records)


def main():
    started=time.monotonic()
    assert not OUT.exists(),"Preserve completed or partial runs; investigate before rerunning."
    before=study.pilot.audit_archive()
    old=json.loads((HERE.parent/"literature_models/results/RUN_MANIFEST.json").read_text())
    for name,digest in old["input_sha256"].items():
        assert study.pilot.sha(study.ROOT/name)==digest,name
    for scenario in study.pilot.SCENARIOS:
        (OUT/scenario/"checkpoints").mkdir(parents=True)
        PANELS[scenario],meta=study.build_panel(scenario)
        (OUT/scenario/"clock.json").write_text(json.dumps(meta,indent=2)+"\n")
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(fit_month,s,m) for s in study.pilot.SCENARIOS for m in FIT_MONTHS]
        for job in as_completed(jobs):job.result()
    with ProcessPoolExecutor(max_workers=5) as pool:
        jobs=[pool.submit(analyze_scenario,s) for s in study.pilot.SCENARIOS]
        for job in as_completed(jobs):job.result()
    pred=pd.concat([pd.read_csv(OUT/s/"predictions.csv.gz") for s in study.pilot.SCENARIOS],ignore_index=True)
    rows=[]
    for keys,group in pred.groupby(["scenario","model","evaluation_month"],sort=False):
        rows.append(dict(zip(["scenario","model","period"],keys),**study.pilot.metrics(group),
            reference_reconstruction_mse=float(group.reference_reconstruction_mse.mean())))
    for period,months in [("Jan_Mar",study.pilot.MONTHS),("Aug_Mar",study.MONTHS)]:
        for keys,group in pred.loc[pred.evaluation_month.isin(months)].groupby(["scenario","model"],sort=False):
            rows.append(dict(zip(["scenario","model"],keys),period=period,**study.pilot.metrics(group),
                reference_reconstruction_mse=float(group.reference_reconstruction_mse.mean())))
    save_csv(pd.DataFrame(rows),OUT/"scores.csv")
    intervals(pred.loc[pred.evaluation_month.isin(study.pilot.MONTHS)]).to_csv(OUT/"paired_block_intervals.csv",index=False)
    for name in ["coefficients","sample_counts","subperiod_and_stress","seed_spread","residual_bins"]:
        save_csv(pd.concat([pd.read_csv(OUT/s/(name+".csv")).assign(scenario=s) for s in study.pilot.SCENARIOS],ignore_index=True),OUT/(name+".csv"))
    correlations=[]
    for scenario,group in pred.loc[pred.evaluation_month.isin(study.pilot.MONTHS)].groupby("scenario"):
        e=group.pivot(index="time",columns="model",values="residual_bp")
        for method in METHODS:
            correlations.append(dict(scenario=scenario,model=method,correlation_to_eq_ols=float(e[method].corr(e.sequential_ols)),
                correlation_to_pca1=float(e[method].corr(e.pca_1))))
    save_csv(pd.DataFrame(correlations),OUT/"residual_correlations.csv")
    after=study.pilot.audit_archive();assert before==after
    inputs=dict(old["input_sha256"])
    for path in list(HERE.glob("*.py"))+[HERE/"PROTOCOL_KO.md"]:
        inputs[str(path.relative_to(study.ROOT))]=study.pilot.sha(path)
    hashes={str(p.relative_to(OUT)):study.pilot.sha(p) for p in OUT.rglob("*") if p.is_file()}
    manifest=dict(stage="A4 nonlinear latent factors",publication_ready=False,
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=study.ROOT,text=True).strip(),
        candidates=GRID,methods=METHODS,fit_count=5*9*len(GRID),scenarios=study.pilot.SCENARIOS,
        input_sha256=inputs,output_sha256=hashes,archive_check=after,elapsed_seconds=time.monotonic()-started,
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__),
        limitations=["exploratory reuse of historical sample", "same-time conditional decomposition, not price forecasting",
          "no causal identification of common or USDT-specific component", "FX clocks unresolved; USDC/USD=1 assumption",
          "fixed-prediction intervals exclude refitting, selection, multiplicity and adaptive research",
          "downstream coefficients are point estimates; no A4 nuisance-refit coefficient confidence intervals",
          "reference-only AE and single-asset conditional loading are adaptations, not exact paper replications"])
    (OUT/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("A4 completed in %.1fs"%manifest["elapsed_seconds"],flush=True)


if __name__=="__main__":
    main()
