"""Literature-motivated A3: frozen candidate set and original-order ablation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import sklearn
import xgboost
from models import Model, candidates, CONTROLS, FAMILIES, METHODS, followup, SEED

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
pilot = followup.pilot
ROOT = pilot.ROOT
GRID = candidates()
BY_ID = {c["candidate_id"]: c for c in GRID}
CONTROL_IDS = {c["family"]: c["candidate_id"] for c in GRID if c["family"] in CONTROLS}
MONTHS = followup.MONTHS
FIT_MONTHS = pd.period_range("2025-07", "2026-03", freq="M").astype(str).tolist()


def save_csv(frame, path):
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None)


def select(validation, structure=None, family=None, shrink=False):
    ids = [c["candidate_id"] for c in GRID if
           (c["family"] == family if family else c["family"] in FAMILIES and c["family"].startswith(structure+"_"))]
    if family in CONTROLS:
        ids = [CONTROL_IDS[family]]
    rows = []
    for cid in ids:
        c = BY_ID[cid]
        base = "sequential_ols" if c["family"].startswith("seq_") else "joint_ols"
        for weight in ([0., .1, .25, .5, 1.] if shrink else [1.]):
            p = validation[CONTROL_IDS[base]] + weight*(validation[cid]-validation[CONTROL_IDS[base]])
            rows.append(dict(candidate_id=cid, weight=weight, validation_mse=float(np.mean((validation.y-p)**2))))
    winner = sorted(rows, key=lambda r:(r["validation_mse"],r["weight"],r["candidate_id"]))[0]
    return winner, rows


def scenario_job(scenario):
    out = OUT / scenario
    out.mkdir(parents=True, exist_ok=True)
    panel, meta = followup.build_panel(scenario)
    data = panel.dropna(subset=["y", "m", "g"]+followup.KCOLS)
    cache, fits, all_candidate_predictions = {}, [], []
    for month in FIT_MONTHS:
        cutoff = pd.Timestamp(month+"-01", tz="UTC")
        end = cutoff+pd.offsets.MonthBegin(1)
        test = data.loc[(data.index >= cutoff)&(data.index < end+pd.Timedelta(hours=12))]
        values = pd.DataFrame({"y":test.y, "m":test.m, "g":test.g})
        for candidate in GRID:
            model = Model(candidate).fit(data, cutoff)
            values[candidate["candidate_id"]] = model.predict(test)
            record = dict(month=month, candidate=candidate, cutoff=str(cutoff),
                train_first=str(model.train.index.min()), train_last=str(model.train.index.max()),
                train_n=len(model.train), test_n=int((test.index < end).sum()), train_rmse=model.train_rmse,
                warnings=model.warnings)
            if candidate["family"].endswith("mlp"):
                record["iterations"] = [m.n_iter_ for m in model.model.networks]
                record["objectives"] = [float(m.loss_) for m in model.model.networks]
            fits.append(record)
        cache[month] = values
        cframe = values.reset_index().rename(columns={values.index.name or "index":"time"})
        cframe["fit_month"] = month
        all_candidate_predictions.append(cframe)
        print("A3 fitted %s %s: 31 candidates" % (scenario, month), flush=True)
    selection, validation_scores, residuals, predictions, bins = [], [], [], [], []
    for month in MONTHS:
        start = pd.Timestamp(month+"-01", tz="UTC")
        end = start+pd.offsets.MonthBegin(1)
        inner = [(start-pd.DateOffset(months=k)).strftime("%Y-%m") for k in [3,2,1]]
        inner = [m for m in inner if m >= "2025-07"]
        validation = pd.concat([cache[m].loc[cache[m].index < pd.Timestamp(m+"-01",tz="UTC")+pd.offsets.MonthBegin(1)] for m in inner])
        assert validation.index.max() < start
        values, chosen = cache[month], {}
        for method in CONTROLS+FAMILIES:
            chosen[method], ranked = select(validation, family=method)
            validation_scores.extend([dict(month=month, method=method, **r) for r in ranked])
        for structure in ["seq", "joint"]:
            for shrink in [False, True]:
                method = structure+("_shrunk" if shrink else "_selected")
                chosen[method], ranked = select(validation, structure=structure, shrink=shrink)
                validation_scores.extend([dict(month=month, method=method, **r) for r in ranked])
        origin_values = values.loc[values.index < end]
        train = data.loc[data.index < start]
        for method, choice in chosen.items():
            cid, weight = choice["candidate_id"], choice["weight"]
            family = BY_ID[cid]["family"]
            base = "sequential_ols" if family.startswith("seq_") else "joint_ols"
            fitted = values[CONTROL_IDS[base]]+weight*(values[cid]-values[CONTROL_IDS[base]])
            e = values.y-fitted
            selection.append(dict(month=month, method=method, chosen_family=family,
                effective_linear=bool(weight==0), validation_months=inner,
                validation_first=str(validation.index.min()), validation_last=str(validation.index.max()),
                validation_n=len(validation), **choice))
            origins = origin_values.index
            pred = pd.DataFrame(dict(time=origins, scenario=scenario, evaluation_month=month,
                model=method, candidate_id=cid, weight=weight, observed_bp=values.y.reindex(origins),
                fitted_bp=fitted.reindex(origins), residual_bp=e.reindex(origins)))
            predictions.append(pred.reset_index(drop=True))
            for feature in ["m","g"]:
                cuts = np.unique(np.quantile(train[feature], [.2,.4,.6,.8]))
                labels = np.searchsorted(cuts, origin_values[feature], side="right")
                for label in np.unique(labels):
                    mask = labels==label
                    bins.append(dict(month=month, method=method, feature=feature, bin=int(label),
                        n=int(mask.sum()), residual_mean_bp=float(e.reindex(origins).to_numpy()[mask].mean()),
                        cuts=json.dumps(cuts.tolist())))
            for h in followup.HORIZONS:
                target_times = origins+pd.Timedelta(hours=h)
                frame = panel.loc[origins, followup.STATES+["m","g"]].copy()
                frame["time"], frame["target_time"] = origins, target_times
                frame["month"], frame["fit_cutoff"] = month, start
                frame["method"], frame["h"] = method, h
                frame["e_origin"], frame["target"] = e.reindex(origins).to_numpy(), e.reindex(target_times).to_numpy()
                frame["y_change"] = panel.y.reindex(target_times).to_numpy()-panel.y.reindex(origins).to_numpy()
                frame["m_change"] = panel.m.reindex(target_times).to_numpy()-panel.m.reindex(origins).to_numpy()
                residuals.append(frame.dropna(subset=["e_origin","target"]).reset_index(drop=True))
    pred, frames = pd.concat(predictions,ignore_index=True), pd.concat(residuals,ignore_index=True)
    assert (pred.groupby("time").size()==len(METHODS)).all()
    assert (frames.groupby(["time","h"]).size()==len(METHODS)).all()
    assert ((frames.target_time-frames.time).dt.total_seconds()==3600*frames.h).all()
    followup.METHODS = METHODS
    coef, _, counts, sensitivity = followup.point_estimates(frames,panel)
    save_csv(pred,out/"predictions.csv.gz")
    save_csv(frames,out/"residual_pairs.csv.gz")
    save_csv(pd.concat(all_candidate_predictions,ignore_index=True),out/"all_candidate_predictions.csv.gz")
    for name, rows in [("coefficients",coef),("sample_counts",counts),("subperiod_and_stress",sensitivity),
                       ("validation_scores",validation_scores),("residual_bins",bins)]:
        save_csv(pd.DataFrame(rows),out/(name+".csv"))
    (out/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    (out/"fits.json").write_text(json.dumps(fits,indent=2)+"\n")
    (out/"clock.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("A3 follow-up completed "+scenario,flush=True)
    return scenario


def intervals(pred, repetitions=2000):
    result=[]
    for scenario, group in pred.groupby("scenario",sort=False):
        errors=group.pivot(index="time",columns="model",values="residual_bp").sort_index()
        errors.index=pd.DatetimeIndex(errors.index)
        assert errors.notna().all().all()
        names=list(errors.columns)
        for block in [3,7]:
            cal,w=followup.calendar_counts(errors.index,block,np.random.default_rng(SEED+block),repetitions)
            n=errors.iloc[:,0].groupby(errors.index.normalize()).size().reindex(cal,fill_value=0).to_numpy()
            sq=(errors**2).groupby(errors.index.normalize()).sum().reindex(cal,fill_value=0).to_numpy()
            ab=errors.abs().groupby(errors.index.normalize()).sum().reindex(cal,fill_value=0).to_numpy()
            brmse=np.sqrt((w@sq)/(w@n)[:,None]); bmae=(w@ab)/(w@n)[:,None]
            for base in ["sequential_ols","joint_ols"]:
                bi=names.index(base)
                for method in names:
                    if method==base: continue
                    mi=names.index(method)
                    dr=brmse[:,mi]-brmse[:,bi]; da=bmae[:,mi]-bmae[:,bi]
                    result.append(dict(scenario=scenario,model=method,baseline=base,block_days=block,
                        delta_rmse_bp=float(np.sqrt((errors[method]**2).mean())-np.sqrt((errors[base]**2).mean())),
                        relative_rmse_pct=float(100*(np.sqrt((errors[method]**2).mean()/(errors[base]**2).mean())-1)),
                        rmse_ci_low=float(np.quantile(dr,.025)),rmse_ci_high=float(np.quantile(dr,.975)),
                        delta_mae_bp=float(errors[method].abs().mean()-errors[base].abs().mean()),
                        mae_ci_low=float(np.quantile(da,.025)),mae_ci_high=float(np.quantile(da,.975))))
    return pd.DataFrame(result)


def main():
    started=time.monotonic()
    before=pilot.audit_archive()
    old=json.loads((HERE.parent/"three_way/results/RUN_MANIFEST.json").read_text())
    for name,digest in old["input_sha256"].items():
        assert pilot.sha(ROOT/name)==digest,name
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/"RUN_MANIFEST.json").exists(),"Preserve completed runs."
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(scenario_job,s) for s in pilot.SCENARIOS]
        for job in as_completed(jobs):
            print("Finished scenario:",job.result(),flush=True)
    pred=pd.concat([pd.read_csv(OUT/s/"predictions.csv.gz") for s in pilot.SCENARIOS],ignore_index=True)
    scores=[]
    for keys,group in pred.groupby(["scenario","model","evaluation_month"],sort=False):
        scores.append(dict(zip(["scenario","model","period"],keys),**pilot.metrics(group)))
    evaluation=pred.loc[pred.evaluation_month.isin(pilot.MONTHS)]
    for keys,group in evaluation.groupby(["scenario","model"],sort=False):
        scores.append(dict(zip(["scenario","model"],keys),period="Jan_Mar",**pilot.metrics(group)))
    for keys,group in pred.groupby(["scenario","model"],sort=False):
        scores.append(dict(zip(["scenario","model"],keys),period="Aug_Mar",**pilot.metrics(group)))
    pd.DataFrame(scores).to_csv(OUT/"scores.csv",index=False)
    intervals(evaluation).to_csv(OUT/"paired_block_intervals.csv",index=False)
    for name in ["coefficients","sample_counts","subperiod_and_stress","residual_bins"]:
        pd.concat([pd.read_csv(OUT/s/(name+".csv")).assign(scenario=s) for s in pilot.SCENARIOS],ignore_index=True).to_csv(OUT/(name+".csv"),index=False)
    after=pilot.audit_archive()
    assert before==after
    inputs=dict(old["input_sha256"])
    for path in [HERE/"PROTOCOL_KO.md",HERE/"REFERENCES_KO.md",HERE/"models.py",HERE/"check_guards.py",Path(__file__).resolve()]:
        inputs[str(path.relative_to(ROOT))]=pilot.sha(path)
    hashes={str(p.relative_to(OUT)):pilot.sha(p) for p in OUT.rglob("*") if p.is_file() and p.name!="RUN_MANIFEST.json"}
    manifest=dict(stage="A3 literature-guided conditional fit plus common downstream point estimates",publication_ready=False,
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        candidates=GRID,methods=METHODS,fit_count=5*9*31,scenarios=pilot.SCENARIOS,
        input_sha256=inputs,output_sha256=hashes,archive_check=after,elapsed_seconds=time.monotonic()-started,
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__),
        limitations=["historical sample previously explored; exploratory model extension",
          "same-time conditional fit, not future-price forecasting or causal decomposition",
          "FX/macro time zones unresolved; USDC/USD=1 assumption",
          "loss intervals condition on fitted predictions and do not adjust for adaptive research/multiple comparisons",
          "downstream A3 coefficients are point-estimate sensitivity, no new nuisance-refit coefficient intervals"])
    (OUT/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Completed A3 in %.0fs"%manifest["elapsed_seconds"],flush=True)


if __name__=="__main__":
    main()
