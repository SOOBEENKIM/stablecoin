"""Reuse every frozen encoder; regularize collinear factor-to-USDT readouts."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import time
import subprocess
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import run_comparison as base
from verify_replay import replay

HERE=Path(__file__).resolve().parent
SOURCE=HERE/"results"
OUT=HERE/"stabilized"


def codes(state, frame, member):
    x=(np.ascontiguousarray(frame[base.study.KCOLS].to_numpy(),dtype=float)-np.array(state["mean"]))/np.array(state["sd"])
    for w,b in zip(member["weights"][:2],member["biases"][:2]):
        x=x@np.array(w)+np.array(b)
        x=np.maximum(x,0) if member["activation"]=="relu" else np.tanh(x)
    return (x-np.array(member["center"]))/np.array(member["scale"])


def refit_month(scenario, month):
    panel=base.PANELS[scenario]
    data=panel.dropna(subset=["y","m","g"]+base.study.KCOLS)
    cutoff=pd.Timestamp(month+"-01",tz="UTC")
    train=data.loc[data.index<cutoff]
    values=pd.read_csv(SOURCE/scenario/(month+"_candidates.csv.gz"),parse_dates=["time"]).set_index("time")
    test=data.loc[values.index]
    fits=json.loads((SOURCE/scenario/(month+"_fits.json")).read_text())
    for fit in fits:
        cid=fit["candidate"]["candidate_id"]
        state=json.loads((SOURCE/scenario/"checkpoints"/(month+"_"+cid+".json")).read_text())
        family=state["family"]
        if family.startswith("ae_"):
            for member in state["members"]:
                z=codes(state,train,member)
                ridge=Ridge(alpha=1.,fit_intercept=True).fit(z,train.y)
                member["first"]=np.r_[ridge.intercept_,ridge.coef_].tolist()
        elif family.startswith("pca_"):
            rank=int(family[-1])
            x=(np.ascontiguousarray(train[base.study.KCOLS].to_numpy(),dtype=float)-np.array(state["mean"]))/np.array(state["sd"])
            pc=x@np.array(state["loading"])[:,:rank]
            center,scale=pc.mean(axis=0),pc.std(axis=0,ddof=1)
            ridge=Ridge(alpha=1.,fit_intercept=True).fit((pc-center)/scale,train.y)
            state["first"]=np.r_[ridge.intercept_-ridge.coef_@(center/scale),ridge.coef_/scale].tolist()
        if family.startswith(("ae_","pca_")):
            g=np.column_stack([np.ones(len(train)),train.g])
            first=replay(state,train)-g@np.array(state["second"])
            state["second"]=np.linalg.lstsq(g,train.y-first,rcond=None)[0].tolist()
            state["readout_alpha"]=1.
            values[cid]=replay(state,test)
            fit["readout_alpha"]=1.
            fit["train_rmse_bp"]=float(np.sqrt(np.mean((train.y-replay(state,train))**2)))
            state["log"]["train_rmse_bp"]=fit["train_rmse_bp"]
            if family.startswith("ae_"):
                for j,member in enumerate(state["members"]):
                    z=codes(state,test,member)
                    values[cid+"_member%d"%j]=np.column_stack([np.ones(len(test)),z])@np.array(member["first"])+np.column_stack([np.ones(len(test)),test.g])@np.array(state["second"])
        (OUT/scenario/"checkpoints"/(month+"_"+cid+".json")).write_text(json.dumps(state,separators=(",",":"))+"\n")
    base.save_csv(values.reset_index(),OUT/scenario/(month+"_candidates.csv.gz"))
    (OUT/scenario/(month+"_fits.json")).write_text(json.dumps(fits,indent=2)+"\n")


def main():
    started=time.monotonic()
    assert not OUT.exists(),"Preserve prior run."
    original=json.loads((SOURCE/"RUN_MANIFEST.json").read_text())
    for name,digest in original["input_sha256"].items():
        assert base.study.pilot.sha(base.study.ROOT/name)==digest,name
    for name,digest in original["output_sha256"].items():
        assert base.study.pilot.sha(SOURCE/name)==digest,name
    before=base.study.pilot.audit_archive()
    for scenario in base.study.pilot.SCENARIOS:
        (OUT/scenario/"checkpoints").mkdir(parents=True)
        base.PANELS[scenario],meta=base.study.build_panel(scenario)
        (OUT/scenario/"clock.json").write_text(json.dumps(meta,indent=2)+"\n")
    with ProcessPoolExecutor(max_workers=5) as pool:
        jobs=[pool.submit(refit_month,s,m) for s in base.study.pilot.SCENARIOS for m in base.FIT_MONTHS]
        for job in as_completed(jobs):job.result()
    print("Stabilized all 720 readouts; frozen encoders reused.",flush=True)
    base.OUT=OUT
    with ProcessPoolExecutor(max_workers=5) as pool:
        jobs=[pool.submit(base.analyze_scenario,s) for s in base.study.pilot.SCENARIOS]
        for job in as_completed(jobs):job.result()
    pred=pd.concat([pd.read_csv(OUT/s/"predictions.csv.gz") for s in base.study.pilot.SCENARIOS],ignore_index=True)
    rows=[]
    for keys,g in pred.groupby(["scenario","model","evaluation_month"],sort=False):
        rows.append(dict(zip(["scenario","model","period"],keys),**base.study.pilot.metrics(g),reference_reconstruction_mse=float(g.reference_reconstruction_mse.mean())))
    for period,months in [("Jan_Mar",base.study.pilot.MONTHS),("Aug_Mar",base.study.MONTHS)]:
        for keys,g in pred.loc[pred.evaluation_month.isin(months)].groupby(["scenario","model"],sort=False):
            rows.append(dict(zip(["scenario","model"],keys),period=period,**base.study.pilot.metrics(g),reference_reconstruction_mse=float(g.reference_reconstruction_mse.mean())))
    base.save_csv(pd.DataFrame(rows),OUT/"scores.csv")
    base.intervals(pred.loc[pred.evaluation_month.isin(base.study.pilot.MONTHS)]).to_csv(OUT/"paired_block_intervals.csv",index=False)
    for name in ["coefficients","sample_counts","subperiod_and_stress","seed_spread","residual_bins"]:
        base.save_csv(pd.concat([pd.read_csv(OUT/s/(name+".csv")).assign(scenario=s) for s in base.study.pilot.SCENARIOS],ignore_index=True),OUT/(name+".csv"))
    after=base.study.pilot.audit_archive();assert before==after
    inputs=dict(original["input_sha256"])
    for path in [Path(__file__).resolve(),HERE/"STABILITY_KO.md",HERE/"verify_replay.py",SOURCE/"RUN_MANIFEST.json"]:
        inputs[str(path.relative_to(base.study.ROOT))]=base.study.pilot.sha(path)
    manifest=dict(stage="A4 fixed-encoder ridge-readout stability follow-up",exploratory_post_result=True,
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=base.study.ROOT).strip(),
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),candidates=base.GRID,methods=base.METHODS,
        fit_count=720,new_neural_fits=0,readout_alpha=1.,scenarios=base.study.pilot.SCENARIOS,
        input_sha256=inputs,output_sha256={str(p.relative_to(OUT)):base.study.pilot.sha(p) for p in OUT.rglob("*") if p.is_file()},
        archive_check=after,elapsed_seconds=time.monotonic()-started,software=original["software"],
        limitations=original["limitations"]+["fixed ridge readout added after numerical instability was observed"])
    (OUT/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Stability follow-up completed in %.1fs"%manifest["elapsed_seconds"],flush=True)


if __name__=="__main__":main()
