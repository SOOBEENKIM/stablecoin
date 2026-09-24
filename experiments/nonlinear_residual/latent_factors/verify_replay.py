"""Independent replay from saved numerical weights, choices and exact timestamps."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from models import study, METHODS

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
VERIFICATION_PATH=HERE/"VERIFICATION.json"
COMPARE_OLD_PCA=True


def replay(state, frame):
    raw=np.ascontiguousarray(frame[study.KCOLS].to_numpy(),dtype=float)
    x=(raw-np.array(state["mean"]))/np.array(state["sd"])
    family=state["family"]
    if family=="sequential_ols":
        first=np.column_stack([np.ones(len(frame)),frame.m])@np.array(state["first"])
    elif family.startswith("pca_"):
        rank=int(family[-1]);z=x@np.array(state["loading"])[:,:rank]
        first=np.column_stack([np.ones(len(frame)),z])@np.array(state["first"])
    elif family.startswith("ae_"):
        outputs=[]
        for member in state["members"]:
            z=x.copy()
            for w,b in zip(member["weights"][:2],member["biases"][:2]):
                z=z@np.array(w)+np.array(b)
                z=np.maximum(z,0) if member["activation"]=="relu" else np.tanh(z)
            z=(z-np.array(member["center"]))/np.array(member["scale"])
            outputs.append(np.column_stack([np.ones(len(frame)),z])@np.array(member["first"]))
        first=np.mean(outputs,axis=0)
    else:
        s=state["scales"]
        z=(x@np.array(state["loading"])[:,0]-s["zmean"])/s["zsd"]
        c=(np.log1p(raw.std(axis=1,ddof=1))-s["cmean"])/s["csd"]
        outputs=[]
        for arr in state["thetas"]:
            t=np.array(arr);beta=t[1]+t[2]*c
            if len(t)>3:
                u,d,v=np.split(t[3:],3);beta+=np.tanh(c[:,None]*u+d)@v
            outputs.append(t[0]+z*beta)
        first=s["ymean"]+s["ysd"]*np.mean(outputs,axis=0)
    return first+np.column_stack([np.ones(len(frame)),frame.g])@np.array(state["second"])


def main():
    manifest=json.loads((OUT/"RUN_MANIFEST.json").read_text())
    for name,digest in manifest["input_sha256"].items():
        assert study.pilot.sha(study.ROOT/name)==digest,name
    for name,digest in manifest["output_sha256"].items():
        assert study.pilot.sha(OUT/name)==digest,name
    grid=manifest["candidates"];family={r["candidate_id"]:r["family"] for r in grid}
    scores=pd.read_csv(OUT/"scores.csv")
    choices_checked=0;replayed=0;qr_checked=0;max_delta=0.
    for scenario in manifest["scenarios"]:
        p=OUT/scenario;panel,_=study.build_panel(scenario)
        cache={m:pd.read_csv(p/(m+"_candidates.csv.gz"),parse_dates=["time"]).set_index("time")
               for m in pd.period_range("2025-07","2026-03",freq="M").astype(str)}
        for month,raw in cache.items():
            fits=json.loads((p/(month+"_fits.json")).read_text());assert len(fits)==len(grid)
            for fit in fits:
                assert pd.Timestamp(fit["train_last"])<pd.Timestamp(fit["cutoff"])
                cid=fit["candidate"]["candidate_id"]
                state=json.loads((p/"checkpoints"/(month+"_"+cid+".json")).read_text())
                expected=replay(state,panel.loc[raw.index])
                delta=float(np.max(np.abs(expected-raw[cid].to_numpy())))
                np.testing.assert_allclose(expected,raw[cid].to_numpy(),atol=1e-7,rtol=1e-12,err_msg=str((scenario,month,cid,delta)))
                max_delta=max(delta,max_delta);replayed+=1
        choices=json.loads((p/"selection.json").read_text())
        for choice in choices:
            valid=pd.concat([cache[m].loc[cache[m].index<pd.Timestamp(m+"-01",tz="UTC")+pd.offsets.MonthBegin(1)] for m in choice["validation_months"]])
            assert len(valid)==choice["validation_n"]
            assert valid.index.max()<pd.Timestamp(choice["month"]+"-01",tz="UTC")
            method=choice["method"]
            if method=="pca_selected":ids=[cid for cid,f in family.items() if f in ["pca_1","pca_2"]]
            elif method in ["ae_selected","ae_reconstruction_selected"]:ids=[cid for cid,f in family.items() if f.startswith("ae_")]
            else:ids=[cid for cid,f in family.items() if f==method]
            ranked=[(float(valid[cid+"_recon"].mean()) if method=="ae_reconstruction_selected"
                     else float(np.mean((valid.y-valid[cid])**2)),cid) for cid in ids]
            best=min(ranked);assert best[1]==choice["candidate_id"]
            np.testing.assert_allclose(best[0],choice["validation_loss"],rtol=1e-10,atol=1e-9)
            choices_checked+=1
        pred=pd.read_csv(p/"predictions.csv.gz",parse_dates=["time"])
        assert (pred.groupby("time").model.nunique()==len(METHODS)).all()
        np.testing.assert_allclose(pred.observed_bp-pred.fitted_bp,pred.residual_bp,atol=1e-9,rtol=1e-10)
        for method in METHODS:
            g=pred.loc[(pred.model==method)&pred.evaluation_month.isin(study.pilot.MONTHS)]
            row=scores.loc[(scores.scenario==scenario)&(scores.model==method)&(scores.period=="Jan_Mar")].iloc[0]
            assert len(g)==row.n
            np.testing.assert_allclose(np.sqrt(np.mean(g.residual_bp**2)),row.rmse_bp,atol=1e-10,rtol=1e-10)
        frames=pd.read_csv(p/"residual_pairs.csv.gz",parse_dates=["time","target_time","fit_cutoff"])
        assert ((frames.target_time-frames.time).dt.total_seconds()==3600*frames.h).all()
        assert (frames.time>=frames.fit_cutoff).all()
        for choice in choices:
            f=frames.loc[(frames.month==choice["month"])&(frames.method==choice["method"])]
            values=cache[choice["month"]];e=values.y-values[choice["candidate_id"]]
            np.testing.assert_allclose(e.reindex(f.time),f.e_origin,atol=1e-9,rtol=1e-10)
            np.testing.assert_allclose(e.reindex(f.target_time),f.target,atol=1e-9,rtol=1e-10)
        old=pd.read_csv(HERE.parent/"three_way/results"/scenario/"residual_pairs.csv.gz",parse_dates=["time"])
        controls=[("sequential_ols","linear")]+([("pca_1","pca")] if COMPARE_OLD_PCA else [])
        for new_name,old_name in controls:
            a=frames.loc[frames.method==new_name].sort_values(["time","h"])
            b=old.loc[old.method==old_name].sort_values(["time","h"])
            assert np.array_equal(a.time.to_numpy(),b.time.to_numpy())
            np.testing.assert_allclose(a[["e_origin","target"]],b[["e_origin","target"]],atol=1e-8,rtol=1e-10)
        study.METHODS=METHODS;samples,_=study.common_samples(frames)
        coeff=pd.read_csv(p/"coefficients.csv");cols=study.SPECS["core"]["M2"]
        for method in ["pca_1","ae_selected","conditional_neural"]:
            frame=samples[("core","all",6,method)];x=frame[cols]
            model=QuantileRegressor(quantile=.1,alpha=0,solver="highs").fit((x-x.mean())/x.std(ddof=1),frame.target)
            saved=coeff.loc[(coeff.method==method)&(coeff.scope=="core")&(coeff["sample"]=="all")&
                (coeff.h==6)&(coeff.q==.1)&(coeff.model=="M2")].set_index("term").reindex(cols)
            np.testing.assert_allclose(model.coef_,saved.effect_bp_per_sd,atol=1e-7,rtol=1e-7);qr_checked+=1
    result=dict(status="passed",scenario_count=5,choices_reconstructed=choices_checked,
        candidate_month_predictions_replayed=replayed,max_replay_difference_bp=max_delta,
        replay_tolerance=dict(atol_bp=1e-7,rtol=1e-12),
        note="Relative tolerance accommodates machine rounding in extreme ill-conditioned candidate outputs; original verifier preserved.",
        independent_quantile_fits=qr_checked,checks=["input/output hashes","actual saved neural weights replayed",
          "past-only fit and model selection","exact horizons and same origin-month residual function",
          "matched observations across all methods","previous EQ control reproduced; PCA1 compared only for original OLS readout",
          "independent quantile solver","saved scores recomputed"])
    VERIFICATION_PATH.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
