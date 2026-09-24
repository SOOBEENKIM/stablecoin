"""Independent reconstruction of A3 choices, times, controls and saved scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from models import followup, METHODS, FAMILIES, CONTROLS

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
pilot=followup.pilot


def main():
    manifest=json.loads((OUT/"RUN_MANIFEST.json").read_text())
    for name,digest in manifest["input_sha256"].items():
        assert pilot.sha(pilot.ROOT/name)==digest,name
    for name,digest in manifest["output_sha256"].items():
        assert pilot.sha(OUT/name)==digest,name
    grid=manifest["candidates"]
    id_family={r["candidate_id"]:r["family"] for r in grid}
    controls={r["family"]:r["candidate_id"] for r in grid if r["family"] in CONTROLS}
    score=pd.read_csv(OUT/"scores.csv")
    selections_checked=0; max_coefficient_difference=0.; checked_coefficients=0
    for scenario in manifest["scenarios"]:
        p=OUT/scenario
        raw=pd.read_csv(p/"all_candidate_predictions.csv.gz",parse_dates=["time"])
        fits=json.loads((p/"fits.json").read_text())
        assert len(fits)==9*31
        for fit in fits:
            assert pd.Timestamp(fit["train_last"]) < pd.Timestamp(fit["cutoff"])
        choices=json.loads((p/"selection.json").read_text())
        for choice in choices:
            vals=pd.concat([raw.loc[(raw.fit_month==m)&(raw.time<pd.Timestamp(m+"-01",tz="UTC")+pd.offsets.MonthBegin(1))]
                for m in choice["validation_months"]])
            assert len(vals)==choice["validation_n"]
            assert vals.time.max()<pd.Timestamp(choice["month"]+"-01",tz="UTC")
            method=choice["method"]
            if method in CONTROLS+FAMILIES:
                ids=[cid for cid,f in id_family.items() if f==method]
                weights=[1.]
            else:
                structure=method.split("_")[0]
                ids=[cid for cid,f in id_family.items() if f in FAMILIES and f.startswith(structure+"_")]
                weights=[0.,.1,.25,.5,1.] if method.endswith("shrunk") else [1.]
            ranked=[]
            for cid in ids:
                base=controls["sequential_ols" if id_family[cid].startswith("seq_") else "joint_ols"]
                for weight in weights:
                    pred=vals[base]+weight*(vals[cid]-vals[base])
                    ranked.append((float(np.mean((vals.y-pred)**2)),weight,cid))
            best=min(ranked)
            assert (best[1],best[2])==(choice["weight"],choice["candidate_id"])
            np.testing.assert_allclose(best[0],choice["validation_mse"],atol=1e-9,rtol=1e-10)
            selections_checked+=1
        pred=pd.read_csv(p/"predictions.csv.gz",parse_dates=["time"])
        assert (pred.groupby("time").model.nunique()==len(METHODS)).all()
        np.testing.assert_allclose(pred.observed_bp-pred.fitted_bp,pred.residual_bp,atol=1e-9,rtol=1e-10)
        for method in METHODS:
            group=pred.loc[(pred.model==method)&pred.evaluation_month.isin(pilot.MONTHS)]
            row=score.loc[(score.scenario==scenario)&(score.model==method)&(score.period=="Jan_Mar")].iloc[0]
            assert len(group)==row.n
            np.testing.assert_allclose(np.sqrt(np.mean(group.residual_bp**2)),row.rmse_bp,atol=1e-10,rtol=1e-10)
        frames=pd.read_csv(p/"residual_pairs.csv.gz",parse_dates=["time","target_time","fit_cutoff"])
        assert ((frames.target_time-frames.time).dt.total_seconds()==3600*frames.h).all()
        assert (frames.time>=frames.fit_cutoff).all()
        # Same origin-month function on both sides, including month boundaries.
        for choice in choices:
            f=frames.loc[(frames.month==choice["month"])&(frames.method==choice["method"])]
            values=raw.loc[raw.fit_month==choice["month"]].set_index("time")
            cid,weight=choice["candidate_id"],choice["weight"]
            base=controls["sequential_ols" if id_family[cid].startswith("seq_") else "joint_ols"]
            residual=values.y-(values[base]+weight*(values[cid]-values[base]))
            np.testing.assert_allclose(residual.reindex(f.time),f.e_origin,atol=1e-9,rtol=1e-10)
            np.testing.assert_allclose(residual.reindex(f.target_time),f.target,atol=1e-9,rtol=1e-10)
        old=pd.read_csv(HERE.parent/"three_way/results"/scenario/"residual_pairs.csv.gz",parse_dates=["time","target_time"])
        for new_name,old_name in [("sequential_ols","linear"),("joint_ols","joint_linear"),("pca","pca")]:
            a=frames.loc[frames.method==new_name].sort_values(["time","h"])
            b=old.loc[old.method==old_name].sort_values(["time","h"])
            assert len(a)==len(b)
            assert np.array_equal(a.time.to_numpy(),b.time.to_numpy())
            np.testing.assert_allclose(a[["e_origin","target"]],b[["e_origin","target"]],atol=1e-8,rtol=1e-10)
        followup.METHODS=METHODS
        samples,_=followup.common_samples(frames)
        coefs=pd.read_csv(p/"coefficients.csv")
        columns=followup.SPECS["core"]["M2"]
        for method in ["sequential_ols","seq_selected","joint_selected"]:
            frame=samples[("core","all",6,method)]
            x=frame[columns]; mean=x.mean(); sd=x.std(ddof=1)
            fitted=QuantileRegressor(quantile=.1,alpha=0,solver="highs").fit((x-mean)/sd,frame.target)
            saved=coefs.loc[(coefs.method==method)&(coefs.scope=="core")&(coefs["sample"]=="all")&
                (coefs.h==6)&(coefs.q==.1)&(coefs.model=="M2")].set_index("term").reindex(columns)
            delta=float(np.max(np.abs(fitted.coef_-saved.effect_bp_per_sd.to_numpy())))
            assert delta<1e-7,delta
            max_coefficient_difference=max(max_coefficient_difference,delta);checked_coefficients+=1
    result=dict(status="passed",scenario_count=len(manifest["scenarios"]),choices_reconstructed=selections_checked,
        independent_quantile_fits=checked_coefficients,max_coefficient_difference_bp_per_sd=max_coefficient_difference,
        checks=["input/output hashes","same samples","exact horizons","past-only selection and training",
          "same origin-month function for current/future residuals","all three controls reproduce previous outputs",
          "saved primary RMSE values","independent quantile solver agreement"])
    (HERE/"VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
