"""Replay stabilized outputs and independently check all Ridge readouts."""
import json
import numpy as np
import pandas as pd
import verify_replay as verify
from stabilize_readout import HERE, OUT, SOURCE, codes, base


def main():
    verify.OUT=OUT
    verify.VERIFICATION_PATH=HERE/"STABILITY_VERIFICATION.json"
    verify.COMPARE_OLD_PCA=False
    verify.main()
    count=0;max_error=0.
    for scenario in base.study.pilot.SCENARIOS:
        panel,_=base.study.build_panel(scenario)
        data=panel.dropna(subset=["y","m","g"]+base.study.KCOLS)
        for month in base.FIT_MONTHS:
            train=data.loc[data.index<pd.Timestamp(month+"-01",tz="UTC")]
            for c in base.GRID:
                if not c["family"].startswith(("ae_","pca_")):continue
                name=month+"_"+c["candidate_id"]+".json"
                state=json.loads((OUT/scenario/"checkpoints"/name).read_text())
                original=json.loads((SOURCE/scenario/"checkpoints"/name).read_text())
                if c["family"].startswith("ae_"):
                    for m,old in zip(state["members"],original["members"]):
                        for key in ["weights","biases","center","scale"]:assert m[key]==old[key]
                    pairs=[(codes(state,train,m),np.array(m["first"])) for m in state["members"]]
                else:
                    rank=int(c["family"][-1])
                    x=(np.ascontiguousarray(train[base.study.KCOLS].to_numpy(),dtype=float)-np.array(state["mean"]))/np.array(state["sd"])
                    pc=x@np.array(state["loading"])[:,:rank]
                    center,scale=pc.mean(axis=0),pc.std(axis=0,ddof=1)
                    raw=np.array(state["first"])
                    pairs=[((pc-center)/scale,np.r_[raw[0]+raw[1:]@center,raw[1:]*scale])]
                for z,coef in pairs:
                    centered=z-z.mean(axis=0)
                    slope=np.linalg.solve(centered.T@centered+np.eye(z.shape[1]),centered.T@(train.y.to_numpy()-train.y.mean()))
                    intercept=float(train.y.mean()-z.mean(axis=0)@slope)
                    err=float(np.max(np.abs(np.r_[intercept,slope]-coef)))
                    assert err<1e-7,(scenario,month,c,err)
                    max_error=max(max_error,err);count+=1
    path=HERE/"STABILITY_VERIFICATION.json"
    result=json.loads(path.read_text())
    result.update(unchanged_encoder_weights_verified=True,independent_ridge_readouts=count,max_ridge_coef_difference=max_error)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
