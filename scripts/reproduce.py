"""Run archived stages in independent scratch copies, preserving the evidence."""
from pathlib import Path
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from verify_archive import ROOT,verify

TICK='option1_tick_learning_20260910'
FOLLOW='option1_model_followup_20260910'
LATEST='option1_factorial_20260910'


def steps(directory,scripts,runtime='analysis'):
    return [(directory,script,[],runtime) for script in scripts]


STAGES={
    'reanalysis-check':[('reanalysis_20260909','recalculate.py',['check'],'analysis')],
    'reanalysis-points':[('reanalysis_20260909','recalculate.py',['points'],'analysis')],
    'pilot':steps('pilot_20260909',['run_pilot.py']),
    'extension':steps('extension_20260909',['timing_checks.py','measurement_diagnostics.py','run_extension.py',
        'analyze_extension.py','supplementary_analysis.py','grid_sensitivity.py']),
    'development':steps('option1_development_20260909',['screen_measurement.py','plot_screening.py']),
    'core':steps('option1_core_20260909',['prepare_data.py','policy.py','economics.py','supplement.py','verify.py','figures.py']),
    'core-checks':steps('option1_core_20260909',['verify.py']),
    'measurement-checks':steps('option1_resolution_20260910',['verify.py']),
    'resolution':steps('option1_resolution_20260910',['measurement.py','policy_robustness.py','prediction_evaluation.py',
        'supplement.py','risk_report.py','verify.py','figures.py']),
    'prepare-data':steps(TICK,['prepare.py']),
    'tick-baselines':steps(TICK,['baselines.py']),
    'tick-learning':steps(TICK,['train.py'],'neural')+steps(TICK,['analyze.py'])+
        steps(TICK,['verify_checkpoints.py'],'neural')+steps(TICK,['verify.py','diagnostics.py','figures.py']),
    'model-followup':steps(FOLLOW,['preflight.py','train.py'],'neural')+steps(FOLLOW,['analyze.py'])+
        steps(FOLLOW,['verify_checkpoints.py'],'neural')+steps(FOLLOW,['verify.py','decompose.py','figures.py']),
    'latest-analysis':steps(LATEST,['analyze.py','verify.py','figures.py']),
    'latest-checkpoints':steps(LATEST,['verify_checkpoints.py'],'neural')+steps(LATEST,['verify.py']),
    'latest-full':steps(LATEST,['preflight.py','train.py'],'neural')+steps(LATEST,['analyze.py'])+
        steps(LATEST,['verify_checkpoints.py'],'neural')+steps(LATEST,['verify.py','figures.py','training_diagnostics.py']),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['list']+list(STAGES))
    parser.add_argument('--neural-image',default='stablecoin-neural:2.9.1')
    args=parser.parse_args()
    if args.stage=='list':
        for name,items in STAGES.items():print(name+': '+', '.join(script for _,script,_,_ in items))
        return
    verify()
    if any(runtime=='neural' for *_,runtime in STAGES[args.stage]):
        check=subprocess.run(['docker','image','inspect',args.neural_image],stdout=subprocess.DEVNULL)
        if check.returncode:raise SystemExit('Build environments/Dockerfile.neural first; see docs/REPRODUCING.md')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=ROOT/'.runs'/(args.stage+'-'+stamp);run.mkdir(parents=True)
    work=run/'research';shutil.copytree(ROOT/'research',work)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MPLCONFIGDIR=str(run/'matplotlib'))
    completed=[]
    for directory,script,arguments,runtime in STAGES[args.stage]:
        if runtime=='analysis':cmd=[sys.executable,script]+arguments;cwd=work/directory
        else:
            cmd=['docker','run','--rm','--pull=never','--network','none','--cpus','2','--memory','4g',
                '--user',f'{os.getuid()}:{os.getgid()}',
                '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1',
                '-v',str(work)+':/research:ro','-v',str(work/directory)+':/work','-w','/work',
                '--entrypoint','python',args.neural_image,'/work/'+script]+arguments
            cwd=ROOT
        print('Running '+directory+'/'+script,flush=True)
        log=run/(directory+'-'+script+'.log')
        with log.open('w') as output:
            result=subprocess.run(cmd,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT)
        completed.append(dict(directory=directory,script=script,runtime=runtime,returncode=result.returncode,log=log.name))
        if result.returncode:
            print(log.read_text()[-6000:]);raise SystemExit('Failed; inspect '+str(log))
    if args.stage=='prepare-data':
        import numpy as np
        with np.load(ROOT/'research'/TICK/'data.npz') as expected,np.load(work/TICK/'data.npz') as actual:
            assert set(expected.files)==set(actual.files)
            for key in expected.files:np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
        completed.append(dict(check='All raw-derived prepared arrays exactly equal the archived arrays',status='passed'))
    verify()
    (run/'RUN.json').write_text(json.dumps(dict(stage=args.stage,status='passed',steps=completed,
        archive_unchanged=True,python=sys.version,neural_image=args.neural_image),indent=2)+'\n')
    print(json.dumps(dict(status='passed',stage=args.stage,run=str(run),archive_unchanged=True),indent=2),flush=True)


if __name__=='__main__':main()
