"""Evidence-first CLI. Missing prerequisite evidence is never interpreted as success."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import time
import math
from .runtime import Run,atomic_json,lock_protocol,command,digest
from .evaluation.report import report,initialize_csvs


def prerequisites(run,stage):
    order=['doctor','audit-data','smoke-1','smoke-8','baseline','diagnostic','short-ablation','pilot','confirm','generalization']
    missing=[s for s in order[:order.index(stage)] if not run.verified(s)]
    return missing


def run_stage(run,stage,require_go=None):
    if run.verified(stage):return 0
    missing=prerequisites(run,stage)
    if missing:
        reason='Preconditions not passed: '+', '.join(missing)
        run.record(stage,'BLOCKED',reason,2)
        return 2
    if stage in ('pilot','confirm','generalization'):
        if not require_go:raise ValueError('versioned GO decision required')
        decision=json.loads(Path(require_go).read_text())
        predecessor={'pilot':'short-ablation','confirm':'pilot','generalization':'confirm'}[stage]
        if (decision.get('status')!='GO_PILOT' or decision.get('config_hash')!=run.config_hash
                or decision.get('from_stage')!=predecessor or decision.get('next_stage')!=stage
                or not decision.get('evidence_hashes')):
            raise ValueError('GO decision absent, stale, wrong stage or unsupported by evidence')
        for p,h in decision['evidence_hashes'].items():
            if digest(p)!=h:raise ValueError('GO evidence changed')
    # All robot stages require a measured P0 acceptance contract, not a manually flipped status.
    acceptance=run.report/'p0_acceptance.json'
    if not acceptance.exists():
        run.record(stage,'BLOCKED','Measured P0 acceptance contract absent',2);return 2
    # This branch is unreachable in the recorded blocked run. Do not imply a completed
    # robot driver merely because the command parses.
    raise NotImplementedError('Robot stage orchestration awaits measured P0 contract; see implementation_audit.md')


def execute(run,args):
    if args.command=='doctor':
        from .doctor import doctor
        d=doctor(run);rc=0 if d['status']=='PASSED' else 2
        run.record('doctor',d['status'],'Missing runtime: '+','.join(d['missing_packages'])+'; '+'; '.join(d['compatibility_issues']),rc,[run.report/'doctor.json'])
        return rc
    if args.command=='audit-data':
        from .data.audit import audit
        d=audit(run.config,run.report);rc=0 if d['status']=='PASSED' else 2
        run.record('audit-data',d['status'],'; '.join(d['blocking_items']),rc,[run.report/'data_audit.json'])
        return rc
    if args.command=='smoke':
        stage=f'smoke-{args.num_envs}'
        if run.verified(stage):return 0
        if args.num_envs==8 and not run.verified('smoke-1'):
            run.record(stage,'BLOCKED','1-env smoke not passed',2);return 2
        attempt=f'{stage}_{time.time_ns()}'
        output=run.report/f'{attempt}.json'
        cmd=[sys.executable,'scripts/smoke_forge.py','--headless','--enable_cameras','--num_envs',str(args.num_envs),'--output',str(output)]
        result=command(cmd,timeout=1800)
        atomic_json(run.report/f'{attempt}_command.json',result)
        ok=result.get('returncode')==0 and output.exists()
        if output.exists():
            measurement=json.loads(output.read_text())
            ok=ok and measurement.get('status')=='PASSED'
            counts=measurement.get('physics_substeps',{})
            decimation=measurement.get('decimation',8)
            run.manifest['steps']['evaluation']+=math.ceil(counts.get('evaluation',0)/decimation)
            run.manifest['steps']['restore']+=math.ceil(counts.get('initialization_restore',0)/decimation)
        failure_text=result.get('stderr','')+result.get('stdout','')+result.get('error','')
        environment_failure=any(x in failure_text for x in ["No module named 'isaaclab'","No module named 'isaacsim'",'EULA','LICENSE','ConnectionError','timed out'])
        status='PASSED' if ok else ('BLOCKED' if environment_failure else 'FAILED_IMPLEMENTATION')
        run.record(stage,status,'Real launcher attempted; see command log',0 if ok else 2,[run.report/f'{attempt}_command.json',output,Path('environment.lock.json')])
        return 0 if ok else 2
    if args.command=='run':return run_stage(run,args.stage,args.require_go)
    if args.command=='evaluate':
        if not args.no_training:raise ValueError('generalization requires --no-training')
        return run_stage(run,'generalization',args.require_go)
    if args.command=='report':print(report(run));return 0
    raise ValueError(args.command)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['doctor','audit-data','smoke','run','evaluate','report'])
    p.add_argument('--config',default='configs/minimal_validation.yaml')
    p.add_argument('--run-id');p.add_argument('--run-dir');p.add_argument('--require-go')
    p.add_argument('--stage',choices=['baseline','diagnostic','short-ablation','pilot','confirm'])
    p.add_argument('--num-envs',type=int,choices=[1,8],default=1)
    p.add_argument('--suite',choices=['generalization']);p.add_argument('--no-training',action='store_true')
    args=p.parse_args()
    if args.command=='run' and not args.stage:p.error('--stage is required')
    run_id=args.run_id or (Path(args.run_dir).name if args.run_dir else os.environ.get('FORCECRITIC_RUN_ID'))
    if not run_id and Path('reports/PROJECT_STATUS.json').exists():
        run_id=json.loads(Path('reports/PROJECT_STATUS.json').read_text()).get('run_id')
    Path('.cache').mkdir(exist_ok=True)
    lock_handle=open('.cache/run.lock','w')
    fcntl.flock(lock_handle,fcntl.LOCK_EX)
    lock_protocol(args.config);run=Run(args.config,run_id);initialize_csvs(run)
    try:return execute(run,args)
    except Exception as e:
        stage=args.stage or ('generalization' if args.command=='evaluate' else args.command)
        run.record(stage,'FAILED_IMPLEMENTATION',str(e),1)
        (run.report/f'{stage}_exception.log').write_text(traceback.format_exc())
        traceback.print_exc();return 1


if __name__=='__main__':sys.exit(main())
