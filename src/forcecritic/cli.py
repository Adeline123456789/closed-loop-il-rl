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
from .runtime import Run,atomic_json,lock_protocol,command,digest,source_hashes
from .evaluation.report import report,initialize_csvs


def prerequisites(run,stage):
    order=['doctor','audit-data','smoke-1','smoke-8','baseline','diagnostic','short-ablation','pilot','confirm','generalization']
    missing=[s for s in order[:order.index(stage)] if not run.verified(s)]
    return missing


def run_stage(run,stage,require_go=None,demonstrations=None):
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
    if stage in ('baseline','diagnostic','short-ablation'):
        if not demonstrations:raise ValueError('--demonstrations accepted manifest required')
        launcher=json.loads(Path('configs/isaac_launcher.json').read_text())
        attempt=f'{stage}_{time.time_ns()}'
        outcome=run.report/f'{attempt}_outcome.json'
        cmd=[sys.executable,'-u','scripts/train_forge.py','--stage',stage,'--config',str(run.config_path),
             '--run-id',run.run_id,'--demonstrations',demonstrations,'--headless','--enable_cameras',
             '--outcome',str(outcome),
             '--kit_args='+launcher['kit_args']]
        if stage=='short-ablation':
            cmd += ['--paired-bundle',str(Path('checkpoints')/run.run_id/'paired_confirmed.pt')]
        log_path=run.report/f'{attempt}.log'
        executed_sources=source_hashes()
        result=command(cmd,timeout=86400,log_path=log_path)
        result['execution_source_hashes']=executed_sources
        atomic_json(log_path.with_suffix('.command.json'),result)
        measured=json.loads(outcome.read_text()) if outcome.exists() else {'status':'FAILED_IMPLEMENTATION'}
        status=measured['status'] if result.get('returncode')==0 else 'FAILED_IMPLEMENTATION'
        accounting=outcome.with_suffix('.accounting.json')
        artifacts=[outcome,accounting,log_path,log_path.with_suffix('.command.json'),Path(demonstrations)]
        reports_by_stage={'baseline':['baseline_validation.json','baseline_training.json'],
            'diagnostic':['d0_initial.json','paired_screening.json','paired_confirmation.json','d2_transfer.json','protocol_effect_thresholds.json'],
            'short-ablation':['short_sampling.json','short_ablation_outcomes.json']}
        artifacts += [run.report/name for name in reports_by_stage[stage]]
        csv_patterns={'baseline':['act0_validation.csv'],'diagnostic':['paired_*.csv','d2_*_normal_validation.csv'],
                      'short-ablation':['short_*_validation.csv']}
        for pattern in csv_patterns[stage]:artifacts.extend(run.result.glob(pattern))
        if accounting.exists():
            counts=json.loads(accounting.read_text())['individual_control_step_equivalents']
            for key,value in counts.items():run.manifest['steps'][key]+=value
        run.record(stage,status,measured.get('reason',measured.get('error','Robot driver failed')),
                   0 if status=='PASSED' else 2,artifacts,executed_sources=executed_sources)
        return 0 if run.verified(stage) else 2
    raise NotImplementedError('Robot stage orchestration awaits measured P0 contract; see implementation_audit.md')


def execute(run,args):
    if args.command=='doctor':
        from .doctor import doctor
        d=doctor(run);rc=0 if d['status']=='PASSED' else 2
        run.record('doctor',d['status'],'Missing runtime: '+','.join(d['missing_packages'])+'; '+'; '.join(d['compatibility_issues']),rc,[run.report/'doctor.json'])
        return rc
    if args.command=='audit-data':
        if args.demonstrations:
            from .data.demonstrations import Demonstrations
            dataset=Demonstrations(args.demonstrations,episode_limit=None)
            if any(r['source']!='real_forge_recording' for r in dataset.rows):
                raise ValueError('Recollection acceptance requires actual same-version recordings')
            d={'status':'PASSED','source':'same_version_recollection','accepted_successful_train_episodes':len(dataset.rows),
               'manifest':args.demonstrations,'manifest_sha256':dataset.manifest_hash,
               'old_hf_alignment':'UNRESOLVED; not admitted by this acceptance'}
            atomic_json(run.report/'recollection_audit.json',d)
            run.record('audit-data','PASSED','Accepted hashed same-version recordings; HF semantics remain unresolved',0,
                       [run.report/'recollection_audit.json',Path(args.demonstrations)])
            return 0
        from .data.audit import audit
        d=audit(run.config,run.report);rc=0 if d['status']=='PASSED' else 2
        run.record('audit-data',d['status'],'; '.join(d['blocking_items']),rc,[run.report/'data_audit.json'])
        return rc
    if args.command=='smoke':
        stage=f'smoke-{args.num_envs}'
        if run.verified(stage) and not args.rerun:return 0
        if args.num_envs==8 and not run.verified('smoke-1'):
            run.record(stage,'BLOCKED','1-env smoke not passed',2);return 2
        attempt=f'{stage}_{time.time_ns()}'
        output=run.report/f'{attempt}.json'
        cmd=[sys.executable,'scripts/smoke_forge.py','--headless','--enable_cameras','--num_envs',str(args.num_envs),'--output',str(output)]
        launcher=json.loads(Path('configs/isaac_launcher.json').read_text())
        cmd.append('--kit_args='+launcher['kit_args'])
        log_path=run.report/f'{attempt}.log'
        executed_sources=source_hashes()
        result=command(cmd,timeout=launcher['first_launch_timeout_seconds'],log_path=log_path)
        result['execution_source_hashes']=executed_sources
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
        reason='Real launcher attempted; see command log'
        if not ok and 'Do you accept the EULA?' in failure_text:
            reason='BLOCKED_LICENSE: Omniverse Kit requested EULA acceptance; no user acceptance was supplied and no scene was created'
        run.record(stage,status,reason,0 if ok else 2,[run.report/f'{attempt}_command.json',log_path,output,Path('environment.lock.json'),Path('configs/isaac_launcher.json')],executed_sources=executed_sources)
        return 0 if run.verified(stage) else 2
    if args.command=='run':return run_stage(run,args.stage,args.require_go,args.demonstrations)
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
    p.add_argument('--demonstrations',help='Accepted same-version demonstration manifest')
    p.add_argument('--rerun',action='store_true',help='Repeat an actual smoke instead of reusing a cached pass')
    p.add_argument('--stage',choices=['baseline','diagnostic','short-ablation','pilot','confirm'])
    p.add_argument('--num-envs',type=int,choices=[1,8],default=1)
    p.add_argument('--suite',choices=['generalization']);p.add_argument('--no-training',action='store_true')
    args=p.parse_args()
    if args.command=='run' and not args.stage:p.error('--stage is required')
    if args.rerun and args.command!='smoke':p.error('--rerun is only for smoke checks')
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
