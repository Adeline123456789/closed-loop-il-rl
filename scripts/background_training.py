"""User-authorized managed continuation; publish measured evidence at stage boundaries."""
import csv
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT / 'src')
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['GIT_TERMINAL_PROMPT'] = '0'
sys.path.insert(0, str(ROOT / 'src'))
from forcecritic.runtime import atomic_json, digest

RUN = '20261023T104759Z'
REPORT = ROOT / 'reports' / RUN
DATA = ROOT / 'data/act_demonstrations'
PYTHON = str(ROOT / '.venv/bin/python')
STATE = REPORT / 'background_job.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def state(status, stage, **extra):
    atomic_json(STATE, dict(status=status, stage=stage, time_utc=now(), pid=os.getpid(),
                           session='forcecritic-training', log='.cache/background_training.log', **extra))
    print(now(), status, stage, extra, flush=True)


def command(args, timeout=None):
    print('COMMAND', ' '.join(args), flush=True)
    try:
        return subprocess.run(args, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print('TIMEOUT', args[0], flush=True)
        return 124


def snapshot():
    dest = REPORT / 'publication_snapshot'
    dest.mkdir(exist_ok=True)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    rows = manifest['episodes']
    runs = []
    csv_rows = []
    for path in sorted(DATA.glob('*/manifest.json')):
        record = json.loads(path.read_text())
        shutil.copyfile(path, dest / (path.parent.name + '_manifest.json'))
        runs.append(dict(collection_run=path.parent.name, status=record['status'], manifest_sha256=digest(path)))
        with (path.parent / 'expert_smoke.csv').open() as source:
            csv_rows.extend(dict(collection_run=path.parent.name, **r) for r in csv.DictReader(source))
    for filename in ('manifest.json', 'collection_plan.json'):
        shutil.copyfile(DATA / filename, dest / filename)
    with (ROOT / 'results' / RUN / 'recollection_episodes.csv').open('w') as target:
        writer = csv.DictWriter(target, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    progress = ROOT / 'checkpoints' / RUN / 'act0.json'
    updates = json.loads(progress.read_text()).get('completed_updates', 0) if progress.exists() else 0
    stages = json.loads((REPORT / 'run_manifest.json').read_text())['stages']
    info = dict(snapshot_time_utc=now(), completed_collection_runs=len(runs), planned_collection_runs=20,
                completed_episodes=len(rows), completed_stable_successes=sum(r['successful'] for r in rows),
                splits={s:dict(episodes=sum(r['split']==s for r in rows),
                               successful=sum(r['split']==s and r['successful'] for r in rows))
                        for s in ('train','validation','test')},
                actor_pretraining_updates=updates, collection_runs=runs,
                stages={k: v['status'] for k,v in stages.items()},
                raw_dataset_location=str(DATA), background_job=json.loads(STATE.read_text()),
                note='Measured snapshot; expert success is not ACT performance. Raw arrays and weights remain local.')
    atomic_json(dest / 'status.json', info)
    text = ('# Current measured publication snapshot\n\n'
            f'UTC: {info["snapshot_time_utc"]}. Collection: {len(runs)}/20 runs, '
            f'{len(rows)} episodes, {info["completed_stable_successes"]} expert successes.\n\n'
            f'ACT optimizer updates saved: {updates}. This does not establish ACT insertion success.\n\n'
            '| Stage | Recorded status |\n|---|---|\n' +
            ''.join(f'| {k} | {v} |\n' for k,v in info['stages'].items()) +
            '\nSee background_job.json for process status and FINAL_REPORT.md for measured results.\n')
    (REPORT / 'PUBLICATION_STATUS.md').write_text(text)
    return text


def publish(refresh_report=True):
    if refresh_report:
        command([PYTHON, '-m', 'forcecritic.cli', 'report', '--run-id', RUN])
    body = snapshot()
    rc = command([PYTHON, 'scripts/publish.py'])
    if rc:
        print('Publication retry; training gates remain unchanged.', flush=True)
        rc = command([PYTHON, 'scripts/publish.py'])
    if rc == 0:
        # A detached job cannot call the interactive connector; use authenticated gh.
        body_path = ROOT / '.cache/background_pr_body.md'
        body_path.write_text('Real FORGE integration and measured evidence.\n\n' + body +
            '\nBackground continuation: ACT baseline → real paired diagnostics → B1/B2/B3 short ablation. '
            'Each step requires accepted predecessor evidence; any failed gate stops the job. '
            'Pilot/confirmation/generalization are not automatically run.\n\n'
            'Data schema and monitoring: DATA_FORMAT.md and TRAINING_GUIDE.md. '
            'Raw datasets, replay, simulator caches and model weights are local.\n')
        # REST avoids older gh pr edit querying the removed classic Projects API.
        request_path = ROOT / '.cache/background_pr_request.json'
        request_path.write_text(json.dumps({'body': body_path.read_text()}))
        rc = command(['gh', 'api', '--method', 'PATCH',
                      'repos/Adeline123456789/closed-loop-il-rl/pulls/1',
                      '--input', str(request_path), '--jq', '.html_url'], timeout=90)
    atomic_json(ROOT / '.cache/background_publication.json',
                dict(time_utc=now(), returncode=rc, head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-pid', type=int)
    parser.add_argument('--adopt-stage', choices=('baseline','diagnostic','short-ablation'))
    args = parser.parse_args()
    if bool(args.adopt_pid) != bool(args.adopt_stage):
        parser.error('--adopt-pid and --adopt-stage must be supplied together')
    lock = (ROOT / '.cache/background_training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (DATA / 'manifest.json').exists():
        raise RuntimeError('Completed demonstration manifest required; no duplicate collection launched')
    if not args.adopt_pid:
        state('RUNNING', 'initial-publication')
        publish()
    stages = [('doctor', ['doctor']),
              ('audit-data', ['audit-data', '--demonstrations', str(DATA / 'manifest.json')]),
              ('smoke-1', ['smoke', '--num-envs', '1']), ('smoke-8', ['smoke', '--num-envs', '8'])]
    stages += [(s, ['run','--stage',s,'--demonstrations',str(DATA/'manifest.json')])
               for s in ('baseline','diagnostic','short-ablation')]
    if args.adopt_pid:
        # Replace only the supervisor while its already-authorized CLI child runs.
        # No second simulator or optimizer is started, and no failed stage is retried.
        state('RUNNING', args.adopt_stage, adopted_cli_pid=args.adopt_pid)
        process = Path(f'/proc/{args.adopt_pid}/stat')
        while process.exists():
            try:
                if process.read_text().split(') ',1)[1].split()[0] == 'Z':
                    break
            except FileNotFoundError:
                break
            time.sleep(10)
        measured = json.loads((REPORT / 'run_manifest.json').read_text())['stages'][args.adopt_stage]
        passed = measured['status'] == 'PASSED'
        state('STAGE_FINISHED' if passed else 'STOPPED', args.adopt_stage,
              recorded_stage_status=measured['status'])
        publish()
        if not passed:
            return 2
        stages = stages[next(i for i,(name,_) in enumerate(stages) if name==args.adopt_stage)+1:]
    for name, args in stages:
        state('RUNNING', name)
        rc = command([PYTHON, '-m', 'forcecritic.cli', *args, '--run-id', RUN])
        state('STAGE_FINISHED' if rc == 0 else 'STOPPED', name, returncode=rc)
        if rc or name in ('smoke-8','baseline','diagnostic','short-ablation'):
            publish()
        if rc:
            return rc
    state('COMPLETED', 'short-ablation', note='Inspect the actual gate; no automatic pilot.')
    publish()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        state('FAILED', 'orchestrator', error=repr(error))
        raise
