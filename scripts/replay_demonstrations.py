"""Reconstruct an original collection batch in a fresh real FORGE process."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
parser.add_argument('--steps', type=int, default=32)
parser.add_argument('--manifest', help='Original training collection-run manifest; defaults to the registered witness')
parser.add_argument('--omit-expert-rng', action='store_true', help='Negative control only: omit the original expert call')
parser.add_argument('--control-step-cap', type=int)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
plan_path = Path('configs/replay_dataset_reconstruction_v1.json')
plan = json.loads(plan_path.read_text())
cap = args.control_step_cap or plan['per_process_control_step_cap']
if not 1 <= cap <= plan['per_process_control_step_cap']:
    raise ValueError('Process cap exceeds the registered engineering budget')
from forcecritic.data.reconstruction import first_collection_batch
manifest_path = Path(args.manifest or plan['manifest'])
manifest, rows, width = first_collection_batch(manifest_path, requested_steps=args.steps)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from forcecritic.envs.forge import ForgeAdapter, configure_camera, TASK
from forcecritic.envs.expert import Expert
from forcecritic.envs.accounting import PhysicsLedger
from forcecritic.diagnostics.progress import DiagnosticProgress
from forcecritic.diagnostics.artifacts import atomic_json
from forcecritic.runtime import digest, source_hashes

result = dict(status='RUNNING', protocol=plan['protocol'], plan_sha256=digest(plan_path),
              manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
              seed=manifest['seed'], parent_ids=[r['parent_id'] for r in rows],
              num_envs=width, requested_steps=args.steps, steps=0,
              expert_rng_reproduced=not args.omit_expert_rng,
              source_hashes=source_hashes(), per_step=[], max_abs_errors={})
ledger = None
observer = None

def save():
    if ledger is not None: result['accounting'] = ledger.record()
    atomic_json(out/'result.json', result)

try:
    save()
    arrays = []
    for row in rows:
        fields = {}
        for key, entry in row['files'].items():
            path = manifest_path.parent/entry['path']
            if digest(path) != entry['sha256']:
                raise ValueError('Recording hash mismatch: '+str(path))
            fields[key] = np.load(path, mmap_mode='r', allow_pickle=False)
        arrays.append(fields)
    cfg = configure_camera(parse_env_cfg(TASK, device=args.device, num_envs=width))
    cfg.seed = manifest['seed']
    ledger = PhysicsLedger(width, cfg.decimation, cap)
    raw_dir = Path('checkpoints/replay_root_cause_20261027')/out.name
    observer = DiagnosticProgress(ledger, out, raw_dir)
    env = gym.make(TASK, cfg=cfg)
    adapter = ForgeAdapter(env, physics_ledger=ledger)
    expert_record = json.loads(Path(plan['expert_evidence']).read_text())
    player = Expert(env, expert_record['path'], manifest['seed'])
    adapter.reset(seed=manifest['seed'])
    for t in range(args.steps):
        before = adapter.current_frame
        rng_before = torch.cuda.get_rng_state(adapter.raw.device)
        predicted = None if args.omit_expert_rng else player.action(adapter.expert_observation)[:, [0,1,2,5]].cpu().numpy()
        if t == 0:
            result['expert_call_changes_cuda_rng'] = not torch.equal(rng_before, torch.cuda.get_rng_state(adapter.raw.device))
        native = np.stack([a['action'][t] for a in arrays])
        actual = {k:before[k].copy() for k in ('rgb','state','wrench','motion','previous_command')}
        if predicted is not None: actual['regenerated_expert_action'] = predicted
        with ledger.using('interaction'):
            transition = adapter.step(native)
        actual.update(ema_action=transition['ema_action'], raw_wrench=transition['raw_wrench'],
                      substep_wrench=transition['substep_wrench'].transpose(1,0,2),
                      target_position=transition['targets'][-1]['ctrl_target_fingertip_midpoint_pos'],
                      target_quaternion=transition['targets'][-1]['ctrl_target_fingertip_midpoint_quat'],
                      observed_after_state=transition['frame']['state'],
                      terminated=transition['terminated'], truncated=transition['truncated'],
                      stable_success=transition['success'])
        errors = {}
        for key, value in actual.items():
            source_key = 'action' if key == 'regenerated_expert_action' else key
            expected = np.stack([a[source_key][t] for a in arrays])
            errors[key] = float(np.max(np.abs(value.astype(float)-expected.astype(float))))
            result['max_abs_errors'][key] = max(result['max_abs_errors'].get(key,0.),errors[key])
        result['per_step'].append(dict(step=t, max_abs_errors=errors))
        result['steps'] = t+1
        done = transition['terminated'] | transition['truncated']
        if not args.omit_expert_rng:
            player.reset_done(torch.as_tensor(done,device=adapter.raw.device))
        if t == 0 or (t+1)%16 == 0 or t == args.steps-1:
            save()
            print('REPLAY', t+1, json.dumps(result['max_abs_errors']), flush=True)
    result['status'] = 'MEASURED'
    result['all_fields_bitwise_equal'] = not any(result['max_abs_errors'].values())
    result['all_nonvisual_fields_bitwise_equal'] = all(v==0 for k,v in result['max_abs_errors'].items() if k!='rgb')
except BaseException as error:
    result.update(status='FAILED_IMPLEMENTATION',error=repr(error))
    raise
finally:
    save()
    if observer is not None: observer.close()
    if 'adapter' in globals(): adapter.close()
    if ledger is not None: ledger.close()
    app.close()
