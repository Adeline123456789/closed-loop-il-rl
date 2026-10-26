"""Record causal demonstrations from the pinned expert in the actual stock task."""
import argparse
import csv
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--acceptance')
parser.add_argument('--episodes', type=int, default=8)
parser.add_argument('--attempt-cap', type=int, default=32)
parser.add_argument('--transition-cap', type=int, default=50000)
parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--num_envs', type=int, choices=[1, 8], default=1)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless or not args.enable_cameras:
    parser.error('--headless --enable_cameras required')
if not 1 <= args.episodes <= args.attempt_cap:
    parser.error('positive successful episode target <= attempt cap required')
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
if (out / 'manifest.json').exists():
    raise FileExistsError('Use a fresh collection run directory; never overwrite earlier episodes')
app = AppLauncher(args).app
from forcecritic.runtime import atomic_json, digest
manifest = {'status': 'RUNNING', 'source': 'same_version_recollection', 'episodes': [],
            'seed': args.seed, 'split': args.split, 'successful_episodes': 0, 'individual_transitions': 0}
atomic_json(out / 'manifest.json', manifest)
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from forcecritic.envs.forge import ForgeAdapter, configure_camera, TASK, CAMERA
    from forcecritic.envs.expert import Expert, EXPERT_SHA256
    from forcecritic.envs.accounting import PhysicsLedger
    from forcecritic.data.demonstrations import FIELDS
    RECORD_FIELDS = FIELDS + ('raw_wrench','substep_wrench','ema_action','target_position',
                              'target_quaternion','observed_after_state','terminated','truncated','stable_success')
    cfg = configure_camera(parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs))
    cfg.seed = args.seed
    ledger = PhysicsLedger(args.num_envs, cfg.decimation, args.transition_cap)
    env = gym.make(TASK, cfg=cfg)
    adapter = ForgeAdapter(env, physics_ledger=ledger)
    player = Expert(env, args.checkpoint, args.seed)
    adapter.reset(seed=args.seed)
    contract = dict(task=TASK, action_stage='submitted_pre_ema', action_axes=['x','y','z','yaw'],
        observation_timing='before_action', wrench_dim=6, history_steps=4, chunk_length=4,
        wrench_frame='incoming_joint', camera=CAMERA, control_dt=adapter.raw.step_dt,
        expert_sha256=EXPERT_SHA256, expert_seventh_action='replaced_by_fixed_minus_one')
    if args.acceptance:
        shutil.copyfile(args.acceptance, out / 'acceptance.json')
        contract.update(acceptance_evidence='acceptance.json', acceptance_sha256=digest(out / 'acceptance.json'))
    manifest['contract'] = contract
    buffers = [{k: [] for k in RECORD_FIELDS} for _ in range(args.num_envs)]
    stable = np.zeros(args.num_envs, dtype=bool)
    attempts = 0
    metrics_path = out / 'expert_smoke.csv'
    with metrics_path.open('w', newline='') as f:
        fields = ['parent_id','seed','split','stable_success','ever_success','final_success','steps','source']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while (manifest['successful_episodes'] < args.episodes and attempts + args.num_envs <= args.attempt_cap
               and manifest['individual_transitions'] + args.num_envs <= args.transition_cap):
            before = adapter.current_frame
            actions = player.action(adapter.expert_observation)
            native = actions[:, [0, 1, 2, 5]].detach().cpu().numpy()
            for i in range(args.num_envs):
                for k in FIELDS:
                    buffers[i][k].append(native[i].copy() if k == 'action' else before[k][i].copy())
            with ledger.using('interaction'):
                row = adapter.step(native)
            for i in range(args.num_envs):
                extra = dict(raw_wrench=row['raw_wrench'][i],substep_wrench=row['substep_wrench'][:,i],
                    ema_action=row['ema_action'][i],target_position=row['targets'][-1]['ctrl_target_fingertip_midpoint_pos'][i],
                    target_quaternion=row['targets'][-1]['ctrl_target_fingertip_midpoint_quat'][i],
                    observed_after_state=row['frame']['state'][i],terminated=row['terminated'][i],
                    truncated=row['truncated'][i],stable_success=row['success'][i])
                for k,value in extra.items():buffers[i][k].append(np.asarray(value).copy())
            manifest['individual_transitions'] += args.num_envs
            stable |= row['success']
            done = row['terminated'] | row['truncated']
            player.reset_done(torch.as_tensor(done, device=adapter.raw.device))
            for i in np.flatnonzero(done):
                parent = f'{args.split}-{args.seed}-{attempts:06d}'
                length = len(buffers[i]['action'])
                successful = bool(stable[i])
                entry = dict(parent_id=parent, collection_run_id=out.name, split=args.split,
                    successful=successful, source='real_forge_recording', files={})
                directory = out / parent
                directory.mkdir()
                for k in RECORD_FIELDS:
                    path = directory / f'{k}.npy'
                    np.save(path, np.stack(buffers[i][k]), allow_pickle=False)
                    entry['files'][k] = dict(path=str(path.relative_to(out)), sha256=digest(path))
                manifest['episodes'].append(entry)
                writer.writerow(dict(parent_id=parent, seed=args.seed, split=args.split,
                    stable_success=int(successful), ever_success=int(row['ever_success'][i]),
                    final_success=int(row['geometric_success'][i]), steps=length, source='real_forge'))
                f.flush()
                manifest['successful_episodes'] += int(successful)
                attempts += 1
                stable[i] = False
                buffers[i] = {k: [] for k in RECORD_FIELDS}
            atomic_json(out / 'manifest.json', manifest)
        manifest['accounting'] = ledger.record()
        manifest.update(status='COLLECTED' if manifest['successful_episodes'] >= args.episodes else 'BUDGET_EXHAUSTED',
            attempts=attempts, partial_episode_lengths=[len(x['action']) for x in buffers],
            note='A collected manifest is not P0 acceptance or an ACT baseline result.')
        atomic_json(out / 'manifest.json', manifest)
    adapter.close()
except BaseException as error:
    import traceback
    traceback.print_exc()
    manifest.update(status='BUDGET_EXHAUSTED' if 'BUDGET_EXHAUSTED' in str(error) else 'FAILED', error=repr(error))
    if 'ledger' in globals():
        manifest['accounting'] = ledger.record()
    atomic_json(out / 'manifest.json', manifest)
    raise
finally:
    app.close()
