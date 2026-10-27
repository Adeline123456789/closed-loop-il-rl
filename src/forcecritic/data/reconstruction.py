"""Validate a cold-start recipe for the historical batched FORGE collector.

An arbitrary episode/seed is not a simulator snapshot. The first synchronous
batch can be reconstructed only with its original width, initialization and
expert RNG consumption. Later batches require the preceding full run prefix.
"""
import json
from pathlib import Path


def first_collection_batch(manifest_path, *, requested_steps, max_steps=149):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'COLLECTED' or manifest.get('source') != 'same_version_recollection':
        raise ValueError('A completed original collection-run manifest is required')
    contract = manifest.get('contract', {})
    if contract.get('action_stage') != 'submitted_pre_ema' or contract.get('action_axes') != ['x','y','z','yaw']:
        raise ValueError('Recorded action contract does not match the native replay adapter')
    width = manifest.get('accounting', {}).get('num_envs')
    if width not in (1, 8):
        raise ValueError('Missing or unsupported original collection batch width')
    seed = manifest.get('seed')
    split = manifest.get('split')
    if not isinstance(seed, int) or split != 'train':
        raise ValueError('Engineering reconstruction is restricted to a seeded training run')
    rows = manifest.get('episodes', [])[:width]
    if len(rows) != width:
        raise ValueError('The complete first synchronous batch is required')
    for index, row in enumerate(rows):
        if row.get('parent_id') != f'{split}-{seed}-{index:06d}' or row.get('split') != split:
            raise ValueError('Episode order/identity changed: later or shuffled episodes cannot start a cold run')
        if row.get('source') != 'real_forge_recording':
            raise ValueError('Unverified recording source')
        if row.get('collection_run_id') != manifest_path.parent.name:
            raise ValueError('Mixed collection runs')
    if not 1 <= requested_steps <= max_steps:
        raise ValueError(f'Choose 1..{max_steps} recorded control steps')
    return manifest, rows, width
