"""Retention must survive the live trainer's atomic latest-file replacement."""
import os
from pathlib import Path
import runpy

import torch

retain = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/retain_act_checkpoints.py'))['retain']


def test_retained_checkpoint_survives_latest_replacement(tmp_path):
    latest = tmp_path / 'act0.pt'
    torch.save({'update': 1000, 'actor': {'weight': torch.tensor([1.])},
                'optimizer': {'step': 1000}, 'torch_rng': torch.get_rng_state()}, latest)
    record = retain(latest)
    replacement = tmp_path / 'replacement.pt'
    torch.save({'update': 1100, 'actor': {'weight': torch.tensor([2.])}}, replacement)
    os.replace(replacement, latest)
    kept = torch.load(record['path'], weights_only=False)
    assert kept['update'] == 1000
    assert kept['actor']['weight'].item() == 1
    assert kept['optimizer']['step'] == 1000
    assert torch.equal(kept['torch_rng'], torch.get_rng_state())
    assert torch.load(latest, weights_only=False)['update'] == 1100


def test_names_come_from_payload_and_nonmilestones_are_not_mislabeled(tmp_path):
    latest = tmp_path / 'act0.pt'
    torch.save({'update': 6800}, latest)
    latest.with_suffix('.json').write_text('{"completed_updates": 6000}')
    assert retain(latest) is None
    record = retain(latest, initial=True)
    assert record['update'] == 6800
    assert Path(record['path']).name == 'act0_update_006800.pt'
    assert not (tmp_path / 'act0_milestones/act0_update_006000.pt').exists()
