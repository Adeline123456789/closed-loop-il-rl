"""Read-only runtime audit. Package availability is not a camera/asset smoke test."""
import importlib.metadata as md
from pathlib import Path
import platform
import shutil
import sys
from .runtime import atomic_json, command


def doctor(run):
    packages={}
    for name in ('torch','torchvision','isaacsim','isaacsim-kernel','isaaclab','numpy','pyarrow'):
        try: packages[name]=md.version(name)
        except md.PackageNotFoundError: packages[name]=None
    gpu=command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'])
    import torch
    cuda={'torch_cuda':torch.version.cuda,'available':torch.cuda.is_available()}
    missing=[name for name in ('isaacsim','isaaclab') if not packages[name]]
    compatibility=[]
    if sys.version_info[:2]!=(3,11):compatibility.append('Isaac candidate requires Python 3.11; this interpreter is CPU verification only')
    if packages['isaacsim'] and not packages['isaacsim'].startswith('5.0.0'):compatibility.append('Isaac Sim version differs from 5.0.0')
    lab=command(['git','-C','.cache/IsaacLab','rev-parse','HEAD'])
    if lab.get('stdout','').strip()!='0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20':compatibility.append('Pinned Isaac Lab checkout missing/mismatched')
    data={'status':'BLOCKED' if missing or compatibility else 'PASSED','missing_packages':missing,
          'compatibility_issues':compatibility,'isaaclab_checkout':lab,
          'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
          'gpu':gpu,'cuda':cuda,'packages':packages,'disk_free_bytes':shutil.disk_usage('.').free,
          'asset_loading':'NOT_RUN','camera_rendering':'NOT_RUN',
          'git_read':command(['git','ls-remote','origin','HEAD']),
          'git_push_dry_run':command(['git','push','--dry-run','origin','HEAD']),
          'isaac_launcher':str(Path('.cache/IsaacLab/isaaclab.sh').absolute()),
          'license_acceptance':'NOT_VERIFIED',
          'note':'doctor PASSED only means import prerequisites found; smoke and interface audits remain mandatory'}
    atomic_json(run.report/'doctor.json',data)
    atomic_json('environment.lock.json',data)
    return data
