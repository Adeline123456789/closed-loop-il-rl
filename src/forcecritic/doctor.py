"""Read-only runtime audit. Package availability is not a camera/asset smoke test."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from .runtime import atomic_json, command,digest


def license_state():
    """Read the vendor's acceptance indicators without importing Kit or accepting terms."""
    try:marker=Path(md.distribution('omniverse-kit').locate_file('omni/EULA_ACCEPTED'))
    except md.PackageNotFoundError:return {'status':'NOT_INSTALLED','accepted':False}
    environment=os.environ.get('OMNI_KIT_ACCEPT_EULA','N').lower() in ('y','yes','1')
    recorded=marker.exists() and (marker.read_text().splitlines() or [''])[0].strip().lower() in ('y','yes','1')
    return {'status':'ACCEPTED' if environment or recorded else 'NOT_ACCEPTED','accepted':bool(environment or recorded),
            'acceptance_marker_exists':marker.exists(),'environment_acceptance_flag':environment,
            'agreement_url':'https://docs.omniverse.nvidia.com/enterprise/latest/common/NVIDIA_Omniverse_License_Agreement.html'}


def doctor(run):
    packages={}
    for name in ('torch','torchvision','isaacsim','isaacsim-kernel','omniverse-kit','isaaclab','numpy','pyarrow'):
        try: packages[name]=md.version(name)
        except md.PackageNotFoundError: packages[name]=None
    gpu=command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'])
    try:
        import torch
        cuda={'torch_cuda':torch.version.cuda,'available':torch.cuda.is_available()}
    except (ImportError,OSError) as error:
        cuda={'torch_cuda':None,'available':False,'import_error':str(error)}
    missing=[name for name in ('torch','torchvision','isaacsim','isaaclab') if not packages[name]]
    compatibility=[]
    if sys.version_info[:2]!=(3,11):compatibility.append('Isaac candidate requires Python 3.11; this interpreter is CPU verification only')
    if gpu.get('returncode')!=0 or not cuda['available']:compatibility.append('CUDA GPU is unavailable to this interpreter')
    if packages['isaacsim'] and not packages['isaacsim'].startswith('5.0.0'):compatibility.append('Isaac Sim version differs from 5.0.0')
    if packages['torch'] and packages['torch'].split('+')[0]!='2.7.0':compatibility.append('Torch differs from audited 2.7.0 candidate')
    if cuda['torch_cuda'] not in (None,'12.8'):compatibility.append('Torch CUDA build differs from audited cu128 candidate')
    lab=command(['git','-C','.cache/IsaacLab','rev-parse','HEAD'])
    if lab.get('stdout','').strip()!='0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20':compatibility.append('Pinned Isaac Lab checkout missing/mismatched')
    sources=json.loads(Path('upstream.lock.json').read_text())
    asset_path=run.report/'asset_access.json'
    data={'status':'BLOCKED' if missing or compatibility else 'PASSED','missing_packages':missing,
          'code_sha':command(['git','rev-parse','HEAD']).get('stdout','').strip(),
          'config_hash':run.config_hash,'upstream_lock_sha256':digest('upstream.lock.json'),
          'hf_sources':sources.get('HF',{}),
          'asset_endpoint_checks':json.loads(asset_path.read_text()) if asset_path.exists() else [],
          'asset_runtime_version':'NOT_RUN; endpoint probes are not loaded asset versions',
          'compatibility_issues':compatibility,'isaaclab_checkout':lab,
          'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
          'gpu':gpu,'cuda':cuda,'packages':packages,'disk_free_bytes':shutil.disk_usage('.').free,
          'asset_loading':'NOT_RUN','camera_rendering':'NOT_RUN',
          'git_read':command(['git','ls-remote','origin','HEAD']),
          'git_push_dry_run':command(['git','push','--dry-run','origin','HEAD']),
          'isaac_launcher':str(Path('.cache/IsaacLab/isaaclab.sh').absolute()),
          'license_acceptance':license_state(),
          'note':'doctor PASSED only means import prerequisites found; smoke and interface audits remain mandatory'}
    atomic_json(run.report/'doctor.json',data)
    atomic_json('environment.lock.json',data)
    return data
