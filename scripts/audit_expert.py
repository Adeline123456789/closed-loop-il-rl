"""Integrity and safe tensor-shape inspection only; never a robot success evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

REVISION='447a6109cec7c228763d1bfc8cf0ab73f6558a1b'
SHA256='10efdf802e8abfccaa13f6e7a18cc2887ca2f3f96b35526bc9984fde419e59f8'
SIZE=213431615
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',default=f'.cache/expert/{REVISION}/forge_peginsert_2026-06-30.pth')
parser.add_argument('--output',required=True)
args=parser.parse_args()
p=Path(args.checkpoint)
with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
if p.stat().st_size!=SIZE or actual!=SHA256:raise RuntimeError('Pinned checkpoint size/hash mismatch')
result={'repository':'angledusgar/forge-peginsert-v1','revision':REVISION,'path':str(p),
        'size':SIZE,'sha256':SHA256,'download_status':'VERIFIED',
        'inspection_command':[sys.executable,*sys.argv],
        'compatibility':'NOT_RUN','project_expert_success':'NOT_RUN'}
# Explicit local constructors only, including the NumPy 1.x name used by this file.
# Do not import or automatically allow any global named by an external checkpoint.
scalar=np._core.multiarray.scalar if hasattr(np,'_core') else np.core.multiarray.scalar
allowed=[(scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype('float64')),type(np.dtype('float32'))]
try:
    with torch.serialization.safe_globals(allowed):
        checkpoint=torch.load(p,map_location='cpu',weights_only=True)
    result.update(safe_inspection='PASSED',checkpoint_keys=list(checkpoint),
        model_tensor_shapes={k:list(v.shape) for k,v in checkpoint.get('model',{}).items() if torch.is_tensor(v)},
        safe_globals_allowlist=['numpy.core.multiarray.scalar','numpy.dtype','float64 dtype','float32 dtype'],
        inspection_note='Tensor shapes and normalization state are metadata; official player rollout remains required.')
except Exception as error:
    result.update(safe_inspection='BLOCKED',inspection_error_type=type(error).__name__)
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='model_tensor_shapes'},indent=2))
raise SystemExit(0 if result['safe_inspection']=='PASSED' else 2)
