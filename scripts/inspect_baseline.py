"""One CPU architecture forward; deliberately no robot performance or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from forcecritic.training.pretrain import baseline_actor
from forcecritic.runtime import source_hashes

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
torch.set_num_threads(2)
actor=baseline_actor(0).eval()
obs={'rgb':torch.zeros(1,3,256,256),'state':torch.zeros(1,8),'history':torch.zeros(1,4,16)}
with torch.no_grad():out=actor(obs)
result={'status':'PASSED','evidence_type':'SOFTWARE_ONLY','real_robot_transitions':0,
        'command':[sys.executable,*sys.argv],'returncode':0,
        'parameters':sum(p.numel() for p in actor.parameters()),
        'trainable_parameters':sum(p.numel() for p in actor.parameters() if p.requires_grad),
        'width':512,'encoder_layers':4,'decoder_layers':7,'attention_heads':8,'visual_grid':[8,8],
        'input_rgb_shape':[1,3,256,256],'output_shape':list(out.shape),'output_finite':bool(torch.isfinite(out).all()),
        'backbone_frozen':all(not p.requires_grad for p in actor.backbone.parameters()),
        'source_hashes':source_hashes(),'robot_baseline_acceptance':'NOT_RUN'}
weights=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
with weights.open('rb') as f:result['pretrained_resnet18_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
result['pretrained_resnet18_bytes']=weights.stat().st_size
assert result['output_shape']==[1,4,4] and result['output_finite'] and result['backbone_frozen']
path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
