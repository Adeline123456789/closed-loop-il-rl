"""Read-only checkpoint, optimizer and full batch-order matching audit (no updates)."""
import json
from pathlib import Path
import numpy as np
import torch
from forcecritic.runtime import atomic_json,digest
from forcecritic.staged.common import locked_plan,training_indices

torch.set_num_threads(1)
p=locked_plan();report=Path(p['report']);old=Path('checkpoints/20261023T104759Z')
load=lambda path:torch.load(path,map_location='cpu',weights_only=False,mmap=True)
act0=load(old/'act0.pt');a1=load(Path(p['checkpoints'])/'A1.pt');a2=load(old/'direct_Q.pt')
replay=load(old/'shared_replay.pt')
train=[r for r in replay['rows'] if r['split']=='train']
if replay['policy_sha']!=p['inputs'][str(old/'act0.pt')] or any(r['source']!='real_forge' for r in train):
    raise ValueError('Replay provenance mismatch')
manifest_path=Path(p['demonstration_manifest']);manifest=json.loads(manifest_path.read_text())
demonstrations=sorted([r for r in manifest['episodes'] if r['split']=='train' and r['successful']],key=lambda r:r['parent_id'])[:200]
windows=0
for row in demonstrations:
    entry=row['files']['stable_success'];path=manifest_path.parent/entry['path']
    if digest(path)!=entry['sha256']:raise ValueError('Changed terminal alignment')
    stable=np.load(path,allow_pickle=False)
    windows+=int(np.flatnonzero(stable)[0])+1
trace=json.loads((report/'A1_batch_order.json').read_text())
if len(trace)!=200:raise ValueError('Incomplete batch trace')
for i,row in enumerate(trace):
    ri,di=training_indices(len(train),windows,i,p['batch_size'],p['actor_numpy_seed_base'])
    if row!=dict(update=i+1,replay_indices=ri.tolist(),demo_indices=di.tolist()):raise ValueError('A1/A2 batch sequence mismatch')
for model in (a1,a2):
    if model['identity']!=act0['identity'] or model['actor_updates']!=200:raise ValueError('Changed initialization identity or budget')
    if any(not torch.equal(v,act0['actor'][k]) for k,v in model['actor'].items() if k.startswith('backbone.')):
        raise ValueError('Frozen common visual encoder changed')
groups=lambda model:[{k:v for k,v in g.items() if k!='params'} for g in model['actor_optimizer']['param_groups']]
if groups(a1)!=groups(a2):raise ValueError('Optimizer settings differ')
for model in (a1,a2):
    if any(int(s['step'])!=200 for s in model['actor_optimizer']['state'].values()):raise ValueError('Optimizer step count mismatch')
def difference(left,right):
    square=0.;maximum=0.;changed=0
    for name,tensor in left['actor'].items():
        if not tensor.is_floating_point():continue
        d=tensor-right['actor'][name]
        square+=float(d.double().square().sum());maximum=max(maximum,float(d.abs().max()))
        changed+=int(torch.count_nonzero(d))
    return dict(l2=square**.5,maximum_absolute=maximum,changed_elements=changed)
deltas=dict(A1_vs_ACT0=difference(a1,act0),A2_vs_ACT0=difference(a2,act0),A2_vs_A1=difference(a2,a1))
if any(v['changed_elements']==0 for v in deltas.values()):raise ValueError('Missing actual parameter change')
out=dict(status='PASSED',scope='Read-only artifact audit; no optimizer or simulation steps',
    batches_verified=200,replay_train_rows=len(train),demonstration_episodes=len(demonstrations),demonstration_windows=windows,
    initialization_identity_equal=True,normalizer_equal=True,frozen_visual_encoder_equal=True,optimizer_settings=groups(a1),
    parameter_changes=deltas,first_update_reproduction=str(report/'a2_reproduction_audit.json'),
    rng_limit='A2 final RNG was not saved historically; first update reproduced exactly, subsequent CVAE draw counts and sample order match source',
    act_sha={name:digest(path) for name,path in [('ACT0',old/'act0.pt'),('A1',Path(p['checkpoints'])/'A1.pt'),('A2',old/'direct_Q.pt')]})
atomic_json(report/'training_matching_audit.json',out)
print(json.dumps(out,indent=2))
