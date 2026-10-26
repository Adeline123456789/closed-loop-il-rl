import json
from pathlib import Path
from ..runtime import digest

CONFIG=Path('configs/stage_v3_phase1.json')

def locked_plan():
    plan=json.loads(CONFIG.read_text())
    lock=json.loads((Path(plan['report'])/'plan_lock.json').read_text())
    if digest(CONFIG)!=lock['sha256']:raise ValueError('Locked plan changed')
    for path,expected in {**plan['inputs'],**plan['unchanged_artifacts'],
        plan['demonstration_manifest']:plan['demonstration_manifest_sha'],
        plan['historical_A2_csv']:plan['historical_A2_csv_sha'],
        plan['historical_A0_csv']:plan['historical_A0_csv_sha']}.items():
        if digest(path)!=expected:raise ValueError('Historical input changed: '+path)
    return plan

def training_indices(size,demo_size,update,batch_size=128,seed_base=140000):
    import numpy as np
    rng=np.random.default_rng(seed_base+update)
    return rng.integers(size,size=batch_size),rng.integers(demo_size,size=batch_size)
