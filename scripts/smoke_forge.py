"""Real Isaac Lab launch only. No CPU fallback. Run from the Isaac Python environment."""
import argparse
import json
from pathlib import Path
import time

parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,default=1,choices=(1,8))
parser.add_argument('--output',required=True)
parser.add_argument('--steps',type=int,default=4)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not args.headless or not args.enable_cameras:
    parser.error('--headless --enable_cameras are required')
launcher=AppLauncher(args)
app=launcher.app
physics_counts={'initialization_restore':0,'evaluation':0}
records=[]
result={'status':'RUNNING','physics_substeps':physics_counts,'records':records}
def save_progress():
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(path)
save_progress()
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # registers real environments
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera,CameraCfg
    from isaaclab_tasks.utils import parse_env_cfg
    cfg=parse_env_cfg('Isaac-Forge-PegInsert-Direct-v0',device=args.device,num_envs=args.num_envs)
    cfg.scene.diagnostic_camera=CameraCfg(prim_path='{ENV_REGEX_NS}/DiagnosticCamera',update_period=0.,height=256,width=256,
        data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=20.955,
        clipping_range=(.1,10.)))
    # Count reset/initialization physics as well as explicit control steps.
    phase=['initialization_restore']
    original_step=sim_utils.SimulationContext.step
    def counted_step(self,*a,**kw):
        physics_counts[phase[0]]+=args.num_envs
        try:return original_step(self,*a,**kw)
        finally:save_progress()
    sim_utils.SimulationContext.step=counted_step
    env=gym.make('Isaac-Forge-PegInsert-Direct-v0',cfg=cfg)
    camera=env.unwrapped.scene.sensors['diagnostic_camera']
    obs,info=env.reset()
    camera.set_world_poses_from_view(env.unwrapped.scene.env_origins+torch.tensor([1.,1.,1.],device=args.device),
                                    env.unwrapped.scene.env_origins+torch.tensor([0.,0.,.4],device=args.device))
    phase[0]='evaluation'
    result['decimation']=cfg.decimation
    for i in range(args.steps):
        a=env.unwrapped.actions.clone()  # hold initialized stock target; not a trained policy
        a[:,6]=-1
        obs,reward,terminated,truncated,info=env.step(a)
        camera.update(env.unwrapped.step_dt,force_recompute=True)
        rgb=camera.data.output['rgb']
        records.append({'step':i,'rgb_shape':list(rgb.shape),'rgb_min':int(rgb.min()),'rgb_max':int(rgb.max()),
                        'wrench_finite':bool(torch.isfinite(env.unwrapped.force_sensor_smooth).all()),
                        'individual_transitions':args.num_envs})
    result={'status':'PASSED','purpose':'environment/camera smoke only; NOT expert evaluation',
            'num_envs':args.num_envs,'control_dt':env.unwrapped.step_dt,'physics_dt':env.unwrapped.physics_dt,
            'records':records,'time':time.time(),'physics_substeps':physics_counts,
            'reset_control_step_equivalents':physics_counts['initialization_restore']/cfg.decimation}
    if any(r['rgb_min']==r['rgb_max'] or not r['wrench_finite'] for r in records):
        raise RuntimeError('invalid camera/wrench smoke')
    save_progress()
    env.close()
except Exception as error:
    result.update(status='FAILED_IMPLEMENTATION',error=repr(error))
    save_progress()
    raise
finally:
    app.close()
