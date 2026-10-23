# Stock FORGE action contract — source verified, runtime NOT_RUN

Source: Isaac Lab v2.2.1, 0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20.
`forge_env.py:_apply_action`, `factory_env.py:_pre_physics_step`, `factory_env_cfg.py:CtrlCfg`.

* ACT emits one K=4 chunk, four native normalized coordinates x/y/z/yaw. Adapter inserts roll=pitch=0 and success-prediction placeholder=-1. No adapter EMA.
* Stock `_pre_physics_step` applies EMA once: alpha*submitted+(1-alpha)*previous. Forge samples alpha in [0.025,0.1] each reset.
* Position targets: noisy fixed-asset observation origin plus 0.05 m times each EMA action. These are pose targets, not velocities.
* Yaw: -pi + (3*pi/2)*(a_yaw+1)/2, then orientation composition with 180-degree roll. One native unit is 3*pi/4 radians before EMA/clipping. Position epsilon .02/.05 corresponds to 1/2.5 mm before EMA; yaw-only epsilon .02/.05 corresponds to .047124/.117810 rad before EMA. These are candidates, NOT locked safe perturbations.
* Desired pose is then clipped relative to measured EE using randomized position/rotation thresholds. Yaw uses the stock continuous interval [-125,235] degrees to respect the joint limit.
* Index 6 is `(action[6]+1)/2` success prediction, NOT gripper. Stock gripper target is 0. Stock success-prediction reward must be removed for learning, as must the original contact penalty when separate cost is used.
* Factory control dt=8*(1/120)=1/15 s. Four-step chunk is about .267 s. No temporal ensemble.
* Stock reset contains whole-batch operations. Single-env diagnosis is mandatory before accepting an 8-env wrapper. Final observations must be captured BEFORE automatic reset.

Required runtime records: submitted action, EMA action/target, clipped final target, measured EE motion. Runtime logging and reward replacement have NOT been validated.

HF card says “executed action” without collection code or a pinned collection checkout; it does not establish pre/post EMA stage. `stock_to_native` rejects ambiguous/post-EMA labels. Never apply inverse EMA without the missing initial controller memory.

Status: BLOCKED_DATA_ALIGNMENT. No claimed HF-to-current-simulator replay acceptance.
