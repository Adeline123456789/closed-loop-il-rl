"""Audit immutable historical inputs and lock the minimal matched-Q completion."""
import json
from pathlib import Path
from forcecritic.runtime import digest,atomic_json,Run

old=Path('reports/20261023T104759Z')
checkpoint=Path('checkpoints/20261023T104759Z')
report=Path('reports/stage_v3_20261026')
config=Path('configs/stage_v3_phase1.json')
if config.exists():raise FileExistsError('The phase-1 plan is immutable')
previous=json.loads((old/'direct_q_plan.json').read_text())
trained=json.loads((old/'direct_q_training.json').read_text())
expected={checkpoint/'act0.pt':previous['act_sha'],checkpoint/'critic0.pt':previous['critic_sha'],
    checkpoint/'shared_replay.pt':previous['replay_sha'],checkpoint/'direct_Q.pt':trained['sha256']}
checks={str(p):dict(expected=h,actual=digest(p)) for p,h in expected.items()}
material=['scripts/train_direct_q_fallback.py','src/forcecritic/models/act.py',
    'src/forcecritic/models/critic.py','src/forcecritic/training/updates.py',
    'src/forcecritic/training/pretrain.py','src/forcecritic/training/live.py',
    'src/forcecritic/data/demonstrations.py']
for p in material:checks[p]=dict(expected=previous['source_hashes'][p],actual=digest(p))
run=Run('configs/minimal_validation.yaml','20261023T104759Z')
passed=all(c['actual']==c['expected'] for c in checks.values()) and run.verified('baseline')
atomic_json(report/'input_audit.json',dict(status='PASSED' if passed else 'BLOCKED',checks=checks))
if not passed:raise ValueError('Historical hash mismatch; no training permitted')
plan=dict(protocol='forcecritic_staged_v3',stage='phase1_matched_q',status='LOCKED',
    parent_protocol='PROTOCOL_STAGE_V3.md',report=str(report),results='results/stage_v3_20261026',
    checkpoints='checkpoints/stage_v3_20261026',inputs={str(p):h for p,h in expected.items()},
    demonstration_manifest='data/act_demonstrations/manifest.json',
    demonstration_manifest_sha=digest('data/act_demonstrations/manifest.json'),
    historical_plan=str(old/'direct_q_plan.json'),historical_plan_sha=digest(old/'direct_q_plan.json'),
    historical_A2_csv='results/20261023T104759Z/direct_Q_validation.csv',
    historical_A2_csv_sha=digest('results/20261023T104759Z/direct_Q_validation.csv'),
    historical_A0_csv='results/20261023T104759Z/direct_B0_validation.csv',
    historical_A0_csv_sha=digest('results/20261023T104759Z/direct_B0_validation.csv'),
    historical_A2_results_seen_before_this_plan=True,
    historical_interpretation='Exploratory until the missing matched A1 is measured; not a newly blinded A2 test',
    arms={'A0':'fixed ACT0, reuse','A1':'BC + step, Q coefficient zero','A2':'reuse direct_Q only after first-update reproduction audit'},
    force_history_in_phase1=True,history_frames=4,history_dim=16,
    phase2_reuse_of_phase1_as_no_force=False,
    actor_updates=200,actor_lr=1e-5,batch_size=128,alpha_bc=1.,eta_step=1.,lambda_cost=1.,
    actor_numpy_seed_base=140000,torch_seed=0,critic_updates_new=0,
    critic_for_audit='direct_Q.pt critic state after the historical 2000 TD updates',
    audit_optimizer_updates=1,audit_model_never_deployed=True,
    reproduction_rtol=1e-5,reproduction_atol=1e-6,
    initial_evaluation_seeds=list(range(21000,21032)),
    extension_seeds=list(range(22000,22032)),extension_round_cap=1,
    extension_only_if='INCONCLUSIVE after matched initial evaluation and implementation audit PASSED',
    max_new_episodes={'A1':64,'A2':32},evaluation_control_step_cap_per_batch=10000,
    minimum_success_difference=.05,look_alpha=.025,max_decision_looks=2,
    decision='GO/NO_GO require >=5 percentage-point signed change, paired 97.5% bootstrap interval excluding zero and exact two-sided discordant binomial p<=0.025; otherwise INCONCLUSIVE',
    bootstrap_resamples=20000,bootstrap_seed=450000,
    no_training_extension=True,implementation_fix_rerun_cap=1,
    reset_protocol='Same original renderer and NoiseSchedule as historical direct_Q evaluation; stochastic pixels mean equal seed is not a proven identical full contact state',
    paired_interpretation='Pair assigned evaluation seed, not trusted D0 checkpoint replay; report marginal Wilson intervals too',
    unchanged_artifacts={str(p):digest(p) for p in [Path('configs/minimal_validation.yaml'),old/'p0_acceptance.json',Path('src/forcecritic/envs/forge.py'),Path('src/forcecritic/diagnostics/live.py')]})
atomic_json(config,plan)
atomic_json(report/'plan_lock.json',dict(config=str(config),sha256=digest(config),input_audit_sha=digest(report/'input_audit.json')))
print('LOCKED',config,digest(config))
