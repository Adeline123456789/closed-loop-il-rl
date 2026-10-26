"""Diagnostic checkpoint sampling and the D0 physical replay admission gate."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from ..runtime import digest
from .artifacts import atomic_json
from ..models.critic import condition_features
from ..training.live import LivePolicy
from ..training.pretrain import baseline_actor
from ..training.baseline import accepted_limits
from .live import NoiseSchedule, snapshot, fidelity_checkpoint
from .pairs import candidates
from .progress import DiagnosticProgress, progress


def sample_checkpoints(adapter, policy, noise, ledger, thresholds, *, hard_count, normal_count, seed_start):
    checkpoints = []
    counts = {True:0, False:0}
    wanted = {True:hard_count, False:normal_count}
    candidate_steps = set(thresholds['checkpoint_control_steps'])
    if not candidate_steps or any(s < 8 or s % 4 for s in candidate_steps):
        raise ValueError('Checkpoint locations must be pre-registered full-chunk boundaries')
    for seed in range(seed_start, seed_start + 128):
        if counts == wanted:
            break
        progress(ledger, seed=seed, selected_hard=counts[True], selected_normal=counts[False],
                 wanted_hard=hard_count, wanted_normal=normal_count)
        noise.seed, noise.tick = seed, 0
        adapter.reset(seed=seed)
        prefix, rewards = [], []
        ended = False
        selected = False
        while not ended and not selected:
            plan = policy(adapter.observation)
            for k in range(4):
                with ledger.using('prefix'):
                    row = adapter.step(plan[:,k])
                prefix.append(plan[0,k].copy())
                rewards.append(float(row['reward'][0]))
                ended = bool(row['terminated'][0] or row['truncated'][0] or row['success'][0] or row['unsafe'][0])
                if ended:
                    break
            if ended or len(prefix) not in candidate_steps:
                continue
            load = float(np.linalg.norm(row['substep_wrench'][:,0,:3],axis=-1).max())
            gain = float((np.mean(rewards[-4:])-np.mean(rewards[-8:-4]))/4)
            hard = bool(load >= thresholds['hard_force_N'] and gain <= thresholds['hard_progress_per_step'])
            normal = bool(gain > thresholds['normal_progress_per_step'])
            if counts[hard] >= wanted[hard] or (not hard and not normal):
                continue
            observation = copy.deepcopy(adapter.observation[0])
            remaining = (adapter.raw.max_episode_length-1-adapter.raw.episode_length_buf).clamp_min(0)
            feature = condition_features(policy.actor,policy.batch([observation]),remaining,
                torch.as_tensor(adapter.controller(),device=policy.device)).cpu()[0]
            checkpoints.append(dict(parent_id=f'diagnostic-{seed}',checkpoint_id=f'diagnostic-{seed}',
                seed=seed, hard=hard, prefix=copy.deepcopy(prefix), snapshot=snapshot(adapter,noise),
                observation=observation,condition=feature,a0=policy(adapter.observation)[0],
                load_N=load,progress_per_step=float(gain),source='real_forge_checkpoint',accepted_methods=[]))
            counts[hard] += 1
            selected = True
    if counts != wanted:
        return {'status':'INCONCLUSIVE','reason':'Predefined hard/normal checkpoint quotas not reached within 128 parents',
                'checkpoints':checkpoints,'counts':counts}
    return {'status':'COLLECTED','checkpoints':checkpoints,'counts':counts}


def admit_checkpoints(adapter, policy, noise, checkpoints, ledger, thresholds, scale, directory, used=0, csv_path=None, policy_sha=None):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    direction = np.zeros((4,4),dtype=np.float32)
    direction[:,2] = .5
    summary = []
    for point_index, checkpoint in enumerate(checkpoints):
        progress(ledger, phase='D0_initial' if len(checkpoints)==8 else 'D0_additional',
                 parent_id=checkpoint['parent_id'], point_index=point_index+1,
                 points_completed=point_index, points_target=len(checkpoints))
        if used + 21 > 1500:
            raise RuntimeError('BUDGET_EXHAUSTED before D0 O/R/P continuations')
        # Establish the canonical checkpoint immediately before its unreset O
        # continuation. The R state used later by D1 must be this measured state,
        # not an earlier, merely similar sample from the checkpoint search.
        from .live import prefix_replay
        reached=prefix_replay(adapter,noise,checkpoint['seed'],checkpoint['prefix'],ledger)
        load=float(np.linalg.norm(reached['last']['substep_wrench'][:,0,:3],axis=-1).max())
        rewards=reached['recent_rewards'];gain=float((np.mean(rewards[-4:])-np.mean(rewards[-8:-4]))/4)
        hard=bool(load>=thresholds['hard_force_N'] and gain<=thresholds['hard_progress_per_step'])
        if hard!=checkpoint['hard'] or (not hard and gain<=thresholds['normal_progress_per_step']):
            return {'status':'INCONCLUSIVE','methods':[],'continuations_used':used,
                    'reason':'Checkpoint class changed during canonical prefix; original search state is not assumed reproduced'}
        checkpoint['snapshot']=snapshot(adapter,noise)
        checkpoint['observation']=copy.deepcopy(adapter.observation[0])
        remaining=(adapter.raw.max_episode_length-1-adapter.raw.episode_length_buf).clamp_min(0)
        checkpoint['condition']=condition_features(policy.actor,policy.batch(adapter.observation),remaining,
            torch.as_tensor(adapter.controller(),device=policy.device)).cpu()[0]
        checkpoint['a0']=policy(adapter.observation)[0]
        plus,minus,_ = candidates(checkpoint['a0'],direction,.02)
        measured = fidelity_checkpoint(adapter,policy,noise,checkpoint['seed'],checkpoint['prefix'],
            [checkpoint['a0'],plus,minus],ledger,thresholds['fidelity_tolerances'],scale,first_at_checkpoint=True)
        used += 21  # three candidates, each O plus three R plus three P
        checkpoint['accepted_methods'] = measured['accepted_methods']
        torch.save(measured,directory/f"{checkpoint['parent_id']}.pt")
        if csv_path is not None:
            import csv
            from ..evaluation.report import SCHEMAS
            with Path(csv_path).open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=SCHEMAS['fidelity.csv'].split(','))
                for row in measured['records']:
                    e=row['errors']
                    writer.writerow(dict(parent_episode=checkpoint['parent_id'],split='admission',policy_sha=policy_sha,
                        restore_method=row['method'],action_variant=row['candidate'],repeat=row['repeat'],
                        pose_error=e['position_m'],velocity_error=e['linear_velocity_m_s'],force_error=e['force_N'],
                        return_error=e['scaled_return'],cost_error=e['scaled_cost'],success_difference=e['success'],
                        env_steps=row['env_steps']+row['original_env_steps'],status='PASSED' if row['passed'] else 'FAILED'))
        summary.append({'parent_id':checkpoint['parent_id'],'hard':checkpoint['hard'],
            'accepted_methods':measured['accepted_methods'],
            'comparisons':[{k:v for k,v in r.items() if k not in ('original','repeated')} for r in measured['records']]})
        atomic_json(directory/'fidelity.json',{'checkpoints':summary,'continuations_used':used,'accounting':ledger.record()})
        progress(ledger, points_completed=point_index+1)
        if not measured['accepted_methods']:
            # The gate is an intersection over parents. More parents cannot
            # make an empty intersection nonempty; stop without claiming 8/8.
            return {'status':'NO_GO_REPLAY','methods':[], 'continuations_used':used,
                    'completed_points':point_index+1,'planned_points':len(checkpoints),
                    'not_run_points':len(checkpoints)-point_index-1,
                    'reason':'Neither replay method passed this parent; the all-parent admission intersection is empty. Remaining parents NOT_RUN.'}
    common = [method for method in ('snapshot_restore','prefix_replay')
              if all(method in c['accepted_methods'] for c in checkpoints)]
    return {'status':'PASSED' if common else 'NO_GO_REPLAY','methods':common,'continuations_used':used,
            'reason':'Original-vs-replay physical equivalence measured; no solver-state completeness is assumed'}


def bootstrap_interval(values, seed=0):
    values = np.asarray(values)
    if len(values)<3:
        raise ValueError('At least three independent paired noise blocks required')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return np.quantile(means,[.025,.975],axis=0)


def screen_pairs(bundle, scale):
    """Endpoint/pair targets retain parent split and continuation version."""
    groups = {}
    for row in bundle['rows']:
        groups.setdefault(row['parent_id'],[]).append(row)
    endpoints, pairs, choices = [], [], []
    for parent, rows in groups.items():
        base = {r['noise_id']:np.array([r['return'],r['cost']]) for r in rows if r['candidate']=='base'}
        variants = sorted({(r['direction'],r['epsilon']) for r in rows if r['candidate']!='base'})
        for row in rows:
            endpoints.append(dict(parent_id=parent,split=row['split'],condition=row['condition'],
                action=row['action'],returns=np.array([row['return'],row['cost']]),policy_sha=bundle['policy_sha']))
        for direction,epsilon in variants:
            selected = [r for r in rows if r['direction']==direction and r['epsilon']==epsilon]
            plus = {r['noise_id']:r for r in selected if r['candidate']=='plus'}
            minus = {r['noise_id']:r for r in selected if r['candidate']=='minus'}
            ids = sorted(base)
            if set(ids)!=set(plus) or set(ids)!=set(minus):
                raise ValueError('Paired endpoint noise blocks are not aligned')
            p = np.array([[plus[i]['return'],plus[i]['cost']] for i in ids])
            m = np.array([[minus[i]['return'],minus[i]['cost']] for i in ids])
            b = np.stack([base[i] for i in ids])
            delta = (p-m)/scale
            interval = bootstrap_interval(delta[:,0]-delta[:,1])
            identifiable = bool(interval[0]>.1 or interval[1]<-.1)
            example = plus[ids[0]]
            pair = dict(parent_id=parent,checkpoint_id=parent,split=example['split'],hard=example['hard'],
                condition=example['condition'],plus=example['action'],minus=minus[ids[0]]['action'],
                delta=(p-m).mean(0),direction=direction,epsilon=epsilon,accepted=False,
                screening_identifiable=identifiable,screening_interval=interval.tolist(),
                policy_sha=bundle['policy_sha'])
            pairs.append(pair)
            for sign,values,action in [('plus',p,example['action']),('minus',m,minus[ids[0]]['action'])]:
                improvement = (values-b)/scale
                ci = bootstrap_interval(improvement[:,0]-improvement[:,1])
                cost_ci = bootstrap_interval(improvement[:,1])
                choices.append(dict(parent_id=parent,split=example['split'],hard=example['hard'],
                    direction=direction,epsilon=epsilon,sign=sign,action=action,
                    improvement_lower=float(ci[0]),cost_upper=float(cost_ci[1]),
                    candidate_screened=bool(identifiable and ci[0]>.1 and cost_ci[1]<=.025)))
    return endpoints,pairs,choices


def diagnostic(run, manifest, make_adapter, device='cuda:0'):
    from .live import collect_pairs
    directory = Path('checkpoints')/run.run_id
    act_path = directory/'act0.pt'
    policy_sha = digest(act_path)
    state = torch.load(act_path,map_location='cpu',weights_only=False)
    fitted = torch.load(directory/'critic0.pt',map_location='cpu',weights_only=False)
    if fitted['policy_sha'] != policy_sha:
        raise ValueError('Critic is not for the frozen ACT0 continuation policy')
    thresholds_path = run.report/'protocol_effect_thresholds.json'
    thresholds = json.loads(thresholds_path.read_text())
    if (thresholds.get('status')!='LOCKED' or thresholds.get('config_hash')!=run.config_hash
            or thresholds.get('source_split') not in ('development','train')):
        raise ValueError('Physical tolerances and selection criteria must be locked from development/train data')
    actor = baseline_actor().to(device)
    actor.load_state_dict(state['actor'])
    policy = LivePolicy(actor,state['identity']['normalizer'],device)
    adapter,ledger = make_adapter(1,accepted_limits(run.report/'p0_acceptance.json'))
    import time
    repair_path = run.report/'diagnostic_repair.json'
    repair = json.loads(repair_path.read_text()) if repair_path.exists() else {}
    # Engineering probes are charged in the shared diagnostic physical budget.
    ledger.cap -= int(repair.get('engineering_control_steps',0))*ledger.decimation
    observer = DiagnosticProgress(ledger,run.report,directory/f'diagnostic_attempt_{time.time_ns()}',
                                  previous_continuations=repair.get('previous_continuations',0))
    adapter.diagnostic = True
    noise = NoiseSchedule(adapter,40000)
    try:
        selected = sample_checkpoints(adapter,policy,noise,ledger,thresholds,
            hard_count=4,normal_count=4,seed_start=40000)
        if selected['status']!='COLLECTED':
            return {'status':'INCONCLUSIVE','reason':selected['reason']}
        checkpoints = selected['checkpoints']
        torch.save(checkpoints,observer.directory/'selected_initial.pt')
        d0 = admit_checkpoints(adapter,policy,noise,checkpoints,ledger,thresholds,fitted['scale'],
                              directory/'fidelity_initial',csv_path=run.result/'fidelity.csv',policy_sha=policy_sha)
        atomic_json(run.report/'d0_initial.json',d0)
        if (directory/'fidelity_initial/fidelity.json').exists():
            atomic_json(run.report/'fidelity_initial.json',json.loads((directory/'fidelity_initial/fidelity.json').read_text()))
        if d0['status']!='PASSED':
            observer.mark(status=d0['status'], reason=d0['reason'])
            return d0
        observer.mark(phase='D0_additional_sampling')
        selected = sample_checkpoints(adapter,policy,noise,ledger,thresholds,
            hard_count=12,normal_count=4,seed_start=50000)
        if selected['status']!='COLLECTED':
            return {'status':'INCONCLUSIVE','reason':selected['reason']}
        extra = selected['checkpoints']
        torch.save(extra,observer.directory/'selected_additional.pt')
        extra_admission = admit_checkpoints(adapter,policy,noise,extra,ledger,thresholds,fitted['scale'],
            directory/'fidelity_additional',used=d0['continuations_used'],csv_path=run.result/'fidelity.csv',policy_sha=policy_sha)
        if (directory/'fidelity_additional/fidelity.json').exists():
            atomic_json(run.report/'fidelity_additional.json',json.loads((directory/'fidelity_additional/fidelity.json').read_text()))
        if extra_admission['status']!='PASSED':return extra_admission
        common = [m for m in d0['methods'] if m in extra_admission['methods']]
        if not common:
            return {'status':'NO_GO_REPLAY','reason':'No single method passed the original comparisons for the D1 parents'}
        method = common[0]
        checkpoints += extra
        for hard,counts in ((True,(8,4,4)),(False,(4,2,2))):
            group = sorted([c for c in checkpoints if c['hard']==hard],key=lambda c:c['seed'])
            offset = 0
            for split,count in zip(('train','validation','test'),counts):
                for c in group[offset:offset+count]:
                    c['split'] = split
                offset += count
        torch.save(checkpoints,directory/'diagnostic_checkpoints.pt')
        observer.mark(phase='D1_screening')
        screening = collect_pairs(adapter,policy,noise,checkpoints,method,ledger,
            run.result/'paired_screening.csv',policy_sha=policy_sha,
            already_used=extra_admission['continuations_used'])
        used = extra_admission['continuations_used'] + screening['continuations']
        endpoints,pairs,choices = screen_pairs(screening,np.asarray(fitted['scale']))
        torch.save(screening,directory/'paired_screening.pt')
        hard_identified = {p['parent_id'] for p in pairs if p['hard'] and p['screening_identifiable']}
        atomic_json(run.report/'paired_screening.json',{
            'status':'SCREENING_ONLY','hard_identifiable':len(hard_identified),'hard_total':16,
            'continuations_used':used,'policy_sha':policy_sha,'thresholds_sha256':digest(thresholds_path),
            'all_pairs':[{k:v for k,v in p.items() if k not in ('condition','plus','minus','delta')} for p in pairs]})
        if len(hard_identified)<4:
            return {'status':'INCONCLUSIVE','reason':'Hard-state finite effect coverage below 4/16; no calibration training'}
        return confirm_and_transfer(run,manifest,adapter,policy,noise,ledger,checkpoints,method,
            endpoints,pairs,choices,fitted,policy_sha,used,thresholds,screening['branch_rows'])
    except RuntimeError as error:
        if 'BUDGET_EXHAUSTED' not in str(error):
            raise
        return {'status':'INCONCLUSIVE','reason':'Diagnostic physical interaction/continuation budget exhausted'}
    finally:
        atomic_json(run.report/'diagnostic_accounting.json',ledger.record())
        observer.mark(producer_finished=True)
        observer.close()
        noise.close()
        adapter.close()
        ledger.close()


def confirm_and_transfer(run,manifest,adapter,policy,noise,ledger,checkpoints,method,
                         endpoints,pairs,choices,fitted,policy_sha,used,thresholds,branch_rows):
    import csv
    from .live import restore,prefix_replay,continuation,diagnose_transfer
    from ..models.critic import TwinCritic
    from ..data.demonstrations import Demonstrations
    progress(ledger, phase='D1_confirmation')
    from ..training.live import evaluate
    from torch.utils.data import default_collate
    by_parent = {c['parent_id']:c for c in checkpoints}
    best = {}
    for choice in choices:
        if choice['split']!='train':
            continue
        parent = choice['parent_id']
        if parent not in best or choice['improvement_lower']>best[parent]['improvement_lower']:
            best[parent] = choice
    ranked = sorted(best.values(),key=lambda c:(-c['improvement_lower'],c['parent_id']))
    selected = [c for c in ranked if c['candidate_screened']][:6]
    selected += [c for c in ranked if not c['candidate_screened']][:2]
    confirmations,whitelist = [],[]
    path = run.result/'paired_confirmation.csv'
    if path.exists():raise FileExistsError('Use a new run for a new diagnostic confirmation attempt')
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['parent_id','split','policy_sha','variant','noise_id','return','cost','success','steps'])
        writer.writeheader()
        for choice in selected:
            checkpoint = by_parent[choice['parent_id']]
            pair = next(p for p in pairs if p['parent_id']==choice['parent_id']
                        and p['direction']==choice['direction'] and p['epsilon']==choice['epsilon'])
            observed = {k:[] for k in ('base','plus','minus')}
            for repeat in range(8):
                for variant,action in [('base',checkpoint['a0']),('plus',pair['plus']),('minus',pair['minus'])]:
                    if used>=1500:raise RuntimeError('BUDGET_EXHAUSTED before independent confirmation')
                    if method=='snapshot_restore':restore(adapter,noise,checkpoint['snapshot'],ledger)
                    else:prefix_replay(adapter,noise,checkpoint['seed'],checkpoint['prefix'],ledger)
                    noise.seed = 800000+checkpoint['seed']*10+repeat
                    actual = continuation(adapter,policy,action,ledger,collect_td=True)
                    for transition in actual['transitions']:
                        transition.update(parent_id=choice['parent_id'],split='train',continuation_sha=policy_sha,
                                          behavior_sha=policy_sha,noise_id=noise.seed)
                        branch_rows.append(transition)
                    used += 1
                    observed[variant].append(actual['returns'])
                    writer.writerow(dict(parent_id=choice['parent_id'],split='train',policy_sha=policy_sha,
                        variant=variant,noise_id=noise.seed,**{'return':float(actual['returns'][0])},
                        cost=float(actual['returns'][1]),success=int(actual['success']),steps=actual['steps']))
                    f.flush()
                    endpoints.append(dict(parent_id=choice['parent_id'],split='train',condition=checkpoint['condition'],
                        action=action,returns=actual['returns'],policy_sha=policy_sha))
            observed = {k:np.stack(v) for k,v in observed.items()}
            scale = np.asarray(fitted['scale'])
            improvement = (observed[choice['sign']]-observed['base'])/scale
            ci = bootstrap_interval(improvement[:,0]-improvement[:,1])
            cost_ci = bootstrap_interval(improvement[:,1])
            delta = (observed['plus']-observed['minus'])/scale
            delta_ci = bootstrap_interval(delta[:,0]-delta[:,1])
            pair['accepted'] = bool(delta_ci[0]>.1 or delta_ci[1]<-.1)
            pair['delta'] = (3*pair['delta']+8*(observed['plus']-observed['minus']).mean(0))/11
            accepted = bool(pair['accepted'] and ci[0]>.1 and cost_ci[1]<=.025)
            confirmation = dict(parent_id=choice['parent_id'],hard=checkpoint['hard'],
                screened=choice['candidate_screened'],accepted=accepted,improvement_ci=ci.tolist(),
                cost_ci=cost_ci.tolist(),delta_ci=delta_ci.tolist(),sign=choice['sign'])
            confirmations.append(confirmation)
            if accepted:
                whitelist.append(dict(checkpoint_id=choice['parent_id'],split='train',continuation_sha=policy_sha))
                checkpoint['independently_confirmed'] = True
                checkpoint['improving_action'] = choice['action']
            atomic_json(run.report/'paired_confirmation.json',{'status':'MEASURED',
                'confirmations':confirmations,'continuations_used':used,'accounting':ledger.record()})
    confirmed_hard = sum(c['accepted'] and c['hard'] for c in confirmations)
    if confirmed_hard<4:
        return {'status':'INCONCLUSIVE','reason':'Fewer than four independent hard parents confirmed; calibration gate not passed'}
    critic = TwinCritic(fitted['condition_dim']).to(policy.device)
    critic.load_state_dict(fitted['critic'])
    accepted_points = sorted([c for c in checkpoints if c.get('independently_confirmed')],
                             key=lambda c:(not c['hard'],c['seed']))
    checkpoint = accepted_points[0]
    dataset = Demonstrations(manifest,normalizer=policy.normalizer)
    rng = np.random.default_rng(99100)
    obs,actions,valid = default_collate([dataset[int(i)] for i in rng.integers(len(dataset),size=run.config['algorithm']['batch_size'])])
    demo = ({k:v.to(policy.device) for k,v in obs.items()},actions.to(policy.device),valid.to(policy.device))
    if used+4>1500:raise RuntimeError('BUDGET_EXHAUSTED before D2')
    progress(ledger, phase='D2_transfer')
    transfer = diagnose_transfer(adapter,policy,critic,checkpoint,checkpoint['improving_action'],demo,noise,ledger,method=method)
    used += 4
    directory = Path('checkpoints')/run.run_id
    torch.save(transfer,directory/'d2_transfer.pt')
    return finish_transfer(run,adapter,policy,noise,ledger,checkpoints,endpoints,pairs,whitelist,
        fitted,policy_sha,used,transfer,confirmations,thresholds,branch_rows)


def finish_transfer(run,adapter,policy,noise,ledger,checkpoints,endpoints,pairs,whitelist,
                    fitted,policy_sha,used,transfer,confirmations,thresholds,branch_rows):
    import csv
    from ..models.critic import TwinCritic
    from ..training.live import evaluate
    directory = Path('checkpoints')/run.run_id
    with (run.result/'act0_validation.csv').open() as f:
        seeds = [int(r['seed']) for r in csv.DictReader(f) if int(r['stable_success'])]
    if len(seeds)<thresholds['minimum_normal_success_starts']:
        return {'status':'INCONCLUSIVE','reason':'Insufficient fixed successful validation starts to measure forgetting'}
    observer=getattr(ledger,'progress',None)
    previous=observer.data['previous_continuations'] if observer else 0
    if used+previous+3*len(seeds)>1500:raise RuntimeError('BUDGET_EXHAUSTED before forgetting evaluation')
    progress(ledger,phase='D2_forgetting')
    outcomes = {}
    for name in ('baseline','q_only','q_bc_step'):
        actor = copy.deepcopy(policy.actor)
        if name!='baseline':actor.load_state_dict(transfer[name]['actor'])
        actor_path = directory/f'd2_{name}.pt'
        torch.save(actor.state_dict(),actor_path)
        outcomes[name] = evaluate(adapter,LivePolicy(actor,policy.normalizer,policy.device),seeds,ledger,
            run.result/f'd2_{name}_normal_validation.csv',arm=name,policy_sha=digest(actor_path),noise=noise)
        used += len(seeds)
    scale = np.asarray(fitted['scale'])
    utility = lambda result: float(np.dot(result['returns']/scale,[1.,-1.]))
    baseline_value = utility(transfer['baseline'])
    direct_gain = utility(transfer['direct_candidate'])-baseline_value
    q_gain = utility(transfer['q_only']['actual'])-baseline_value
    qbc_gain = utility(transfer['q_bc_step']['actual'])-baseline_value
    critic = TwinCritic(fitted['condition_dim']).to(policy.device)
    critic.load_state_dict(fitted['critic'])
    judgments = []
    with torch.no_grad():
        for pair in pairs:
            if pair['split']!='train' or not pair['accepted']:continue
            condition = pair['condition'].reshape(1,-1).to(policy.device)
            predictions = []
            for action in (pair['plus'],pair['minus']):
                q = critic(condition,torch.as_tensor(action[None],device=policy.device))
                predictions.append(float((q[:,:,0].min(1).values-q[:,:,1].max(1).values).item()))
            observed = float(np.dot(pair['delta']/scale,[1.,-1.]))
            judgments.append(dict(parent_id=pair['parent_id'],predicted=predictions[0]-predictions[1],observed=observed,
                misranked=bool(observed*(predictions[0]-predictions[1])<0)))
    misranked = len({r['parent_id'] for r in judgments if r['misranked']})
    report = dict(status='MEASURED',direct_gain_scaled=direct_gain,q_only_gain_scaled=q_gain,
        q_bc_step_gain_scaled=qbc_gain,judgments=judgments,forgetting_outcomes=outcomes,
        continuations_used=used,accounting=ledger.record(),
        q_only_direction_cosine=transfer['q_only']['direction_cosine'],
        q_bc_direction_cosine=transfer['q_bc_step']['direction_cosine'])
    from ..evaluation.report import SCHEMAS
    with (run.result/'actor_step.csv').open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=SCHEMAS['actor_step.csv'].split(','))
        for name in ('q_only','q_bc_step'):
            item=transfer[name]
            before=item['action_before'].numpy();after=item['action_after'].numpy()
            delta=item['actual']['returns']-transfer['baseline']['returns']
            forgetting=float(np.mean([x['stable_success'] for x in outcomes['baseline']])-
                             np.mean([x['stable_success'] for x in outcomes[name]]))
            writer.writerow(dict(parent_episode=transfer['parent_id'],split='train',policy_sha=policy_sha,arm=name,
                before_action=json.dumps(before.tolist()),after_action=json.dumps(after.tolist()),
                delta_action=json.dumps((after-before).tolist()),q_grad_norm=item['q_grad_norm'],
                bc_grad_norm=item['bc_grad_norm'],q_bc_dot=item['q_bc_dot'],
                angle=float(np.degrees(np.arccos(np.clip(item['direction_cosine'],-1,1)))),
                saturation=item['saturation'],real_return_change=float(delta[0]),real_cost_change=float(delta[1]),
                forgetting=forgetting,status='MEASURED'))
    atomic_json(run.report/'d2_transfer.json',report)
    if direct_gain<=.1:
        return {'status':'INCONCLUSIVE','reason':'Confirmed candidate did not reproduce a useful direct effect at the D2 noise block'}
    if misranked<thresholds['minimum_misranked_confirmed_points']:
        return {'status':'NO_GO_CALIBRATION','reason':'Ordinary critic not shown to have a repeated action-ranking problem'}
    if q_gain>.1 and qbc_gain<=0:
        return {'status':'NO_GO_CALIBRATION','reason':'BC/step objective, rather than the ordinary Q direction, cancels the measured improvement'}
    if transfer['q_only']['direction_cosine']>0:
        return {'status':'INCONCLUSIVE','reason':'Real ACT transfer measured, but critic error is not isolated as the primary cause'}
    bundle = dict(status='INDEPENDENTLY_CONFIRMED',policy_sha=policy_sha,source='real_forge',
        d2_measured_passed=True,whitelist=whitelist,pairs=pairs,endpoints=endpoints,branch_rows=branch_rows,
        return_scale=np.asarray(fitted['scale']).tolist(),
        checkpoints=[{k:v for k,v in c.items() if k not in ('snapshot','prefix')} for c in checkpoints],
        confirmations=confirmations,continuations_used=used,thresholds=thresholds)
    torch.save(bundle,directory/'paired_confirmed.pt')
    return {'status':'PASSED','reason':'Physical replay, independent effects and real Critic-to-ACT transfer measured; same-data short ablations admitted'}
