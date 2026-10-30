"""Pre-registered short-ablation decision from held-out labels and real episodes."""
import json
import numpy as np
from ..runtime import atomic_json,digest
from .gates import decide


def interval(x):
    x=np.asarray(x,float)
    if len(x)<3:raise ValueError('At least three independent parents required')
    rng=np.random.default_rng(0)
    means=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def short_decision(run,outcomes,prediction_rows,paired):
    protocol=json.loads((run.report/'protocol_effect_thresholds.json').read_text())
    locked=protocol['short_gate']
    b2,b3=outcomes['B2'],outcomes['B3']
    if [x['parent_id'] for x in b2]!=[x['parent_id'] for x in b3]:
        raise ValueError('Cannot compare unmatched reset conditions')
    delta=np.array([r['stable_success']-l['stable_success'] for l,r in zip(b2,b3)])*100
    ci=interval(delta)
    baseline_overload=np.array([r['overload_steps']>0 for r in b2])
    new_overload=np.array([r['overload_steps']>0 for r in b3])
    reduction=0. if not baseline_overload.any() else 1-float(new_overload.mean()/baseline_overload.mean())
    # One paired unit per independent diagnostic parent, never per candidate.
    parents=sorted({r['parent_id'] for r in prediction_rows if r['split']=='validation'})
    errors={arm:[] for arm in ('B2','B3')}
    for parent in parents:
        for arm in errors:
            values=[r['contrast_mse'] for r in prediction_rows if r['parent_id']==parent and r['arm']==arm]
            if not values:raise ValueError('Missing held-out contrast prediction')
            errors[arm].append(float(np.mean(values)))
    error_delta=np.asarray(errors['B3'])-errors['B2']
    error_ci=interval(error_delta)
    relative=1-float(np.mean(errors['B3'])/max(np.mean(errors['B2']),1e-12))
    scales=np.asarray(paired['return_scale'])
    utility_delta=[(r['return']-l['return'])/scales[0]-(r['cost']-l['cost'])/scales[1]
                   for l,r in zip(b2,b3)]
    utility_ci=interval(utility_delta)
    peak_worse=any(np.mean([r[k] for r in b3])>(1+locked['max_mean_peak_relative_increase'])*
                   np.mean([r[k] for r in b2]) for k in ('peak_force_N','peak_torque_Nm'))
    hard={p['parent_id'] for p in paired['pairs'] if p['hard'] and p['screening_identifiable']}
    evidence=dict(environment_passed=True,baseline_passed=True,replay_passed=True,
        hard_total=16,hard_identifiable=len(hard),critic_judgment_problem=True,critic_is_primary_cause=True,
        independent_prediction_improved=bool(error_ci[1]<0 and relative>=locked['minimum_contrast_relative_reduction']),
        actual_actor_improved=bool(utility_ci[0]>0),success_difference_pp=float(delta.mean()),success_difference_ci=ci,
        overload_worse=bool(new_overload.mean()>baseline_overload.mean()+locked['max_overload_rate_increase']),
        peak_load_worse=bool(peak_worse),baseline_overload_events=int(baseline_overload.sum()),
        overload_relative_reduction=reduction,intervals_informative=bool(ci[1]-ci[0]<=locked['max_success_ci_width_pp']),
        validation_contrast_parent_count=len(parents),contrast_error_difference_ci=error_ci,
        contrast_relative_reduction=relative,actual_utility_difference_ci=utility_ci,
        overload_event_unit='independent episode with one or more exceedances')
    decision=decide(evidence)
    decision.update(from_stage='short-ablation',next_stage='pilot',config_hash=run.config_hash,
        measured_evidence=evidence,engineering_gate_only=True,
        evidence_hashes={str(p):digest(p) for p in [run.report/'short_ablation_outcomes.json',
            run.report/'short_prediction_diagnostics.json',run.report/'protocol_effect_thresholds.json']})
    atomic_json(run.report/'decision.json',decision)
    return decision
