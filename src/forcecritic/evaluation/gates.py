"""Fail-closed preregistered engineering decisions, not significance claims."""
import math


def decide(e):
    required=['environment_passed','baseline_passed','replay_passed','hard_total','hard_identifiable',
              'critic_judgment_problem','critic_is_primary_cause','independent_prediction_improved',
              'actual_actor_improved','success_difference_pp','success_difference_ci',
              'overload_worse','peak_load_worse','baseline_overload_events','overload_relative_reduction',
              'intervals_informative']
    if any(k not in e or e[k] is None for k in required):
        return {'status':'INCONCLUSIVE','reason':'Required measured evidence missing'}
    if not e['environment_passed'] or not e['baseline_passed']:
        return {'status':'BLOCKED','reason':'Environment/baseline not accepted'}
    if not e['replay_passed']:
        return {'status':'NO_GO_CALIBRATION','reason':'Neither snapshot nor prefix replay accepted'}
    if e['hard_total']<16 or e['hard_identifiable']<4 or e['hard_identifiable']/e['hard_total']<.25:
        return {'status':'INCONCLUSIVE','reason':'Insufficient difficult-state effect coverage'}
    if not e['critic_judgment_problem'] or not e['critic_is_primary_cause']:
        return {'status':'NO_GO_CALIBRATION','reason':'Calibration not shown necessary; direct loop may still be useful'}
    if not e['independent_prediction_improved'] or not e['actual_actor_improved']:
        return {'status':'INCONCLUSIVE','reason':'Prediction and real ACT behavior must both improve'}
    if e['overload_worse'] or e['peak_load_worse']:
        return {'status':'NO_GO_CALIBRATION','reason':'Measured load deterioration'}
    if not e['intervals_informative']:
        return {'status':'INCONCLUSIVE','reason':'Intervals too wide'}
    delta=e['success_difference_pp']
    if not math.isfinite(delta):return {'status':'INCONCLUSIVE','reason':'Nonfinite metric'}
    # At least ten baseline events is a predeclared engineering minimum, not statistical proof.
    alternative=delta>=-3 and e['baseline_overload_events']>=10 and e['overload_relative_reduction']>=.2
    if delta>=5 or alternative:return {'status':'GO_PILOT','reason':'Engineering threshold met; not a significance claim'}
    return {'status':'INCONCLUSIVE','reason':'No incremental calibration signal above fixed threshold'}
