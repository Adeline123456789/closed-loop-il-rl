"""Predeclared paired-seed decisions and complete marginal robot metrics."""
import csv
import math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,norm

def read_rows(path):
    path=Path(path)
    return list(csv.DictReader(path.open())) if path.exists() else []

def arm_metrics(rows):
    if not rows:return dict(status='NOT_RUN',episodes=0,success_rate=None)
    n=len(rows);k=sum(int(r['stable_success']) for r in rows);p=k/n;z=norm.ppf(.975)
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    numeric=lambda name:np.asarray([float(r[name]) for r in rows])
    steps=numeric('steps');times=[float(r['steps'])*float(r['control_dt']) for r in rows if int(r['stable_success'])]
    reasons={reason:sum(r['termination_reason']==reason for r in rows) for reason in sorted({r['termination_reason'] for r in rows})}
    return dict(status='PASSED',episodes=n,successes=k,success_rate=p,wilson95=[float(center-radius),float(center+radius)],
        mean_return=float(numeric('return').mean()),mean_cost=float(numeric('cost').mean()),
        mean_peak_force_N=float(numeric('peak_force_N').mean()),max_peak_force_N=float(numeric('peak_force_N').max()),
        mean_peak_torque_Nm=float(numeric('peak_torque_Nm').mean()),max_peak_torque_Nm=float(numeric('peak_torque_Nm').max()),
        overload_episode_rate=float(np.mean(numeric('overload_steps')>0)),
        overload_step_rate=float(numeric('overload_steps').sum()/steps.sum()),
        protection_stop_rate=sum(r['termination_reason']=='unsafe' for r in rows)/n,
        termination_reason_counts=reasons,
        failure_reasons={reason:sum(r['termination_reason']==reason and not int(r['stable_success']) for r in rows) for reason in reasons if any(r['termination_reason']==reason and not int(r['stable_success']) for r in rows)},
        mean_first_stable_success_seconds=float(np.mean(times)) if times else None,
        success_time_definition='Time to first 3-step stable-success detection, conditional on success; failures censored',
        evaluation_control_steps=int(steps.sum()))

def matched_decision(a1,a2,seeds,look,plan):
    if look not in (1,2):raise ValueError('At most two decision looks')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Expected nonempty unique preregistered seeds')
    def validated(rows):
        if len(rows)!=len(seeds) or len({int(r['seed']) for r in rows})!=len(rows):raise ValueError('Missing or duplicate seed')
        by_seed={int(r['seed']):r for r in rows}
        if set(by_seed)!=set(seeds):raise ValueError('Changed evaluation starts')
        if any(r['source']!='real_forge' or int(r['stable_success']) not in (0,1) for r in rows):raise ValueError('Non-real or invalid outcomes')
        return np.array([int(by_seed[s]['stable_success']) for s in seeds])
    x,y=validated(a1),validated(a2);d=y-x
    wins=int((d==1).sum());losses=int((d==-1).sum())
    probability=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.
    rng=np.random.default_rng(plan['bootstrap_seed']+look)
    means=d[rng.integers(len(d),size=(plan['bootstrap_resamples'],len(d)))].mean(1)
    alpha=plan['look_alpha'];interval=np.quantile(means,[alpha/2,1-alpha/2])
    delta=float(d.mean());threshold=plan['minimum_success_difference']
    decision='INCONCLUSIVE'
    if probability<=alpha:
        if delta>=threshold and interval[0]>0:decision='GO'
        elif delta<=-threshold and interval[1]<0:decision='NO_GO'
    return dict(status=decision,look=look,episodes_per_arm=len(d),success_difference=delta,
        paired_seed_bootstrap_interval=interval.tolist(),interval_level=1-alpha,exact_discordant_p=probability,
        A2_only_success=wins,A1_only_success=losses,paired_seed_caveat=plan['paired_interpretation'],
        reason='Locked success-effect/interval/exact-test rule applied; equal-seed pairing is not D0 contact-state recovery',
        extension_allowed=decision=='INCONCLUSIVE' and look==1)
