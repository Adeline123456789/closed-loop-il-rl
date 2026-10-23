"""Episode-cluster summaries from raw records, with policy/split/seed separation."""
from collections import defaultdict
import math
import numpy as np

ARMS=('B0','B1','B2','B3')
BINARY=('stable_success','ever_success','final_success','overload')


def numeric(row,key,optional=False):
    raw=row.get(key)
    if optional and raw in (None,'','NA'):return None
    try:value=float(raw)
    except (TypeError,ValueError) as exc:raise ValueError(f'invalid {key}: {raw}') from exc
    if not math.isfinite(value):raise ValueError(f'nonfinite {key}')
    if key in BINARY and value not in (0.,1.):raise ValueError(f'nonbinary {key}')
    return value


def completed_rows(rows):
    accepted=[]
    for row in rows:
        if row.get('status')!='COMPLETED':continue
        if row.get('arm') not in ARMS or row.get('split') not in ('train','validation','test'):
            raise ValueError('invalid arm/split in completed episode')
        if not all(row.get(k) for k in ('parent_episode','policy_sha','seed')):
            raise ValueError('completed episode missing parent/policy/seed')
        for key in BINARY:numeric(row,key)
        if numeric(row,'stable_success')>numeric(row,'ever_success'):
            raise ValueError('stable success without ever success')
        for key in ('env_steps','force_peak','torque_peak','completion_time'):
            value=numeric(row,key,optional=key!='env_steps')
            if value is not None and value<0:raise ValueError(f'negative {key}')
        accepted.append(row)
    return accepted


def cluster_interval(rows,key,seed=0,repeats=2000):
    """Bootstrap parent episodes; repeated rows never become independent parents."""
    clusters=defaultdict(list)
    for row in rows:clusters[row['parent_episode']].append(numeric(row,key))
    if len(clusters)<2:return ('NA','NA')
    sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
    rng=np.random.default_rng(seed)
    indices=rng.integers(len(sums),size=(repeats,len(sums)))
    rates=sums[indices].sum(1)/counts[indices].sum(1)
    return tuple(float(x) for x in np.quantile(rates,[.025,.975]))


def summary_rows(rows):
    rows=completed_rows(rows)
    groups=defaultdict(list)
    for r in rows:groups[(r['arm'],r['seed'],r['split'],r['policy_sha'])].append(r)
    result=[]
    for arm in ARMS:
        keys=sorted(k for k in groups if k[0]==arm) or [(arm,'NA','NA','NA')]
        for key in keys:
            group=groups.get(key,[])
            out=dict(zip(('arm','seed','split','policy_sha'),key))
            out.update(status='COMPLETED' if group else 'NOT_RUN',episodes=len(group),
                       independent_parents=len({r['parent_episode'] for r in group}))
            for field in BINARY:
                out[field+'_rate']=sum(numeric(r,field) for r in group)/len(group) if group else 'NA'
                out[field+'_ci_low'],out[field+'_ci_high']=cluster_interval(group,field) if group else ('NA','NA')
            for field in ('force_peak','torque_peak'):
                values=[numeric(r,field,optional=True) for r in group]
                # Missing load measurements must not turn into an apparently low mean.
                out[field+'_observed_episodes']=sum(x is not None for x in values)
                out[field+'_mean_all_episodes']=float(np.mean(values)) if values and all(x is not None for x in values) else 'NA'
                out[field+'_max_all_episodes']=max(values) if values and all(x is not None for x in values) else 'NA'
            times=[numeric(r,'completion_time',optional=True) for r in group if numeric(r,'stable_success')==1]
            out['successful_completion_time_denominator']=sum(v is not None for v in times)
            out['successful_completion_time_mean']=float(np.mean(times)) if times and all(v is not None for v in times) else 'NA'
            out['env_steps']=sum(numeric(r,'env_steps') for r in group) if group else 0
            result.append(out)
    return result


def paired_comparisons(rows):
    """Match evaluation parents; refuse to silently pool checkpoints or incomplete pairs."""
    rows=completed_rows(rows);result=[]
    for seed,split in sorted({(r['seed'],r['split']) for r in rows if r['split']!='train'}):
        subset=[r for r in rows if (r['seed'],r['split'])==(seed,split)]
        for left,right in (('B1','B0'),('B2','B1'),('B3','B2')):
            groups={arm:[r for r in subset if r['arm']==arm] for arm in (left,right)}
            record={'comparison':left+'-'+right,'seed':seed,'split':split,'status':'INCONCLUSIVE',
                    'paired_parents':0,'success_difference_pp':'NA','success_ci_low_pp':'NA','success_ci_high_pp':'NA',
                    'overload_difference_pp':'NA','reason':'missing or mixed policy versions'}
            if any(not g or len({r['policy_sha'] for r in g})!=1 for g in groups.values()):
                result.append(record);continue
            parents={arm:{r['parent_episode'] for r in g} for arm,g in groups.items()}
            if parents[left]!=parents[right]:
                record['reason']='unmatched evaluation parents; no selective intersection reported'
                result.append(record);continue
            differences=[];overloads=[]
            for parent in sorted(parents[left]):
                means={arm:{metric:np.mean([numeric(r,metric) for r in g if r['parent_episode']==parent])
                            for metric in ('stable_success','overload')} for arm,g in groups.items()}
                differences.append(means[left]['stable_success']-means[right]['stable_success'])
                overloads.append(means[left]['overload']-means[right]['overload'])
            record.update(status='COMPLETED',paired_parents=len(differences),
                          success_difference_pp=float(np.mean(differences)*100),
                          overload_difference_pp=float(np.mean(overloads)*100),reason='paired parent bootstrap; not an automatic GO')
            if len(differences)>1:
                rng=np.random.default_rng(0);x=np.asarray(differences)
                samples=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)*100
                record['success_ci_low_pp'],record['success_ci_high_pp']=map(float,np.quantile(samples,[.025,.975]))
            result.append(record)
    return result
