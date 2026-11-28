"""Common, operational Q3 memory envelopes from accepted resource observations."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from .rpe_host_profiles import q3_cpu_measurement_key
from .rpe_q3_resources import valid_q3_execution_receipt

GIB = 1024**3


def learning_policy(value):
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError('Invalid Q3 memory learning policy')
    defaults = dict(min_samples=2, floor_bytes=GIB, margin_numerator=3,
                    margin_denominator=2, headroom_bytes=0, preferred_methods_by_host={})
    if set(value)-defaults.keys():
        raise ValueError('Unknown Q3 memory learning setting')
    policy = dict(defaults, **value)
    if (any(type(policy[k]) is not int or policy[k]<1 for k in
            ('min_samples','floor_bytes','margin_numerator','margin_denominator'))
            or type(policy['headroom_bytes']) is not int or policy['headroom_bytes']<0
            or policy['margin_numerator']<policy['margin_denominator']
            or not isinstance(policy['preferred_methods_by_host'],dict)
            or any(not isinstance(h,str) or not isinstance(ms,list) or any(not isinstance(m,str) or not m for m in ms)
                   for h,ms in policy['preferred_methods_by_host'].items())):
        raise ValueError('Invalid Q3 memory learning bounds')
    return policy


def apply_policy(distributed, value):
    policy = learning_policy(value)
    pbs = distributed['pbs']
    if policy and not set(policy['preferred_methods_by_host']) <= {w['host_id'] for w in pbs['workers']}:
        raise ValueError('Q3 memory learning preference unknown host')
    distributed.setdefault('pbs_pinned_settings',deepcopy(pbs))
    pbs['q3_memory_learning'] = policy
    if policy:
        pbs['q3_measurement_preference'] = dict(pbs.get('q3_measurement_preference',{}), **policy['preferred_methods_by_host'])


def read(path):
    return json.loads(path.read_bytes(), parse_int=lambda v: int(v) if len(v)<=1000 else v)


def observation(row, leaf):
    """Only a durable accepted, clean, exact-owner Q3 result can teach the controller."""
    if row['state']!='ACCEPTED' or row['request'].get('preparation'):
        return None
    try:
        request=row['request']; host=row['host_id']; attempt=row['attempt_id']
        runtime=row['eligibility']['execution_profiles'][host]
        receipt=read(leaf/'PROCESS_RETURN.json'); outcome=read(leaf/'OUTCOME.json')
        key=q3_cpu_measurement_key(request,runtime)
        cls=q3_cpu_measurement_key(request,runtime,memory_class=True)
        dp=runtime.get('dp_workers')
        cpu=runtime.get('task_cpu_count')
        # Legacy eligibility omits the enclosing worker's reserved CPU count.
        # Use the exact claim envelope retained by JobStore, not a guess.
        assigned=(row.get('detail') or {}).get('pbs_launch_resources',{}).get('cpu')
        if cpu is None:
            cpu=assigned if assigned is not None else dp
        if (type(dp) is not int or type(cpu) is not int or cpu-dp not in (0,1)
                or (assigned is not None and assigned!=cpu)):
            return None
        if not (leaf/'Q3_EXECUTION_RESOURCES.json').is_file():
            return legacy_observation(row,leaf,runtime,receipt,outcome,key,cls,cpu)
        sample=read(leaf/'Q3_EXECUTION_RESOURCES.json')
        if (not key or not valid_q3_execution_receipt(sample,request,receipt,outcome,policy_host=host,attempt=attempt)
                or sample['key']!=key or sample['host_id']!=host or sample['request_sha256']!=row['request_sha256']
                or sample['dp_workers']!=runtime['dp_workers'] or sample['task_cpu_count']!=cpu
                or sample.get('memory_class_key',cls)!=cls):
            return None
        return dict(host=host,measurement_key=key,memory_class_key=cls,method=sample['method'],
                    route_kind=sample['route_kind'],dp_workers=sample['dp_workers'],task_cpu_count=cpu,
                    job_id=row['job_id'],attempt_id=attempt,peak_rss_bytes=sample['peak_rss_bytes'])
    except (OSError,ValueError,KeyError,TypeError):
        return None


def legacy_observation(row,leaf,runtime,receipt,outcome,key,cls,cpu):
    """Accept the previous RSS-only transport without inventing a CPU timing."""
    req=row['request']; route=req['route']; sample=read(leaf/'OWNED_RESOURCES.json')
    if (not key or sample.get('schema')!='RPE_OWNED_Q3_MEMORY_V1'
            or any(sample[k]!=row[k] for k in ('job_id','attempt_id','request_sha256'))
            or any(sample[k]!=req[k] for k in ('method','N_X','N_Y'))
            or (sample['route_kind'],sample['view_commit'])!=(route['kind'],route['view_commit'])
            or (sample['interpreter'],sample['runtime_root'],sample['route_view'],sample['adapter_source'])!=
                (runtime['interpreter'],runtime['root'],runtime['route_views'][route['view_commit']],(runtime.get('adapter_manifest') or {}).get('commit'))
            or (sample['dp_workers'],sample['task_cpu_count'])!=(runtime['dp_workers'],cpu)
            or sample['returncode']!=0 or receipt['returncode']!=0 or receipt.get('signal') is not None
            or type(sample['samples']) is not int or sample['samples']<2
            or type(sample['peak_rss_bytes']) is not int or sample['peak_rss_bytes']<=0
            or outcome['request_id']!=row['job_id'] or outcome['status'] not in
                ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD','INCONCLUSIVE')):
        return None
    return dict(host=row['host_id'],measurement_key=key,memory_class_key=cls,method=req['method'],
        route_kind=route['kind'],dp_workers=runtime['dp_workers'],task_cpu_count=cpu,
        job_id=row['job_id'],attempt_id=row['attempt_id'],peak_rss_bytes=sample['peak_rss_bytes'])


def imported_observation(leaf, host):
    """The independent child's exporter checked the exact bound runtime before export."""
    try:
        sample,request,receipt,outcome=(read(leaf/name) for name in
            ('Q3_EXECUTION_RESOURCES.json','REQUEST.json','PROCESS_RETURN.json','OUTCOME.json'))
        if not (leaf/'IMPORTED.json').is_file() or not valid_q3_execution_receipt(sample,request,receipt,outcome,policy_host=host):
            return None
        cls=sample.get('memory_class_key',sample['key'])
        if not isinstance(cls,str) or len(cls)!=64:return None
        return dict(host=host,measurement_key=sample['key'],memory_class_key=cls,method=sample['method'],
            route_kind=sample['route_kind'],dp_workers=sample['dp_workers'],task_cpu_count=sample['task_cpu_count'],
            job_id=sample['job_id'],attempt_id=sample['attempt_id'],peak_rss_bytes=sample['peak_rss_bytes'])
    except (OSError,ValueError,KeyError,TypeError):
        return None


def envelopes(observations, policy, settings):
    profiles={w['host_id']:w['profile'] for w in settings['workers']}
    groups=defaultdict(dict)
    for item in observations:
        group=tuple(item[k] for k in ('host','memory_class_key','method','route_kind','dp_workers','task_cpu_count'))
        if item['host'] in profiles:
            previous=groups[group].get(item['job_id'])
            if previous is None or item['peak_rss_bytes']>previous['peak_rss_bytes']:
                groups[group][item['job_id']]=item
    rows={}
    for (host,cls,method,route,dp,cpus), jobs in groups.items():
        if len(jobs)<policy['min_samples']:
            continue
        peak=max(item['peak_rss_bytes'] for item in jobs.values())
        n,d=policy['margin_numerator'],policy['margin_denominator']
        memory=max(policy['floor_bytes'],(peak*n+d-1)//d+policy['headroom_bytes'])
        # A too-large measurement is not silently shrunk to fit the node.
        # Keep the larger requirement for admission (no new assignment will fit).
        for key in sorted({item['measurement_key'] for item in jobs.values()}):
            rows.setdefault(key,[]).append(dict(host=host,measurement_key=key,method=method,
                route_kind=route,dp_workers=dp,task_cpu_count=cpus,memory_bytes=memory))
    return rows


def preserve_owners(rows, settings):
    from .rpe_pbs import owned_resource_need
    profiles={w['host_id']:w['profile'] for w in settings['workers']}
    owners={}
    for row in rows:
        host=row['host_id']; attempt=row['attempt_id']
        if (host not in profiles or not attempt or row['request'].get('preparation')
                or row['state'] not in ('ASSIGNED','STARTING','RUNNING','UNKNOWN','COMPLETED_PENDING_TRANSFER')):
            continue
        # New workers already have their durable launch envelope in Queue.
        if (row.get('detail') or {}).get('pbs_launch_resources'):
            continue
        cpu,memory=owned_resource_need(row,profiles[host],settings,host)
        owners[attempt]=dict(host=host,job_id=row['job_id'],request_sha256=row['request_sha256'],cpu=cpu,memory_bytes=memory)
    return owners


class MemoryLearner:
    def __init__(self, path, settings):
        self.path=Path(path)
        self.policy=learning_policy(settings.get('q3_memory_learning'))
        self.observations={}
        if self.policy and self.path.is_file():
            saved=read(self.path)
            if saved.get('schema')!='RPE_Q3_MEMORY_LEARNING_V1':
                raise ValueError('Unknown Q3 memory learning state')
            self.observations=saved['observations']
            settings['q3_owned_reservations']=dict(settings.get('q3_owned_reservations',{}),**saved['owners'])
            settings['q3_learned_memory']=envelopes(self.observations.values(),self.policy,settings)

    def update(self, item, owners, settings):
        return self.update_many((item,), owners, settings)

    def update_many(self, items, owners, settings):
        if not self.policy:
            return False
        observations = dict(self.observations)
        for item in items:
            if item:
                observations.setdefault(item['attempt_id'], item)
        if len(observations) == len(self.observations):
            return False
        from .rpe_q3_coordinator import atomic_json
        learned=envelopes(observations.values(),self.policy,settings)
        snapshots=(preserve_owners(owners,settings) if learned!=settings.get('q3_learned_memory',{})
                   else settings.get('q3_owned_reservations',{}))
        atomic_json(self.path,dict(schema='RPE_Q3_MEMORY_LEARNING_V1',observations=observations,owners=snapshots))
        self.observations=observations
        settings['q3_owned_reservations']=snapshots
        settings['q3_learned_memory']=learned
        return True
