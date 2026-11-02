"""Pure, inactive host-only PSI amendment evaluator. Never starts a process."""
import math
from .identity import require, Stop
from .resources import HEADROOM, INITIAL_CGROUP_NS_INO

HOST_STOP_PERCENT = 1.0
HOST_ONLY_MIN_AVAILABLE = int(120.25 * 2**30)
POLICY = {
    'schema_version':'h4-host-only-pressure-proposal-v1',
    'host_full_avg10_stop_percent':HOST_STOP_PERCENT,
    'host_only_min_available_bytes':HOST_ONLY_MIN_AVAILABLE,
    'nonroot_full_avg10_stop_percent':0.0,
    'observation_max_age_seconds':5,
    'approved':False,
    'runtime_authorization':False,
    'production_wiring_present':False,
}


def evaluate(observation, baseline_oom, expected_nonroot_paths, *, now):
    """Evaluate the proposed PSI exception; all live role/owner guards stay external.

    Only a complete initial-namespace v2 hierarchy can qualify. Missing data,
    any non-root PSI, OOM delta or insufficient effective headroom remain STOP.
    An allowed result is a policy simulation, never runtime authorization.
    """
    def result(reason, *, host=None):
        return dict(schema_version=POLICY['schema_version'],
            would_stop=reason is not None,reason=reason,
            host_full_avg10=host,host_full_avg10_stop_percent=HOST_STOP_PERCENT,
            host_only_min_available_bytes=HOST_ONLY_MIN_AVAILABLE,
            approved=False,runtime_authorization=False,
            production_wiring_present=False)
    try:
        require(type(observation) is dict and type(now) in (int,float) and
                math.isfinite(now),'pressure observation/type')
        age=now-observation['observed_at']
        require(type(observation['observed_at']) in (float,int) and math.isfinite(age) and
                0<=age<=5,'pressure observation age')
        available=observation['available']
        require(type(available) is int and available>=0,'pressure available memory')
        scopes=observation['psi_full_by_scope'];hierarchy=observation['hierarchy']
        require(type(scopes) is dict and type(hierarchy) is list and
                observation['cgroup_namespace']=='cgroup:[%d]'%INITIAL_CGROUP_NS_INO,
                'complete initial cgroup namespace')
        require(type(expected_nonroot_paths) is list and expected_nonroot_paths and
                all(type(p) is str and p.startswith('/sys/fs/cgroup/') for p in expected_nonroot_paths) and
                len(set(expected_nonroot_paths))==len(expected_nonroot_paths),'nonroot scope baseline')
        require(all(type(row) is dict and row.get('v2') is True and type(row.get('is_root')) is bool and
                type(row.get('path')) is str for row in hierarchy),'v2 hierarchy proof')
        nonroot=[r['path'] for r in hierarchy if not r['is_root']]
        roots=[r for r in hierarchy if r['is_root']]
        require(len(roots)==1 and roots[0]['path']=='/sys/fs/cgroup' and
                len(set(nonroot))==len(nonroot) and set(nonroot)==set(expected_nonroot_paths),
                'exact visible nonroot hierarchy')
        require(set(scopes)=={'host',*nonroot},'missing/extra pressure scope')
        require(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=100
                    for v in scopes.values()),'invalid pressure value')
        require(type(observation['psi_full_avg10']) in (float,int) and
                observation['psi_full_avg10']==max(scopes.values()),'aggregated pressure binding')
        require(type(baseline_oom) is dict and set(baseline_oom)==set(nonroot) and
                all(type(v) is int and v>=0 for v in baseline_oom.values()),'OOM baseline coverage')
        events=observation['oom_events']
        require(type(events) is dict and set(events)==set(baseline_oom) and
                all(type(v) is int and v>=0 for v in events.values()),'OOM scope coverage')
        host=scopes['host']
        if available<HEADROOM:return result('memory_headroom',host=host)
        if events!=baseline_oom:return result('oom_or_changed_cgroup',host=host)
        if any(scopes[p]!=0 for p in nonroot):return result('nonroot_memory_pressure',host=host)
        if host>=HOST_STOP_PERCENT:return result('host_memory_pressure',host=host)
        if host>0 and available<HOST_ONLY_MIN_AVAILABLE:
            return result('host_pressure_insufficient_effective_headroom',host=host)
        return result(None,host=host)
    except (Stop,KeyError,TypeError,ValueError,OverflowError) as exc:
        return result('pressure_observation_invalid: '+str(exc)[:160])
