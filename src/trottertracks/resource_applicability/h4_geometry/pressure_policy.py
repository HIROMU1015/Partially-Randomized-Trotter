"""Closed pressure profile, activated only by the source-bound user amendment."""
import time
from .identity import require
from .pressure_proposal import evaluate as evaluate_predicate
from .resources import INITIAL_CGROUP_NS_INO

PARAMETERS={
    'schema_version':'h4-memory-pressure-policy-v1',
    'mode':'bounded_host_only',
    'host_full_avg10_stop_percent':1.0,
    'nonroot_full_avg10_stop_percent':0.0,
    'host_only_min_available_bytes':int(120.25*2**30),
    'zero_host_headroom_bytes':16*2**30,
    'observation_max_age_seconds':5,
}

GRACE_PARAMETERS={**PARAMETERS,
    'schema_version':'h4-memory-pressure-policy-v2',
    'mode':'bounded_host_only_grace',
    'host_full_avg10_stop_percent':5.0,
    'host_warning_percent':1.0,
    'host_warning_continuous_seconds':30.0,
    'fresh_launch_host_required_below_percent':1.0,
    'host_only_min_available_bytes':int(152.25*2**30),
    'observation_interval_max_seconds':5,
}

def verify(profile):
    expected=GRACE_PARAMETERS if type(profile) is dict and profile.get('schema_version')=='h4-memory-pressure-policy-v2' else PARAMETERS
    require(type(profile) is dict and set(profile)==set(expected) and
            all(type(profile[k]) is type(v) and profile[k]==v for k,v in expected.items()),
            'closed user-approved pressure profile')
    return dict(profile)


class PressureGuard:
    """Use a complete kernel-observed baseline; retain original OOM counters."""
    def __init__(self,baseline,profile):
        self.profile=verify(profile)
        require(type(baseline) is dict and baseline.get('cgroup_namespace')=='cgroup:[%d]'%INITIAL_CGROUP_NS_INO and
                type(baseline.get('hierarchy')) is list and type(baseline.get('oom_events')) is dict,
                'pressure baseline kernel scope')
        self.paths=[row['path'] for row in baseline['hierarchy'] if row['is_root'] is False]
        self.baseline_oom=dict(baseline['oom_events'])
        self.expected_hierarchy=[(row['path'],row['v2'],row['is_root']) for row in baseline['hierarchy']]
        result=evaluate_predicate(baseline,self.baseline_oom,self.paths,now=baseline['observed_at'])
        require(not result['would_stop'],'pressure baseline: '+str(result['reason']))
        self.grace=None
        if self.profile['schema_version']=='h4-memory-pressure-policy-v2':
            require(result['host_full_avg10']==0 or baseline['available']>=self.profile['host_only_min_available_bytes'],
                    'pressure baseline effective headroom')
            from .pressure_grace import GraceGuard
            # Baseline is the original launch OOM/hierarchy identity, not a
            # timer tick. Worker startup must not create an artificial gap.
            self.grace=GraceGuard(self.baseline_oom,self.paths)

    def decision(self,observation,*,now=None):
        now=time.monotonic() if now is None else now
        try:
            hierarchy=[(row['path'],row['v2'],row['is_root']) for row in observation['hierarchy']]
        except (KeyError,TypeError):hierarchy=None
        if hierarchy!=self.expected_hierarchy:
            return dict(mode=self.profile['mode'],reason='pressure_hierarchy_changed',host_only_exception=False)
        result=(self.grace.evaluate(observation,now=now) if self.grace is not None else
                evaluate_predicate(observation,self.baseline_oom,self.paths,now=now))
        value=dict(mode=self.profile['mode'],reason=result['reason'],
            host_full_avg10=result['host_full_avg10'],
            host_full_avg10_stop_percent=self.profile['host_full_avg10_stop_percent'],
            host_only_min_available_bytes=self.profile['host_only_min_available_bytes'],
            host_only_exception=not result['would_stop'] and result['host_full_avg10']>0)
        if self.grace is not None:
            value.update(host_warning_percent=self.profile['host_warning_percent'],
                host_warning_continuous_seconds=self.profile['host_warning_continuous_seconds'],
                warning_active=result['warning_active'],warning_elapsed_seconds=result['warning_elapsed_seconds'])
        return value
