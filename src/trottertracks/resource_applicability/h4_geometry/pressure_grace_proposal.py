"""Inactive bounded host-only PSI grace proposal. Never authorizes a run."""
import math
from .identity import Stop,require
from .pressure_proposal import evaluate as original_evaluate

PARAMETERS=dict(schema_version='h4-host-pressure-grace-proposal-v1',
    host_warning_percent=1.0,host_immediate_stop_percent=5.0,
    host_warning_continuous_seconds=30.0,
    nonroot_full_avg10_stop_percent=0.0,
    host_only_min_available_bytes=int(152.25*2**30),
    observation_max_age_seconds=5,observation_interval_max_seconds=5,
    approved=False,runtime_authorization=False,production_wiring_present=False)


class ProposedGraceGuard:
    """O(visible cgroup scopes), monotonic elapsed time, no sleep or writes.

    A warning gets grace only while every original scope/OOM/freshness check
    passes and effective available memory stays above the four-worker budget.
    One invalid observation ends grace; the first failure remains latched.
    """
    def __init__(self,baseline_oom,expected_nonroot_paths):
        self.baseline_oom=dict(baseline_oom)
        self.paths=list(expected_nonroot_paths)
        self.last=None;self.warning_since=None;self.hierarchy=None;self.first_failure=None

    def evaluate(self,observation,*,now):
        host=None;reason=None;warning=False
        if self.first_failure is not None:return self._result(self.first_failure,host,None,False)
        try:
            require(type(now) in (int,float) and math.isfinite(now),'grace monotonic time')
            if self.last is not None:
                require(0<now-self.last<=PARAMETERS['observation_interval_max_seconds'],
                        'grace observation interval')
            self.last=now
            original=original_evaluate(observation,self.baseline_oom,self.paths,now=now)
            host=original['host_full_avg10']
            reason=original['reason']
            if reason not in (None,'host_memory_pressure'):return self._stop(reason,host,now)
            hierarchy=tuple((row['path'],row['v2'],row['is_root']) for row in observation['hierarchy'])
            if self.hierarchy is None:self.hierarchy=hierarchy
            require(hierarchy==self.hierarchy,'grace hierarchy changed')
            if host>0 and observation['available']<PARAMETERS['host_only_min_available_bytes']:
                return self._stop('host_pressure_insufficient_effective_headroom',host,now)
            if host>=PARAMETERS['host_immediate_stop_percent']:
                return self._stop('host_memory_pressure_severe',host,now)
            warning=host>=PARAMETERS['host_warning_percent']
            if warning:
                if self.warning_since is None:self.warning_since=now
                elapsed=now-self.warning_since
                if elapsed>=PARAMETERS['host_warning_continuous_seconds']:
                    return self._stop('host_memory_pressure_sustained',host,now)
            else:self.warning_since=None;elapsed=0.0
            return self._result(None,host,elapsed,warning)
        except (Stop,KeyError,TypeError,ValueError,OverflowError) as exc:
            return self._stop('pressure_observation_invalid: '+str(exc)[:160],host,now)

    def _stop(self,reason,host,now):
        self.first_failure=reason
        elapsed=None if self.warning_since is None or type(now) not in (int,float) or not math.isfinite(now) else now-self.warning_since
        return self._result(reason,host,elapsed,False)

    def _result(self,reason,host,elapsed,warning):
        return dict(schema_version=PARAMETERS['schema_version'],would_stop=reason is not None,
            reason=reason,host_full_avg10=host,warning_active=warning,
            warning_elapsed_seconds=elapsed,first_failure=self.first_failure,
            approved=False,runtime_authorization=False,production_wiring_present=False)
