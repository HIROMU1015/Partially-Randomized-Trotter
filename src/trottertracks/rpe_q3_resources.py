"""Read-only resource offers and gradual slot trials for one owned Q3 worker.

Affinity/cgroup/allocation bounds describe what this process may use; host-wide
idle memory is only a pressure observation. No scheduler or OS control is changed.
The cold-start offer is empirical, and never presented as measured H4 capacity.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

from .rpe_q3_timing import stamp


SCHEMA = 'RPE_Q3_RESOURCE_OFFER_V1'
OCCUPIED = {'ASSIGNED', 'STARTING', 'RUNNING', 'UNKNOWN', 'COMPLETED_PENDING_TRANSFER'}


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _positive(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def _cpus(value):
    result = set()
    for item in (value or '').split(','):
        if not item:
            continue
        parts = item.split('-')
        lo, hi = int(parts[0]), int(parts[-1])
        if lo < 0 or hi < lo or hi-lo > 1000000:
            raise ValueError('Invalid CPU set')
        result.update(range(lo, hi+1))
    return result


def resource_snapshot(profile, *, proc_root='/proc', cgroup_root='/sys/fs/cgroup', affinity=None, environ=None):
    """Observe process/allocation bounds; do not equate idle host CPUs to a grant."""
    proc, groups = Path(proc_root), Path(cgroup_root)
    env = os.environ if environ is None else environ
    if affinity is None:
        try:
            affinity = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            affinity = None
    cpu_bounds = []
    if affinity:
        cpu_bounds.append(dict(kind='PROCESS_AFFINITY_NONEXCLUSIVE', value=len(affinity)))
    group_path = ''
    for line in (_read(proc/'self/cgroup') or '').splitlines():
        if line.startswith('0::'):
            group_path = line[3:].lstrip('/')
    selected = groups/group_path
    # cgroup v2 limits can be imposed by any ancestor of this process's group.
    ancestors = []
    if selected.is_relative_to(groups):
        ancestors = [selected] + [p for p in selected.parents if p.is_relative_to(groups)]
    memory_bounds, memory_remaining = [], []
    for directory in ancestors:
        cpu_set = _read(directory/'cpuset.cpus.effective')
        if cpu_set:
            try:
                count = len(_cpus(cpu_set))
                if count: cpu_bounds.append(dict(kind='CGROUP_CPUSET', value=count, path=str(directory)))
            except ValueError:
                pass
        quota = (_read(directory/'cpu.max') or '').split()
        if len(quota) == 2 and quota[0] != 'max':
            q, period = _positive(quota[0]), _positive(quota[1])
            if q and period: cpu_bounds.append(dict(kind='CGROUP_CPU_QUOTA', value=q/period, path=str(directory)))
        maximum = _positive(_read(directory/'memory.max'))
        if maximum:
            memory_bounds.append(dict(kind='CGROUP_MEMORY_MAX', value=int(maximum), path=str(directory)))
            used = _positive(_read(directory/'memory.current')) or 0
            memory_remaining.append(max(0, int(maximum-used)))
    for name in ('SLURM_CPUS_PER_TASK', 'PBS_NP', 'NSLOTS'):
        amount = _positive(env.get(name))
        if amount: cpu_bounds.append(dict(kind='ALLOCATION_ENV_'+name, value=amount))
    policy = profile.get('slot_policy') or {}
    declared = policy.get('resource_limits') or {}
    amount = _positive(declared.get('cpu_count'))
    if amount: cpu_bounds.append(dict(kind='DECLARED_RESEARCH_CPU_BOUND', value=amount))
    meminfo = {}
    for line in (_read(proc/'meminfo') or '').splitlines():
        key, _, value = line.partition(':')
        try: meminfo[key] = int(value.split()[0])*1024
        except (ValueError, IndexError): pass
    amount = _positive(declared.get('memory_bytes'))
    if amount:
        memory_bounds.append(dict(kind='DECLARED_RESEARCH_MEMORY_BOUND', value=int(amount)))
    # A scheduler's per-task memory grant is a bound, not global MemAvailable.
    for name in ('SLURM_MEM_PER_NODE',):
        amount = _positive(env.get(name))
        if amount: memory_bounds.append(dict(kind='ALLOCATION_ENV_'+name, value=int(amount*1024*1024)))
    return dict(schema='RPE_Q3_RESOURCE_SNAPSHOT_V1', clock=stamp(),
        machine_cpu_count=os.cpu_count(), machine_mem_total_bytes=meminfo.get('MemTotal'),
        machine_mem_available_bytes=meminfo.get('MemAvailable'),
        allowed_cpu_count=min((x['value'] for x in cpu_bounds), default=None), cpu_bounds=cpu_bounds,
        allowed_memory_bytes=min((x['value'] for x in memory_bounds), default=meminfo.get('MemTotal')),
        memory_bounds=memory_bounds,
        remaining_group_memory_bytes=min(memory_remaining, default=None),
        memory_scope='BOUNDED_ALLOCATION' if memory_bounds else 'HOST_PHYSICAL_NONEXCLUSIVE_NO_RESERVATION',
        cpu_scope='PROCESS_ALLOWANCE_NONEXCLUSIVE_NOT_MACHINE_IDLE_CPUS',
        observational_only=True)


def owned_tree_sample(pid, *, proc_root='/proc'):
    """Sample only explicitly owned supervisor/child PIDs, without process scans."""
    proc = Path(proc_root); todo = [pid]; seen = set(); rss = 0; cpu = 0.
    ticks = os.sysconf('SC_CLK_TCK')
    while todo:
        current = todo.pop()
        if type(current) is not int or current <= 0 or current in seen: continue
        seen.add(current)
        directory = proc/str(current)
        for line in (_read(directory/'status') or '').splitlines():
            if line.startswith('VmRSS:'):
                rss += int(line.split()[1])*1024
        text = _read(directory/'stat')
        if text and ')' in text:
            values = text.rsplit(')', 1)[1].split()
            try: cpu += (float(values[11])+float(values[12]))/ticks
            except (IndexError, ValueError): pass
        try: todo.extend(map(int, (_read(directory/'task'/str(current)/'children') or '').split()))
        except ValueError: pass
    return dict(rss_bytes=rss, cpu_seconds=cpu, sampled_pids=len(seen))


def decide_slots(profile, previous, resources, jobs, feedback=None, *, now=None):
    """Pure bounded one-slot trials; reductions drain, and never signal a job."""
    now = time.monotonic() if now is None else float(now)
    policy = profile.get('slot_policy') or {}
    initial = policy.get('initial_slots', profile.get('slots', 1))
    if type(initial) is not int or initial < 1: raise ValueError('Positive initial_slots required')
    state = deepcopy(previous or {})
    occupied = sum(job.get('state') in OCCUPIED for job in jobs)
    if policy.get('mode') != 'auto':
        occupied = sum(job.get('state') in ('ASSIGNED','STARTING','RUNNING') for job in jobs)
        capacity = profile.get('slots', 1)
        if type(capacity) is not int or capacity < 1: raise ValueError('Positive fixed slots required')
        return state, dict(schema=SCHEMA, mode='FIXED', capacity=capacity,
            slots=max(0,capacity-occupied), occupied=occupied, stage='EXPLICIT_FIXED_PROFILE', reasons=['Legacy/explicit worker slots retained'])
    target = max(1, int(state.get('target_slots', initial)))
    peak = max([float(state.get('peak_job_rss_bytes', 0))] + [float(j.get('peak_rss_bytes', 0)) for j in jobs])
    dp = max([int(profile.get('dp_workers') or 1)] + [int(j.get('dp_workers') or 1) for j in jobs])
    cpu = _positive(resources.get('allowed_cpu_count'))
    observed_cpu = max([float(state.get('peak_job_cpu_cores', 0))] + [float(j.get('observed_cpu_cores', 0)) for j in jobs])
    cpu_per_job = max(dp, observed_cpu)
    cpu_limit = max(1, int(cpu//min(cpu_per_job,cpu))) if cpu else initial
    memory = resources.get('allowed_memory_bytes')
    free = [x for x in (resources.get('remaining_group_memory_bytes'), resources.get('machine_mem_available_bytes')) if x is not None]
    # Peak includes sampled owned children. Reserve one measured job footprint
    # of headroom while increasing; no estimate is manufactured before sampling.
    memory_limit = int(memory//(peak*1.5)) if memory and peak else initial
    if free and peak:
        memory_limit = min(memory_limit, occupied+int(max(0,min(free))//(peak*1.5)))
    limits = [max(0,cpu_limit), max(0,memory_limit)]
    explicit = policy.get('max_slots')
    if explicit is not None:
        if type(explicit) is not int or explicit < 1: raise ValueError('Positive max_slots required')
        limits.append(explicit)
    ceiling = min(limits)
    reasons = []; stage = state.get('stage', 'COLD_START_UNMEASURED')
    if ceiling < target:
        target = ceiling; state.pop('trial', None); stage = 'RESOURCE_REDUCTION_DRAIN'
        reasons.append('Observed process allowance or memory headroom decreased; existing jobs drain')
    feedback = deepcopy(feedback or state.get('feedback') or {})
    prior = state.get('rate_anchor') or {}
    domain = feedback.get('controller_invocation')
    same_domain = domain is not None and domain == prior.get('controller_invocation')
    count = int(feedback.get('usable_results', 0)); before_count = int(prior.get('usable_results', 0))
    elapsed = _positive(feedback.get('measurement_seconds'))
    previous_elapsed = _positive(prior.get('measurement_seconds')) or 0
    progress = same_domain and count > before_count and elapsed and elapsed > previous_elapsed
    rate = (count-before_count)/(elapsed-previous_elapsed) if progress else None
    wait = feedback.get('decision_wait_seconds'); prior_wait = prior.get('decision_wait_seconds')
    latency = max(0,float(wait)-float(prior_wait))/(count-before_count) if progress and wait is not None and prior_wait is not None else None
    if progress:
        baseline = state.get('baseline')
        trial = state.get('trial')
        if trial and rate is not None:
            base_rate = float(trial.get('rate') or 0)
            base_wait = trial.get('decision_wait_per_result')
            no_rate_regression = rate >= base_rate
            no_wait_regression = base_wait is None or latency is None or latency <= base_wait
            improvement = rate > base_rate or (base_wait is not None and latency is not None and latency < base_wait)
            useful = no_rate_regression and no_wait_regression and improvement
            if not useful:
                target = max(1,min(ceiling,int(trial['from_slots']))); stage='MEASURED_TRIAL_REVERT_DRAIN'
                reasons.append('Extra slot did not improve usable result rate/selected-result latency')
                state.pop('trial',None)
            else:
                stage='MEASURED_THROUGHPUT_AND_WAIT' if latency is not None else 'MEASURED_THROUGHPUT_WAIT_UNMEASURED'
                reasons.append('Trial retained using new usable results; latency missing is recorded')
                state.pop('trial',None)
        else:
            stage='MEASURED_THROUGHPUT_AND_WAIT' if latency is not None else 'MEASURED_THROUGHPUT_WAIT_UNMEASURED'
        # A one-slot trial establishes the next measurement, rather than using
        # machine-wide free CPU as evidence that more Q3s are useful.
        if not trial and peak and target < ceiling and int(feedback.get('ready_jobs',0)) > 0 and rate:
            state['trial']=dict(from_slots=target, rate=rate, decision_wait_per_result=latency)
            target += 1; stage='RESOURCE_BOUNDED_EXPLORATORY_TRIAL'
            reasons.append('One extra slot trial after usable completion with resource headroom and READY work')
        state['baseline']=dict(rate=rate, decision_wait_per_result=latency, slots=target)
    if not same_domain or progress:
        state['rate_anchor'] = deepcopy(feedback)
    if not reasons:
        reasons.append('Insufficient new usable completions for another slot trial' if peak else 'Cold start: per-job memory and useful throughput not measured')
    state.update(schema='RPE_Q3_SLOT_STATE_V1', target_slots=target, peak_job_rss_bytes=peak, peak_job_cpu_cores=observed_cpu,
                 stage=stage, feedback=feedback, decision_clock=stamp(), monotonic=now)
    report=dict(schema=SCHEMA, mode='AUTO', capacity=target, slots=max(0,target-occupied),
                occupied=occupied, stage=stage, reasons=reasons, resources=resources,
                limits=dict(cpu_slots=cpu_limit,memory_slots=memory_limit,explicit_max_slots=explicit,memory_sample_headroom_factor=1.5),
                dp_workers_per_q3=dp, peak_job_rss_bytes=peak, peak_job_cpu_cores=observed_cpu, feedback=feedback,
                memory_estimate_scope='SAMPLED_OWNED_RSS_HEURISTIC_NOT_A_PEAK_MEMORY_GUARANTEE',
                trial=state.get('trial'), running_jobs_may_exceed_reduced_capacity=occupied>target)
    return state, report
