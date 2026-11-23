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
from collections import Counter

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


def controller_cpu_count():
    """Use the central process allowance, including any OS quota."""
    report = resource_snapshot({}, environ={})
    return max(1, int(report['allowed_cpu_count'] or report['machine_cpu_count'] or 1))


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
    memberships = (_read(proc/'self/cgroup') or '').splitlines()
    v1 = []
    for line in memberships:
        parts = line.split(':',2)
        if len(parts) != 3:continue
        if line.startswith('0::'):
            group_path = parts[2].lstrip('/')
        elif parts[1]:
            v1.append((parts[1].split(','),parts[2].lstrip('/')))
    version = 'v1' if v1 and not any(line.startswith('0::') for line in memberships) else 'v2'
    boundary_reads = dict(cpu=False,memory=False)
    missing_controls = []
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
        cpu_raw = _read(directory/'cpu.max')
        quota = (cpu_raw or '').split()
        if len(quota) == 2 and _positive(quota[1]) and (quota[0] == 'max' or _positive(quota[0])):
            boundary_reads['cpu'] = True
            if quota[0] != 'max':
                cpu_bounds.append(dict(kind='CGROUP_CPU_QUOTA', value=float(quota[0])/float(quota[1]), path=str(directory)))
        elif version == 'v2' and memberships and cpu_raw is None:
            # The v2 mount root has no cpu.max. Other groups expose it
            # precisely when their parent delegates the CPU controller.
            enabled = _read(directory.parent/'cgroup.subtree_control')
            if directory == groups:
                readable = _read(directory/'cgroup.controllers') is not None
            else:
                readable = enabled is not None and 'cpu' not in enabled.split() and directory.is_dir()
            if readable:
                boundary_reads['cpu'] = True
            else:
                missing_controls.append('cpu')
        elif cpu_raw is not None:
            missing_controls.append('cpu')
        memory_raw = _read(directory/'memory.max')
        if memory_raw is not None:boundary_reads['memory']=True
        maximum = _positive(memory_raw)
        if maximum:
            memory_bounds.append(dict(kind='CGROUP_MEMORY_MAX', value=int(maximum), path=str(directory)))
            used_raw = _read(directory/'memory.current')
            if used_raw is None:missing_controls.append('memory_usage')
            used = _positive(used_raw) or 0
            memory_remaining.append(max(0, int(maximum-used)))
    if version == 'v1':
        boundary_reads = dict(cpu=False,memory=False)
        for controllers,relative in v1:
            for controller in ('cpu','cpuset','memory'):
                if controller not in controllers:continue
                candidates=[groups/','.join(controllers),groups/controller]
                mount=next((candidate for candidate in candidates if candidate.is_dir()),candidates[0])
                current=mount/relative
                chain=([current]+[p for p in current.parents if p.is_relative_to(mount)]) if current.is_relative_to(mount) else []
                observed=False
                for directory in chain:
                    if controller=='cpu':
                        quota_raw=_read(directory/'cpu.cfs_quota_us')
                        period_raw=_read(directory/'cpu.cfs_period_us')
                        if quota_raw is not None and _positive(period_raw):
                            observed=True;boundary_reads['cpu']=True
                            quota,period=_positive(quota_raw),_positive(period_raw)
                            if quota:cpu_bounds.append(dict(kind='CGROUP_V1_CPU_QUOTA',value=quota/period,path=str(directory)))
                    elif controller=='cpuset':
                        text=_read(directory/'cpuset.cpus')
                        if text is not None:
                            observed=True
                            try:
                                count=len(_cpus(text))
                                if count:cpu_bounds.append(dict(kind='CGROUP_V1_CPUSET',value=count,path=str(directory)))
                            except ValueError:pass
                    else:
                        text=_read(directory/'memory.limit_in_bytes')
                        if text is not None:
                            observed=True;boundary_reads['memory']=True
                            maximum=_positive(text)
                            # Linux uses huge sentinel values for an unlimited v1 group.
                            if maximum and maximum<2**60:
                                memory_bounds.append(dict(kind='CGROUP_V1_MEMORY_MAX',value=int(maximum),path=str(directory)))
                                used_raw=_read(directory/'memory.usage_in_bytes')
                                if used_raw is None:missing_controls.append('memory_usage')
                                used=_positive(used_raw) or 0
                                memory_remaining.append(max(0,int(maximum-used)))
                if not observed:missing_controls.append(controller)
    complete = all(boundary_reads.values()) and not missing_controls
    for name in ('SLURM_CPUS_PER_TASK', 'PBS_NP', 'NSLOTS'):
        amount = _positive(env.get(name))
        if amount: cpu_bounds.append(dict(kind='ALLOCATION_ENV_'+name, value=amount))
    policy = profile.get('slot_policy') or {}
    declared = policy.get('resource_limits') or {}
    cpu_reserve = policy.get('controller_cpu_reserve', 0)
    if type(cpu_reserve) is not int or cpu_reserve < 0:
        raise ValueError('Controller CPU reserve must be a non-negative integer')
    if cpu_reserve and cpu_bounds:
        observed = min(bound['value'] for bound in cpu_bounds)
        cpu_bounds.append(dict(kind='CONTROLLER_CPU_HEADROOM', value=max(1, observed-cpu_reserve)))
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
        cgroup_version=version, allocation_bounds_complete=complete,
        boundary_detection='OBSERVED' if complete else 'PARTIAL_OR_UNREADABLE',
        missing_controls=sorted(set(missing_controls+[name for name,seen in boundary_reads.items() if not seen])),
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


def resource_class(request, evaluator=None):
    preparation = request.get('preparation')
    if preparation:
        return 'preparation:' + preparation['stage'] + ':' + preparation['provider']
    return 'q3:' + ('finite_support' if evaluator == 'RPE_FINITE_SUPPORT_MSE_V1' else 'count_dp')


def measured_resource_class(request, evaluator=None, runtime=None):
    """Group comparable requests for capacity estimates, not result identity."""
    category = resource_class(request, evaluator)
    if request.get('preparation'):
        return category
    runtime = runtime or {}
    shots = tuple(request.get('N_X', ())) + tuple(request.get('N_Y', ()))
    if not shots or any(type(value) is not int or value < 1 for value in shots):
        return category
    adapter = (runtime.get('adapter_manifest') or {}).get('commit', '')
    view = request.get('route', {}).get('view_commit', '')
    return ':'.join(map(str, (category, request.get('method', ''), view, adapter,
                              runtime.get('dp_workers', ''), len(shots), max(shots).bit_length())))


def _measured_class_slots(waiting, offered, peaks, jobs, resources, cpu, memory, explicit):
    """One class per offer, with existing attempts charged at their own size."""
    def footprint(kind):
        sample = peaks[kind]
        cpu_need = min(cpu, math.ceil(max(1, sample.get('dp_workers', 0), sample.get('observed_cpu_cores', 0))))
        return cpu_need, math.ceil(sample['peak_rss_bytes'] * 1.5)

    free_cpu, free_memory = cpu, memory
    occupied = [job for job in jobs if job['state'] in OCCUPIED]
    for job in occupied:
        kind = job.get('resource_class')
        if not peaks.get(kind, {}).get('peak_rss_bytes'):
            return 0, [], len(occupied), len(occupied)
        cpu_need, rss = footprint(kind)
        free_cpu -= cpu_need
        free_memory -= rss
    for key in ('remaining_group_memory_bytes', 'machine_mem_available_bytes'):
        if resources.get(key) is not None:
            free_memory = min(free_memory, resources[key])
    counts = Counter(waiting)
    for kind in counts:
        if kind not in offered:
            continue
        cpu_need, rss = footprint(kind)
        cpu_slots = max(0, int(free_cpu // cpu_need))
        memory_slots = max(0, int(free_memory // rss))
        fits = min(cpu_slots, memory_slots, counts[kind])
        if explicit is not None:
            fits = min(fits, explicit-len(occupied))
        if fits > 0:
            return fits, [kind], len(occupied)+cpu_slots, len(occupied)+memory_slots
    return 0, [], len(occupied), len(occupied)


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
    feedback = deepcopy(feedback or state.get('feedback') or {})
    unmeasured_class = False
    offered_classes = None
    measured_demand = feedback.get('capacity_ready_jobs', feedback.get('ready_jobs', 0))
    isolated = None
    pack_classes = policy.get('pack_measured_task_classes') is True
    if policy.get('separate_task_resources'):
        peaks = state.setdefault('task_resource_peaks', {})
        active = set(feedback.get('ready_task_classes', ()))
        occupied_classes = {job.get('resource_class') for job in jobs if job['state'] in OCCUPIED}
        for job in jobs:
            kind = job.get('resource_class')
            if kind is None:
                continue
            if job['state'] in OCCUPIED:
                active.add(kind)
            entry = peaks.setdefault(kind, {})
            for metric in ('peak_rss_bytes', 'observed_cpu_cores', 'dp_workers'):
                entry[metric] = max(entry.get(metric, 0), job.get(metric, 0))
        active = sorted(active) or state.get('task_resource_classes', [])
        state['task_resource_classes'] = active
        if 'ready_task_class_order' in feedback:
            waiting = feedback['ready_task_class_order']
            cold_running = {kind for kind in occupied_classes if kind is not None and
                            not peaks.get(kind, {}).get('peak_rss_bytes')}
            previous_trial = state.get('isolated_resource_class')
            if previous_trial is not None and previous_trial in occupied_classes:
                isolated = previous_trial
            elif cold_running:
                isolated = min(cold_running)
            elif waiting and not peaks.get(waiting[0], {}).get('peak_rss_bytes') and (not pack_classes or not occupied):
                # Give the highest-priority cold request a turn after current
                # work drains. Do not use a measured offer to start it.
                isolated = waiting[0]
            if isolated is not None:
                offered_classes = [isolated]
                state['isolated_resource_class'] = isolated
            else:
                state.pop('isolated_resource_class', None)
                offered_classes = sorted({k for k in waiting if peaks.get(k, {}).get('peak_rss_bytes')})
                measured_demand = sum(k in offered_classes for k in waiting)
            active = sorted(set(offered_classes).union(k for k in occupied_classes if k is not None))
        selected = [peaks.get(kind, {}) for kind in active]
        peak = max((job.get('peak_rss_bytes', 0) for job in selected), default=0)
        # Keep every relevant measured peak even if another class is new.
        unmeasured_class = any(not job.get('peak_rss_bytes') for job in selected)
        observed_cpu = max((job.get('observed_cpu_cores', 0) for job in selected), default=0)
        dp = max([int(profile.get('dp_workers') or 1)] + [int(job.get('dp_workers') or 1) for job in selected])
    else:
        peak = max([float(state.get('peak_job_rss_bytes', 0))] + [float(j.get('peak_rss_bytes', 0)) for j in jobs])
        dp = max([int(profile.get('dp_workers') or 1)] + [int(j.get('dp_workers') or 1) for j in jobs])
        observed_cpu = max([float(state.get('peak_job_cpu_cores', 0))] + [float(j.get('observed_cpu_cores', 0)) for j in jobs])
    cpu = _positive(resources.get('allowed_cpu_count'))
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
    if resources.get('allocation_bounds_complete') is False or unmeasured_class:
        ceiling = min(ceiling, initial)
    if isolated is not None:
        ceiling = min(ceiling, 1)
    reasons = []; stage = state.get('stage', 'COLD_START_UNMEASURED')
    if ceiling < target:
        target = ceiling; state.pop('trial', None); stage = 'RESOURCE_REDUCTION_DRAIN'
        reasons.append('Observed process allowance or memory headroom decreased; existing jobs drain')
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
        if not trial and peak and target < ceiling and int(measured_demand) > target and rate:
            state['trial']=dict(from_slots=target, rate=rate, decision_wait_per_result=latency)
            target += 1; stage='RESOURCE_BOUNDED_EXPLORATORY_TRIAL'
            reasons.append('One extra slot trial after usable completion with resource headroom and READY work')
        state['baseline']=dict(rate=rate, decision_wait_per_result=latency, slots=target)
    if not same_domain or progress:
        state['rate_anchor'] = deepcopy(feedback)
    if resources.get('allocation_bounds_complete') is False:
        reasons.append('Allocation boundary detection incomplete; do not grow above the initial offer')
    if isolated is not None:
        stage = 'ISOLATED_RESOURCE_CLASS_TRIAL'
        reasons.append('A class without a resource sample runs without other work on the host')
    if not reasons:
        reasons.append('Insufficient new usable completions for another slot trial' if peak else 'Cold start: per-job memory and useful throughput not measured')
    if (pack_classes and offered_classes is not None and isolated is None and cpu and memory
            and resources.get('allocation_bounds_complete') is not False and not unmeasured_class):
        slots, offered_classes, cpu_limit, memory_limit = _measured_class_slots(
            waiting, offered_classes, peaks, jobs, resources, cpu, memory, explicit)
        target = occupied + slots
        state.pop('trial', None)
        stage = 'MEASURED_TASK_HEADROOM'
        reasons = ['Offer fitting work by task class after reserving the costs of existing attempts']
    state.update(schema='RPE_Q3_SLOT_STATE_V1', target_slots=target, peak_job_rss_bytes=peak, peak_job_cpu_cores=observed_cpu,
                 stage=stage, feedback=feedback, decision_clock=stamp(), monotonic=now)
    report=dict(schema=SCHEMA, mode='AUTO', capacity=target, slots=max(0,target-occupied),
                occupied=occupied, stage=stage, reasons=reasons, resources=resources,
                limits=dict(cpu_slots=cpu_limit,memory_slots=memory_limit,explicit_max_slots=explicit,memory_sample_headroom_factor=1.5),
                dp_workers_per_q3=dp, peak_job_rss_bytes=peak, peak_job_cpu_cores=observed_cpu, feedback=feedback,
                memory_estimate_scope='SAMPLED_OWNED_RSS_HEURISTIC_NOT_A_PEAK_MEMORY_GUARANTEE',
                trial=state.get('trial'), running_jobs_may_exceed_reduced_capacity=occupied>target)
    if offered_classes is not None:
        report['admitted_task_classes'] = offered_classes
    if isolated is not None:
        report['isolated_resource_class'] = isolated
    return state, report
