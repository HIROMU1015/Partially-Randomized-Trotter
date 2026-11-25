"""Read-only resource offers and gradual slot trials for one owned Q3 worker.

Affinity/cgroup/allocation bounds describe what this process may use; host-wide
idle memory is only a pressure observation. No scheduler or OS control is changed.
The cold-start offer is empirical, and never presented as measured H4 capacity.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
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



def _fixed_workload_document(root, name, sha):
    """Read only the declared immutable planning input (never scientific execution)."""
    path = Path(root) / name.removeprefix('workspace/')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('Workload input hash differs')
    return json.loads(raw)


def preparation_workload(request):
    """Physical work counts, independent of configuration ID or target host.

    Unknown/older schemas retain their existing resource policy. Counts are
    operational proxies, not RZ estimates or scientific admission decisions.
    """
    prep = request.get('preparation') or {}
    inputs = prep.get('inputs', {})
    if not isinstance(inputs, dict) or not prep.get('controller_root'):
        return None
    candidates = [(name, sha) for name, sha in inputs.items() if name.startswith('workspace/') and
                  (name.endswith('/df_prices/INPUTS.json') or name.endswith('/PF_COMPILE.INPUT.json'))]
    if len(candidates) != 1:
        return None
    name, sha = candidates[0]
    return _workload_from_input(prep['controller_root'], name, sha, prep.get('stage'), prep.get('provider'),
                                (prep.get('source') or {}).get('commit'), bool(prep.get('parallel_df_price_axes')),
                                prep.get('df_event_layout'))


@lru_cache(maxsize=2048)
def _workload_from_input(root, name, sha, stage, provider, source, parallel_axes, df_event_layout=None):
    try:
        doc = _fixed_workload_document(root, name, sha)
        cells = []; event_shapes = []
        if name.endswith('/df_prices/INPUTS.json'):
            family = 'DF_EVENT_COUNT_V1'
            for cell in doc['cells']:
                req = cell['request']; ctx = req['physical_context']; physical = ctx['physical_candidate']
                binding = ctx['binding']
                shape = [physical['system'], physical['method'], physical['formula'], binding['basis'],
                         binding['rank'], binding['compiler'], binding['control']]
                count = ctx['structure']['event_count']
                if type(count) is not int or count < 1:
                    return None
                paths = len(cell['indices']); axes = len(req['axes'])
                if paths and axes:
                    cells.append((shape, count, paths, axes, count + physical['q'] * (physical['prefix'] + 1)))
                    event_shapes.append((physical['prefix'], physical.get('K')))
        else:
            family = 'PF_STEP_COUNT_V1'
            configuration = doc['value']['configuration']; spec = configuration['spec']; binding = spec['binding']
            shape = [binding['system'], spec['method'], spec['formula'], binding['basis'],
                     binding['rank'], binding['compiler'], binding['control']]
            rounds = {r['m']: r for r in configuration['rounds']}
            for m in doc['rounds']:
                physical = rounds[m].get('physical', rounds[m]); count = physical.get('q_PF')
                axes = len(doc['axes_by_round'][str(m)])
                if type(count) is not int or count < 1:
                    return None
                if axes:cells.append((shape, count, 1, axes, count * (spec['L_D'] + 1)))
        if not cells or any(c[0] != cells[0][0] for c in cells):
            return None
        result = dict(family=family, shape=cells[0][0], stage=stage, provider=provider, source=source,
                    peak_units=max(c[4] for c in cells), total_units=sum(c[4]*c[2]*c[3] for c in cells),
                    path_units=sum(c[1]*c[2] for c in cells), axis_jobs=sum(c[2]*c[3] for c in cells),
                    path_count=sum(c[2] for c in cells),
                    parallel_axes=parallel_axes)
        if df_event_layout and family == 'DF_EVENT_COUNT_V1':
            result['df_event_layout'] = df_event_layout
            result['event_shapes'] = [list(x) for x in sorted(set(event_shapes),key=repr)]
            result['path_workers'] = result['path_count']
        return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
        return None



@lru_cache(maxsize=1024)
def _event_path_counts(root, name, sha):
    doc = _fixed_workload_document(root, name, sha)
    return [(cell['m'], cell['request']['physical_context']['structure']['event_count'], tuple(cell['indices']))
            for cell in doc['cells']]


def owned_preparation_observation(attempt, assignment, profile, peak):
    """Read only size/usage from this attempt; no new result or raw file is made."""
    request = assignment['request']; prep = request.get('preparation') or {}
    if not prep or type(peak) is not int or peak <= 0:return None
    observation = dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                       request_sha256=assignment['request_sha256'], peak_rss_bytes=peak,
                       cpu_count=max(1, profile.get('task_cpu_count') or profile.get('dp_workers') or 1))
    attempt = Path(attempt)
    # Count each completed raw path once, even with both axis hard links.
    inputs = prep.get('inputs') or {}
    if not isinstance(inputs, dict):return observation
    for name, sha in inputs.items():
        if not name.startswith('workspace/') or not name.endswith('/df_prices/INPUTS.json'):continue
        try:
            cells = _event_path_counts(str(attempt/'workspace'), name, sha)
            base = (attempt/name).parent
            data = base/'calibration'/'acquisition'
            if not data.exists():data = base/'holdout'/'acquisition'
            units = size = 0
            for m, count, indices in cells:
                for index in indices:
                    path = data/f'm{m}'/f'p{index}'
                    # Paired workers store a single axes_XY trajectory. Old
                    # split workers may have X/Y hard links or separate files.
                    candidates=[path/'events_PRIVATE.json', *[path/axis/'events_PRIVATE.json'
                                for axis in ('axes_XY','axes_X','axes_Y')]]
                    unique={}
                    for event_file in candidates:
                        if not event_file.is_file():continue
                        parent=event_file.parent
                        if not ((parent/'STRUCTURAL.json').is_file() or
                                parent==path and (path/'SHARED_EVENTS.json').is_file()):continue
                        stat=event_file.stat();unique[(stat.st_dev,stat.st_ino)]=stat.st_size
                    if unique:
                        units += count;size += sum(unique.values())
            if units and size:observation.update(stored_path_units=units, unique_event_bytes=size)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):pass
    try:
        receipt = json.loads((attempt/'PROCESS_RETURN.json').read_text())
        outcome = json.loads((attempt/'OUTCOME.json').read_text())
        observation['completed'] = (receipt.get('returncode') == 0 and receipt.get('signal') is None
                                    and outcome.get('status') == 'PREPARATION_COMPLETE')
        elapsed = receipt.get('wall_seconds')
        if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0:
            observation['elapsed_seconds'] = elapsed
    except (OSError, ValueError, TypeError):pass
    return observation

def _near_scale_envelope(target, samples, *, margin=1.0):
    """Fixed overhead plus variable workload; never let a tiny sample dominate.

    All values are normalized to comparable workers before fitting. Adjacent
    logarithmic scales (including the closest observed scale when outside the
    range) determine the variable term. A single scale cannot identify an
    intercept, so retain conservative proportional growth in that case.
    """
    if not samples:return None
    distinct = {size for size, _ in samples}
    if len(distinct) == 1:
        return margin * max(value * max(1, target/size) for size, value in samples)
    overhead = min(value for _, value in samples)
    distance = min(abs(math.log2(target/size)) for size, _ in samples)
    neighbors = [(size,value) for size,value in samples
                 if abs(math.log2(target/size)) <= distance + 1]
    # Only measured variation supplies a slope. Do not extrapolate fixed
    # interpreter memory or launch/collection time as if it grew per event.
    estimates = [overhead + max(0, value-overhead)*target/size for size,value in neighbors]
    if target >= max(distinct) and all(value <= overhead for _,value in neighbors):
        estimates += [value*target/size for size,value in neighbors]
    return margin * max(estimates)


def preparation_forecast(request, observations, *, cpu_count=None, host=None):
    """Conservative envelope from comparable owned attempts, including partial peaks.

    Time only uses completed observations. Storage is based on unique path
    records (not X/Y hard-link counts). A smaller trial cannot silently lower
    the static reservation. An unfinished peak is a lower bound, not a timing sample.
    """
    if not observations:return {}
    target = preparation_workload(request)
    if target is None:return {}
    forecasts = {}; legacy_storage=[]; matching_storage=[];memory_samples=[];layout_memory=[];time_samples=[]
    keys = ('family', 'shape', 'provider', 'stage', 'source', 'parallel_axes')
    def add(key, value):
        forecasts[key] = max(forecasts.get(key, 0), math.ceil(value))
    for observation in observations or ():
        sample = observation.get('workload') or {}
        if any(sample.get(k) != target[k] for k in keys):continue
        if not all(type(sample.get(k)) is int and sample[k] > 0 for k in ('peak_units','total_units','path_units','axis_jobs')):
            continue
        workers = observation.get('cpu_count')
        peak = observation.get('peak_rss_bytes')
        if type(workers) is int and workers > 0 and type(peak) is int and peak > 0:
            active = min(sample.get('path_workers',sample['axis_jobs']), workers)
            target_active = min(target.get('path_workers',target['axis_jobs']), cpu_count or workers)
            item = (sample['peak_units'], peak / active * target_active)
            if target.get('df_event_layout') == sample.get('df_event_layout'):
                layout_memory.append(item)
            elif not sample.get('df_event_layout'):
                memory_samples.append(item)
        stored = observation.get('unique_event_bytes'); units = observation.get('stored_path_units')
        if type(stored) is int and stored > 0 and type(units) is int and units > 0:
            estimate = stored * target['path_units']/units
            if target.get('df_event_layout'):
                if (sample.get('df_event_layout') == target['df_event_layout']
                        and sample.get('event_shapes') == target.get('event_shapes')):
                    matching_storage.append(estimate)
                elif not sample.get('df_event_layout'):
                    legacy_storage.append(estimate)
            elif not sample.get('df_event_layout'):
                legacy_storage.append(estimate)
        elapsed = observation.get('elapsed_seconds')
        if (observation.get('completed') is True and host == observation.get('host') and
                type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed > 0 and
                type(workers) is int and workers > 0):
            # Legacy/new layout timings are not interchangeable once the
            # target has measured its own completed work.
            if target.get('df_event_layout') == sample.get('df_event_layout') or not sample.get('df_event_layout'):
                time_samples.append((sample['total_units'], elapsed * max(1, workers/(cpu_count or workers)),
                                     sample.get('df_event_layout') == target.get('df_event_layout')))
    comparable_layout = any(.5 <= size/target['peak_units'] <= 2 for size,_ in layout_memory)
    if estimate := _near_scale_envelope(target['peak_units'], layout_memory if comparable_layout else layout_memory+memory_samples, margin=1.5):
        add('memory_bytes', estimate)
    matching_times = [(size,value) for size,value,matching in time_samples if matching]
    timed = matching_times if any(.5 <= size/target['total_units'] <= 2 for size,_ in matching_times) else [(size,value) for size,value,_ in time_samples]
    if estimate := _near_scale_envelope(target['total_units'],timed):add('seconds', estimate)
    if matching_storage or legacy_storage:add('storage_bytes',max(matching_storage or legacy_storage))
    return forecasts

def resource_class(request, evaluator=None):
    preparation = request.get('preparation')
    if preparation:
        return 'preparation:' + preparation['stage'] + ':' + preparation['provider']
    return 'q3:' + ('finite_support' if evaluator == 'RPE_FINITE_SUPPORT_MSE_V1' else 'count_dp')


def measured_resource_class(request, evaluator=None, runtime=None, *, preparation_detail=False, workload=None):
    """Group comparable requests for capacity estimates, not result identity."""
    category = resource_class(request, evaluator)
    runtime = runtime or {}
    if preparation := request.get('preparation'):
        if not preparation_detail:
            return category
        # Source and the fixed resource footprint affect memory measurements.
        # A previous provider's peak must not govern every later provider.
        source = (preparation.get('source') or {}).get('commit', '')
        category = ':'.join(map(str, (category, source, runtime.get('task_cpu_count', ''),
                                      runtime.get('task_memory_bytes', ''), runtime.get('dp_workers', ''))))
        if workload:
            # Powers-of-two classes reuse nearby scales instead of making a
            # class for every candidate. They are never scientific identities.
            factors = [workload[k].bit_length() for k in ('peak_units', 'total_units', 'axis_jobs', 'path_units')]
            shape = hashlib.sha256(json.dumps(workload['shape'], separators=(',', ':')).encode()).hexdigest()
            category += ':' + ':'.join(map(str, (workload['family'], shape, workload['parallel_axes'], *factors)))
            if workload.get('df_event_layout'):category += ':'+workload['df_event_layout']
        return category
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
        if not target:
            # A measured request that cannot fit must not prevent an
            # independent, unmeasured class from getting its first sample.
            isolated = next((kind for kind in waiting if not peaks.get(kind, {}).get('peak_rss_bytes')), None)
            if isolated is not None:
                target = 1
                offered_classes = [isolated]
                state['isolated_resource_class'] = isolated
                stage = 'ISOLATED_RESOURCE_CLASS_TRIAL'
                reasons = ['No measured class fits; sample one unmeasured class on the idle host']
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
