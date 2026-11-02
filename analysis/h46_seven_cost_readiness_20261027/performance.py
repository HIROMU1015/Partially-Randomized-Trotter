"""Execution-only optimizations. No scientific imports, sampling or compile."""
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time


def cpu_capacity():
    """Affinity/quota ceiling, without assigning cores or changing cgroups."""
    slots = float(len(os.sched_getaffinity(0)))
    root = Path('/sys/fs/cgroup')
    path = root
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            path = root / line[3:].lstrip('/')
    for p in (path, *path.parents):
        if p != root and root not in p.parents:
            continue
        try:
            quota, period = (p / 'cpu.max').read_text().split()
            if quota != 'max':
                slots = min(slots, int(quota) / int(period))
        except FileNotFoundError:
            pass
    return max(1, math.ceil(slots))


def own_process_cpu():
    """CPU counters only for this controller's living multiprocessing children."""
    out = {}
    for child in mp.active_children():
        try:
            fields = Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[1]) == os.getpid():
                out[child.pid] = (int(fields[19]), int(fields[11]) + int(fields[12]))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return out


def admission(observation, active, capacity, own_cpu=0., footprint=None):
    """Idle plus our measured CPU use estimates capacity available to this job.

    Counting only idle CPU would mistake our own productive workers for other
    users' load. Reductions affect subsequent submissions; no process is killed.
    """
    available = max(0., observation['idle_cpu'] + own_cpu)
    target = min(capacity, math.floor(available + 1e-9))
    if target == 0 and capacity == 1 and available > 0:
        target = 1  # A fractional CPU quota can still run one process.
    if footprint:
        if observation['available_ram'] < footprint:
            raise RuntimeError('Available memory below observed worker footprint')
        # Active allocations are already deducted from available RAM. The 2x
        # estimate accommodates the retained circuit and transient next build.
        target = min(target, active + int(observation['available_ram'] // (2 * footprint)))
    return max(0, target)


class AdaptiveAdmission:
    def __init__(self, resource_reader, log_path, footprint):
        self.reader, self.log_path, self.footprint = resource_reader, Path(log_path), footprint
        self.observations = 0

    def note_result(self, value):
        self.footprint = max(self.footprint, value.get('peak_RSS_bytes', self.footprint))

    def __call__(self, active, capacity):
        before = own_process_cpu();start = time.monotonic()
        observation = self.reader()
        after = own_process_cpu();elapsed = max(time.monotonic() - start, 1e-9)
        ticks = sum(max(0, value[1] - before[pid][1]) for pid, value in after.items()
                    if pid in before and value[0] == before[pid][0])
        own_cpu = ticks / os.sysconf('SC_CLK_TCK') / elapsed
        target = admission(observation, active, capacity, own_cpu, self.footprint)
        with self.log_path.open('a') as stream:
            stream.write(json.dumps(dict(monotonic=time.monotonic(), observation=observation,
                active=active, pool_capacity=capacity, own_cpu=own_cpu, observed_worker_footprint=self.footprint,
                desired_inflight=target)) + '\n')
        self.observations += 1
        return target


def order_jobs(jobs, cost_proxy):
    """Longest structural work first, deterministic ties, unchanged job IDs.

    This is a scheduling estimate, not a scientific cost prediction or bound.
    Representative correctness work is the caller's separate first wave.
    """
    return sorted(jobs, key=lambda j: (-cost_proxy(j[0]) * (8 if j[0]['system']=='H4' else 12)**2,
                                      j[0]['id'], j[1], j[2]))


class ScheduleMemo:
    """Keep seven small immutable schedules; return fresh descriptors to callers."""
    def __init__(self, original):
        self.original, self.cache, self.hits, self.misses = original, {}, 0, 0

    def __call__(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in self.cache:
            self.cache[key] = tuple(dict(v) for v in self.original(*args, **kwargs));self.misses += 1
        else:
            self.hits += 1
        return [dict(v) for v in self.cache[key]]


class ComposeMemo:
    """One exact prior evolution per worker; retain no growing circuit cache.

    Only full numerical input equality can skip a build. Component specs are
    matched by object identity and held alive, so different registries/specs
    produce a safe miss. Seeds/cohorts stay in the original sampler and output.
    The original wrapper rechecks the cached circuit's numerical fingerprint.
    """
    def __init__(self, original):
        self.original, self.key, self.value = original, None, None
        self.references = None
        self.hits = self.misses = 0

    def signature(self, row, prep, requests, scalar):
        pieces, held = [], [prep]
        for request in requests:
            if request.preparation is not prep:
                raise ValueError('Different request preparation')
            occurrence = request.rte_occurrence
            held.append(occurrence.component_specs)
            pieces.append(dict(time=float(request.step_time).hex(),
                component_specs_identity=id(occurrence.component_specs),
                events=[event.to_dict() for event in occurrence.events],
                flags={name:getattr(occurrence,name) for name in (
                    'controlled','ancilla_qubit','preserve_event_order','cancel_adjacent_equal_bases',
                    'control_diagonal_only','identity_as_relative_ancilla_phase','tail_id','tail_hash','occurrence_rte_steps')},
                basis_plan=None if request.rte_basis_plan is None else request.rte_basis_plan.to_dict()))
        payload = dict(candidate=row, prep_identity=id(prep), preparation_hash=prep.preparation_hash,
                       scalar=float(scalar).hex(), pieces=pieces)
        # Full bytes, rather than a digest alone, establish equality. All
        # floats preserve Python's round-trip representation, including -0.0.
        return json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False), held

    def __call__(self, row, prep, requests, scalar, fingerprint):
        try:
            key, held = self.signature(row, prep, requests, scalar)
        except (AttributeError, TypeError, ValueError):
            self.misses += 1
            return self.original(row, prep, requests, scalar, fingerprint)
        if key == self.key:
            self.hits += 1
            return self.value
        # Release the old circuit before building the new one to avoid keeping
        # two large cached evolutions during a miss.
        self.key = self.value = self.references = None
        value = self.original(row, prep, requests, scalar, fingerprint)
        self.key, self.value, self.references = key, value, held
        self.misses += 1
        return value
