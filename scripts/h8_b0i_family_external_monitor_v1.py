"""Independent resource observations in the existing metadata launcher.

No science imports, extra process, relaxed limit, or automatic retry. A driver
callback can be delayed only while this bound observer proves continuous <=5s
observations. Compile children also wait for their own observed identity.
"""
import json
import errno
import math
import os
from pathlib import Path
import time

CONFIG = dict(enabled=True, role='EXISTING_METADATA_LAUNCHER',
    interval_seconds=1.0, maximum_staleness_seconds=5.0)
HEARTBEAT = 'external_resource_observation_v1.json'
PHASE = 'driver_phase_v1.json'


def child_cpu_total(seen,carry,contract_sha,driver_identity,allowed):
    """Exact completed waves plus observed tasks not covered by that receipt."""
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in seen.values()):
        raise ValueError('INVALID_CHILD_CPU_OBSERVATION')
    if carry is None:return sum(seen.values())
    if (carry.get('contract_sha256')!=contract_sha or carry.get('driver_identity')!=driver_identity or
            carry.get('all_reaped') is not True):raise ValueError('COMPILE_CPU_CARRY_BINDING_OR_CLEANUP')
    completed=carry.get('completed_names');base=carry.get('child_cpu_seconds')
    if (not isinstance(completed,list) or any(not isinstance(x,str) for x in completed) or
            len(completed)!=len(set(completed)) or not set(completed)<=allowed):
        raise ValueError('H8_COMPLETED_CPU_TASK_BINDING')
    if type(base) not in (int,float) or not math.isfinite(base) or base<0:
        raise ValueError('INVALID_CHILD_CPU_CARRY')
    return base+sum(v for k,v in seen.items() if k not in completed)


def same(left, right):
    return isinstance(left, dict) and isinstance(right, dict) and all(
        left.get(k) == right.get(k) for k in ('pid', 'start_ticks', 'uid', 'boot_id'))


def read_small(path, limit=65536):
    with Path(path).open('rb') as handle:
        raw = handle.read(limit+1)
    if len(raw) > limit: raise RuntimeError('OBSERVER_RECORD_CAP')
    return json.loads(raw)


def atomic(path, value):
    path = Path(path)
    raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()+b'\n'
    if len(raw) > 65536: raise RuntimeError('OBSERVER_RECORD_CAP')
    pending = path.with_name('.'+path.name+'.pending.'+str(os.getpid()))
    fd = os.open(pending, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            n = os.write(fd, view); view = view[n:]
        os.fsync(fd)
    finally: os.close(fd)
    os.replace(pending, path)


def verify_record(value, grant, ticket, own, now):
    c = grant.contract; budget = c['budget']; maximum = budget['monitor_max_seconds']
    if (value.get('schema') != 'H8_EXTERNAL_RESOURCE_OBSERVATION_V1' or
            value.get('SOURCE') != c['SOURCE'] or
            value.get('contract_sha256') != grant.contract_sha or
            value.get('authorization_sha256') != grant.authorization_sha or
            not same(value.get('launcher_identity'), ticket['parent_identity']) or
            value.get('driver_identity') != ticket['identity']):
        raise RuntimeError('OBSERVER_RUN_BINDING')
    for name in ('monotonic', 'max_interval_seconds', 'max_observation_seconds'):
        v = value.get(name)
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise RuntimeError('OBSERVER_FINITE_TIMING_REQUIRED')
    if not 0 <= now-value['monotonic'] <= maximum:
        raise RuntimeError('EXTERNAL_OBSERVATION_STALE')
    if value['max_interval_seconds'] > maximum or value['max_observation_seconds'] > maximum:
        raise RuntimeError('EXTERNAL_OBSERVATION_GAP')
    if type(value.get('sequence')) is not int or value['sequence'] < 1:
        raise RuntimeError('OBSERVER_SEQUENCE_REQUIRED')
    rows = [value.get('driver')]+value.get('workers', [])
    if value.get('first_stop') is None and not any(isinstance(row, dict) and row.get('identity') == own for row in rows):
        raise LookupError('OWN_IDENTITY_NOT_YET_OBSERVED')
    return value


class Observer:
    def __init__(self, grant, identity, launcher_identity, g, clock=time.monotonic):
        self.grant, self.identity, self.launcher, self.g = grant, identity, launcher_identity, g
        self.clock = clock; self.started = self.last = clock(); self.sequence = 0
        self.maximum_gap = self.maximum_duration = 0.; self.first_stop = None
        self.worker_first_seen = {}
        self.worker_affinity_confirmed = set()
        self.resource_last_seen = {}
        self.child_cpu_seen = {}
        self.control = Path(grant.contract['control_directory'])
        self.baseline = grant.contract['external_monitor_memory_baseline']
        self.log = self.control/'external_observations_v1.jsonl'
        fd = os.open(self.log, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600); os.close(fd)

    def process(self, expected, allowed, *, worker_bootstrap=False):
        g = self.g; current = g.identity(expected['pid'])
        if current is None: return None
        if current != expected: raise g.Stop('OBSERVER_OWN_IDENTITY_CHANGED')
        self.resource_last_seen.setdefault(expected['pid'],self.clock())
        root = Path('/proc')/str(expected['pid'])
        try:
            status = dict(line.split(':', 1) for line in (root/'status').read_text().splitlines() if ':' in line)
            if status['State'].lstrip().startswith('Z'): return None
            rss = int(status['VmHWM'].split()[0])*1024
            size = int(status['VmSize'].split()[0])*1024
            fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
            cpu = (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
            affinity = os.sched_getaffinity(expected['pid'])
        except KeyError as error:
            if error.args[0] not in ('VmHWM','VmSize'):raise
            current=g.identity(expected['pid'])
            if current is not None and current!=expected:raise g.Stop('OBSERVER_OWN_IDENTITY_CHANGED')
            if current is not None and self.clock()-self.resource_last_seen[expected['pid']]>5:
                raise g.Stop('EXTERNAL_MEMORY_OBSERVATION_MISSING')
            return None  # No fresh resource lease is issued for missing data.
        except FileNotFoundError:return None
        except ProcessLookupError as error:
            if error.errno!=errno.ESRCH:raise
            return None
        budget = self.grant.contract['budget']
        if rss > budget['rss_bytes']: raise g.Stop('EXTERNAL_RSS_CAP')
        if size > budget['address_space_bytes']: raise g.Stop('EXTERNAL_AS_CAP')
        if cpu > budget['cpu_seconds']: raise g.Stop('EXTERNAL_CPU_CAP')
        if int(status['Threads']) != 1:
            raise g.Stop('EXTERNAL_THREAD_COUNT_CHANGED:threads='+status['Threads'].strip()+
                ';state='+status['State'].strip()+';affinity='+str(sorted(affinity)))
        if len(affinity) != 1 or not affinity <= set(allowed):
            first = self.worker_first_seen.setdefault(expected['pid'], self.clock())
            if (worker_bootstrap and expected['pid'] not in self.worker_affinity_confirmed
                    and affinity == {self.grant.contract['parallel_compile']['driver_cpu']}
                    and self.clock()-first < 5):
                return None  # No lease/compile until this child is on its worker CPU.
            raise g.Stop('EXTERNAL_AFFINITY_CHANGED:actual='+str(sorted(affinity))+
                ';allowed='+str(sorted(allowed)))
        if worker_bootstrap: self.worker_affinity_confirmed.add(expected['pid'])
        self.resource_last_seen[expected['pid']]=self.clock()
        return dict(identity=expected, peak_RSS_bytes=rss, AS_bytes=size,
            CPU_seconds=cpu, affinity=sorted(affinity))

    def step(self):
        g = self.g; c = self.grant.contract; now = self.clock(); phase = 'PRELAUNCH'
        self.maximum_gap = max(self.maximum_gap, now-self.last); self.last = now
        driver = None; workers = []; memory = filesystem = None
        try:
            if self.maximum_gap > c['budget']['monitor_max_seconds']:
                raise g.Stop('EXTERNAL_OBSERVATION_STALE')
            if c['budget']['wall_seconds'] is not None and self.clock()-self.started > c['budget']['wall_seconds']: raise g.Stop('EXTERNAL_WALL_CAP')
            phase_path = self.control/PHASE
            if phase_path.exists():
                record = read_small(phase_path)
                if record['contract_sha256'] != self.grant.contract_sha or record['identity'] != self.identity:
                    raise g.Stop('DRIVER_PHASE_BINDING')
                phase = record['phase']
            driver = self.process(self.identity, [c['parallel_compile']['driver_cpu']])
            if driver is None: return False
            for candidate in c['new_candidates']:
                p,q=candidate['prefix'],candidate['q']
                for axis in ('cosine','sine'):
                    path = self.control/f'observed_compile_started_{p}_{q}_{axis}.json'
                    if not path.exists(): continue
                    record = read_small(path); expected = record['identity']
                    if expected is None: raise g.Stop('COMPILE_IDENTITY_NOT_RECORDED')
                    if expected['parent_pid'] != self.identity['pid'] or expected['uid'] != self.identity['uid']:
                        raise g.Stop('EXTERNAL_COMPILE_OWNERSHIP')
                    if record['cpu'] not in c['parallel_compile']['worker_cpus']:
                        raise g.Stop('EXTERNAL_COMPILE_CPU_BINDING')
                    row = self.process(expected, [record['cpu']], worker_bootstrap=True)
                    if row is not None:
                        workers.append(row);self.child_cpu_seen[record['name']]=row['CPU_seconds']
            carry_path=self.control/'observed_compile_cpu_carry_v1.json'
            allowed={f"{r['prefix']}_{r['q']}_{axis}" for r in c['new_candidates'] for axis in ('cosine','sine')}
            child_cpu=child_cpu_total(self.child_cpu_seen,read_small(carry_path) if carry_path.exists() else None,
                self.grant.contract_sha,self.identity,allowed)
            if driver['CPU_seconds']+child_cpu>c['budget']['cpu_seconds']:
                raise g.Stop('EXTERNAL_AGGREGATE_STAGE_CPU_CAP')
            memory = g.d0.memory_snapshot(); g.validate_memory_observation(memory, self.baseline)
            g.d0.check_memory_snapshot(memory, c['budget'], self.baseline)
            filesystem = g.filesystem_snapshot(self.grant.output.parent)
            g.check_filesystem(filesystem, c['budget'], now=now)
        except Exception as error:
            if self.first_stop is None:
                self.first_stop = dict(reason=type(error).__name__+':'+str(error)[:700], phase=phase, monotonic=now)
        duration = self.clock()-now
        self.maximum_duration = max(self.maximum_duration, duration)
        if duration > c['budget']['monitor_max_seconds'] and self.first_stop is None:
            self.first_stop = dict(reason='EXTERNAL_OBSERVATION_IO_STALE', phase=phase, monotonic=now)
        self.sequence += 1
        value = dict(schema='H8_EXTERNAL_RESOURCE_OBSERVATION_V1', SOURCE=c['SOURCE'],
            contract_sha256=self.grant.contract_sha, authorization_sha256=self.grant.authorization_sha,
            launcher_identity=self.launcher, driver_identity=self.identity, sequence=self.sequence,
            monotonic=now, interval_seconds=now-(self.last if self.sequence==1 else self.previous),
            observation_seconds=duration, max_interval_seconds=self.maximum_gap,
            max_observation_seconds=self.maximum_duration, phase=phase, first_stop=self.first_stop,
            driver=driver, workers=workers, memory=memory, filesystem=filesystem)
        self.previous = now
        atomic(self.control/HEARTBEAT, value)
        # Combined launcher stdout + observer journal remains within the old log cap.
        row = dict(sequence=self.sequence, monotonic=now, phase=phase,
            interval_seconds=value['interval_seconds'], observation_seconds=duration,
            driver=driver, workers=workers, first_stop=self.first_stop,
            effective_available_bytes=memory['effective_available_bytes'] if memory else None)
        raw = json.dumps(row, allow_nan=False, separators=(',', ':')).encode()+b'\n'
        if self.log.stat().st_size+(self.control/'launch_stdout.log').stat().st_size+len(raw) > c['log_cap_bytes']:
            raise g.Stop('COMBINED_OBSERVER_LAUNCH_LOG_CAP')
        fd = os.open(self.log, os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        try:
            view = memoryview(raw)
            while view:
                n = os.write(fd, view); view = view[n:]
        finally: os.close(fd)
        if self.first_stop is not None: raise g.Stop(self.first_stop['reason'])
        return True

    def summary(self):
        atomic(self.control/'external_monitor_summary_v1.json', dict(
            SOURCE=self.grant.contract['SOURCE'], observations=self.sequence,
            max_interval_seconds=self.maximum_gap, max_observation_seconds=self.maximum_duration,
            first_stop=self.first_stop, role='EXISTING_METADATA_LAUNCHER', extra_processes=0))


def monitor_class(g):
    class ObservedMonitor(g.Monitor):
        def __init__(self, grant, **kwargs):
            self.external_grant = grant; self.external_control = Path(grant.contract['control_directory'])
            self.external_own = g.identity(); self.external_ticket = read_small(g.marker_path(grant.output))
            self.external_last_sequence = 0; self.external_phase_written = None
            self.external_callback_last = time.monotonic()
            super().__init__(grant, **kwargs)

        def observation(self):
            started = time.monotonic()
            while True:
                try:
                    value = verify_record(read_small(self.external_control/HEARTBEAT),
                        self.external_grant, self.external_ticket, self.external_own, time.monotonic())
                    if value['sequence'] < self.external_last_sequence: raise RuntimeError('OBSERVER_SEQUENCE_REVERSED')
                    self.external_last_sequence = value['sequence']
                    return value
                except (FileNotFoundError, LookupError):
                    if self.external_last_sequence or time.monotonic()-started >= 5: raise
                    time.sleep(.02)

        def tick(self, *, force=False):
            now = self.clock()
            if not force and now-self.last < .5: return
            try:
                value = self.observation()
                if value['first_stop'] is not None: self.fail('EXTERNAL_OBSERVER:'+value['first_stop']['reason'])
                if self.external_own == self.external_ticket['identity'] and self.phase != self.external_phase_written:
                    atomic(self.external_control/PHASE, dict(identity=self.external_own,
                        contract_sha256=self.external_grant.contract_sha, phase=self.phase, monotonic=now))
                    self.external_phase_written = self.phase
            except Exception as error: self.fail(type(error).__name__+':'+str(error))
            callback_gap = now-self.last
            # The independent log proves observation continuity across delayed
            # Python callbacks; all original resource checks still execute here.
            if callback_gap > self.budget['monitor_max_seconds']: self.last = self.clock()
            super().tick(force=True)
            self.history[-1].update(driver_callback_interval_seconds=callback_gap,
                external_observer_sequence=value['sequence'],
                external_max_interval_seconds=value['max_interval_seconds'],
                external_max_observation_seconds=value['max_observation_seconds'])

        def fail(self, reason):
            if self.first_stop is None:
                try:
                    value = self.observation()
                    if value['first_stop'] is not None:
                        self.first_stop = dict(value['first_stop'])
                except Exception: pass
            return super().fail(reason)
    return ObservedMonitor
