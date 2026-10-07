"""Independent stdlib observer. Production role requires a separate approval.

The private socket carries bounded JSON, never pickle or external data. Only
the exact registered identities are signalled, through pidfds. No affinity,
cgroup, host settings, or foreign processes are changed.
"""
import json
import math
import os
from pathlib import Path
import resource
import select
import signal
import socket
import subprocess
import sys
import threading
import time

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
    __package__ = 'trottertracks.resource_applicability.h4_geometry'

from .identity import Stop, require
from .resources import HEADROOM, ROLE_CAP, WALL_CAP, observe_memory, fsync_directory

AS_CAP = 256 * 2**20
RSS_CAP = 64 * 2**20
FRAME_CAP = 8192
DEADLINE = 5
PERIOD = 1
TERMINAL_RESERVE = 2 * FRAME_CAP


def process_sample(pid):
    """Bracket status with stat to reject exit/reuse races and zombie owners."""
    root = Path('/proc') / str(pid)
    raw = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    require(raw[0] not in ('Z', 'X'), 'owned process exited')
    fields = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines() if ':' in line)
    uid = [int(x) for x in fields['Uid'].split()]
    require(len(uid) == 4 and len(set(uid)) == 1, 'owned UID ambiguity')
    def kib(name):
        value = fields[name].split()
        require(len(value) == 2 and value[1] == 'kB', 'missing process memory')
        return int(value[0]) * 1024
    value = dict(pid=pid, start=raw[19], parent=int(raw[1]), uid=uid[0],
                 rss=kib('VmRSS'), address_space=kib('VmSize'))
    after = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    require(after[19] == raw[19] and after[1] == raw[1] and after[0] not in ('Z', 'X'),
            'owned identity changed during observation')
    return value


def identity(sample):
    return {k: sample[k] for k in ('pid', 'start', 'parent', 'uid')}


class OwnedIdentity:
    def __init__(self, expected, parent, *, sampler=process_sample):
        require(expected['uid'] == os.getuid() and expected['parent'] == parent, 'foreign ownership')
        self.expected, self.sampler = expected, sampler
        self.fd = os.pidfd_open(expected['pid'])
        try:
            self.sample()
        except BaseException:
            self.close()
            raise

    def sample(self):
        require(self.fd is not None and not select.select([self.fd], [], [], 0)[0], 'owned process exited')
        result = self.sampler(self.expected['pid'])
        require(identity(result) == self.expected, 'owned identity lost')
        return result

    def terminate(self, sig=signal.SIGTERM):
        try:
            self.sample()
        except (Stop, OSError, KeyError):
            return False  # never signal a missing/reused/foreign process
        signal.pidfd_send_signal(self.fd, sig)
        return True

    def terminate_after_parent_exit(self, parent_owner):
        """A registered worker remains owned after kernel reparenting.

        Relax only parent equality, only with the original parent's stable
        pidfd proving exit. PID/starttime/UID and the child's pidfd still match.
        """
        if (parent_owner.fd is None or self.expected['parent'] != parent_owner.expected['pid'] or
                not select.select([parent_owner.fd], [], [], 0)[0]):
            return self.terminate()
        try:
            require(self.fd is not None and not select.select([self.fd], [], [], 0)[0], 'owned process exited')
            current = self.sampler(self.expected['pid'])
            require(all(current[k] == self.expected[k] for k in ('pid','start','uid')), 'orphan ownership lost')
        except (Stop, OSError, KeyError):return False
        signal.pidfd_send_signal(self.fd, signal.SIGTERM)
        return True

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class ObservationState:
    """Pure policy; timings, phase and the first failure are separate fields."""
    def __init__(self, baseline, started, *, prior_wall=0, workers=12, wall_started=None):
        self.baseline = baseline['oom_events']
        self.last = started
        self.started = started if wall_started is None else wall_started
        self.prior_wall, self.workers = prior_wall, workers
        self.phase = dict(sequence=0, name='observer_start', monotonic=started)
        self.first_failure = None

    def set_phase(self, phase, now):
        require(set(phase) == {'sequence', 'name', 'monotonic'} and
                type(phase['sequence']) is int and phase['sequence'] > self.phase['sequence'] and
                isinstance(phase['name'], str) and len(phase['name']) <= 128 and
                type(phase['monotonic']) in (float, int) and math.isfinite(phase['monotonic']) and
                0 <= now - phase['monotonic'] <= DEADLINE, 'driver phase sequence/timestamp')
        self.phase = phase

    def evaluate(self, observation, samples, observer_sample, begun, ended):
        record = dict(kind='observation', monotonic=ended, interval_seconds=ended-self.last,
                      observation_started=begun,
                      observation_duration_seconds=ended-begun,
                      staleness_seconds=None if observation is None else ended-observation['observed_at'],
                      driver_phase=dict(self.phase), phase_age_seconds=ended-self.phase['monotonic'],
                      wall_seconds=self.prior_wall+ended-self.started, processes=samples,
                      observer=observer_sample)
        reason = None
        interval, duration, stale = (record[k] for k in
                                     ('interval_seconds', 'observation_duration_seconds', 'staleness_seconds'))
        if not 0 <= duration <= DEADLINE: reason = 'observation_duration'
        elif not 0 <= interval <= DEADLINE: reason = 'observation_interval'
        elif stale is None: reason = 'missing_observation'
        elif not 0 <= stale <= DEADLINE: reason = 'observation_staleness'
        elif not 0 <= record['wall_seconds'] <= WALL_CAP: reason = 'cumulative_wall'
        elif observation['available'] < HEADROOM: reason = 'memory_headroom'
        elif observation['psi_full_avg10'] != 0: reason = 'memory_pressure'
        elif observation['oom_events'] != self.baseline: reason = 'oom_or_changed_cgroup'
        elif not 1 <= len(samples) <= self.workers+1: reason = 'owned_process_count'
        elif any(not 0 <= s['rss'] <= ROLE_CAP or not 0 <= s['address_space'] <= ROLE_CAP for s in samples):
            reason = 'owned_role_rss_as'
        elif len({s['pid'] for s in samples}) != len(samples): reason = 'duplicate_owned_process'
        elif not 0 <= observer_sample['rss'] <= RSS_CAP or not 0 <= observer_sample['address_space'] <= AS_CAP:
            reason = 'observer_rss_as'
        self.last = ended
        if reason and self.first_failure is None:
            self.first_failure = dict(reason=reason, **record)
        record['first_failure'] = self.first_failure
        return record


class Trace:
    def __init__(self, fd, cap):
        require(type(cap) is int and TERMINAL_RESERVE < cap <= 3 * 2**30, 'observer output cap')
        self.fd, self.cap, self.written = fd, cap, 0

    def write(self, record, *, terminal=False):
        data = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'
        require(len(data) <= FRAME_CAP, 'observer trace frame')
        require(self.written+len(data) <= self.cap-(0 if terminal else TERMINAL_RESERVE), 'observer output budget')
        view = memoryview(data)
        while view:
            count = os.write(self.fd, view)
            require(count > 0, 'observer trace write')
            view = view[count:]
        os.fsync(self.fd)  # first failure durable before any signal
        self.written += len(data)


def send(sock, value):
    data = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    require(len(data) <= FRAME_CAP and sock.send(data) == len(data), 'observer private frame')


def receive(sock):
    data, _anc, flags, _address = sock.recvmsg(FRAME_CAP)
    require(data and not flags & socket.MSG_TRUNC, 'observer pipe EOF/frame')
    value = json.loads(data)
    require(isinstance(value, dict), 'observer wire object')
    return value


def observer_main(sock_fd, trace_fd):
    sock = socket.socket(fileno=sock_fd)
    config = receive(sock)
    synthetic = config['scope'] == 'SYNTHETIC_ONLY'
    require(synthetic or config['runtime_authorization'] is True, 'production observer unapproved')
    if not synthetic and config.get('role_cpus') is not None:
        require(len(config['role_cpus'])==1 and set(config['role_cpus'])<=set(config['allowed_cpus']), 'observer CPU approval')
        os.sched_setaffinity(0,set(config['role_cpus']))
        require(set(os.sched_getaffinity(0))==set(config['role_cpus']), 'observer own CPU binding')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = AS_CAP if hard == resource.RLIM_INFINITY else min(AS_CAP, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    trace = Trace(trace_fd, config['output_cap'])
    driver = OwnedIdentity(config['driver'], config['driver']['parent'])
    require(driver.expected['pid'] == os.getppid(), 'observer direct parent handshake')
    workers = {}
    state = None
    try:
        start = time.monotonic()
        baseline = observe_memory()
        state = ObservationState(baseline, start, prior_wall=config['prior_wall'], workers=config['workers'],
                                 wall_started=config['wall_started'])
        record = state.evaluate(baseline, [driver.sample()], process_sample(os.getpid()), start, time.monotonic())
        require(state.first_failure is None, 'observer initial observation')
        trace.write(record)
        send(sock, {'kind': 'ready', 'identity': identity(process_sample(os.getpid()))})
        next_poll = time.monotonic()+PERIOD
        period = PERIOD
        delay = 0
        while True:
            ready = select.select([sock], [], [], max(0, next_poll-time.monotonic()))[0]
            if ready:
                message = receive(sock)
                command = message.get('command')
                if command == 'phase': state.set_phase(message['phase'], time.monotonic())
                elif command == 'own':
                    expected = message['identity']
                    require(expected['pid'] not in workers and len(workers) < config['workers'] and
                            expected['pid'] != os.getpid(), 'observer worker registration')
                    workers[expected['pid']] = OwnedIdentity(expected, driver.expected['pid'])
                elif command == 'shutdown':
                    require(not workers, 'shutdown with live registered workers')
                    trace.write({'kind': 'shutdown', 'written_before_final': trace.written,
                        'wall_seconds':state.prior_wall+time.monotonic()-state.started}, terminal=True)
                    send(sock, {'kind': 'ack', 'first_failure': state.first_failure})
                    break
                elif command == 'stop_children':
                    for owner in workers.values(): owner.terminate()
                elif command == 'reap':
                    require(all(select.select([owner.fd], [], [], 0)[0] for owner in workers.values()),
                            'cannot release live registered workers')
                    for owner in workers.values(): owner.close()
                    workers.clear()
                elif command == 'synthetic_fault':
                    require(synthetic and message['fault'] in ('io_delay', 'fast_period'), 'synthetic-only fault')
                    value = message['value']
                    require(type(value) in (float, int) and math.isfinite(value), 'synthetic timing value')
                    if message['fault'] == 'io_delay':
                        require(0 <= value <= 5.25, 'bounded synthetic I/O delay'); delay = value
                    else:
                        require(0.05 <= value <= PERIOD, 'bounded synthetic period'); period = value
                    next_poll = time.monotonic()+period
                else: require(command == 'ping', 'unknown observer command')
                send(sock, {'kind': 'ack', 'first_failure': state.first_failure, 'last_observation': record})
            if time.monotonic() >= next_poll:
                begun = time.monotonic()
                if delay: time.sleep(delay)
                observation = observe_memory()
                samples = [driver.sample(), *(owner.sample() for owner in workers.values())]
                record = state.evaluate(observation, samples, process_sample(os.getpid()), begun, time.monotonic())
                trace.write(record, terminal=state.first_failure is not None)
                if state.first_failure: raise Stop(state.first_failure['reason'])
                next_poll = time.monotonic()+period
    except BaseException as exc:
        failure = state.first_failure if state and state.first_failure else {
            'reason': type(exc).__name__+': '+str(exc), 'driver_phase': state.phase if state else None,
            'monotonic': time.monotonic(),
            'interval_seconds': time.monotonic()-state.last if state else None,
            'observation_duration_seconds': time.monotonic()-begun if 'begun' in locals() else None,
            'staleness_seconds': None}
        try:
            trace.write({'kind': 'first_stop', 'first_failure': failure}, terminal=True)
        finally:
            for owner in workers.values(): owner.terminate_after_parent_exit(driver)
            if not synthetic: driver.terminate()
    finally:
        for owner in workers.values(): owner.close()
        driver.close(); sock.close(); os.close(trace_fd)


class IndependentObserver:
    """Spawned only from explicitly scoped synthetic cases or reviewed new role."""
    def __init__(self, python, trace_path, *, scope, workers=12, prior_wall=0,
                 output_cap=4*2**20, runtime_authorization=False, wall_started=None,allowed_cpus=None,role_cpus=None):
        wall_started = time.monotonic() if wall_started is None else wall_started
        require(scope == 'SYNTHETIC_ONLY' or (scope == 'PRODUCTION' and runtime_authorization is True),
                'independent observer production role not approved')
        path = Path(trace_path)
        require(path.is_absolute() and path.is_relative_to('/home/AbeHiromu') and
                not any(p.is_symlink() for p in [path, *path.parents]) and
                path.parent.stat().st_uid == os.getuid(), 'private observer output path')
        trace_fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
        fsync_directory(path.parent)
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.sock, self.process, self.owner = parent, None, None
        self.trace_path = path
        self.lock = threading.Lock()
        self.children = {}
        self.phase_sequence = 0
        self.first_failure = None
        self.last_phase = None
        try:
            self.process = subprocess.Popen([python, '-P', '-B', str(Path(__file__).absolute()),
                    '--private-observer', str(child.fileno()), str(trace_fd)],
                    pass_fds=(child.fileno(), trace_fd), stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                    cwd=str(Path(__file__).absolute().parents[4]), env=dict(os.environ))
            self.owner = OwnedIdentity(identity(process_sample(self.process.pid)), os.getpid())
            child.close(); os.close(trace_fd); trace_fd = None
            parent.settimeout(DEADLINE)
            send(parent, dict(scope=scope, runtime_authorization=runtime_authorization,
                 driver=identity(process_sample(os.getpid())), workers=workers, prior_wall=prior_wall,
                 wall_started=wall_started, output_cap=output_cap,allowed_cpus=allowed_cpus,role_cpus=role_cpus))
            response = receive(parent)
            require(response == {'kind': 'ready', 'identity': self.owner.expected}, 'observer ready ownership')
        except BaseException as exc:
            if trace_fd is not None: os.close(trace_fd)
            child.close()
            try:self.capture_failure(exc)
            finally:self.close(abort=True)
            raise

    def capture_failure(self, exc):
        """Recover the child's durable first cause before a secondary EOF error.

        Parent-only failures (including observer death) have their own bounded
        durable file. Both are covered by the pre-reserved output allowance.
        """
        if self.first_failure is not None:
            return
        failure = None
        try:
            fd = os.open(self.trace_path, os.O_RDONLY|os.O_NOFOLLOW)
            try:
                size = os.fstat(fd).st_size
                os.lseek(fd, max(0, size-2*FRAME_CAP), os.SEEK_SET)
                tail = os.read(fd, 2*FRAME_CAP)
            finally:os.close(fd)
            for line in reversed(tail.splitlines()):
                try:row = json.loads(line)
                except (ValueError, UnicodeError):continue
                if isinstance(row, dict) and row.get('kind') == 'first_stop':
                    failure = row['first_failure'];break
        except OSError:pass
        self.first_failure = failure or dict(reason=type(exc).__name__+': '+str(exc),
            monotonic=time.monotonic(), driver_phase=self.last_phase,
            interval_seconds=None, observation_duration_seconds=None, staleness_seconds=None)
        data = json.dumps({'kind':'driver_first_stop','first_failure':self.first_failure},
                          sort_keys=True,separators=(',', ':')).encode()+b'\n'
        require(len(data) <= FRAME_CAP, 'driver first-stop frame cap')
        fd = os.open(str(self.trace_path)+'.driver-first-stop.json',
                     os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
        try:
            require(os.write(fd,data) == len(data), 'driver first-stop write')
            os.fsync(fd)
        finally:os.close(fd)
        fsync_directory(self.trace_path.parent)

    def request(self, command, **values):
        requested = time.monotonic()
        with self.lock:
            try:
                remaining = DEADLINE-(time.monotonic()-requested)
                require(remaining > 0, 'observer request interval')
                self.sock.settimeout(remaining)
                self.owner.sample()
                send(self.sock, dict(command=command, **values))
                remaining = DEADLINE-(time.monotonic()-requested)
                require(remaining > 0, 'observer request duration')
                self.sock.settimeout(remaining)
                reply = receive(self.sock)
                require(reply.get('kind') == 'ack', 'observer acknowledgement')
                if reply.get('first_failure') and self.first_failure is None:
                    self.first_failure = reply['first_failure']
                require(self.first_failure is None, 'observer first STOP: '+str(self.first_failure))
                if command == 'ping':
                    require(0 <= time.monotonic()-reply['last_observation']['monotonic'] <= DEADLINE,
                            'observer observation staleness')
                return reply
            except BaseException as exc:
                try:self.capture_failure(exc)
                finally:
                    self.stop_children(local_only=True)
                    if self.owner is not None:self.owner.terminate()
                raise Stop('independent observer failed closed: '+str(self.first_failure)) from exc

    def phase(self, name):
        self.phase_sequence += 1
        self.last_phase = dict(sequence=self.phase_sequence, name=name, monotonic=time.monotonic())
        return self.request('phase', phase=self.last_phase)

    def own_child(self, pid):
        owner = OwnedIdentity(identity(process_sample(pid)), os.getpid())
        try:
            self.request('own', identity=owner.expected)
            self.children[pid] = owner
        except BaseException:
            owner.terminate(); owner.close(); raise

    def poll(self):
        return self.request('ping')

    def stop_children(self, *, local_only=False):
        for owner in self.children.values(): owner.terminate()

    def close(self, *, abort=False):
        self.stop_children(local_only=True)
        for owner in self.children.values(): owner.close()
        self.children.clear()
        if self.process is not None:
            if abort:self.sock.close()  # EOF releases a child still awaiting config
            if not abort and self.process.poll() is None:
                try:
                    self.request('reap')
                    self.request('shutdown')
                except (Stop, OSError): abort = True
            if abort and self.owner is not None: self.owner.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if self.owner is not None: self.owner.terminate(signal.SIGKILL)
                self.process.wait(timeout=2)
        if self.owner is not None: self.owner.close()
        self.sock.close()


if __name__ == '__main__':
    require(len(sys.argv) == 4 and sys.argv[1] == '--private-observer', 'private observer entry only')
    observer_main(int(sys.argv[2]), int(sys.argv[3]))
