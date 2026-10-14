"""Own-process gates and bounded artifacts. No authorization is issued here."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import secrets
import signal
import stat
import sys
import time

HOME_ROOT = Path('/home/AbeHiromu')
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'QISKIT_NUM_PROCS')
_spec = importlib.util.spec_from_file_location('final_frozen_d0',
    Path(__file__).with_name('d0_saved_value_reanalysis_v1.py'))
d0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d0)
_TOKEN = object()


class Stop(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def home_path(path, *, existing=False):
    p = Path(path)
    if (not p.is_absolute() or p.resolve() != p or p == HOME_ROOT or
            not p.is_relative_to(HOME_ROOT) or p.is_symlink()):
        raise Stop('CANONICAL_HOME_PATH_REQUIRED')
    if existing and (not p.is_file() or p.stat().st_uid != os.getuid()):
        raise Stop('OWN_REGULAR_FILE_REQUIRED')
    return p


def read_bytes(path, *, cap=16*2**20):
    p = home_path(path, existing=True)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise Stop('OWN_REGULAR_FILE_REQUIRED')
        if info.st_size > cap:
            raise Stop('INPUT_SIZE_CAP')
        data = bytearray()
        while True:
            chunk = os.read(fd, min(2**20, cap+1-len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > cap:
                raise Stop('INPUT_SIZE_CAP')
        if len(data) != info.st_size:
            raise Stop('INPUT_CHANGED_DURING_READ')
        return bytes(data)
    finally:
        os.close(fd)


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    root = Path('/proc')/str(pid)
    try:
        text = (root/'stat').read_text()
        fields = text.rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]), parent_pid=int(fields[1]),
            uid=root.stat().st_uid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


class Grant:
    def __init__(self, token, contract, authorization, contract_sha, authorization_sha, output, synthetic):
        if token is not _TOKEN:
            raise Stop('VALIDATED_GRANT_REQUIRED')
        self.contract = contract
        self.authorization = authorization
        self.contract_sha = contract_sha
        self.authorization_sha = authorization_sha
        self.output = output
        self.synthetic = synthetic
        self.binding_digest = d0.digest((contract, authorization))

    def require(self, scope):
        if self.contract['scope'] != scope or d0.digest((self.contract, self.authorization)) != self.binding_digest:
            raise Stop('GRANT_SCOPE_OR_MUTATION')


def authorization_gate(contract, authorization, *, contract_sha, authorization_sha,
        output, scope, synthetic=False, consumed=False):
    """First gate: runs before input mapping, scalar loaders or scientific APIs."""
    if (contract.get('approved') is not True or contract.get('runtime_authorization') is not True or
            authorization.get('approved_by_user') is not True or
            authorization.get('runtime_authorization') is not True or
            contract.get('scope') != scope or authorization.get('scope') != scope or
            authorization.get('contract_sha256') != contract_sha or
            authorization.get('SOURCE') != contract.get('SOURCE') or
            authorization.get('automatic_retry') is not False or
            contract.get('automatic_retry') is not False or
            contract.get('next_stage_authorized') is not False):
        raise Stop('EXPLICIT_NEW_EXECUTION_AUTHORIZATION_REQUIRED')
    if bool(contract.get('synthetic_only')) != synthetic or bool(authorization.get('synthetic_only')) != synthetic:
        raise Stop('SYNTHETIC_AUTHORIZATION_SCOPE')
    for field in ('input_receipts', 'budget', 'binding', 'profile', 'source_hashes', 'allowed_cpus', 'carry'):
        if authorization.get(field+'_digest') != d0.digest(contract[field]):
            raise Stop('AUTH_BINDING:'+field)
    if (not isinstance(authorization.get('approval_evidence_sha256'), str) or
            len(authorization['approval_evidence_sha256']) != 64 or
            not isinstance(authorization.get('review_commit'), str) or len(authorization['review_commit']) != 40):
        raise Stop('APPROVAL_EVIDENCE_AND_REVIEW_REQUIRED')
    output = home_path(output)
    if authorization.get('exclusive_output') != str(output) or contract.get('exclusive_output') != str(output):
        raise Stop('OUTPUT_BINDING')
    if not consumed and (output.exists() or marker_path(output).exists()):
        raise Stop('ONE_SHOT_ALREADY_USED')
    return Grant(_TOKEN, contract, authorization, contract_sha, authorization_sha, output, synthetic)


def marker_path(output):
    return output.with_name(output.name+'.oneshot')


def runtime_gate(grant, root):
    c = grant.contract
    root = home_path(root)
    for name, expected in c['source_hashes'].items():
        p = root/name
        if not p.resolve().is_relative_to(root) or sha(read_bytes(p)) != expected:
            raise Stop('SOURCE_BYTES_SHA')
    if not grant.synthetic:
        # Git objects are checked without checkout, fetch, configuration or maintenance.
        import subprocess
        for name, expected in c['source_hashes'].items():
            fixed = subprocess.check_output(['git', '-C', str(root), 'show', c['SOURCE']+':'+name],
                env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))
            if sha(fixed) != expected:
                raise Stop('SOURCE_COMMIT_BINDING')
    p = c['profile']
    if sys.version != p['python'] or sys.executable != p['executable']:
        raise Stop('RUNTIME_PROFILE')
    for name, expected in p['runtime_file_hashes'].items():
        if sha(Path(name).read_bytes()) != expected:
            raise Stop('RUNTIME_FILE_SHA')
    if any(os.environ.get(k) != '1' for k in THREAD_ENV):
        raise Stop('THREAD_ENV_REQUIRED')
    if len(list(Path('/proc/self/task').iterdir())) != 1:
        raise Stop('SINGLE_THREAD_REQUIRED')
    if not grant.synthetic:
        cpus = c['allowed_cpus']
        if not cpus or not set(os.sched_getaffinity(0)) <= set(cpus):
            raise Stop('APPROVED_OWN_AFFINITY_REQUIRED')


def load_inputs(grant, mapping):
    expected = {v['logical_name']: v for v in grant.contract['input_receipts']}
    if set(mapping) != set(expected):
        raise Stop('INPUT_MAPPING_SCOPE')
    data = {}
    for name, receipt in expected.items():
        value = read_bytes(mapping[name])
        if len(value) != receipt['bytes'] or sha(value) != receipt['sha256']:
            raise Stop('INPUT_BYTES_SHA:'+name)
        data[name] = value
    return data


def filesystem_snapshot(parent):
    stat = os.statvfs(parent)
    # The actual output filesystem, including nested mount points.
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    visible = [x for x in mounts if len(x) >= 4 and parent.is_relative_to(x[1])]
    if not visible:
        raise Stop('FILESYSTEM_MOUNT_UNRESOLVED')
    mount = max(visible, key=lambda x: len(x[1]))
    options = mount[3].split(',')
    quota_enabled = any('quota' in x and not x.startswith('no') for x in options)
    return dict(available_bytes=stat.f_bavail*stat.f_frsize, available_inodes=stat.f_favail,
        block_bytes=stat.f_frsize, mount=mount[1], fsid=stat.f_fsid,
        quota_enabled=quota_enabled, quota_options=options)


def check_filesystem(snapshot, budget, quota=None, *, now=None):
    for field in ('available_bytes', 'available_inodes'):
        if type(snapshot.get(field)) is not int or snapshot[field] < 0:
            raise Stop('FILESYSTEM_OBSERVATION_REQUIRED:'+field)
    if type(snapshot.get('quota_enabled')) is not bool:
        raise Stop('QUOTA_STATE_OBSERVATION_REQUIRED')
    if snapshot['available_bytes'] < budget['fresh_available_bytes'] or snapshot['available_inodes'] < budget['fresh_available_inodes']:
        raise Stop('FILESYSTEM_BLOCK_OR_INODE_HEADROOM')
    if snapshot['quota_enabled']:
        if (quota is None or any(type(quota.get(k)) is not int or quota[k] < 0
                for k in ('available_bytes', 'available_inodes')) or
                not isinstance(now, (int, float)) or isinstance(now, bool) or not math.isfinite(now) or
                not isinstance(quota.get('observed_monotonic'), (int, float)) or
                isinstance(quota['observed_monotonic'], bool) or not math.isfinite(quota['observed_monotonic']) or
                quota.get('uid') != os.getuid() or quota.get('fsid') != snapshot['fsid'] or
                quota.get('mount') != snapshot['mount'] or quota.get('all_user_group_project_limits_checked') is not True or
                now is None or not 0 <= now-quota.get('observed_monotonic', -math.inf) <= 5 or
                quota.get('available_bytes', 0) < budget['fresh_available_bytes'] or
                quota.get('available_inodes', 0) < budget['fresh_available_inodes']):
            raise Stop('FRESH_EFFECTIVE_QUOTA_RECEIPT_REQUIRED')


def validate_memory_observation(snapshot, baseline=None):
    """Do not let absent OOM counters or nonfinite PSI masquerade as zero.

    Keep the frozen D0 policy and arithmetic unchanged. oom_group_kill may be
    absent on older kernels, but its presence must stay consistent with admission.
    """
    if (type(snapshot.get('effective_available_bytes')) is not int or
            type(snapshot.get('host_oom_kill')) is not int or snapshot['host_oom_kill'] < 0 or
            type(snapshot.get('cgroups')) is not list):
        raise Stop('MEMORY_OBSERVATION_REQUIRED')
    def pressure(value):
        return (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100)
    if not pressure(snapshot.get('host_full_psi_avg10')):
        raise Stop('FINITE_HOST_PSI_OBSERVATION_REQUIRED')
    for index, group in enumerate(snapshot['cgroups']):
        events = group.get('events', {})
        if (not group.get('path') or not pressure(group.get('full_psi_avg10')) or
                any(type(events.get(k)) is not int or events[k] < 0 for k in ('oom', 'oom_kill')) or
                ('oom_group_kill' in events and
                    (type(events['oom_group_kill']) is not int or events['oom_group_kill'] < 0))):
            raise Stop('CGROUP_PSI_OOM_OBSERVATION_REQUIRED')
        if baseline is not None and index < len(baseline['cgroups']):
            old = baseline['cgroups'][index]['events']
            if ('oom_group_kill' in events) != ('oom_group_kill' in old):
                raise Stop('CGROUP_OOM_COUNTER_COVERAGE_CHANGED')


class Artifacts:
    """Exclusive durable marker, owned dirfd and bounded incremental JSON."""
    def __init__(self, grant):
        self.grant = grant
        self.root = grant.output
        self.bytes = 0
        self.files = 0
        self.receipts = []
        self.dirfd = None
        ticket = dict(contract_sha256=grant.contract_sha, authorization_sha256=grant.authorization_sha,
            SOURCE=grant.contract['SOURCE'], output=str(self.root), identity=identity(),
            parent_identity=identity(os.getppid()), nonce=secrets.token_hex(16))
        fd = os.open(marker_path(self.root), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, json.dumps(ticket, sort_keys=True).encode()+b'\n')
            os.fsync(fd)
        finally:
            os.close(fd)
        self.root.mkdir(mode=0o700)
        self.dirfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.fsync(self.dirfd)
        parentfd = os.open(self.root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parentfd)
        finally:
            os.close(parentfd)
        self.ticket = ticket

    def write(self, name, value, guard=lambda: None, *, terminal=False):
        if '/' in name or name in ('.', '..'):
            raise Stop('OUTPUT_NAME')
        limit = self.grant.contract['budget']['output_bytes']
        if terminal:
            limit += self.grant.contract['budget']['terminal_reserve_bytes']
        inode_limit = self.grant.contract['budget']['output_inode_cap']
        if terminal: inode_limit += self.grant.contract['budget']['terminal_inode_reserve']
        if self.files >= inode_limit:
            raise Stop('OUTPUT_INODE_CAP')
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.dirfd)
        self.files += 1
        checksum = hashlib.sha256(); written_bytes = 0
        try:
            for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False,
                    separators=(',', ':')).iterencode(value):
                payload = chunk.encode()
                if self.bytes+len(payload)+1 > limit:
                    raise Stop('OUTPUT_BYTES_CAP')
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                self.bytes += len(payload)
                checksum.update(payload); written_bytes += len(payload)
                guard()
            os.write(fd, b'\n'); self.bytes += 1
            checksum.update(b'\n'); written_bytes += 1
            os.fsync(fd)
            self.receipts.append(dict(name=name, bytes=written_bytes, sha256=checksum.hexdigest()))
        finally:
            os.close(fd)

    def close(self):
        if self.dirfd is not None:
            os.close(self.dirfd); self.dirfd = None


class Monitor:
    """Single process/thread: periodic signal plus explicit bounded checkpoints."""
    def __init__(self, grant, *, memory=d0.memory_snapshot, filesystem=filesystem_snapshot,
            quota=None, clock=time.monotonic):
        self.grant = grant; self.budget = grant.contract['budget']
        self.memory = memory; self.filesystem = filesystem; self.quota = quota; self.clock = clock
        self.started = clock(); self.last = self.started; self.first_stop = None; self.phase = 'PRELAUNCH'
        self.max_interval = 0.; self.max_observation = 0.; self.observations = 0
        self.history = []
        self.admission = memory(); validate_memory_observation(self.admission)
        self.affinity = os.sched_getaffinity(0)
        self.tick(force=True); self.handlers = {}; self.timer_installed = False

    def fail(self, reason):
        if self.first_stop is None:
            self.first_stop = dict(reason=str(reason)[:1000], phase=self.phase, monotonic=self.clock())
        raise Stop(self.first_stop['reason'])

    def tick(self, *, force=False):
        now = self.clock()
        if not force and now-self.last < .5:
            return
        gap = now-self.last; self.max_interval = max(self.max_interval, gap)
        if gap > self.budget['monitor_max_seconds']:
            self.fail('OBSERVATION_STALE')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if now-self.started > self.budget['wall_seconds']: self.fail('WALL_CAP')
        if usage.ru_utime+usage.ru_stime > self.budget['cpu_seconds']: self.fail('CPU_CAP')
        if usage.ru_maxrss*1024 > self.budget['rss_bytes']: self.fail('RSS_CAP')
        own_status = dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        if int(own_status['VmSize'].split()[0])*1024 > self.budget['address_space_bytes']:
            self.fail('AS_CAP')
        if os.sched_getaffinity(0) != self.affinity or len(list(Path('/proc/self/task').iterdir())) != 1:
            self.fail('OWN_AFFINITY_OR_THREAD_CHANGED')
        try:
            memory = self.memory(); filesystem = self.filesystem(self.grant.output.parent)
            validate_memory_observation(memory, self.admission)
            d0.check_memory_snapshot(memory, self.budget, self.admission)
            check_filesystem(filesystem, self.budget,
                self.quota() if self.quota else None, now=now)
        except Exception as error:
            self.fail(type(error).__name__+':'+str(error))
        duration = self.clock()-now; self.max_observation = max(self.max_observation, duration)
        if duration > self.budget['monitor_max_seconds']: self.fail('OBSERVATION_IO_STALE')
        # Start-to-start interval includes previous observation IO, so a slow
        # observation followed by blocked Python cannot hide a >5-second gap.
        self.last = now; self.observations += 1
        self.history.append(dict(phase=self.phase, monotonic=now, interval_seconds=gap,
            observation_seconds=duration, peak_RSS_bytes=usage.ru_maxrss*1024,
            CPU_seconds=usage.ru_utime+usage.ru_stime, effective_available_bytes=memory['effective_available_bytes'],
            host_memory_pressure=memory['host_full_psi_avg10'], filesystem_available_bytes=filesystem['available_bytes'],
            filesystem_available_inodes=filesystem['available_inodes']))
        self.history = self.history[-64:]

    def install(self):
        resource.setrlimit(resource.RLIMIT_AS, (self.budget['address_space_bytes'], self.budget['address_space_bytes']))
        resource.setrlimit(resource.RLIMIT_CPU, (self.budget['cpu_seconds'], self.budget['cpu_seconds']+2))
        def periodic(*_):
            signal.setitimer(signal.ITIMER_REAL, 0)
            try: self.tick(force=True)
            except Exception as error: self.fail(type(error).__name__+':'+str(error))
            finally:
                if self.first_stop is None: signal.setitimer(signal.ITIMER_REAL, .5)
        def stop_signal(sig, _): self.fail('OWN_SIGNAL:'+str(sig))
        for sig, handler in [(signal.SIGALRM, periodic), (signal.SIGXCPU, stop_signal),
                (signal.SIGTERM, stop_signal), (signal.SIGINT, stop_signal)]:
            self.handlers[sig] = signal.signal(sig, handler)
        signal.setitimer(signal.ITIMER_REAL, .5); self.timer_installed = True

    def close(self):
        if self.timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in self.handlers.items(): signal.signal(sig, handler)
            self.timer_installed = False
