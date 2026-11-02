"""Owned resource accounting. No host/cgroup/job mutations."""
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import re
import signal
import time
import threading
from .identity import Stop, require

GiB = 2**30
ROLE_CAP = 8 * GiB
HEADROOM = 16 * GiB
OUTPUT_CAP = 10 * GiB
WALL_CAP = 72 * 3600
# Linux v6.8 include/linux/proc_ns.h: PROC_CGROUP_INIT_INO.
# Unknown/private cgroup namespaces remain blocked rather than hiding ancestors.
INITIAL_CGROUP_NS_INO = 0xEFFFFFFB
MANAGED_WRITE_CONTEXT=threading.local()


def cpus(text):
    result = set()
    for item in text.strip().split(','):
        bounds = item.split('-')
        require(len(bounds) in (1, 2), 'CPU list')
        low, high = int(bounds[0]), int(bounds[-1])
        require(0 <= low <= high, 'CPU range')
        result.update(range(low, high + 1))
    require(bool(result), 'empty CPU list')
    return result


def admission(available, observed_at, requested, explicit_cpus, process_cpus, *, now=None):
    now = time.monotonic() if now is None else now
    require(0 <= now - observed_at <= 5, 'stale memory observation')
    require(type(available) is int and available >= 0, 'memory observation')
    require(type(requested) is int and requested >= 1, 'worker request')
    permitted = set(explicit_cpus) & set(process_cpus)
    for w in range(min(12, requested, len(permitted)), 0, -1):
        if available >= (8 + 8*w + 16)*GiB:
            return w
    raise Stop('even one worker requires 32 GiB; no launch')


def effective_available(host_available, limits):
    require(type(host_available) is int and host_available >= 0, 'host available')
    effective = host_available
    for maximum, current in limits:
        require(type(current) is int and current >= 0, 'cgroup current')
        if maximum is not None:
            require(type(maximum) is int and maximum >= current, 'cgroup limit/current')
            effective = min(effective, maximum - current)
    return effective


def cgroup_hierarchy(cgroup_text, mountinfo, namespace_link):
    """Prove visible hierarchy roots; return every member/non-root ancestor/root.

    '/' in mountinfo alone is insufficient inside a private cgroup namespace.
    The initial namespace marker plus full-root mounts and no shadow mounts is
    required. No ENOENT-based root inference or host-only fallback is used.
    """
    require(namespace_link == 'cgroup:[%d]' % INITIAL_CGROUP_NS_INO,
            'initial cgroup namespace not proven; hidden ancestors need review')
    memberships = []
    for line in cgroup_text.splitlines():
        parts = line.split(':', 2)
        require(len(parts) == 3 and parts[0].isdigit(), 'cgroup membership format')
        _hierarchy, controllers, path = parts
        if controllers == '' or 'memory' in controllers.split(','):
            member = PurePosixPath(path)
            require(member.is_absolute() and str(member) == path and '..' not in member.parts
                    and '\\' not in path, 'cgroup membership path visibility')
            memberships.append((controllers == '', PurePosixPath(path)))
    require(memberships and len({v2 for v2, member in memberships}) == len(memberships), 'ambiguous memory membership')
    mounts = []
    for line in mountinfo.splitlines():
        parts = line.split(' - ', 1)
        require(len(parts) == 2, 'mountinfo format')
        before, after = parts
        fields, suffix = before.split(), after.split()
        require(len(fields) >= 6 and len(suffix) >= 3, 'mountinfo fields')
        mounts.append((fields, suffix))
    directories = []
    for v2, member in memberships:
        matching = []
        for fields, suffix in mounts:
            kind, options = suffix[0], suffix[2]
            if (v2 and kind == 'cgroup2') or (not v2 and kind == 'cgroup' and 'memory' in options.split(',')):
                matching.append((fields, suffix))
        require(len(matching) == 1, 'missing/ambiguous cgroup memory mount')
        fields, suffix = matching[0]
        root, mount = PurePosixPath(fields[3]), Path(fields[4])
        require(root == PurePosixPath('/'), 'hidden upstream cgroup limits need review')
        require('\\' not in fields[3]+fields[4] and mount.is_absolute()
                and str(mount) == fields[4] and '..' not in mount.parts, 'escaped cgroup mount needs review')
        for other_fields, _other_suffix in mounts:
            if other_fields is fields:
                continue
            other_mount = Path(other_fields[4])
            require(not other_mount.is_relative_to(mount), 'shadow cgroup mount hides limits/interfaces')
        p = mount / str(member.relative_to(root))
        while True:
            directories.append({'path':p, 'v2':v2, 'hierarchy_root':mount, 'is_root':p == mount})
            if p == mount:
                break
            p = p.parent
    return directories


def cgroup_directories(cgroup_text, mountinfo, *, namespace_link=None):
    """Compatibility inventory; root-aware observer uses cgroup_hierarchy."""
    if namespace_link is None:
        namespace_link = os.readlink('/proc/self/ns/cgroup')
    return [(item['path'], item['v2']) for item in cgroup_hierarchy(cgroup_text, mountinfo, namespace_link)]


def nonnegative_integer(text, label):
    value = text.strip()
    require(re.fullmatch(r'[0-9]+', value) is not None, 'invalid nonnegative '+label)
    return int(value)


def psi_full_average(text):
    rows = [line.split() for line in text.splitlines() if line.startswith('full ')]
    require(len(rows) == 1, 'memory PSI full record')
    pairs = [word.split('=') for word in rows[0][1:]]
    require(all(len(pair) == 2 for pair in pairs), 'memory PSI fields')
    fields = dict(pairs)
    require(len(fields) == len(pairs) and {'avg10','avg60','avg300','total'} <= set(fields), 'memory PSI fields')
    for key in ('avg10','avg60','avg300'):
        try:
            value = float(fields[key])
        except ValueError as exc:
            raise Stop('invalid memory PSI average') from exc
        require(math.isfinite(value) and 0 <= value <= 100, 'invalid memory PSI average')
    nonnegative_integer(fields['total'], 'PSI total')
    return float(fields['avg10'])


def memory_oom_events(text):
    pairs = [line.split() for line in text.splitlines()]
    require(pairs and all(len(pair) == 2 for pair in pairs), 'memory.events fields')
    fields = dict(pairs)
    require(len(fields) == len(pairs) and {'oom','oom_kill'} <= set(fields), 'missing/duplicate memory OOM fields')
    values = {key:nonnegative_integer(value, 'memory.events '+key) for key,value in fields.items()}
    return values['oom'] + values['oom_kill']


def observe_memory():
    started = time.monotonic()
    membership = Path('/proc/self/cgroup').read_text()
    mountinfo = Path('/proc/self/mountinfo').read_text()
    namespace = os.readlink('/proc/self/ns/cgroup')
    dirs = cgroup_hierarchy(membership, mountinfo, namespace)
    meminfo = [line.split() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')]
    require(len(meminfo) == 1 and len(meminfo[0]) == 3 and meminfo[0][2] == 'kB', 'host MemAvailable fields/unit')
    available = nonnegative_integer(meminfo[0][1], 'host MemAvailable') * 1024
    allowed = next(l.split(':',1)[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('Cpus_allowed_list:'))
    # Host PSI covers global/root pressure. Every non-root pressure is also read.
    avg10 = psi_full_average(Path('/proc/pressure/memory').read_text())
    pressure_scopes = {'host': avg10}
    limits, events, hierarchy = [], {}, []
    for entry in dirs:
        p, v2 = entry['path'], entry['v2']
        item = {'path':str(p), 'v2':v2, 'is_root':entry['is_root'], 'maximum':None, 'current':None}
        if v2 and entry['is_root']:
            require('memory' in (p/'cgroup.controllers').read_text().split(), 'root memory controller unavailable')
            item['interface_policy'] = 'TRUE_V2_ROOT_NO_MEMORY_CONTROL_INTERFACES_HOST_PSI'
            hierarchy.append(item)
            continue  # proven root only; non-root ENOENT/permission errors propagate
        if v2:
            raw_maximum = (p/'memory.max').read_text().strip()
            maximum = None if raw_maximum == 'max' else nonnegative_integer(raw_maximum, 'memory.max')
            current = nonnegative_integer((p/'memory.current').read_text(), 'memory.current')
            events[str(p)] = memory_oom_events((p/'memory.events').read_text())
            pressure_scopes[str(p)] = psi_full_average((p/'memory.pressure').read_text())
            avg10 = max(avg10, pressure_scopes[str(p)])
        else:
            raw_maximum = nonnegative_integer((p/'memory.limit_in_bytes').read_text(), 'memory.limit_in_bytes')
            maximum = None if raw_maximum >= 2**60 else raw_maximum
            current = nonnegative_integer((p/'memory.usage_in_bytes').read_text(), 'memory.usage_in_bytes')
            events[str(p)] = nonnegative_integer((p/'memory.failcnt').read_text(), 'memory.failcnt')
        limits.append((maximum, current))
        item.update(maximum=maximum, current=current, interface_policy='ALL_REQUIRED_MEMORY_INTERFACES_CHECKED')
        hierarchy.append(item)
    require(Path('/proc/self/cgroup').read_text() == membership and Path('/proc/self/mountinfo').read_text() == mountinfo
            and os.readlink('/proc/self/ns/cgroup') == namespace, 'cgroup visibility changed during observation')
    return dict(available=effective_available(available, limits), observed_at=started,
                process_cpus=cpus(allowed), oom_events=events, psi_full_avg10=avg10,
                host_available=available, psi_full_by_scope=pressure_scopes,
                cgroup_namespace=namespace, hierarchy=hierarchy,
                root_pressure_policy='HOST_MEMORY_PSI_PLUS_ALL_NONROOT_MEMORY_PSI')


def require_inherited_address_space(cap):
    require(type(cap) is int and cap in (ROLE_CAP, 32*GiB), 'closed owned AS cap')
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    require(hard == resource.RLIM_INFINITY or hard >= cap, 'inherited AS hard limit too small')


def limit_owned_address_space(cap=ROLE_CAP, *, preserve_hard=False):
    require_inherited_address_space(cap)
    require(type(preserve_hard) is bool, 'AS hard-limit mode')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    selected = (cap, hard if preserve_hard else cap)
    resource.setrlimit(resource.RLIMIT_AS, selected)
    require(resource.getrlimit(resource.RLIMIT_AS) == selected, 'owned AS limit did not bind')


class Monitor:
    """Only explicitly registered children, identified by PID + starttime + parent."""
    def __init__(self, workers, observation, *, clock=time.monotonic,pressure_profile=None,pressure_baseline=None):
        self.workers, self.clock = workers, clock
        self.last = observation['observed_at']
        self.baseline = observation['oom_events']
        self.pressure_guard=None
        if pressure_profile is not None:
            from .pressure_policy import PressureGuard
            self.pressure_guard=PressureGuard(observation if pressure_baseline is None else pressure_baseline,pressure_profile)
            self.baseline=self.pressure_guard.baseline_oom
        self.children = {}
        self.check(observation, [0])

    def own_child(self, pid):
        raw = Path('/proc/%d/stat' % pid).read_text().rsplit(')',1)[1].split()
        require(int(raw[1]) == os.getpid(), 'refuse foreign child')
        self.children[pid] = raw[19]

    def stop_children(self):
        for pid, start in list(self.children.items()):
            try:
                raw = Path('/proc/%d/stat' % pid).read_text().rsplit(')',1)[1].split()
                if raw[19] == start and int(raw[1]) == os.getpid():
                    os.kill(pid, signal.SIGTERM)
            except FileNotFoundError:
                pass

    def check(self, observation, rss):
        now = self.clock()
        require(0 <= now-self.last <= 5 and 0 <= now-observation['observed_at'] <= 5, 'monitor interval/freshness')
        self.last = now
        require(observation['available'] >= HEADROOM, 'memory headroom pressure')
        if self.pressure_guard is None:require(observation['psi_full_avg10'] == 0, 'memory PSI pressure')
        else:
            decision=self.pressure_guard.decision(observation,now=now)
            require(decision['reason'] is None,'memory PSI pressure: '+str(decision['reason']))
        require(observation['oom_events'].keys() == self.baseline.keys() and
                all(observation['oom_events'][k] == v for k, v in self.baseline.items()), 'OOM or changed cgroup')
        require(len(rss) <= self.workers+1 and all(0 <= v <= ROLE_CAP for v in rss) and
                sum(rss) <= ROLE_CAP*(self.workers+1), 'own RSS budget')

    def poll(self):
        try:
            rss = []
            for pid in [os.getpid(), *self.children]:
                if pid != os.getpid():
                    raw = Path('/proc/%d/stat' % pid).read_text().rsplit(')',1)[1].split()
                    require(raw[19] == self.children[pid] and int(raw[1]) == os.getpid(), 'child ownership lost')
                lines = Path('/proc/%d/status' % pid).read_text().splitlines()
                rss.append(int(next(l for l in lines if l.startswith('VmRSS:')).split()[1])*1024)
            self.check(observe_memory(), rss)
        except (Stop, MemoryError, OSError):
            self.stop_children()
            raise


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class OutputBudget:
    """Conservative cumulative bytes written (includes temporary files and journal).

    Rewrites never reclaim budget. Stage handoff consumes existing budget only
    through a separately reviewed launch; no automatic resume is supported.
    """
    def __init__(self, root, *, cap=OUTPUT_CAP, handoff=False, prior_charge=0,file_limits=None):
        self.root, self.cap = Path(root), cap
        require(type(prior_charge) is int and 0 <= prior_charge < cap and
                (not handoff or prior_charge == 0), 'explicit fresh-run prior charge')
        require(not prior_charge or prior_charge+128 <= cap, 'prior charge plus new journal row')
        self.thread_lock=threading.Lock()
        self.file_limits=file_limits
        self.cached_charge=None;self.cached_offset=0
        require(self.root.is_absolute(), 'absolute own output root')
        require(not any(p.is_symlink() for p in [self.root, *self.root.parents]), 'output symlink')
        self.root.mkdir(parents=True,exist_ok=handoff,mode=0o700)
        fsync_directory(self.root.parent)
        self.fd = os.open(self.root/'byte-budget.journal', os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW |
                          (0 if handoff else os.O_CREAT | os.O_EXCL), 0o600)
        if prior_charge:
            require(prior_charge+128 <= self.cap, 'prior charge plus new journal row')
            require(os.write(self.fd, (str(prior_charge+128)+'\n').encode().ljust(128, b' '))==128,'prior carry journal write')
            os.fsync(self.fd)

    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None

    def reserve(self, size):
        require(type(size) is int and size >= 0, 'byte reservation')
        with self.thread_lock:
            self._reserve_locked(size)

    def _reserve_locked(self,size):
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            journal_size=os.fstat(self.fd).st_size
            if self.cached_charge is None:
                os.lseek(self.fd,0,os.SEEK_SET);raw=b''
                while part:=os.read(self.fd,65536):raw+=part
                require(len(raw)%128==0,'ambiguous byte reservation STOP')
                charges=[int(raw[i:i+128].strip()) for i in range(0,len(raw),128)]
                require(all(v>=128 for v in charges),'invalid/refunded byte-journal row')
                self.cached_charge=sum(charges)
                self.cached_offset=len(raw)
            # This object is the only writer; foreign append/truncation stops.
            # No O(N^2) whole-journal reread for hundreds of thousands of rows.
            require(journal_size==self.cached_offset,'foreign byte-journal writer/size change STOP')
            used=self.cached_charge
            charge = 2*size + 128  # temp + final, journal itself
            require(used + charge <= self.cap, 'output budget before write')
            require(os.write(self.fd, (str(charge)+'\n').encode().ljust(128, b' '))==128,'byte reservation journal write')
            os.fsync(self.fd)
            self.cached_charge+=charge;self.cached_offset+=128
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def write(self, name, data):
        require(type(data) is bytes and isinstance(name, str), 'output type')
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and len(relative.parts)==1, 'output escape')
        if self.file_limits is not None:
            matches=[cap for prefix,cap in self.file_limits.items() if name.startswith(prefix)]
            require(len(matches)==1 and len(data)<=matches[0],'sealed per-file output cap')
        self.reserve(len(data))
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        tmp = name + '.pending'
        MANAGED_WRITE_CONTEXT.root=self.root
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # link is atomic and exclusive, unlike overwrite-capable rename.
            os.link(tmp, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            os.fsync(directory)
            os.unlink(tmp, dir_fd=directory)
            os.fsync(directory)
        finally:
            MANAGED_WRITE_CONTEXT.root=None
            os.close(directory)


class WallBudget:
    def __init__(self, prior_consumed, *, clock=time.monotonic):
        require(0 <= prior_consumed < WALL_CAP, 'cumulative wall budget')
        self.prior, self.clock, self.start = prior_consumed, clock, clock()

    def consumed(self):
        used = self.prior + self.clock() - self.start
        require(0 <= used <= WALL_CAP, 'cumulative 72h STOP')
        return used
