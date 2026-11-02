"""Owned resource accounting. No host/cgroup/job mutations."""
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import time
import threading
from .identity import Stop, require

GiB = 2**30
ROLE_CAP = 8 * GiB
HEADROOM = 16 * GiB
OUTPUT_CAP = 10 * GiB
WALL_CAP = 72 * 3600


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


def cgroup_directories(cgroup_text, mountinfo):
    """Resolve every applicable v1 memory/v2 mount and ancestors to its mount root."""
    memberships = []
    for line in cgroup_text.splitlines():
        _hierarchy, controllers, path = line.split(':', 2)
        if controllers == '' or 'memory' in controllers.split(','):
            memberships.append((controllers == '', PurePosixPath(path)))
    directories = []
    for line in mountinfo.splitlines():
        before, after = line.split(' - ', 1)
        fields, suffix = before.split(), after.split()
        kind, options = suffix[0], suffix[2]
        for v2, member in memberships:
            if (v2 and kind != 'cgroup2') or (not v2 and (kind != 'cgroup' or 'memory' not in options.split(','))):
                continue
            root, mount = PurePosixPath(fields[3]), Path(fields[4])
            require(root==PurePosixPath('/'), 'hidden upstream cgroup limits need review')
            require('\\' not in fields[3]+fields[4], 'escaped cgroup mount needs review')
            require(member.is_relative_to(root), 'cgroup namespace/mount mismatch')
            p = mount / str(member.relative_to(root))
            while True:
                directories.append((p, v2))
                if p == mount:
                    break
                p = p.parent
    require(bool(directories), 'missing cgroup memory hierarchy')
    return list(dict.fromkeys(directories))


def observe_memory():
    started = time.monotonic()
    fields = {l.split(':')[0]: l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines()}
    available = int(fields['MemAvailable'].split()[0]) * 1024
    allowed = next(l.split(':',1)[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('Cpus_allowed_list:'))
    dirs = cgroup_directories(Path('/proc/self/cgroup').read_text(), Path('/proc/self/mountinfo').read_text())
    limits, events = [], {}
    for p, v2 in dirs:
        if v2:
            maximum = (p/'memory.max').read_text().strip()
            limits.append((None if maximum == 'max' else int(maximum), int((p/'memory.current').read_text())))
            e = dict(line.split() for line in (p/'memory.events').read_text().splitlines())
            events[str(p)] = int(e.get('oom', 0)) + int(e.get('oom_kill', 0))
        else:
            # Kernel v1 unlimited sentinel is not a usable physical reservation.
            maximum = int((p/'memory.limit_in_bytes').read_text())
            limits.append((None if maximum >= 2**60 else maximum, int((p/'memory.usage_in_bytes').read_text())))
            events[str(p)] = int((p/'memory.failcnt').read_text())
    psi = Path('/proc/pressure/memory').read_text()
    full = next(l for l in psi.splitlines() if l.startswith('full '))
    avg10 = float(dict(x.split('=') for x in full.split()[1:])['avg10'])
    for p, v2 in dirs:
        if v2:
            local = next(l for l in (p/'memory.pressure').read_text().splitlines() if l.startswith('full '))
            avg10 = max(avg10, float(dict(x.split('=') for x in local.split()[1:])['avg10']))
    return dict(available=effective_available(available, limits), observed_at=started,
                process_cpus=cpus(allowed), oom_events=events, psi_full_avg10=avg10)


def limit_owned_address_space():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = ROLE_CAP if hard == resource.RLIM_INFINITY else min(ROLE_CAP, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


class Monitor:
    """Only explicitly registered children, identified by PID + starttime + parent."""
    def __init__(self, workers, observation, *, clock=time.monotonic):
        self.workers, self.clock = workers, clock
        self.last = observation['observed_at']
        self.baseline = observation['oom_events']
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
        require(observation['psi_full_avg10'] == 0, 'memory PSI pressure')
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
    def __init__(self, root, *, cap=OUTPUT_CAP, handoff=False):
        self.root, self.cap = Path(root), cap
        self.thread_lock=threading.Lock()
        require(self.root.is_absolute(), 'absolute own output root')
        require(not any(p.is_symlink() for p in [self.root, *self.root.parents]), 'output symlink')
        self.root.mkdir(parents=True,exist_ok=handoff)
        fsync_directory(self.root.parent)
        self.fd = os.open(self.root/'byte-budget.journal', os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW |
                          (0 if handoff else os.O_CREAT | os.O_EXCL), 0o600)

    def close(self):
        os.close(self.fd)

    def reserve(self, size):
        require(type(size) is int and size >= 0, 'byte reservation')
        with self.thread_lock:
            self._reserve_locked(size)

    def _reserve_locked(self,size):
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            raw = b''
            while part := os.read(self.fd, 65536):
                raw += part
            require(len(raw) % 128 == 0, 'ambiguous byte reservation STOP')
            used = sum(int(raw[i:i+128].strip()) for i in range(0, len(raw), 128))
            charge = 2*size + 128  # temp + final, journal itself
            require(used + charge <= self.cap, 'output budget before write')
            os.write(self.fd, (str(charge)+'\n').encode().ljust(128, b' '))
            os.fsync(self.fd)
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def write(self, name, data):
        require(type(data) is bytes and isinstance(name, str), 'output type')
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and len(relative.parts)==1, 'output escape')
        self.reserve(len(data))
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        tmp = name + '.pending'
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
            os.close(directory)


class WallBudget:
    def __init__(self, prior_consumed, *, clock=time.monotonic):
        require(0 <= prior_consumed < WALL_CAP, 'cumulative wall budget')
        self.prior, self.clock, self.start = prior_consumed, clock, clock()

    def consumed(self):
        used = self.prior + self.clock() - self.start
        require(0 <= used <= WALL_CAP, 'cumulative 72h STOP')
        return used
