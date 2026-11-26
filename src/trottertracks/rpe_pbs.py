"""PBS allocations attached to the existing controller's durable task queue.

PBS submission is not a scientific task. Only an allocated worker's live pull
offer participates in matching; queued/held PBS jobs contribute no free slots.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import tempfile
import threading
from statistics import median
import time
import uuid

from .rpe_q3_timing import stamp, append
from .rpe_q3_worker import atomic, digest, read
from .rpe_pbs_worker import runtime_cpu_footprint, task_cpu_footprint
from .rpe_worker_storage import SAMPLE_SECONDS, prefix_caches


TERMINAL = {'F', 'X', 'FINISHED_SAVED_WORKER', 'SUBMIT_REJECTED', 'SUPERSEDED_UNSUBMITTED'}
GIB = 1024**3
# The controller and the independently pinned worker use the same proof lifetime.
Q3_WORKER_PROOF_SECONDS = 120


def standby_capacity(pool, waves=None):
    """Unstarted backlog uses task resources as weights, not as reservations."""
    cpu, memory = pool['cpu_count'], max(0, pool['memory_bytes'] - GIB)
    if waves is None:
        return cpu // 4, memory // 4
    if type(waves) is not int or waves < 1:
        raise ValueError('PBS standby waves must be a positive integer')
    return cpu * waves, memory * waves

# Measure the same shared paths from the gateway, where directory walks do
# not occupy a PBS compute slot or depend on a compute node's NFS latency.
_STORAGE_MONITOR = """
import json, select, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1])/'src'))
from trottertracks.rpe_worker_storage import SAMPLE_SECONDS, report
profile = json.loads(sys.stdin.readline())
while True:
    report(profile['policy'], spools=profile['spools'],
           cache_directories=profile['caches'], refresh=True)
    if select.select([sys.stdin], [], [], max(1., SAMPLE_SECONDS-2.))[0]:
        break
"""

_STAGE_ARTIFACT = """
import hashlib, os, sys, tempfile
from pathlib import Path

path, expected = Path(sys.argv[1]), sys.argv[2]
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=path.parent, prefix=expected+'.transfer-', delete=False) as output:
    temporary = Path(output.name)
    digest = hashlib.sha256()
    try:
        for block in iter(lambda: sys.stdin.buffer.read(1024 * 1024), b''):
            digest.update(block)
            output.write(block)
        if digest.hexdigest() != expected:
            raise ValueError('PBS artifact stream differs')
        output.flush()
        os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open('rb') as existing:
                if path.is_symlink() or hashlib.file_digest(existing, 'sha256').hexdigest() != expected:
                    raise ValueError('Conflicting cached PBS artifact')
        print(expected)
    finally:
        temporary.unlink()
"""

_CHECK_STORAGE_LAUNCHER = """
import hashlib, json, os, stat, subprocess, sys, tempfile
from pathlib import Path

path, expected = Path(sys.argv[1]), sys.argv[2]
with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as launcher:
    info = os.fstat(launcher.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError('PBS storage launcher must be a regular file owned by this user')
    if hashlib.sha256(launcher.read()).hexdigest() != expected:
        raise ValueError('PBS storage launcher differs from the selected deployment')
    if not info.st_mode & stat.S_IXUSR:
        os.fchmod(launcher.fileno(), stat.S_IMODE(info.st_mode) | stat.S_IXUSR)
with tempfile.TemporaryDirectory(prefix='rpe-pbs-launcher-check-') as directory:
    profile = Path(directory)/'PROFILE.json'
    profile.write_text(json.dumps(dict(pbs=dict(allocation_dir=directory, compute_storage_sampling=True))))
    subprocess.run([str(path), '--check-imports', '-B', sys.argv[3], '--profile', str(profile)],
                   stdout=subprocess.DEVNULL, check=True, timeout=12)
"""


class SharedSSH:
    """One authenticated transport and reconnect deadline per login host."""
    def __init__(self, alias):
        self.alias = alias

    def command(self):
        alias = self.alias
        base = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
        # Check the effective user policy before using even an existing master.
        config = subprocess.check_output([*base, '-G', alias], text=True, timeout=5)
        if any(line in ('proxycommand /usr/bin/false', 'proxycommand /bin/false')
               for line in config.splitlines()):
            raise OSError('PBS SSH access is paused')
        root = Path(tempfile.gettempdir())/f'rpe-pbs-ssh-{os.getuid()}'
        root.mkdir(mode=0o700, exist_ok=True)
        if root.is_symlink() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
            raise ValueError('PBS SSH control sockets require a private directory')
        key = hashlib.sha256(('rpe-pbs-control-v2\0'+config).encode()).hexdigest()[:16]
        socket = root/key
        retry = socket.with_suffix('.retry.json')
        options = [*base, '-S', str(socket)]
        with socket.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not subprocess.run([*options, '-O', 'check', alias], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=5).returncode:
                return [*options, '-o', 'ControlMaster=no', '-o', 'ProxyCommand=/bin/false', '-T', alias]
            previous = read(retry) if retry.is_file() else {}
            now = time.time()
            if now < previous.get('retry_after', 0):
                raise OSError('PBS SSH reconnect backoff active')
            try:
                subprocess.run([*options, '-o', 'ControlMaster=yes', '-o', 'ControlPersist=60',
                    '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', '-NTf', alias],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=12, check=True)
            except (OSError, subprocess.SubprocessError):
                delay = min(60, max(5, previous.get('delay', 0)*2))
                atomic(retry, dict(delay=delay, retry_after=time.time()+delay))
                raise
            retry.unlink(missing_ok=True)
        # If the master disappears, a channel must never silently authenticate.
        return [*options, '-o', 'ControlMaster=no', '-o', 'ProxyCommand=/bin/false', '-T', alias]


class Gateway:
    """A private persistent SSH JSON-lines connection; reconnect is read/replay."""
    def __init__(self, settings, ssh=None):
        self.settings = settings
        self.ssh = ssh or (SharedSSH(settings['ssh_alias']) if 'ssh_alias' in settings else None)
        self.process = None
        self.call_lock = threading.Lock()
        self.responses = queue.Queue()
        self.storage_monitors = {}
        self.storage_monitor_started = {}
        self.storage_monitor_sample_paths = {}

    def _start(self):
        config = self.settings
        if config.get('gateway_sha256'):
            self.verify_gateway_file(config['gateway_script'], config['gateway_sha256'])
        remote = [config.get('gateway_python', '/usr/bin/python3'), '-B',
                  config['gateway_script'], '--root', config['remote_root']]
        command = config.get('gateway_command') or [*self.ssh.command(), shlex.join(remote)]
        self.responses = queue.Queue()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        process, responses = self.process, self.responses

        def collect():
            try:
                for line in process.stdout:
                    responses.put(line)
            finally:
                responses.put(None)
        threading.Thread(target=collect, daemon=True).start()

    def call(self, payload):
        with self.call_lock:
            return self._call(payload)

    def _call(self, payload):
        if self.process is None or self.process.poll() is not None:
            self._start()
        try:
            self.process.stdin.write(json.dumps(payload, separators=(',', ':'))+'\n')
            self.process.stdin.flush()
            try:
                line = self.responses.get(timeout=self.settings.get('request_timeout_seconds', 20))
            except queue.Empty as error:
                raise TimeoutError(f"PBS gateway {payload['action']} response timed out") from error
            if line is None:
                raise OSError('PBS gateway SSH connection closed')
            value = json.loads(line)
            if not value.get('ok'):
                raise ValueError(value.get('error', 'PBS gateway failed'))
            return value['result']
        except (OSError, ValueError, queue.Empty):
            self.close()
            raise

    def storage_report(self, policy):
        config = self.settings
        remote = [config.get('gateway_python', '/usr/bin/python3'), '-B',
                  str(Path(config['transport_root'])/'scripts/run_rpe_q3_worker.py'), '--storage-limit-report']
        root = Path(config['remote_root'])
        for worker in config['workers']:
            remote.extend(('--storage-spool', str(root/'workers'/worker['host_id'])))
        if config.get('namespace_by_run', True):
            remote.extend(('--storage-pbs-runs', str(root.parent)))
        output = subprocess.check_output(
            [*self.ssh.command(), shlex.join(remote)], input=json.dumps(policy), text=True, timeout=15)
        return json.loads(output)

    def verify_gateway_file(self, path, expected):
        output = subprocess.check_output(
            [*self.ssh.command(), shlex.join(['sha256sum', '--', path])],
            text=True, timeout=12)
        if not output.split() or output.split()[0] != expected:
            raise ValueError('PBS gateway file differs from the selected deployment')

    def verify_storage_launcher(self, launcher):
        config = self.settings
        command = [config['gateway_python'], '-B', '-c', _CHECK_STORAGE_LAUNCHER,
                   launcher['path'], launcher['sha256'],
                   str(Path(config['transport_root'])/'scripts/run_rpe_pbs_worker.py')]
        subprocess.run([*self.ssh.command(), shlex.join(command)], check=True,
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=20)

    def stage_artifact(self, source, destination, sha):
        config = self.settings
        command = [config['gateway_python'], '-B', '-c', _STAGE_ARTIFACT, str(destination), sha]
        with source.open('rb') as stream:
            completed = subprocess.run(
                [*self.ssh.command(), shlex.join(command)], stdin=stream, capture_output=True,
                timeout=config.get('request_timeout_seconds', 20), check=True)
        if completed.stdout.strip() != sha.encode():
            raise ValueError('PBS artifact acknowledgement differs')

    def monitor_storage(self, profiles):
        """Keep the shared sample fresh only for running allocations in this run."""
        for host in self.storage_monitors.keys() - profiles.keys():
            self._stop_storage_monitor(self.storage_monitors.pop(host))
            self.storage_monitor_started.pop(host, None)
            self.storage_monitor_sample_paths.pop(host, None)
        for host, profile in profiles.items():
            previous = self.storage_monitors.get(host)
            sample_path = profile['storage_limit']['sample_path']
            if previous is not None and previous.poll() is None:
                if self.storage_monitor_sample_paths.get(host) == sample_path:
                    continue
                self._stop_storage_monitor(previous)
                self.storage_monitor_started.pop(host, None)
            now = time.monotonic()
            if now - self.storage_monitor_started.get(host, float('-inf')) < SAMPLE_SECONDS:
                continue
            config = self.settings
            remote = [config.get('gateway_python', '/usr/bin/python3'), '-B', '-c',
                      _STORAGE_MONITOR, config['transport_root']]
            process = subprocess.Popen(
                [*self.ssh.command(), shlex.join(remote)], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            self.storage_monitors[host] = process
            self.storage_monitor_started[host] = now
            self.storage_monitor_sample_paths[host] = sample_path
            try:
                process.stdin.write(json.dumps(dict(policy=profile['storage_limit'],
                    spools=[profile['work_dir']], caches=prefix_caches(profile)))+'\n')
                process.stdin.flush()
            except OSError:
                self._stop_storage_monitor(self.storage_monitors.pop(host))
                raise

    @staticmethod
    def _stop_storage_monitor(process):
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self):
        self.monitor_storage({})
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
            self.process = None


def q3_walltime_workers(rule, settings):
    """Resolve the runtime through the release's existing fixed worker profiles."""
    return [worker for worker in settings.get('workers', [])
            if any((route['route_kind'], route['view_commit'],
                    route['profile'].get('adapter_manifest', {}).get('commit'))
                   == (rule['route_kind'], rule['view_commit'], rule['adapter_source'])
                   for route in worker['profile'].get('route_profiles', []))]


def apply_walltime_policy(settings, policy, overrides, *, computation_source=None):
    """Resolve one release policy into the existing immutable PBS run settings."""
    if policy['schema'] != 'RPE_PBS_WALLTIME_POLICY_V1':
        raise ValueError('Unsupported PBS walltime policy')
    if (policy['computation_source'] != computation_source
            or policy['transport_source'] != settings.get('transport_source')):
        raise ValueError('PBS walltime policy SOURCE differs from selected release')
    q3 = deepcopy(policy['q3'])
    preparation = deepcopy(policy['preparation'])
    rules = q3 + preparation
    ids = {row['id'] for row in rules}
    if (len(ids) != len(rules) or set(overrides) - ids
            or any(type(value) is not int or value < 0 for value in overrides.values())):
        raise ValueError('PBS overrides require known rule IDs and nonnegative integer seconds')
    for rows, keys in ((q3, ('method', 'route_kind', 'view_commit')),
                       (preparation, ('stage', 'provider', 'source'))):
        if len({tuple(row[key] for key in keys) for row in rows}) != len(rows):
            raise ValueError('Ambiguous PBS walltime estimate')
    for row in rules:
        if type(row['seconds']) is not int or row['seconds'] <= 0:
            raise ValueError('Release PBS estimates must be positive integer seconds')
        if 'memory_bytes' in row and (type(row['memory_bytes']) is not int or row['memory_bytes'] <= 0):
            raise ValueError('Release PBS memory estimates must be positive integer bytes')
        row['seconds'] = overrides.get(row['id'], row['seconds'])
    for row in q3:
        row['hosts'] = [w['host_id'] for w in q3_walltime_workers(row, settings)]
        if not row['hosts']:
            raise ValueError('PBS walltime rule has no matching fixed route profile: ' + row['id'])
    settings.update(walltime_policy=dict(id=policy['id'], sha256=digest(policy)),
                    q3_task_seconds=q3, preparation_task_seconds=preparation, task_seconds={})
    if 'allocation_sizing' in policy:
        sizing = policy['allocation_sizing']
        memory = sizing['q3_startup_memory_bytes_per_task']
        if type(memory) is not int or memory <= 0:
            raise ValueError('PBS Q3 startup memory estimate must be positive integer bytes')
        settings['allocation_sizing'] = deepcopy(sizing)


def task_rule(request, settings, host_id=None):
    preparation = request.get('preparation') or {}
    stage = preparation.get('stage', 'q3')
    rules = settings.get('preparation_task_seconds')
    if rules is not None and preparation:
        provider = preparation.get('provider')
        source = (preparation.get('source') or {}).get('commit')
        matches = [rule for rule in rules if (rule['stage'], rule['provider'], rule['source'])
                   == (stage, provider, source)]
        return matches[0] if len(matches) == 1 else None
    if not preparation and 'q3_task_seconds' in settings:
        route = request.get('route') or {}
        values = (request.get('method'), route.get('kind'), route.get('view_commit'))
        matches = [rule for rule in settings['q3_task_seconds']
                   if values == (rule['method'], rule['route_kind'], rule['view_commit'])
                   and (host_id is None or host_id in rule['hosts'])]
        return matches[0] if len(matches) == 1 else None
    return dict(seconds=(settings.get('task_seconds') or {}).get(stage))


def task_estimate(request, settings, host_id=None):
    from .rpe_q3_resources import preparation_forecast
    baseline = (task_rule(request, settings, host_id) or {}).get('seconds')
    forecast = preparation_forecast(request, settings.get('preparation_resource_observations'), host=host_id).get('seconds')
    return max(baseline or 0, forecast or 0) or None


def task_memory_estimate(request, settings, host_id=None):
    return (task_rule(request, settings, host_id) or {}).get('memory_bytes')


def known_independent_search_jobs(run_dir):
    """Find only scheduler jobs bound to a different registered search."""
    run_dir=Path(run_dir).resolve()
    central=next((p for p in run_dir.parents if p.name=='.rpe-central'),None)
    if central is None:
        return []
    try:
        registry=json.loads((central/'state/RUNS.json').read_text())
    except (OSError,ValueError):
        return []
    jobs=set()
    for run_id,record in registry.items():
        directory=Path(record.get('run_directory','')).resolve()
        if directory==run_dir or not run_id.startswith('search-'):
            continue
        try:
            state=json.loads((directory/'distributed/pbs/STATE.json').read_text())
            records=state['allocations'].values()
        except (OSError,ValueError,KeyError,TypeError):
            continue
        for allocation in records:
            try:
                if Path(allocation['spec']['run_id']).resolve()!=directory:
                    continue
                observed=allocation['observed'];scheduler=observed['scheduler']
                if (allocation['status'] in ('Q','R','H','W','E') and
                        scheduler['job_state'] in ('Q','R','H','W','E') and
                        scheduler['Job_Name']=='RPEsearch-'+run_id[7:12] and observed.get('job_id')
                        and not observed.get('pilot_token')):
                    jobs.add(observed['job_id'])
            except (OSError,ValueError,KeyError,TypeError):
                continue
    return sorted(jobs)


def usable_memory(profile, settings):
    amount, unit = re.fullmatch(r'(\d+)([kmgt]?b)?', settings['resources']['mem'].lower()).groups()
    memory = int(amount) * (1024 ** ('b', 'kb', 'mb', 'gb', 'tb').index(unit or 'b'))
    limit = profile.get('slot_policy', {}).get('resource_limits', {}).get('memory_bytes', memory)
    return min(memory, limit) // GIB * GIB - GIB


def measured_q3_memory(request, runtime, host, estimates):
    """Match a measured envelope to the fixed evaluator route and shot range."""
    if not estimates or request.get('preparation'):
        return None
    x, y = request.get('N_X', ()), request.get('N_Y', ())
    if not x or len(x) != len(y):
        return None
    route = request['route']
    risk = (route.get('horizon_risk_source') or {}).get('commit')
    adapter = (runtime.get('adapter_manifest') or {}).get('commit')
    peak_shots = max(max(x), max(y))
    return max((row['memory_bytes'] for row in estimates
                if (row['host'], row['method'], row['route_kind'], row['view_commit'],
                    row['horizon_risk_commit'], row['adapter_source'], row['dp_workers'], row['rounds'])
                == (host, request['method'], route['kind'], route.get('view_commit'),
                    risk, adapter, runtime.get('dp_workers'), len(x))
                and row.get('interpreter', runtime.get('interpreter')) == runtime.get('interpreter')
                and row['min_peak_shots'] <= peak_shots <= row['max_peak_shots']), default=None)


def resource_need(request, profile, settings, host):
    from .rpe_q3_worker import select_host_profile

    memory = task_memory_estimate(request, settings, host)
    # A resumed controller may import the original run's profile selector,
    # whose route key predates the horizon binding. Give it only the matching
    # horizon before accounting for that specific worker's CPUs.
    route = request.get('route') or {}
    horizon = (route.get('horizon_risk_source') or {}).get('commit')
    if profile.get('route_profiles'):
        profile = dict(profile, route_profiles=[row for row in profile['route_profiles']
                       if row.get('horizon_risk_commit') == horizon])
    runtime = select_host_profile(profile, {'request': request})
    if memory is None:
        # A stage-specific resource declaration is already a verified profile
        # input; do not override it with the entire node. Unmeasured stages
        # retain conservative isolation until comparable observations exist.
        declared = runtime.get('task_memory_bytes') if request.get('preparation') else None
        memory = declared if type(declared) is int and declared > 0 else (
            usable_memory(profile, settings) if request.get('preparation') else settings['allocation_sizing']['q3_startup_memory_bytes_per_task'])
    measured = measured_q3_memory(request, runtime, host, settings.get('q3_measured_memory'))
    if measured is not None:
        memory = measured
    from .rpe_q3_resources import preparation_forecast
    cpu = runtime_cpu_footprint(profile, runtime)
    forecast = preparation_forecast(request, settings.get('preparation_resource_observations'), cpu_count=cpu, host=host)
    return (cpu, max(1, memory, forecast.get('memory_bytes', 0), (profile.get('pbs') or {}).get('task_memory_bytes') or 0))


def packed_resources(requests, profile, settings, host, *, selected=None):
    resources = settings['resources']
    limits = profile.get('slot_policy', {}).get('resource_limits', {})
    cpu_limit = min(resources['ncpus'], limits.get('cpu_count', resources['ncpus']))
    memory_limit = usable_memory(profile, settings)
    cpus = memory = count = reserved_storage = 0
    storage_limit = (profile.get('storage_limit') or {}).get('limit_bytes')
    from .rpe_q3_resources import preparation_forecast
    max_slots = profile.get('slot_policy', {}).get('max_slots') or cpu_limit
    # A preparation task without a memory estimate needs the whole pool. Give
    # tasks with a usable estimate a chance to share it before that fallback.
    ordered = sorted(requests, key=lambda request: bool(request.get('preparation'))
                     and task_memory_estimate(request, settings, host) is None)
    for request in ordered:
        if count >= max_slots:
            break
        try:
            cpu, ram = resource_need(request, profile, settings, host)
        except (KeyError, ValueError):
            continue
        storage = preparation_forecast(request, settings.get('preparation_resource_observations'),
                                       cpu_count=cpu, host=host).get('storage_bytes', 0)
        if (cpus + cpu <= cpu_limit and memory + ram <= memory_limit
                and (storage_limit is None or reserved_storage + storage <= storage_limit)):
            reserved_storage += storage
            cpus += cpu
            memory += ram
            count += 1
            if selected is not None:
                selected.append(request)
    return cpus, (memory + GIB - 1) // GIB + 1, count


def apply_preparation_resources(distributed, overrides):
    """Set operational reservations and fill missing fixed preparation estimates."""
    pbs = distributed['pbs']
    distributed.setdefault('pbs_pinned_settings', deepcopy(pbs))
    for override in overrides:
        host = override['host']
        worker = next(entry for entry in pbs['workers'] if entry['host_id'] == host)
        source, provider = override['source'], override['provider']
        slots, memory = override['task_cpu_count'], override['memory_bytes']
        interpreter = override.get('interpreter')
        stage = override.get('stage')
        seconds = override.get('seconds')
        if interpreter is not None and not Path(interpreter).is_absolute():
            raise ValueError('PBS preparation interpreter must be a fixed absolute path')
        if type(slots) is not int or not 0 < slots <= pbs['resources']['ncpus']:
            raise ValueError('PBS preparation CPU reservation is outside the configured pool')
        if type(memory) is not int or not 0 < memory <= usable_memory(worker['profile'], pbs):
            raise ValueError('PBS preparation memory reservation is outside the configured pool')
        configured = distributed['preparation_profiles'][host]
        if not isinstance(configured, dict):
            configured = read(Path(configured))
            distributed['preparation_profiles'][host] = configured
        for profile in (worker['profile'], configured):
            matches = [rule for rule in profile['preparation_profiles']
                       if (rule.get('source_commit'), rule.get('provider')) == (source, provider)
                       and (stage is None or rule['stage'] == stage)]
            if len(matches) != 1:
                raise ValueError('PBS preparation profile is missing or ambiguous: '+provider)
            matches[0]['profile']['task_cpu_count'] = slots
            if interpreter is not None:
                matches[0]['profile']['interpreter'] = interpreter
        estimates = pbs['preparation_task_seconds']
        matches = [rule for rule in estimates if (rule['source'], rule['provider']) == (source, provider)
                   and (stage is None or rule['stage'] == stage)]
        if seconds is not None:
            if matches or stage is None or type(seconds) is not int or seconds <= 0:
                raise ValueError('New PBS preparation walltime requires an unused stage and positive seconds')
            estimates.append(dict(id=f'{stage}-{provider}-{source}', stage=stage, provider=provider,
                                  source=source, seconds=seconds, memory_bytes=memory))
        elif len(matches) == 1:
            matches[0]['memory_bytes'] = memory
        else:
            raise ValueError('PBS preparation walltime rule is missing or ambiguous: '+provider)


def apply_q3_measured_memory(distributed, estimates):
    pbs = distributed['pbs']
    profiles = {row['host_id']: row['profile'] for row in pbs['workers']}
    integers = ('dp_workers', 'rounds', 'min_peak_shots', 'max_peak_shots', 'memory_bytes')
    sources = ('view_commit', 'adapter_source')
    for row in estimates:
        if (row['host'] not in profiles or not row['method'] or not row['route_kind']
                or any(not re.fullmatch('[0-9a-f]{40}', row[key]) for key in sources)
                or ('horizon_risk_commit' not in row or row['horizon_risk_commit'] is not None
                    and not re.fullmatch('[0-9a-f]{40}', row['horizon_risk_commit']))
                or ('interpreter' in row and not Path(row['interpreter']).is_absolute())
                or any(type(row[key]) is not int or row[key] <= 0 for key in integers)
                or row['min_peak_shots'] > row['max_peak_shots']
                or row['memory_bytes'] > usable_memory(profiles[row['host']], pbs)):
            raise ValueError('Invalid measured Q3 memory envelope')
    distributed.setdefault('pbs_pinned_settings', deepcopy(pbs))
    pbs['q3_measured_memory'] = deepcopy(estimates)


def fits_request(request, report, settings, host_id=None):
    """Require a declared conservative runtime before starting scientific work."""
    pbs = (report or {}).get('pbs')
    if pbs is None:
        return True
    artificial = request.get('route', {}).get('kind') == 'artificial' or request.get('preparation', {}).get('artificial')
    estimate = task_estimate(request, settings, host_id)
    if estimate is None and artificial:
        estimate = pbs.get('minimum_task_seconds', 1)
    if not isinstance(estimate, (int, float)) or estimate <= 0:
        return False
    slot_memory = pbs.get('sized_task_memory_bytes')
    if slot_memory is not None:
        required = task_memory_estimate(request, settings, host_id)
        if required is None or required > slot_memory:
            return False
    return (pbs.get('accepting', False) and
            pbs.get('remaining_seconds', 0) >= estimate + settings.get('drain_seconds', 30))


def apply_q3_adapter(distributed, selection):
    """Switch the declared Q3 routes on one PBS worker, preserving the pinned settings."""
    pbs = distributed['pbs']
    distributed.setdefault('pbs_pinned_settings', deepcopy(pbs))
    worker = next(entry for entry in pbs['workers'] if entry['host_id'] == selection['host_id'])
    root, manifest = selection['root'], selection['manifest']
    if not Path(root).is_absolute() or not manifest.get('files'):
        raise ValueError('PBS Q3 adapter must have an absolute root and fixed files')
    rules = {rule['id']: rule for rule in pbs['q3_task_seconds']}
    for identity in selection['rule_ids']:
        rule = rules[identity]
        if selection['host_id'] not in rule['hosts'] or rule['hosts'] != [selection['host_id']]:
            raise ValueError('PBS Q3 rule must belong to the selected worker')
        matches = [route['profile'] for route in worker['profile'].get('route_profiles', [])
                   if (route['route_kind'], route['view_commit']) == (rule['route_kind'], rule['view_commit'])
                   and route['profile'].get('adapter_manifest', {}).get('commit')
                   in (rule['adapter_source'], manifest['commit'])]
        if len(matches) != 1:
            raise ValueError('PBS Q3 profile is missing or ambiguous: '+identity)
        matches[0].update(adapter_root=root, adapter_manifest=deepcopy(manifest))
        rule['adapter_source'] = manifest['commit']


def apply_q3_horizon(distributed, selection):
    """Add a verified horizon profile without replacing the original routes."""
    from .rpe_host_profiles import select_host_profile

    pbs = distributed['pbs']
    distributed.setdefault('pbs_pinned_settings', deepcopy(pbs))
    host = selection['host_id']
    worker = next(entry for entry in pbs['workers'] if entry['host_id'] == host)
    profile = worker['profile']
    root, source = selection['transport_root'], selection['transport_source']
    if not Path(root).is_absolute():
        raise ValueError('PBS horizon worker root must be absolute')
    for row in selection['route_profiles']:
        new_profile = row['profile']
        risk = row['horizon_risk_commit']
        route = dict(kind=row['route_kind'], view_commit=row['view_commit'])
        rule = next((rule for rule in pbs['q3_task_seconds'] if
                     (rule['route_kind'], rule['view_commit']) == (route['kind'], route['view_commit'])
                     and host in rule['hosts'] and rule['seconds'] > 0), None)
        if rule is None:
            raise ValueError('PBS horizon profile needs a fixed route resource estimate')
        previous = select_host_profile(profile, {'request': {'method': rule['method'], 'route': route}})
        manifest = new_profile['adapter_manifest']
        if (new_profile['adapter_root'] != root or manifest['commit'] != source or not manifest['files']
                or risk in previous['route_views'] or not Path(new_profile['route_views'][risk]).is_absolute()):
            raise ValueError('PBS horizon profile needs its fixed adapter and risk view')
        previous.update(adapter_root=root, adapter_manifest=manifest)
        previous['route_views'][risk] = new_profile['route_views'][risk]
        if previous != new_profile:
            raise ValueError('PBS horizon profile changes the existing host runtime or resource settings')
        if any((r['route_kind'], r['view_commit'], r.get('horizon_risk_commit')) ==
               (row['route_kind'], row['view_commit'], risk) for r in profile['route_profiles']):
            raise ValueError('PBS horizon profile already configured')
        profile['route_profiles'].append(deepcopy(row))
    pbs.update(transport_root=root, transport_source=source)


def pbs_transport_paths(value, paths):
    """Relocate only exact operational paths in the new allocation profile."""
    if isinstance(value, dict):
        return {key: pbs_transport_paths(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [pbs_transport_paths(item, paths) for item in value]
    return paths.get(value, value) if isinstance(value, str) else value


class PbsManager:
    def __init__(self, executor, settings, *, gateway=None):
        self.executor = executor
        self.settings = deepcopy(settings)
        original = deepcopy(getattr(executor, 'distributed', {}).get('pbs_pinned_settings', settings))
        namespace = digest(dict(controller_run=str(executor.run_dir.resolve())))[:20]
        for configuration in (self.settings, original):
            if configuration.get('namespace_by_run', True):
                configuration['remote_root'] = configuration['remote_root'].rstrip('/')+'/'+namespace
        self.directory = executor.run_dir/'distributed'/'pbs'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory/'MANAGER.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixed = self.directory/'SETTINGS.json'
        if fixed.exists() and read(fixed) != original:
            self.lock.close()
            raise ValueError('Existing run retains its pinned PBS settings')
        if not fixed.exists(): atomic(fixed, original)
        self.path = self.directory/'STATE.json'
        self.state = read(self.path) if self.path.exists() else dict(allocations={}, requests={})
        dispatch = getattr(executor, 'distributed', {}).get('pbs_dispatch_gateway')
        self.transport_paths = {}
        if dispatch:
            self.settings.update(gateway_script=dispatch['path'], gateway_sha256=dispatch['sha256'])
            transport = dispatch.get('transport')
            if transport:
                if (not Path(transport['root']).is_absolute()
                        or transport['source'] != self.settings['transport_source']):
                    raise ValueError('PBS replacement transport must retain the fixed source')
                self.transport_paths = transport.get('paths', {})
                if any(not Path(old).is_absolute() or not Path(new).is_absolute()
                       for old, new in self.transport_paths.items()):
                    raise ValueError('PBS replacement transport requires absolute paths')
                self.settings['transport_root'] = transport['root']
                self.settings['workers']=pbs_transport_paths(self.settings['workers'],self.transport_paths)
        self.bounded_results = bool(dispatch and dispatch.get('bounded_results'))
        self.compute_storage_sampling = bool(dispatch and dispatch.get('compute_storage_sampling'))
        self.defer_q3_evidence = bool(dispatch and dispatch.get('defer_q3_evidence'))
        self.batch_q3_completions = bool(self.defer_q3_evidence and self.bounded_results
                                         and dispatch.get('batch_q3_completions'))
        self.evidence_directory = self.directory/'deferred-evidence'
        self.evidence_gateway = None
        self.evidence_pool = None
        self.evidence_pending = None
        self.evidence_retry_after = {}
        self.offline_plan = getattr(executor, 'distributed', {}).get('pbs_offline_plan')
        self.offline_imported = {}
        self.offline_pending = None
        self.offline_progress = {}
        self.offline_progress_next = 0.
        self.offline_staged = set()
        self.offline_ranks = {}
        self.gateway = gateway or Gateway(self.settings)
        self.bulk_gateway = Gateway(self.settings) if self.bounded_results else None
        # Only gateways deployed with the selected-attempt harvest protocol
        # may receive an owners filter.
        self.selective_harvest = bool(dispatch and dispatch.get('selective_harvest'))
        self.harvest_gateway = Gateway(self.settings) if self.selective_harvest else None
        self.harvest_pool = None
        self.harvest_pending = None
        self.live_harvest = bool(self.selective_harvest and dispatch.get('live_harvest'))
        self.live_harvest_pending = None
        self.live_harvest_checked = {}
        self.harvest_retry_after = {}
        self.ensure_retry_after = {}
        self.storage_launcher = (getattr(executor, 'distributed', {})
                                 .get('pbs_gateway_storage_launcher'))
        self.storage_launcher_verified = False
        self.stop = threading.Event()
        self.thread = None
        self.allocations = []
        self.responses = {}
        self.artifact_pool = None
        self.artifact_transfers = {}
        self.prefetched_inputs = {}
        self.q3_worker_proofs = {}
        completion_workers = getattr(executor, 'distributed', {}).get('pbs_gateway_completion_workers', 0)
        self.completion_pool = (ThreadPoolExecutor(max_workers=completion_workers, thread_name_prefix='pbs-receive')
                                if completion_workers else None)
        self.completion_workers = completion_workers
        self.pending_completions = {}
        self.split_exchange = bool(dispatch and dispatch.get('split_exchange') and completion_workers)
        if (self.bounded_results and not self.split_exchange
                or (self.defer_q3_evidence or self.offline_plan)
                and (not self.bounded_results or not self.selective_harvest)):
            self.lock.close()
            raise ValueError('PBS bounded and deferred transfers require their completion and harvest lanes')
        self.completion_gateway = Gateway(self.settings) if self.split_exchange else None
        self.completion_responses = {}
        self.completion_thread = None
        self.split_status = bool(self.split_exchange and dispatch.get('split_status'))
        self.status_gateway = Gateway(self.settings) if self.split_status else None
        self.status_thread = None
        self.status_queue = queue.Queue()
        self.status_pending = set()
        self.status_lock = threading.Lock()
        self.controls = {}
        self.storage_checks = {}
        self.last_error = None
        self.configured = {worker['host_id']:worker for worker in self.settings['workers']}
        if len(self.configured) != len(settings['workers']):
            raise ValueError('PBS host IDs must be distinct')
        self.hold_node = getattr(executor, 'distributed', {}).get('pbs_hold_node') is True
        self.retain_between_batches = False
        self.detached = False
        if self.hold_node and (len(self.configured) != 1 or not (
                self.settings.get('allocation_sizing') or {}).get('per_task_resources')):
            self.lock.close()
            raise ValueError('PBS node reuse requires one worker with per-task resource accounting')

    def log(self, event, **fields):
        append(self.directory/'EVENTS.jsonl', event, **fields)

    def save(self):
        atomic(self.path, self.state)

    def readiness_wait_seconds(self):
        # PBS queue latency varies by queue and resource request. Keep the
        # wait bounded, including when the gateway or scheduler is unavailable.
        samples = [a['submit_to_ready_seconds'] for a in self.state['allocations'].values()
                   if 'submit_to_ready_seconds' in a]
        return max(15., min(30., median(samples[-10:]) if samples else 20.))

    def accepting_allocation(self, host, job_id):
        return any(allocation['host_id'] == host and allocation['status'] == 'R'
                   and allocation['job_id'] == job_id
                   and (allocation.get('worker') or {}).get('accepting')
                   and not self.state['allocations'][allocation['token']].get('resize_requested')
                   and not self.state['allocations'][allocation['token']].get('launcher_rollover_requested')
                   for allocation in self.allocations)

    def session_control(self, host, record, *, stopping, release_when_idle):
        """Retain ownership on admission pause; only final drains send STOP."""
        policy_reader=getattr(self.executor,'_host_admission_policy',None)
        paused=False
        if policy_reader is not None:
            policy,error=policy_reader()
            paused=not self.executor._host_admission(host,policy,error)['accepting']
        stop=bool(stopping or record.get('resize_requested') or record.get('storage_maintenance_requested'))
        return dict(stop=stop,pause=bool(paused and not stop),release_when_idle=bool(release_when_idle and (stop or not paused)))

    def startup_jobs(self, *, except_host=None):
        """Offer a sized allocation a brief first chance at its checked work."""
        now = stamp()
        jobs = set()
        for record in list(self.state['allocations'].values()):
            spec = record['spec']
            submitted = record['submitted_clock']
            reason = spec['reason']
            status = record['status']
            # qsub can identify an allocation before qstat has its first
            # record. A missing initial storage sample is also transitional.
            starting = status in ('INTENT', 'Q', 'R') or (
                status == 'HISTORY_MISSING_RECONCILE_SPOOL'
                and (record.get('observed') or {}).get('job_id') is not None)
            if (starting
                    and spec['profile']['host_id'] != except_host
                    and not record.get('resize_requested')
                    and record.get('worker_storage_hold') in (None, 'STORAGE_USAGE_UNAVAILABLE')
                    and now['clock_id'] == submitted['clock_id']
                    and now['monotonic'] - submitted['monotonic'] < reason.get('startup_wait_seconds', 0)):
                jobs.update(reason.get('startup_jobs', ()))
        return jobs

    def preparation_readiness_wait_seconds(self, host):
        # Queue wait and worker-side preflight have different budgets. A live
        # worker may need several artifact RPCs before it can report readiness.
        # Bound the extended wait so a stalled controller cannot hold a lane.
        rpc_timeout = self.settings.get('request_timeout_seconds', 20)
        worker = self.configured[host]['profile']
        rpc_timeout = max(rpc_timeout, (worker.get('pbs') or {}).get('rpc_timeout_seconds', 15))
        base = self.readiness_wait_seconds() + 3 * rpc_timeout
        ready = any(allocation['host_id'] == host and allocation['status'] == 'R'
                    and (allocation.get('worker') or {}).get('accepting')
                    for allocation in self.allocations)
        if ready and self.last_error is None:
            return max(base, 300.)
        return base

    def _recent_preparation_completion(self, seconds):
        # A newly accepted, successful required stage may shortly deliver an
        # audit or a ready candidate. Never hold an allocation for a long stage
        # with no completion in sight or for optional price checks.
        with self.executor.store._connection() as database:
            rows = database.execute(
                "SELECT request, detail FROM jobs WHERE state='ACCEPTED' "
                "AND method LIKE 'PREPARATION:%' AND completed_epoch>=?",
                (time.time() - seconds,))
            return any(not json.loads(row[0])['preparation'].get('optional_price_validation')
                       and json.loads(row[1] or '{}').get('outcome_status') == 'PREPARATION_COMPLETE'
                       for row in rows)

    def _storage_accepting(self, worker):
        policy = worker['profile'].get('storage_limit')
        if policy is None:
            return True
        if self.compute_storage_sampling and self.hold_node and self.storage_launcher:
            # Attach the worker so it can collect the first compute-node sample.
            # Its own offer and launch checks still enforce the original limit.
            return True
        host = worker['host_id']
        sampled, report = self.storage_checks.get(host, (float('-inf'), None))
        interval = max(30, SAMPLE_SECONDS) if self.compute_storage_sampling else SAMPLE_SECONDS
        if time.monotonic() - sampled >= interval:
            try:
                new = self.gateway.storage_report(policy)
                used = new['used_bytes']
                free = new.get('filesystem_free_bytes')
                minimum = new.get('filesystem_min_free_bytes')
                if used is None:
                    reason = 'STORAGE_USAGE_UNAVAILABLE'
                elif type(used) is not int or used < 0:
                    raise ValueError('Invalid PBS storage usage')
                elif used >= policy['limit_bytes']:
                    reason = 'STORAGE_LIMIT_REACHED'
                elif free is not None and minimum is not None and free < minimum:
                    reason = 'FILESYSTEM_SPACE_LOW'
                else:
                    reason = None
                if (new['limit_bytes'] != policy['limit_bytes']
                        or new['reason'] != reason
                        or new['accepting_new_jobs'] is not (reason is None)):
                    raise ValueError('PBS storage report differs from selected limit')
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                new = dict(limit_bytes=policy['limit_bytes'], used_bytes=None,
                           accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
            if report is None or (report['accepting_new_jobs'], report['reason']) != (new['accepting_new_jobs'], new['reason']):
                self.log('PBS_STORAGE_ADMISSION', host_id=host, **new)
            report = new
            self.storage_checks[host] = (time.monotonic(), report)
            self.state.setdefault('storage', {})[host] = report
        return report['accepting_new_jobs']

    def _reuse_q3_worker_proofs(self, token, worker, request_clock, payload):
        """Reuse recent facts from this exact compute worker, never the gateway."""
        starts = worker.get('start_clock') or {}
        identity = (worker.get('job_id'), worker.get('hostname'), worker.get('pid'),
                    worker.get('source_sha'), starts.get('clock_id'), starts.get('monotonic'))
        sent = request_clock.get('monotonic')
        if (not worker.get('accepting') or not payload.get('accepting_new_jobs', True) or None in identity
                or request_clock.get('clock_id') != starts.get('clock_id') or sent is None
                or sent < starts['monotonic']):
            self.q3_worker_proofs.pop(token, None)
            return set()
        previous, proofs = self.q3_worker_proofs.get(token, (None, {}))
        if previous != identity:
            proofs = {}
        now = time.monotonic()
        observed = worker.get('monotonic')
        elapsed = max(0, observed - sent) if observed is not None else 0
        received = payload.get('host_fingerprints') or {}
        errors = payload.get('readiness_errors') or {}
        checked = payload.get('readiness_checked_monotonic') or {}
        ages = {}
        for job, facts in received.items():
            finished = checked.get(job, sent)
            if type(finished) in (int, float) and starts['monotonic'] <= finished <= sent:
                age = elapsed + sent - finished
                if age < Q3_WORKER_PROOF_SECONDS:
                    ages[job] = age
        proofs = {job: entry for job, entry in proofs.items()
                  if now - entry[0] < Q3_WORKER_PROOF_SECONDS and job not in errors and job not in received}
        reused = set(proofs)
        received = {job: facts for job, facts in received.items() if job in ages}
        for job, facts in received.items():
            if (isinstance(facts, dict) and facts.get('schema') == 'RPE_Q3_CURRENT_HOST_FINGERPRINT_V1'
                    and job not in errors):
                proofs[job] = now - ages[job], facts
        self.q3_worker_proofs[token] = identity, proofs
        payload['host_fingerprints'] = {**{job: facts for job, (_, facts) in proofs.items()}, **received}
        return reused

    def _dispatch(self, token, request):
        host = self.state['allocations'][token]['spec']['profile']['host_id']
        if request['payload'].get('host_id') != host:
            raise ValueError('PBS request host differs from its saved allocation')
        request_id = request['request_id']
        saved = self.directory/'responses'/token/(request_id+'.json')
        saved.parent.mkdir(parents=True, exist_ok=True)
        identity = digest(request)
        if saved.exists():
            record = read(saved)
            if record['request_sha256'] != identity:
                raise ValueError('Reused PBS RPC ID has changed content')
            return record['response']
        # Queue/attempt operations themselves are idempotent if the controller
        # crashes after RPC commit but before saving this response.
        try:
            payload = deepcopy(request['payload'])
            allocation = next((a for a in self.allocations if a['token'] == token), None)
            detail = payload.get('detail') or {}
            if request['endpoint'] == '/v1/state' and detail.get('pbs_attempt_exit'):
                observed = allocation or self.state['allocations'][token].get('observed', {})
                row = self.executor.store.row(payload['job_id'])
                if (payload['state'] != 'UNKNOWN' or not row or payload['attempt_id'] != row['attempt_id']
                        or not self._attempt_released(detail, token, observed.get('job_id'), row)):
                    raise ValueError('PBS attempt release differs from its saved owner')
            if request['endpoint'] == '/v1/poll':
                spec = self.state['allocations'][token]['spec']
                replacement = spec.get('reason', {}).get('retained_replacement')
                if replacement:
                    # Replays belong to the worker that received the original
                    # assignment. Treat its durable owners as already sent.
                    payload.setdefault('durable_assignments', []).extend(replacement['old_owners'])
                if (allocation or {}).get('retained_replacement'):
                    return self._record_rpc_response(token, request_id, identity, dict(ok=True,
                        result=dict(assignments=[], stop=False, eligibility_challenges=[])))
            if request['endpoint'] == '/v1/poll':
                admission = (payload.get('resource_report') or {}).get('pbs')
                if admission is not None:
                    # Identity comes from the manager's saved allocation.
                    admission['allocation_token'] = token
                    admission['other_allocation_owners'] = (
                        self.state['allocations'][token]['spec'].get('reason', {})
                        .get('retained_replacement', {}).get('old_owners', []))
            if request['endpoint'] == '/v1/poll' and allocation is not None:
                admission = (payload.get('resource_report') or {}).get('pbs')
                current = allocation.get('worker') or {}
                if admission is not None:
                    admission['remaining_seconds'] = min(admission.get('remaining_seconds',0), current.get('remaining_seconds',0))
                    admission['accepting'] = admission.get('accepting',False) and current.get('accepting',False)
                    spec = self.state['allocations'][token]['spec']
                    if spec['profile'].get('pbs', {}).get('per_task_resources'):
                        observed = current.get('resources') or {}
                        admission['resource_pool'] = dict(
                            cpu_count=min(spec['resources']['ncpus'], observed.get('cpu_count') or 0),
                            memory_bytes=min(int(spec['resources']['mem'][:-2]) * GIB,
                                             observed.get('memory_bytes') or 0))
                    else:
                        admission.pop('resource_pool', None)
                    if not admission['accepting'] or not admission.get('resource_pool'):
                        admission.pop('ready_buffer', None)
            if request['endpoint'] == '/v1/poll' and (allocation is None or allocation['status'] != 'R' or
                    not (allocation.get('worker') or {}).get('accepting', False) or
                    self.state['allocations'][token].get('resize_requested') or
                    self.state['allocations'][token].get('storage_maintenance_requested') or
                    self.state['allocations'][token].get('launcher_rollover_requested')):
                # A delayed file survives worker/PBS termination; it cannot
                # constitute a current free-slot offer.
                payload.update(slots=0, capacity=0, accepting_new_jobs=False)
                ((payload.get('resource_report') or {}).get('pbs') or {}).pop('ready_buffer', None)
            reused = set()
            if request['endpoint'] == '/v1/poll':
                reused = self._reuse_q3_worker_proofs(
                    token, (allocation or {}).get('worker') or {}, request.get('clock') or {}, payload)
            result = self.executor.rpc(request['endpoint'], payload)
            if request['endpoint'] == '/v1/poll':
                assignments = {item['job_id'] for item in result.get('assignments', [])}
                claimed = assignments & reused
                if claimed:
                    self.log('PBS_Q3_ASSIGNED_WITH_PREFETCHED_PROOF', host_id=host, jobs=sorted(claimed))
                cached = self.q3_worker_proofs.get(token)
                if cached:
                    for job_id in assignments:
                        cached[1].pop(job_id, None)
                result = self._prefetch_preparation_inputs(token, payload, result)
                cached_challenges = payload.get('pbs_cached_challenges')
                if isinstance(cached_challenges, dict):
                    challenges = result.get('eligibility_challenges', ())
                    compact = []
                    for challenge in challenges:
                        job = challenge['job_id']
                        known = cached_challenges.get(job)
                        if (known and challenge['request'].get('route', {}).get('kind') != 'preparation'
                                and digest(challenge) == known):
                            compact.append(dict(job_id=job, pbs_challenge_ref=known))
                        else:
                            compact.append(challenge)
                    result['eligibility_challenges'] = compact
                record = self.state['allocations'][token]
                storage = (payload.get('resource_report') or {}).get('storage') or {}
                ready = payload['slots'] > 0 and payload.get('accepting_new_jobs', True)
                if ready:
                    previous = record.get('first_usable_offer_clock') or {}
                    now = stamp()
                    if previous.get('clock_id') != now['clock_id']:
                        record['first_usable_offer_clock'] = now
                    submitted = record['submitted_clock']
                    if now['clock_id'] == submitted['clock_id'] and 'submit_to_usable_seconds' not in record:
                        record['submit_to_usable_seconds'] = max(0., now['monotonic'] - submitted['monotonic'])
                    record['last_usable_offer_clock'] = now
                else:
                    record.pop('first_usable_offer_clock', None)
                    record.pop('last_usable_offer_clock', None)
                errors = payload.get('readiness_errors') or {}
                error_hash = digest(errors) if errors else None
                if (error_hash != record.get('worker_readiness_error_sha256') or
                        storage.get('reason') != record.get('worker_storage_hold')):
                    self.log('PBS_WORKER_ADMISSION', token=token, host_id=host,
                             storage_reason=storage.get('reason'), readiness_errors=errors)
                    record['worker_storage_hold'] = storage.get('reason')
                    record['worker_readiness_error_sha256'] = error_hash
            response = dict(ok=True, result=result)
        except (OSError, ValueError, KeyError) as error:
            response = dict(ok=False, error=type(error).__name__+': '+str(error))
        return self._record_rpc_response(token, request_id, identity, response)

    def _shared_artifact(self, profile, sha):
        cache = profile.get('artifact_cache_directory')
        if not cache or not self.settings.get('stage_artifacts_over_ssh'):
            return None
        source = self.executor.artifact_cache.path(sha)
        try:
            if source.is_symlink() or not source.is_file() or source.stat().st_size < 1024 * 1024:
                return None
        except OSError:
            return None  # the usual worker transfer will report a missing source
        return source, Path(cache) / sha[:2] / sha

    def _start_artifact_transfer(self, paths, sha):
        if self.artifact_pool is None:
            self.artifact_pool = ThreadPoolExecutor(max_workers=len(self.configured),
                                                    thread_name_prefix='pbs-artifact')
        return self.artifact_pool.submit(self.gateway.stage_artifact, *paths, sha)

    def _prefetch_preparation_inputs(self, token, payload, result):
        """Wait only the affected challenge until its inputs reach the worker cache."""
        if ('cached_artifact_sha256' not in payload or result.get('stop')
                or not payload.get('accepting_new_jobs', True)
                or not self.settings.get('stage_artifacts_over_ssh')
                or not result.get('eligibility_challenges')):
            return result
        from .rpe_preparation_worker import metadata_bundle_hashes

        reported = set(payload['cached_artifact_sha256']) | set(payload.get('cached_metadata_bundle_sha256', ()))
        profile = self.state['allocations'][token]['spec']['profile']
        ready = []
        for challenge in result['eligibility_challenges']:
            request = challenge['request']
            pending = False
            if request.get('preparation'):
                for sha in metadata_bundle_hashes(challenge):
                    if sha in reported:
                        continue
                    paths = self._shared_artifact(profile, sha)
                    if paths is None:
                        continue
                    key = token, sha
                    entry = self.prefetched_inputs.get(key)
                    if entry is None:
                        entry = dict(future=self._start_artifact_transfer(paths, sha))
                        self.prefetched_inputs[key] = entry
                    future = entry['future']
                    if not future.done():
                        pending = True
                        continue
                    if 'worker_visibility_deadline' not in entry:
                        try:
                            future.result()
                        except (OSError, ValueError, subprocess.SubprocessError) as error:
                            entry['worker_visibility_deadline'] = 0.
                            self.log('PBS_SHARED_INPUT_PREFETCH_FALLBACK', token=token, sha256=sha,
                                     error=type(error).__name__)
                        else:
                            entry['worker_visibility_deadline'] = time.monotonic() + self.settings.get(
                                'request_timeout_seconds', 20)
                            self.log('PBS_SHARED_INPUT_PREFETCHED', token=token, sha256=sha)
                    pending |= time.monotonic() < entry['worker_visibility_deadline']
            if not pending:
                ready.append(challenge)
        return dict(result, eligibility_challenges=ready)

    def _dispatch_ready(self, token, request):
        """Bulk inputs use the worker's shared cache, leaving the poll lane free."""
        if 'spool_bytes' in request:
            raw = self._read_bulk(self.bulk_gateway, token, 'spool', request['request_id'],
                                  size=request['spool_bytes'])
            decoded = json.loads(raw)
            if (decoded['request_id'] != request['request_id']
                    or decoded['endpoint'] not in ('/v1/complete', '/v1/artifacts/put')):
                raise ValueError('PBS bulk request identity changed')
            request = decoded
        if request['endpoint'] == '/v1/complete':
            payload = request['payload']
            if 'pbs_evidence_status_batch' in payload:
                return self._batch_evidence_status(token, payload)
            if payload.keys() & {'pbs_evidence', 'pbs_evidence_status'}:
                return self._complete_with_deferred_evidence(token, request)
        if (not self.settings.get('stage_artifacts_over_ssh')
                or request['endpoint'] != '/v1/artifacts/get'):
            return self._dispatch(token, request)
        from .rpe_preparation_artifacts import artifact_hash
        profile = self.state['allocations'][token]['spec']['profile']
        payload = request['payload']
        if payload.get('host_id') != profile['host_id']:
            raise ValueError('PBS artifact request host differs from its allocation')
        hashes = payload.get('hashes')
        if not isinstance(hashes, list) or len(hashes) != 1:
            return self._dispatch(token, request)
        sha = artifact_hash(hashes[0])
        if (token, sha) in self.prefetched_inputs:
            # A GET for an already prefetched digest is the compute node's
            # direct observation of a miss. Return bytes through the ordinary
            # hash-checked transfer path instead of another empty ACK.
            return self._dispatch(token, request)
        paths = self._shared_artifact(profile, sha)
        if paths is None:
            return self._dispatch(token, request)
        ident = (token, request['request_id'])
        saved = self.directory / 'responses' / token / (ident[1] + '.json')
        if saved.exists():
            return self._dispatch(token, request)
        fingerprint = digest(request)
        transfer = self.artifact_transfers.get(ident)
        if transfer is None:
            future = self._start_artifact_transfer(paths, sha)
            self.artifact_transfers[ident] = (fingerprint, future)
            return None
        observed, future = transfer
        if observed != fingerprint:
            raise ValueError('Reused PBS artifact RPC ID has changed content')
        if not future.done():
            return None
        del self.artifact_transfers[ident]
        try:
            future.result()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.log('PBS_SHARED_ARTIFACT_FALLBACK', token=token, sha256=sha, error=type(error).__name__)
            return self._dispatch(token, request)
        response = dict(ok=True, result=dict(blobs={}))
        saved.parent.mkdir(parents=True, exist_ok=True)
        atomic(saved, dict(request_sha256=fingerprint, response=response, clock=stamp()))
        self.log('PBS_SHARED_ARTIFACT_STAGED', token=token, sha256=sha,
                 bytes=paths[0].stat().st_size)
        return response

    def _collect_offline_history(self, token):
        from .rpe_q3_worker import safe_id

        from .rpe_pbs_offline import independent_plan
        if independent_plan(self.offline_plan):
            return self._collect_independent_history(token)
        gateway = Gateway(self.settings)
        try:
            progress = gateway.call(dict(action='offline_results', token=token))
            history = self.directory/'offline-history'/'jobs'
            copied = 0
            for key, item in progress['jobs'].items():
                method, job = key.split('/')
                safe_id(job)
                if method not in self.offline_plan['round']['methods'] or not any(
                        part['id'] == item['part'] and part['method'] == method
                        for part in self.offline_plan['parts']):
                    raise ValueError('Offline PBS result lies outside the fixed scope')
                target = history/method/job
                if (target/'OUTCOME.json').exists():
                    continue
                names = ('REQUEST.json', 'PROCESS_RETURN.json', 'OUTCOME.json')
                if set(item['files']) != set(names):
                    raise ValueError('PBS offline result is missing its fixed receipt')
                files = {name: self._read_bulk(gateway, token, 'offline', name,
                            size=item['files'][name], method=method, job_id=job) for name in names}
                request = json.loads(files['REQUEST.json'])
                receipt = json.loads(files['PROCESS_RETURN.json'])
                if (request.get('request_id') != job or request.get('method') != method
                        or request.get('optimizer_source') != self.executor.computation_source
                        or receipt.get('returncode') != 0 or receipt.get('signal') is not None):
                    self.log('PBS_OFFLINE_RESULT_NOT_IMPORTED', job_id=job, method=method)
                    continue
                target.mkdir(parents=True, exist_ok=True)
                for name in names:
                    temporary = target/(name+'.tmp')
                    temporary.write_bytes(files[name])
                    temporary.replace(target/name)
                copied += 1
            reports = self.directory/'offline-history'/'coverage'
            reports.mkdir(parents=True, exist_ok=True)
            atomic(reports/(safe_id(token)+'.json'), progress['coverage'])
            if copied:
                self.log('PBS_OFFLINE_RESULTS_IMPORTED', count=copied, token=token)
        finally:
            gateway.close()

    @staticmethod
    def _receive_independent_file(gateway, token, kind, name, target, *, size, sha=None, **owner):
        """Stream a bounded file directly to disk, never a whole result in RAM."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            received, identity = 0, {}
            checksum = hashlib.sha256()
            try:
                while received < size:
                    chunk = gateway.call(dict(action='read_bulk_chunk', token=token, kind=kind,
                        name=name, offset=received, size=size, **owner, **identity))
                    raw = base64.b64decode(chunk['data'], validate=True)
                    if chunk['size'] != size or not raw or received+len(raw) > size:
                        raise ValueError('Independent result transfer size changed')
                    identity = {'file_identity':chunk['file_identity']}
                    stream.write(raw)
                    checksum.update(raw)
                    received += len(raw)
                if sha is not None and checksum.hexdigest() != sha:
                    raise ValueError('Independent artifact SHA differs')
                stream.flush()
                os.fsync(stream.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

    def _collect_independent_history(self, token):
        from .rpe_pbs_offline import pbs_units, unit_members
        from .rpe_q3_worker import safe_id
        from .rpe_preparation_queue import normalize_preparation_request
        gateway = Gateway(self.settings)
        try:
            progress = gateway.call(dict(action='offline_results', token=token))
            history = self.directory/'offline-history'
            parts = {p['id']:p for p in pbs_units(self.offline_plan)}
            names = ('REQUEST.json','PROCESS_RETURN.json','OUTCOME.json')
            copied = 0
            held = {}
            for key,item in progress['jobs'].items():
                try:
                    method, job = map(safe_id,key.split('/'))
                    part = parts.get(item['part'])
                    preparation = method.startswith('PREPARATION-')
                    methods = {p['method'] for p in unit_members(self.offline_plan,part)} if part else set()
                    if not part or (not preparation and method not in methods) or set(item['files']) != set(names):
                        raise ValueError('Independent result outside assigned block')
                    target = history/'jobs'/method/job
                    if (target/'IMPORTED.json').is_file():
                        continue
                    staging = target/'incoming'
                    staging.mkdir(parents=True,exist_ok=True)
                    for name in names:
                        self._receive_independent_file(gateway,token,'offline',name,staging/name,
                            size=item['files'][name],method=method,job_id=job)
                    request, receipt, outcome = (read(staging/name) for name in names)
                    if (request.get('request_id') != job or request.get('method') not in methods
                            or receipt.get('returncode') != 0 or receipt.get('signal') is not None):
                        raise ValueError('Independent result receipt differs')
                    if preparation:
                        origin = item.get('origin') or {}
                        if (normalize_preparation_request(request) != request
                                or origin.get('request_sha256') != digest(request)
                                or outcome.get('request_sha256') != digest(request)
                                or outcome.get('request_id') != job or outcome.get('status') != 'PREPARATION_COMPLETE'
                                or not origin.get('attempt_id') or not origin.get('completion_sha256')
                                or set(item.get('artifacts',{})) != set(outcome.get('artifacts',{}).values())):
                            raise ValueError('Independent preparation result differs from its accepted receipt')
                        cache = self.executor.artifact_cache
                        for sha, size in item['artifacts'].items():
                            if not cache.missing([sha]):
                                continue
                            path = staging/sha
                            self._receive_independent_file(gateway,token,'offline_artifact',sha,path,size=size,
                                sha=sha,part=item['part'],method=method,job_id=job)
                            cache.put_file(path,sha)
                            path.unlink()
                        atomic(staging/'ORIGIN.json',origin)
                    elif request.get('optimizer_source') != self.executor.computation_source:
                        raise ValueError('Independent Q3 SOURCE differs')
                    for name in ('REQUEST.json','PROCESS_RETURN.json','ORIGIN.json','OUTCOME.json'):
                        if (staging/name).is_file():
                            (staging/name).replace(target/name)
                    atomic(target/'IMPORTED.json',dict(part=item['part'],token=token))
                    try:
                        staging.rmdir()
                    except OSError:
                        pass
                    copied += 1
                except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.SubprocessError) as error:
                    held[item.get('part','unknown')]=str(error)
                    self.log('PBS_INDEPENDENT_RESULT_PENDING', token=token, job=key, reason=str(error))
            coverage = deepcopy(progress.get('coverage') or {})
            for part,reason in held.items():
                row = coverage.get('parts',{}).get(part)
                if row is not None:
                    row['observed_termination'] = row.pop('termination',None)
                    row['import_pending'] = reason
            report_root = history/'reports'
            for part, entry in coverage.get('parts',{}).items():
                if part not in parts:
                    raise ValueError('Independent coverage outside assigned blocks')
                root = report_root/part
                root.mkdir(parents=True,exist_ok=True)
                revision = entry.get('report_revision',0)
                mark = root/'REVISION.json'
                if mark.is_file() and read(mark).get('revision') == revision:
                    continue
                for name,size in entry.get('reports',{}).items():
                    if name not in ('SCOPE_MAP.json','ROUND_RESULT.json'):
                        raise ValueError('Unknown independent report')
                    self._receive_independent_file(gateway,token,'offline_report',name,root/name,
                                                  size=size,part=part)
                atomic(mark,dict(revision=revision))
            reports = history/'coverage'
            reports.mkdir(parents=True,exist_ok=True)
            atomic(reports/(safe_id(token)+'.json'),coverage)
            from .rpe_pbs_offline import publish_partition_view
            publish_partition_view(self.executor.run_dir)
            if copied:
                self.log('PBS_INDEPENDENT_RESULTS_IMPORTED',count=copied,token=token)
            with self.executor.state.cv:
                self.executor.state.cv.notify_all()
            if held:
                raise RuntimeError('Independent receipts pending: '+str(len(held)))
        finally:
            gateway.close()

    def _advance_offline_history(self):
        if self.offline_pending is not None:
            token, revision, future = self.offline_pending
            if not future.done():
                return

            self.offline_pending = None
            try:
                future.result()
            except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
                self.harvest_retry_after[token] = time.monotonic()+30
                self.log('PBS_OFFLINE_IMPORT_PENDING', token=token, error=str(error))
            else:
                self.offline_imported[token] = revision
        for allocation in self.allocations:
            token = allocation['token']
            revision = (allocation.get('worker') or {}).get('offline_revision', 0)
            from .rpe_pbs_offline import independent_plan
            terminal_unread = (independent_plan(self.offline_plan) and allocation['status'] in TERMINAL
                               and token not in self.offline_imported)
            if ((revision > self.offline_imported.get(token, 0) or terminal_unread)
                    and time.monotonic() >= self.harvest_retry_after.get(token, 0)):
                if self.harvest_pool is None:
                    self.harvest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rpe-pbs-harvest')
                self.offline_pending = (token, revision, self.harvest_pool.submit(self._collect_offline_history, token))
                return

    def _poll_delay(self):
        from .rpe_pbs_offline import independent_plan
        usual = self.settings.get('poll_seconds',.25)
        if not independent_plan(getattr(self,'offline_plan',None)):
            return usual
        if self.independent_finished():
            return usual
        # While the node is computing its own backlog, fast per-job
        # exchanges have no purpose. Restore the normal cadence when an
        # allocation returns to the central queue for extra work.
        live = [row.get('worker') or {} for row in self.allocations if row['status']=='R']
        if any(row.get('independent_active') is False for row in live):
            return usual
        return max(usual, float(self.offline_plan['settings'].get('poll_seconds',10)))

    def independent_coverage(self):
        from .rpe_pbs_offline import independent_plan
        if not independent_plan(self.offline_plan):
            return None
        parts = {}
        for file in (self.directory/'offline-history'/'coverage').glob('*.json'):
            try:
                for key, row in read(file).get('parts', {}).items():
                    old = parts.get(key, {})
                    if (row.get('termination') or not old.get('termination') and
                            row.get('report_revision', 0) >= old.get('report_revision', 0)):
                        parts[key] = row
            except (OSError, ValueError):
                continue
        return parts

    def independent_finished(self):
        from .rpe_pbs_offline import all_parts_finished
        coverage = self.independent_coverage()
        return coverage is not None and all_parts_finished(self.offline_plan, coverage)

    def _independent_demand(self, stopping):
        from .rpe_pbs_offline import independent_plan, pbs_units
        if not independent_plan(getattr(self,'offline_plan',None)) or self.independent_finished():
            return False
        active = [a for a in self.allocations if a['status'] not in TERMINAL]
        owned = {r['spec']['profile']['host_id'] for r in self.state['allocations'].values()
                 if r['status'] not in TERMINAL}
        for allocation in active:
            token, host = allocation['token'], allocation['host_id']
            self.controls[token] = self.session_control(host, self.state['allocations'][token],
                stopping=stopping, release_when_idle=stopping)
            self._offline_contact(token)
        if not stopping:
            policy,error = self.executor._host_admission_policy()
            pending = len([b for b in pbs_units(self.offline_plan)
                           if not (self.independent_coverage().get(b['id'], {}).get('termination'))])
            for host, worker in self.configured.items():
                if host in owned or len(owned) >= pending:
                    continue
                if not self.executor._host_admission(host,policy,error)['accepting']:
                    continue
                if not (self.compute_storage_sampling and self.storage_launcher) and not self._storage_accepting(worker):
                    continue
                try:
                    self._submit(host,worker,dict(independent_blocks=True, uncompleted_blocks=pending))
                    owned.add(host)
                except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
                    self.log('PBS_INDEPENDENT_SUBMIT_HELD',host_id=host,reason=str(error))
        self.state['heartbeat'] = stamp()
        self.save()
        return True

    def offline_priority(self):
        from .rpe_pbs_offline import independent_plan
        if independent_plan(self.offline_plan) or not self.offline_plan or self.last_error is None:
            return {}
        methods = self.offline_plan['round']['methods']
        fallback = len(self.offline_plan['parts'])
        return {method: fallback+rank for rank, method in enumerate(reversed(methods))} | self.offline_ranks

    def bind_offline_points(self, points):
        if not self.offline_plan:
            return
        ranks = {(part['method'], digest(part['width'])): rank
                 for rank, part in enumerate(reversed(self.offline_plan['parts']))}
        selected = {}
        for point in points:
            configuration = point.get('evaluation_configuration_id')
            method = point['method']
            key = method, digest(point.get('requested_delta_pf'))
            if configuration and key in ranks:
                selected[method, configuration] = ranks[key]
        self.offline_ranks = selected

    def _offline_contact(self, token):
        if time.monotonic() >= self.offline_progress_next:
            with self.executor.store._connection() as db:
                self.offline_progress = dict(db.execute(
                    "SELECT method,count(*) FROM jobs WHERE state='ACCEPTED' GROUP BY method"))
            self.offline_progress_next = time.monotonic()+30
        self.controls[token].update(contact=int(time.time()), accepted_by_method=self.offline_progress)
        from .rpe_pbs_offline import independent_plan
        if independent_plan(self.offline_plan):
            status = self.executor.run_dir/'PBS_PARTITION_STATUS.json'
            if status.is_file():
                self.controls[token]['partition_best'] = read(status).get('best_by_method',{})

    def _deferred_evidence_owner(self, host, payload):
        job, attempt, sha = (payload[key] for key in ('job_id', 'attempt_id', 'request_sha256'))
        row = self.executor.store.row(job)
        if (not self.defer_q3_evidence or not row or self.executor._preparation_request(row['request'])
                or (host, attempt, sha) != (row['host_id'], row['attempt_id'], row['request_sha256'])):
            return None
        return row, self.evidence_directory/job/(attempt+'.json')

    def _batch_evidence_status(self, token, payload):
        owners = payload['pbs_evidence_status_batch']
        host = self.state['allocations'][token]['spec']['profile']['host_id']
        keys = ('job_id', 'attempt_id', 'request_sha256')
        if (not self.batch_q3_completions or payload['host_id'] != host or not isinstance(owners, list)
                or any(not isinstance(owner, dict) or any(not isinstance(owner.get(k), str) for k in keys)
                       for owner in owners)):
            raise ValueError('Invalid PBS evidence status batch')
        ready = []
        for owner in owners:
            verified = self._deferred_evidence_owner(host, owner)
            if verified and verified[0]['state'] == 'ACCEPTED' and not verified[1].exists():
                ready.append({key: owner[key] for key in keys})
        return dict(ok=True, result=dict(evidence_complete=ready))

    def _complete_with_deferred_evidence(self, token, request):
        """Commit the exact outcome before retrieving the worker's large evidence."""
        from .rpe_q3_coordinator import _artifact_destination

        payload = request['payload']
        job, attempt, sha = (payload[key] for key in ('job_id', 'attempt_id', 'request_sha256'))
        host = self.state['allocations'][token]['spec']['profile']['host_id']
        verified = self._deferred_evidence_owner(host, payload)
        if verified is None or payload['host_id'] != host:
            raise ValueError('PBS deferred Q3 evidence does not match the assigned attempt')
        row, marker = verified
        if payload.get('pbs_evidence_status') is True:
            if row['state'] != 'ACCEPTED':
                raise ValueError('PBS Q3 core is not yet accepted')
            # Only the receipt's sender and the existing JobStore own the
            # decision. An observation must never submit or adopt another job.
            return dict(ok=True, result=dict(ok=True, duplicate=True, evidence_pending=marker.exists()))
        evidence = payload['pbs_evidence']
        files = payload['files']
        if (not isinstance(evidence, dict) or not evidence or evidence.keys() & files.keys()
                or not {'OUTCOME.json', 'PROCESS_RETURN.json'} <= files.keys()
                or any(type(size) is not int or size < 0 for size in evidence.values())):
            raise ValueError('Invalid PBS deferred evidence manifest')
        directory = self.executor.run_dir/'jobs'/row['method']/job
        destinations = [_artifact_destination(directory, name) for name in evidence]
        receipt = dict(token=token, host_id=host, job_id=job, attempt_id=attempt, request_sha256=sha,
                       files=evidence)
        if marker.exists() and read(marker) != receipt:
            raise ValueError('PBS evidence manifest changed for the same attempt')
        if row['state'] == 'ACCEPTED':
            # A terminated worker may already have been recovered by ordinary
            # harvest. The returned outcome must still match the accepted one.
            for name in ('OUTCOME.json', 'PROCESS_RETURN.json'):
                if (directory/name).read_bytes() != base64.b64decode(files[name], validate=True):
                    raise ValueError('Repeated PBS completion changed its outcome or exit receipt')
            result = dict(ok=True, duplicate=True)
            if not marker.exists() and all(path.is_file() and path.stat().st_size == evidence[name]
                                           for path, name in zip(destinations, evidence)):
                final = dict(ok=True, result=dict(result, evidence_pending=False))
                return self._record_rpc_response(token, request['request_id'], digest(request), final)
        else:
            result = None
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            atomic(marker, receipt)
        if result is None:
            result = self.executor.rpc('/v1/complete', {k: v for k, v in payload.items() if k != 'pbs_evidence'})
            if not result.get('duplicate'):
                self.log('PBS_Q3_CORE_ACCEPTED', job_id=job, attempt_id=attempt,
                         deferred_bytes=sum(evidence.values()),
                         completion_rpc_clock=request.get('clock'))
        # A short-lived reply unlocks the worker's collection lane. The second
        # acknowledgement allows it to clean original copies after archiving.
        return dict(ok=True, result=dict(result, evidence_pending=marker.exists()))

    def _archive_evidence(self, marker, receipt):
        from .rpe_q3_coordinator import install_files, publish_staged_files

        token, job, attempt = (receipt[key] for key in ('token', 'job_id', 'attempt_id'))
        row = self.executor.store.row(job)
        if (not row or row['state'] != 'ACCEPTED' or
                any(row[key] != receipt[key] for key in ('host_id', 'attempt_id', 'request_sha256'))):
            raise ValueError('Deferred PBS evidence has no accepted original attempt')
        root = self.executor.run_dir
        transfer = root/'distributed'/'transfers'/job/attempt
        target = root/'jobs'/row['method']/job
        for name, size in receipt['files'].items():
            if self.stop.is_set():
                return
            raw = self._read_bulk(self.evidence_gateway, token, 'attempt', name, size=size,
                                  job_id=job, attempt_id=attempt)
            install_files(transfer, {name: base64.b64encode(raw).decode('ascii')})
        publish_staged_files(transfer, target)
        marker.unlink()
        self.log('PBS_DEFERRED_EVIDENCE_COLLECTED', job_id=job, attempt_id=attempt,
                 bytes=sum(receipt['files'].values()))

    def _advance_deferred_evidence(self):
        if self.evidence_pending is not None:
            marker, future = self.evidence_pending
            if not future.done():
                return
            self.evidence_pending = None
            try:
                future.result()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                self.evidence_retry_after[marker] = time.monotonic() + self.settings.get('request_timeout_seconds', 20)
                self.log('PBS_DEFERRED_EVIDENCE_PENDING', job_id=marker.parent.name, error=str(error))
            else:
                self.evidence_retry_after.pop(marker, None)
        for marker in self.evidence_directory.glob('*/*.json'):
            if time.monotonic() < self.evidence_retry_after.get(marker, 0):
                continue
            receipt = read(marker)
            row = self.executor.store.row(receipt['job_id'])
            if not row or row['state'] != 'ACCEPTED':
                continue  # The ordinary worker or end-of-session harvest owns recovery.
            if self.evidence_pool is None:
                self.evidence_gateway = Gateway(self.settings)
                self.evidence_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pbs-evidence')
            self.evidence_pending = marker, self.evidence_pool.submit(self._archive_evidence, marker, receipt)
            return

    @staticmethod
    def _read_bulk(gateway, token, kind, name, *, size=None, **owner):
        result = bytearray()
        identity = {}
        while size is None or len(result) < size:
            chunk = gateway.call(dict(action='read_bulk_chunk', token=token, kind=kind,
                                      name=name, offset=len(result), **owner,
                                      **identity, **({'size': size} if size is not None else {})))
            if size is None:
                size = chunk['size']
            if type(size) is not int or size < 0 or chunk['size'] != size:
                raise ValueError('Invalid PBS bulk transfer size')
            identity = {'file_identity': chunk['file_identity']}
            raw = (chunk['text'].encode('ascii') if kind == 'spool' else
                   base64.b64decode(chunk['data'], validate=True))
            if len(raw) > 1024*1024 or len(result)+len(raw) > size or (not raw and len(result) < size):
                raise ValueError('PBS bulk transfer made no valid progress')
            result.extend(raw)
        return result

    def _fetch_harvest(self, gateway, token, owners=None, *, live=False):
        request = dict(action='harvest', token=token)
        if owners is not None:
            request['owners'] = [dict(job_id=job, attempt_id=attempt) for job, attempt in sorted(owners)]
        if live:
            request['completed_only'] = True
        if self.bounded_results:
            request['manifest_only'] = True
        data = gateway.call(request)
        if not self.bounded_results:
            missing = self.executor.artifact_cache.missing(data.get('artifact_hashes', []))
            blobs = gateway.call(dict(request, hashes=missing))['blobs'] if missing else {}
            return data, blobs
        hashes = set()
        for item in data['attempts']:
            assignment = item['assignment']
            files = {}
            for name, size in item['files'].items():
                raw = self._read_bulk(gateway, token, 'attempt', name, size=size,
                                      job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])
                if name == 'OUTCOME.json':
                    hashes.update(json.loads(raw).get('artifacts', {}).values())
                files[name] = base64.b64encode(raw).decode('ascii')
            item['files'] = files
        missing = self.executor.artifact_cache.missing(sorted(hashes))
        blobs = {sha: base64.b64encode(self._read_bulk(gateway, token, 'artifact', sha)).decode('ascii')
                 for sha in missing}
        return data, blobs

    def _accept_harvested(self, allocation, data, blobs, owners, *, live=False):
        if owners is not None and data['host_id'] != allocation['host_id']:
            raise ValueError('PBS harvest belongs to a different worker')
        if blobs:
            self.executor.artifact_cache.put(blobs)
        count = 0
        for item in data['attempts']:
            assignment = item['assignment']
            job, attempt = assignment['job_id'], assignment['attempt_id']
            if owners is not None and (job, attempt) not in owners:
                raise ValueError('Unrequested PBS attempt returned by harvest')
            row = self.executor.store.row(job)
            if row is not None and row['state'] == 'ACCEPTED':
                continue
            if live and item['state'].get('state') != 'COMPLETED_PENDING_TRANSFER':
                raise ValueError('PBS did not provide a locally verified completion')
            try:
                self.executor.rpc('/v1/complete', dict(host_id=data['host_id'], job_id=job, attempt_id=attempt,
                    request_sha256=assignment['request_sha256'], files=item['files']))
                count += 1
            except (OSError, ValueError) as error:
                if not live:
                    raise
                self.log('PBS_LIVE_RESULT_PENDING', token=allocation['token'], job_id=job, attempt_id=attempt,
                         error=f'{type(error).__name__}: {error}')
        return count

    def _harvest(self, allocation, *, fetched=None, owners=None, revisit=False):
        token = allocation['token']
        record = self.state['allocations'][token]
        if record.get('harvested') and not revisit:
            return
        data, blobs = fetched if fetched is not None else self._fetch_harvest(self.gateway, token)
        self._accept_harvested(allocation, data, blobs, owners)
        ended = allocation['status'] in TERMINAL or (allocation.get('worker') or {}).get('state') == 'EXITED'
        for item in data.get('incomplete', []) if ended else []:
            assignment = item['assignment']; row = self.executor.store.row(assignment['job_id'])
            if (owners is not None and (assignment['job_id'], assignment['attempt_id']) not in owners):
                raise ValueError('Unrequested incomplete PBS attempt returned by harvest')
            if row is None or row['attempt_id'] != assignment['attempt_id'] or row['host_id'] != data['host_id']:
                continue
            detail = item.get('state', {}).get('detail') or {}
            admission_exit = detail.get('pbs_admission_exit') or {}
            released_before_launch = (detail.get('resource_released') is True and
                detail.get('automatic_retry') is False and detail.get('scientific_execution_started') is False and
                row.get('started_epoch') is None and admission_exit.get('token') == token and
                all(admission_exit.get(k) == row[k] for k in ('job_id', 'attempt_id', 'request_sha256')))
            released_attempt = self._attempt_released(detail, token, allocation.get('job_id'), row)
            if (item['state'].get('state') == 'UNKNOWN'
                    and (detail.get('prelaunch_rejection') or released_before_launch or released_attempt)
                    and row['state'] not in ('ACCEPTED', 'CANCELLED')
                    and row['request_sha256'] == assignment['request_sha256']):
                self.executor.rpc('/v1/state', dict(host_id=data['host_id'], job_id=row['job_id'],
                    attempt_id=row['attempt_id'], state='UNKNOWN', detail=detail))
                continue
            if row is not None and row['state'] not in ('ACCEPTED','UNKNOWN','CANCELLED'):
                self.executor.rpc('/v1/state', dict(host_id=data['host_id'], job_id=assignment['job_id'],
                    attempt_id=assignment['attempt_id'], state='UNKNOWN',
                    detail=dict(reason='PBS_ENDED_WITHOUT_TASK_EXIT_RECEIPT', automatic_retry=False)))
        if ended and owners is None:
            # A shared pilot can expose several historical sessions with the
            # same host ID. An old session cannot declare a newer attempt lost.
            for row in self.executor._owned_attempts():
                if row['host_id'] == data['host_id'] and row['state'] != 'UNKNOWN':
                    self.executor.rpc('/v1/state',dict(host_id=row['host_id'], job_id=row['job_id'],
                        attempt_id=row['attempt_id'], state='UNKNOWN',detail=dict(
                            reason='PBS_ENDED_WITHOUT_COMPLETE_ORIGINAL_ATTEMPT',automatic_retry=False)))
        record['harvested'] = ended
        if ended and self.settings.get('gateway_sha256'):
            record['harvest_gateway_sha256'] = self.settings['gateway_sha256']
        self.log('PBS_SAVED_ARTIFACTS_RECONCILED', token=token, attempts=len(data['attempts']))

    @staticmethod
    def _attempt_released(detail, token, pbs_job_id, row):
        proof = detail.get('pbs_attempt_exit') or {}
        return (bool(pbs_job_id) and proof.get('pbs_job_id') == pbs_job_id and proof.get('token') == token
                and proof.get('session_ended') is True and type(proof.get('operator_stop')) is bool
                and detail.get('resource_released') is True and detail.get('automatic_retry') is False
                and all(proof.get(k) == row.get(k) for k in ('job_id', 'attempt_id', 'request_sha256')))

    def _collect_live_results(self, allocation, owners):
        data, blobs = self._fetch_harvest(self.harvest_gateway, allocation['token'], owners, live=True)
        return self._accept_harvested(allocation, data, blobs, owners, live=True)

    def _advance_harvest(self):
        """Keep remote archive reads out of the live worker exchange."""
        if self.harvest_pending is not None:
            allocation, owners, revisit, future = self.harvest_pending
            if not future.done():
                return
            self.harvest_pending = None
            try:
                self._harvest(allocation, fetched=future.result(), owners=owners, revisit=revisit)
            except Exception as error:
                token = allocation['token']
                self.harvest_retry_after[token] = time.monotonic() + self.settings.get('request_timeout_seconds', 20)
                self.log('PBS_HARVEST_PENDING', token=token, error=f'{type(error).__name__}: {error}')
            self.save()
        if self.live_harvest_pending is not None:
            token, future = self.live_harvest_pending
            if not future.done():
                return
            self.live_harvest_pending = None
            try:
                if count := future.result():
                    self.log('PBS_LIVE_RESULTS_COLLECTED', token=token, attempts=count)
            except Exception as error:
                self.log('PBS_LIVE_HARVEST_PENDING', token=token, error=f'{type(error).__name__}: {error}')
        if self.stop.is_set():
            return
        # Capacity released for an UNKNOWN attempt is unrelated to its
        # recoverability. Read ownership from the ledger independently.
        with self.executor.store._connection() as db:
            owned = [dict(row) for row in db.execute(
                "SELECT job_id, host_id, attempt_id, method, assigned_epoch, state, "
                "(state='UNKNOWN' AND completion_sha256 IS NULL AND started_epoch IS NULL "
                "AND json_extract(detail,'$.reason')='ValueError: Existing adapter dry-check failed' "
                "AND json_type(detail,'$.prelaunch_rejection') IS NULL) AS needs_prelaunch_proof "
                "FROM jobs WHERE host_id IS NOT NULL "
                "AND attempt_id IS NOT NULL AND state NOT IN ('QUEUED','CANCELLED','ACCEPTED')")]
        revision = self.settings.get('gateway_sha256')
        for allocation in self.allocations:
            token = allocation['token']
            record = self.state['allocations'][token]
            revisit = bool(record.get('harvested'))
            if (allocation['status'] not in TERMINAL | {'HISTORY_MISSING_RECONCILE_SPOOL'}
                    or (revisit and (not revision or record.get('harvest_gateway_sha256') == revision))
                    or time.monotonic() < self.harvest_retry_after.get(token, 0)):
                continue
            owners = {(row['job_id'], row['attempt_id']) for row in owned
                      if row['host_id'] == allocation['host_id']
                      and (not revisit or row['needs_prelaunch_proof'])}
            if not owners:
                if revisit:
                    continue
                ended = allocation['status'] in TERMINAL or (allocation.get('worker') or {}).get('state') == 'EXITED'
                if ended:
                    record['harvested'] = True
                    self.log('PBS_SAVED_ARTIFACTS_RECONCILED', token=token, attempts=0, source='central ownership ledger')
                continue
            if self.harvest_pool is None:
                self.harvest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rpe-pbs-harvest')
            future = self.harvest_pool.submit(self._fetch_harvest, self.harvest_gateway, token, owners)
            self.harvest_pending = (allocation, owners, revisit, future)
            return
        if self.live_harvest:
            self._advance_live_harvest(owned)

    def _advance_live_harvest(self, owned):
        # The worker still uploads normally. If acknowledgements lag, the
        # gateway can pick up several durable preparation receipts at once.
        now = time.time()
        monotonic = time.monotonic()
        grace = self.settings.get('request_timeout_seconds', 20)
        checked = self.live_harvest_checked
        eligible = {}
        for row in owned:
            if row['method'].startswith('PREPARATION:') and row['assigned_epoch'] is not None and (
                    row['state'] == 'COMPLETED_PENDING_TRANSFER' or now-row['assigned_epoch'] >= grace):
                eligible[(row['job_id'], row['attempt_id'])] = row
        self.live_harvest_checked = {owner: at for owner, at in checked.items() if owner in eligible}
        for allocation in self.allocations:
            if allocation['status'] != 'R':
                continue
            candidates = [owner for owner, row in eligible.items()
                          if row['host_id'] == allocation['host_id'] and monotonic-checked.get(owner, -grace) >= grace]
            if not candidates:
                continue
            candidates.sort(key=lambda owner: (checked.get(owner, 0), eligible[owner]['assigned_epoch'], owner))
            owners = set(candidates[:16])
            self.live_harvest_checked.update(dict.fromkeys(owners, monotonic))
            if self.harvest_pool is None:
                self.harvest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rpe-pbs-harvest')
            future = self.harvest_pool.submit(self._collect_live_results, allocation, owners)
            self.live_harvest_pending = (allocation['token'], future)
            return

    def _demand(self):
        # This is advisory sizing, not reservation or assignment. Copy the
        # mutable inputs briefly; the actual queue owner is checked at claim.
        with self.executor.state.cv:
            preparation = {job: dict(record, failures=dict(record.get('failures', {})),
                                     failure_epochs=dict(record.get('failure_epochs', {})))
                           for job, record in self.executor.preparation_pending.items()}
            observed_hosts = tuple(self.executor.hosts)
        return self._estimate_demand(preparation, observed_hosts)

    def _estimate_demand(self, preparation, observed_hosts):
        now = time.time()
        # A failure never grants eligibility. It merely delays another
        # preflight, including when the retained allocation has no session yet.
        def retry_due(record, host):
            return (host not in record.get('failures', {}) or
                    now - record.get('failure_epochs', {}).get(host, 0) >= 60)

        def estimate():
            policy, error = self.executor._host_admission_policy()
            admission = {host: self.executor._host_admission(host, policy, error) for host in self.configured}
            hosts = [host for host, rule in admission.items() if rule['accepting'] and rule.get('max_slots') != 0]
            def compatible(request, eligibility, *, unchecked=False):
                from .rpe_q3_resources import preparation_forecast
                artificial = self.executor.store.artificial(request)
                result = []
                for host in hosts:
                    if not (self.executor._host_supports_stage(host, request) and
                            (artificial or (task_estimate(request, self.settings, host) or 0) > 0) and
                            (unchecked or (eligibility is None and artificial) or
                             host in (eligibility or {}).get('eligible_host_ids', []))):continue
                    forecast = preparation_forecast(request, self.settings.get('preparation_resource_observations'), host=host)
                    if forecast:
                        profile = self.configured[host]['profile']
                        storage_limit = (profile.get('storage_limit') or {}).get('limit_bytes')
                        if type(storage_limit) is int and forecast.get('storage_bytes', 0) > storage_limit:continue
                        try:
                            cpu, memory = resource_need(request, profile, self.settings, host)
                            if cpu > self.settings['resources']['ncpus'] or memory > usable_memory(profile, self.settings):continue
                        except (KeyError, ValueError):continue
                    result.append(host)
                return result

            # A task can already be reserved on another host before a PBS
            # allocation exists to check it. Retain that eligible demand if
            # there is no compatible free slot outside PBS.
            checks = {}
            budgets = self.executor.store.preparation_reservations()
            for job, record in preparation.items():
                request = record['request']
                row = self.executor.store.row(job)
                if row is not None and row['state'] not in ('QUEUED', 'CANCELLED'):
                    continue
                stage = request['preparation']['stage']
                budget = record.get('budget')
                needs_reservation = row is None or row['state'] == 'CANCELLED'
                if needs_reservation and (not record.get('approved', True) or
                        (budget not in (None, 'UNLIMITED') and budgets.get(stage, 0) >= budget)):
                    continue
                usable = [host for host in compatible(request, None, unchecked=True)
                          if retry_due(record, host)]
                if usable:
                    checks[job] = (usable, request)
                    if needs_reservation and budget not in (None, 'UNLIMITED'):
                        budgets[stage] = budgets.get(stage, 0) + 1
            ready = []
            if hasattr(self.executor, '_ready_queue_snapshot'):
                queue, queue_rows = self.executor._ready_queue_snapshot()
            else:
                queue = self.executor._ready_queue()
                queue_rows = {task.job.name:self.executor.store.row(task.job.name) for task in queue}
            for task in queue:
                row = queue_rows.get(task.job.name)
                if row is None or row['state'] != 'QUEUED':
                    continue
                usable = compatible(row['request'], row['eligibility'])
                unchecked, _ = checks.pop(task.job.name, ([], None))
                if usable or unchecked:
                    ready.append((task, usable, unchecked, row['request']))
            available = {host:self.executor._offer_free(host,policy,error) for host in observed_hosts if host not in self.configured}
            startup = self.startup_jobs()
            matched = set() if self.hold_node else self.executor._match_ready_jobs(
                [task for task, _, _, _ in ready if task.job.name not in startup],
                {host:count for host,count in available.items() if count > 0})
            outstanding = [(usable, request) for task, usable, _, request in ready
                           if usable and task.job.name not in matched]
            pending = [(unchecked, request) for task, usable, unchecked, request in ready
                       if not usable and task.job.name not in matched]
            unreserved = []
            for job, value in checks.items():
                row = self.executor.store.row(job)
                if row is None or row['state'] in ('QUEUED', 'CANCELLED'):
                    unreserved.append(value)
            return outstanding, pending + unreserved, len(ready) + len(unreserved), len(matched)
        return estimate()

    def _submit(self, host, worker, demand, requests=()):
        token = uuid.uuid4().hex
        if self.hold_node and (alongside:=known_independent_search_jobs(self.executor.run_dir)):
            demand=dict(demand,alongside_search_job_ids=alongside)
        profile = deepcopy(worker['profile'])
        resources = deepcopy(self.settings['resources'])
        sizing = self.settings.get('allocation_sizing')
        if demand.get('independent_blocks'):
            policy = profile.setdefault('slot_policy', {})
            capacity = min(resources['ncpus'], policy.get('max_slots') or resources['ncpus'])
            policy['initial_slots'] = capacity
            if sizing and sizing.get('per_task_resources'):
                profile.setdefault('pbs',{})['per_task_resources'] = True
            demand = dict(demand, requested_task_capacity=capacity)
        elif sizing and sizing.get('per_task_resources'):
            packed = []
            cpus, gib, capacity = packed_resources(requests, profile, self.settings, host, selected=packed)
            if not capacity:
                raise ValueError('No PBS task fits the configured CPU and memory pool')
            if self.hold_node:
                policy = profile.get('slot_policy') or {}
                capacity = min(resources['ncpus'], policy.get('max_slots') or resources['ncpus'])
            else:
                resources.update(ncpus=cpus, mem=f'{gib}gb')
            profile.setdefault('pbs', {})['per_task_resources'] = True
            profile.setdefault('slot_policy', {})['initial_slots'] = capacity
            demand = dict(demand, requested_task_capacity=capacity, per_task_resources=True)
            if self.hold_node:
                demand['held_node'] = True
            # Hold only work that passed the PBS-specific input check and fit
            # this actual request. No queue owner, attempt or budget is changed.
            startup_jobs = []
            for request in (packed if not self.hold_node else ()):
                job = request.get('request_id')
                row = self.executor.store.row(job) if job else None
                if row and row['state'] == 'QUEUED' and host in (row.get('eligibility') or {}).get('eligible_host_ids', ()):
                    startup_jobs.append(job)
            if startup_jobs:
                samples = [a.get('submit_to_usable_seconds', a.get('submit_to_ready_seconds'))
                           for a in self.state['allocations'].values()]
                measured = [seconds for seconds in samples if seconds is not None]
                demand.update(startup_jobs=startup_jobs, startup_wait_seconds=min(
                    300., max(measured[-6:], default=30.) + 3 * self.readiness_wait_seconds()))
        elif sizing:
            cost = task_cpu_footprint(profile)
            maximum = int(min(resources['ncpus'], (profile.get('slot_policy') or {}).get(
                'resource_limits', {}).get('cpu_count', resources['ncpus']))) // cost
            capacity = min(maximum, max(1, len(requests)))
            if capacity < 1:
                raise ValueError('PBS CPU limit does not fit one configured task')
            resources['ncpus'] = capacity * cost
            policy = profile.setdefault('slot_policy', {})
            if policy.get('max_slots') is not None:
                capacity = min(capacity, policy['max_slots'])
                resources['ncpus'] = capacity * cost
            if capacity < 1:
                raise ValueError('PBS slot limit does not fit one configured task')
            estimates = [task_memory_estimate(request, self.settings, host) for request in requests[:capacity]]
            if estimates and all(value is not None for value in estimates):
                amount, unit = re.fullmatch(r'(\d+)([kmgt]?b)?', resources['mem'].lower()).groups()
                maximum_memory = int(amount) * (1024 ** ('b', 'kb', 'mb', 'gb', 'tb').index(unit or 'b'))
                task_memory = max(*estimates, (profile.get('pbs') or {}).get('task_memory_bytes') or 0)
                unit = 1024**3
                task_gib = (task_memory + unit - 1) // unit
                overhead_gib = 1
                capacity = min(capacity, (maximum_memory // unit - overhead_gib) // task_gib)
                if capacity < 1:
                    raise ValueError('PBS memory limit does not fit one compatible task')
                resources.update(ncpus=capacity * cost, mem=f'{capacity * task_gib + overhead_gib}gb')
                policy['initial_slots'] = capacity
                profile.setdefault('pbs', {}).update(task_memory_bytes=task_gib * unit,
                                                     sized_task_memory_bytes=task_memory)
            elif requests and all(request.get('route', {}).get('kind') != 'preparation'
                                and task_estimate(request, self.settings, host) for request in requests):
                amount, unit = re.fullmatch(r'(\d+)([kmgt]?b)?', resources['mem'].lower()).groups()
                memory = int(amount) * (1024 ** ('b', 'kb', 'mb', 'gb', 'tb').index(unit or 'b'))
                per_task = sizing['q3_startup_memory_bytes_per_task']
                if (profile.get('pbs') or {}).get('task_memory_bytes'):
                    per_task = max(per_task, profile['pbs']['task_memory_bytes'])
                policy['initial_slots'] = max(1, min(capacity, memory // per_task))
            demand = dict(demand, requested_cpu_per_task=cost, requested_task_capacity=capacity,
                          initial_slots=policy.get('initial_slots', profile.get('slots', 1)))
        profile['host_id'] = host
        profile['work_dir'] = self.settings['remote_root']+'/workers/'+host
        profile.setdefault('prefix_cache_directory', self.settings['remote_root']+'/cache/'+host+'/dp')
        profile.setdefault('artifact_cache_directory', self.settings['remote_root']+'/cache/'+host+'/artifacts')
        if self.settings.get('namespace_by_run', True):
            pool = Path(self.settings['remote_root']).parent/'preparation-metadata-cache'
            storage = profile.get('storage_limit') or {}
            if any(pool.is_relative_to(Path(root)) for root in storage.get('roots', [])):
                profile.setdefault('preparation_metadata_cache_directory', str(pool))
        profile['pbs'] = dict(profile.get('pbs') or {}, worker_source=self.settings['transport_source'],
            worker_root=self.settings['transport_root'], drain_seconds=self.settings.get('drain_seconds', 30))
        profile['pbs'].setdefault('rpc_timeout_seconds', max(15., 2 * self.settings.get('request_timeout_seconds', 20)))
        if self.bounded_results:
            profile['pbs']['bounded_results'] = True
        if self.compute_storage_sampling:
            profile['pbs']['compute_storage_sampling'] = True
        if self.defer_q3_evidence:
            profile['pbs']['defer_q3_evidence'] = True
        if self.batch_q3_completions:
            profile['pbs']['batch_q3_completions'] = True
        if getattr(self.executor, 'distributed', {}).get('pbs_q3_memory_backfill') is True:
            profile['pbs']['q3_memory_backfill'] = True
        if self.offline_plan:
            raw = json.dumps(self.offline_plan, sort_keys=True, separators=(',', ':')).encode()
            sha = hashlib.sha256(raw).hexdigest()
            path = self.directory/'offline-plans'/f'{sha}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(raw)
                temporary.replace(path)
            profile['pbs']['offline_plan'] = dict(sha256=sha,
                path=str(Path(self.settings['remote_root'])/'offline-plans'/path.name))
            from .rpe_pbs_offline import independent_plan
            if independent_plan(self.offline_plan):
                profile['pbs']['independent_root'] = str(Path(self.settings['remote_root'])/'independent-search')
        if (waves := getattr(self.executor, 'distributed', {}).get('pbs_standby_waves')) is not None:
            standby_capacity(dict(cpu_count=resources['ncpus'], memory_bytes=GIB), waves)
            profile['pbs']['standby_waves'] = waves
        collection_workers = getattr(self.executor, 'distributed', {}).get('pbs_collection_workers')
        if collection_workers is not None:
            profile['pbs']['collection_workers'] = collection_workers
        deployment=worker.get('deployment')or self.settings.get('deployment')
        if deployment:
            profile['pbs']['deployment']=deepcopy(deployment)
        if worker.get('task_deployments'):
            profile['pbs']['task_deployments'] = deepcopy(worker['task_deployments'])
        if self.transport_paths:
            profile = pbs_transport_paths(profile, self.transport_paths)
        interpreter = self.settings['python']
        if self.storage_launcher and profile.get('storage_limit'):
            if not self.storage_launcher_verified:
                self.gateway.verify_storage_launcher(self.storage_launcher)
                self.storage_launcher_verified = True
            interpreter = self.storage_launcher['path']
        spec = dict(token=token, profile=profile, resources=resources,
            python=interpreter, transport_root=self.settings['transport_root'],
            run_id=str(self.executor.run_dir), reason=demand)
        # Save intent BEFORE making the remote call. On lost qsub/SSH response
        # the same token is reconciled, never replaced with a new submission.
        started = stamp()
        self.state['allocations'][token] = dict(spec=spec, status='INTENT', clock=started, submitted_clock=started)
        self.save()
        self.log('PBS_SUBMISSION_INTENT', token=token, host_id=host, resources=spec['resources'], reason=demand)
        self._ensure_spec(spec)
        return spec

    def _ensure_spec(self, spec):
        reference = spec.get('profile', {}).get('pbs', {}).get('offline_plan')
        if reference and reference['sha256'] not in self.offline_staged:
            sha = reference['sha256']
            source = self.directory/'offline-plans'/f'{sha}.json'
            self.gateway.stage_artifact(source, reference['path'], sha)
            self.offline_staged.add(sha)
        return self.gateway.call(dict(action='ensure', spec=spec))

    def _rollover_storage_launcher(self, allocation, record):
        """Drain an obsolete idle worker session, not its physical reservation."""
        if (not self.storage_launcher or not allocation.get('pilot_token')
                or allocation.get('retained_replacement') or allocation['status'] != 'R'
                or record['spec']['python'] == self.storage_launcher['path']
                or record.get('storage_maintenance_requested')):
            return
        if not self.storage_launcher_verified:
            if time.monotonic() < getattr(self, '_launcher_rollover_retry_after', 0):
                return
            self._launcher_rollover_retry_after = time.monotonic() + 60
            try:
                self.gateway.verify_storage_launcher(self.storage_launcher)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.log('PBS_LAUNCHER_ROLLOVER_DEFERRED',token=allocation['token'],reason=str(error))
                return
            self.storage_launcher_verified=True
        token, host = allocation['token'], allocation['host_id']
        old = record['spec'].get('reason',{}).get('retained_replacement',{}).get('old_owners',[])
        report = dict(pbs=dict(allocation_token=token,other_allocation_owners=old))
        if not record.get('launcher_rollover_requested'):
            record['launcher_rollover_requested']=True
            self.log('PBS_LAUNCHER_ROLLOVER_DRAIN',token=token,host_id=host,
                     job_id=allocation['job_id'],reservation_retained=True)
        owned = self.executor._offer_owners(host,self.executor._owned_attempts(),report)
        if not owned:
            record['storage_maintenance_requested']=True
            self.log('PBS_LAUNCHER_ROLLOVER_IDLE',token=token,host_id=host,
                     job_id=allocation['job_id'],reservation_retained=True)

    def _connect_retained_replacement(self):
        """Open a session on the new node while its predecessor only drains."""
        previous = next((row for row in self.allocations if row.get('retained_replacement')), None)
        if previous is None:
            return
        pilot = previous['retained_replacement']
        matching = [record for record in self.state['allocations'].values()
                    if (record['spec'].get('reason', {}).get('retained_replacement') or {}).get('pilot_token') == pilot
                    or record.get('observed', {}).get('pilot_token') == pilot]
        if matching:
            if any(record['status'] not in TERMINAL for record in matching):
                return
            # An ended session does not mean the reservation ended. Only a
            # proved session exit on the still-running replacement may reattach.
            ended = next((record for record in reversed(matching)
                          if record.get('launcher_rollover_requested')
                          and record.get('storage_maintenance_requested')
                          and record.get('observed',{}).get('session_end')
                          and record.get('observed',{}).get('reservation_status') == 'R'),None)
            if ended is None:
                return
            owners = (ended['spec'].get('reason',{}).get('retained_replacement') or {}).get('old_owners',[])
        else:
            with self.executor.store._connection() as db:
                owners = [list(row) for row in db.execute(
                    "SELECT job_id,attempt_id,request_sha256 FROM jobs WHERE host_id=? AND state='ASSIGNED'", (previous['host_id'],))]
        host = previous['host_id']
        worker = self.configured[host]
        outstanding, pending, queued, free = self._demand()
        requests = [request for hosts, request in outstanding + pending if host in hosts]
        if not requests or not self._storage_accepting(worker):
            return
        demand = dict(ready_tasks=queued, existing_host_free_slots=free,
            retained_replacement=dict(pilot_token=pilot, predecessor=previous['pilot_token'], old_owners=owners))
        try:
            self._submit(host, worker, demand, requests)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            self.log('PBS_RETAINED_REPLACEMENT_HELD', host_id=host, reason=f'{type(error).__name__}: {error}')

    def _assigned_pilot_replay(self, host, owned):
        """Select a closed session's live node for durable assignment replay only."""
        if not self.hold_node:
            return None
        attempts = [row for row in owned if row['host_id'] == host]
        if not attempts or any(row['state'] != 'ASSIGNED' for row in attempts):
            return None
        closed = [a for a in self.allocations if a['host_id'] == host and a.get('session_end')]
        if not closed:
            return None
        previous = max(closed, key=lambda a: a['session_end']['epoch'])
        if (previous.get('status') != 'F' or not previous.get('pilot_token') or not previous.get('job_id')
                or (previous.get('scheduler') or {}).get('job_state') != 'R'
                or (previous.get('worker') or {}).get('state') != 'EXITED'):
            return None
        rows = [self.executor.store.row(row['job_id']) for row in attempts]
        if any(row is None or row['state'] != 'ASSIGNED' or row['started_epoch'] is not None
               or (row['host_id'], row['attempt_id']) != (host, attempt['attempt_id'])
               for row, attempt in zip(rows, attempts)):
            return None
        return dict(token=previous['pilot_token'], job_id=previous['job_id']), [row['request'] for row in rows]

    def _flush_responses(self):
        if not self.responses:
            return
        started = time.monotonic()
        responses = sum(len(batch) for batch in self.responses.values())
        self.gateway.call(dict(action='exchange', responses=self.responses, controls={},
                               responses_only=True))
        self.responses = {}
        self.log('PBS_RESPONSE_FLUSH', replies=responses, seconds=time.monotonic()-started)

    def _timed_dispatch(self, token, request):
        started = time.monotonic()
        result = self._dispatch_ready(token, request)
        elapsed = time.monotonic()-started
        if elapsed >= 1:
            self.log('PBS_REQUEST_TIMING', token=token, endpoint=request.get('endpoint', 'bulk'),
                     request_id=request['request_id'], seconds=elapsed,
                     replied=result is not None)
        return result

    def _record_rpc_response(self, token, request_id, request_sha256, response):
        path = self.directory/'responses'/token/(request_id+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            previous = read(path)
            if previous['request_sha256'] != request_sha256:
                raise ValueError('Reused PBS RPC ID has changed content')
            return previous['response']
        atomic(path, dict(request_sha256=request_sha256, response=response, clock=stamp()))
        return response

    def _collect_completed_requests(self, responses):
        for (token, request_id), (request, future) in tuple(self.pending_completions.items()):
            if not future.done():
                continue
            del self.pending_completions[token, request_id]
            try:
                result = future.result()
            except Exception as error:
                self.log('PBS_COMPLETION_EXCHANGE_PENDING', token=token, request_id=request_id,
                         error=str(error))
                continue
            if result is not None:
                responses.setdefault(token, {})[request_id] = result

    def _submit_completion(self, token, request):
        key = token, request['request_id']
        if key in self.pending_completions:
            if request != self.pending_completions[key][0]:
                raise ValueError('In-flight PBS completion RPC changed content')
        elif len(self.pending_completions) < self.completion_workers:
            self.pending_completions[key] = request, self.completion_pool.submit(
                self._timed_dispatch, token, request)

    def _exchange_completions(self):
        self._collect_completed_requests(self.completion_responses)
        exclude = {}
        for token, request_id in self.pending_completions:
            exclude.setdefault(token, []).append(request_id)
        remaining = self.completion_workers-len(self.pending_completions)
        if not remaining and not self.completion_responses:
            return
        start = time.monotonic()
        data = self.completion_gateway.call(dict(
            action='exchange', request_mode='completion', limit=remaining,
            exclude=exclude, responses=self.completion_responses, controls={},
            **({'inline_q3_completions': True} if self.batch_q3_completions else {}),
            **({'bounded_results': True} if self.bounded_results else {})))
        count = sum(len(rows) for rows in data['requests'].values())
        if count or self.completion_responses:
            self.log('PBS_COMPLETION_EXCHANGE_TIMING', seconds=time.monotonic()-start,
                     requests=count, replies=sum(len(rows) for rows in self.completion_responses.values()),
                     inline_cores=data.get('inline_cores', 0), inline_bytes=data.get('inline_bytes', 0))
        self.completion_responses = {}
        for token, requests in data['requests'].items():
            for request in requests:
                if 'spool_bytes' not in request and request['endpoint'] not in ('/v1/complete', '/v1/artifacts/put'):
                    raise ValueError('Non-completion request on PBS completion exchange')
                self._submit_completion(token, request)

    def _run_completions(self):
        try:
            while not self.stop.is_set():
                try:
                    self._exchange_completions()
                except Exception as error:
                    self.log('PBS_COMPLETION_EXCHANGE_PENDING', error=str(error))
                self.stop.wait(self._poll_delay())
        finally:
            self.completion_pool.shutdown(wait=True)
            self._collect_completed_requests(self.completion_responses)
            try:
                if self.completion_responses:
                    self.completion_gateway.call(dict(action='exchange', responses_only=True,
                                                       responses=self.completion_responses, controls={}))
            except Exception as error:
                self.log('PBS_COMPLETION_EXCHANGE_PENDING', error=str(error))
            finally:
                self.completion_gateway.close()

    def _queue_status(self, token, request):
        key = token, request['request_id']
        with self.status_lock:
            if key in self.status_pending:
                return
            self.status_pending.add(key)
            self.status_queue.put((token, request))

    def _run_status(self):
        """A slow status burst cannot delay acquisition of the next offer."""
        try:
            while True:
                item = self.status_queue.get()
                if item is None:
                    break
                batch = [item]
                closing = False
                while len(batch) < 32:
                    try:
                        item = self.status_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        closing = True
                        break
                    batch.append(item)
                replies = {}
                started = time.monotonic()
                try:
                    for token, request in batch:
                        result = self._timed_dispatch(token, request)
                        replies.setdefault(token, {})[request['request_id']] = result
                    self.status_gateway.call(dict(action='exchange', responses=replies,
                                                  controls={}, responses_only=True))
                    self.log('PBS_STATUS_EXCHANGE_TIMING', replies=len(batch),
                             seconds=time.monotonic()-started)
                except Exception as error:
                    # The request file and any committed reply survive and are
                    # replayed on the next snapshot or the next Resume.
                    self.log('PBS_STATUS_EXCHANGE_PENDING', error=str(error))
                finally:
                    with self.status_lock:
                        self.status_pending.difference_update(
                            (token, request['request_id']) for token, request in batch)
                if closing:
                    break
        finally:
            self.status_gateway.close()

    def _dispatch_requests(self, requests_by_token):
        if not self.split_exchange:
            self._collect_completed_requests(self.responses)
        for token, requests in requests_by_token.items():
            if self.split_exchange:
                # Apply finished state before reserving replacement work, but
                # do not hold assignment replies behind other status traffic.
                def priority(row):
                    if row is None:
                        return 3
                    if row['endpoint'] == '/v1/state' and row['payload'].get('state') == 'COMPLETED_PENDING_TRANSFER':
                        return 0
                    return 1 if row['endpoint'] == '/v1/poll' else 2
                requests = sorted(requests, key=priority)
            for request in requests:
                if request is None:
                    continue
                endpoint = request['endpoint']
                request_id = request['request_id']
                if request_id in self.responses.get(token, {}):
                    continue
                if self.completion_pool is not None and not self.split_exchange and endpoint == '/v1/complete':
                    self._submit_completion(token, request)
                    continue
                if getattr(self, 'split_status', False) and endpoint in ('/v1/state', '/v1/transfer-metric'):
                    self._queue_status(token, request)
                    continue
                if endpoint in ('/v1/poll', '/v1/complete', '/v1/artifacts/get', '/v1/artifacts/put'):
                    self._flush_responses()
                result = self._timed_dispatch(token, request)
                if result is not None:
                    self.responses.setdefault(token, {})[request_id] = result
                    if self.split_exchange and endpoint == '/v1/poll':
                        self._flush_responses()
        if not self.split_exchange:
            self._collect_completed_requests(self.responses)
        self._flush_responses()

    def cycle(self):
        if self.defer_q3_evidence:
            self._advance_deferred_evidence()
        for token, record in self.state['allocations'].items():
            if record['status'] != 'INTENT' or time.monotonic() < self.ensure_retry_after.get(token, 0):
                continue
            self.ensure_retry_after[token] = time.monotonic() + self.settings.get('request_timeout_seconds', 20)
            try:
                if (self.live_harvest and self.storage_launcher and record['spec']['python'] != self.storage_launcher['path']
                        and self.gateway.call(dict(action='intent_present', token=token))['present'] is False):
                    record['status'] = 'SUPERSEDED_UNSUBMITTED'
                    self.save()
                    self.log('PBS_UNSUBMITTED_INTENT_SUPERSEDED', token=token)
                    continue
                self._ensure_spec(record['spec'])
                record.pop('ensure_hold',None)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                # An unresolved allocation intent cannot prevent receipt
                # exchange/harvest for other allocations already owned.
                record['ensure_hold']=dict(reason=type(error).__name__+': '+str(error),retry_after=self.ensure_retry_after[token])
                self.save()
                self.log('PBS_ALLOCATION_ENSURE_HELD',token=token,reason=record['ensure_hold']['reason'])
        exchange_started = time.monotonic()
        response_count = sum(len(batch) for batch in self.responses.values())
        request = dict(action='exchange', responses=self.responses, controls=self.controls)
        if self.bounded_results:
            request['bounded_results'] = True
        if self.split_exchange:
            request['request_mode'] = 'control'
        if self.split_status:
            with self.status_lock:
                pending = tuple(self.status_pending)
            exclude = {}
            for token, request_id in pending:
                exclude.setdefault(token, []).append(request_id)
            request['exclude'] = exclude
        data = self.gateway.call(request)
        elapsed = time.monotonic()-exchange_started
        if elapsed >= 1:
            self.log('PBS_EXCHANGE_TIMING', seconds=elapsed, replies=response_count,
                     requests=sum(len(batch) for batch in data['requests'].values()))
        self.responses, self.controls = {}, {}
        self.allocations = data['allocations']
        if self.offline_plan:
            self._advance_offline_history()
        for allocation in self.allocations:
            token = allocation['token']
            if token not in self.state['allocations']:
                # Separate runs use different remote roots. Never adopt an
                # unrelated directory merely because it is accessible.
                raise ValueError('Unowned PBS allocation in this run namespace')
            record = self.state['allocations'][token]
            if record.get('status') != allocation['status']:
                self.log('PBS_ALLOCATION_STATE', token=token, job_id=allocation['job_id'],
                    status=allocation['status'], worker=allocation['worker'])
            now = stamp()
            worker = allocation.get('worker') or {}
            started = record.get('submitted_clock') or {}
            if (allocation['status'] == 'R' and worker.get('accepting')
                    and 'submit_to_ready_seconds' not in record and now['clock_id'] == started.get('clock_id')):
                record['submit_to_ready_seconds'] = max(0., now['monotonic'] - started['monotonic'])
            if allocation['status'] == 'R' and worker.get('accepting'):
                first = record.get('first_accepting_clock') or {}
                if first.get('clock_id') != now['clock_id']:
                    record['first_accepting_clock'] = now
            # Older gateways report the shared pilot's physical state only
            # inside scheduler metadata. Keep it visible after a run detaches.
            session = allocation.get('session_status')
            reservation = allocation.get('reservation_status') or (allocation.get('scheduler') or {}).get('job_state')
            record.update(status=allocation['status'], observed=allocation, clock=now,
                          reservation_status=reservation or allocation['status'],
                          session_status=session or (allocation['status'] if allocation.get('pilot_token') else None))
            if allocation['status'] != 'R':
                self.q3_worker_proofs.pop(token, None)
            if not self.selective_harvest and (
                    allocation['status'] in TERMINAL or allocation['status'] == 'HISTORY_MISSING_RECONCILE_SPOOL'):
                self._harvest(allocation)
        if self.storage_launcher:
            profiles = {}
            for record in self.state['allocations'].values():
                spec = record['spec']
                if (record['status'] != 'R' or spec['python'] != self.storage_launcher['path']
                        or spec['profile'].get('pbs', {}).get('compute_storage_sampling')):
                    continue
                profile = deepcopy(spec['profile'])
                sample = (Path(self.settings['remote_root'])/'allocations'/spec['token']/
                          'STORAGE_USAGE_GATEWAY.json')
                profile['storage_limit']['sample_path'] = str(sample)
                profiles[profile['host_id']] = profile
            self.gateway.monitor_storage(profiles)
        self._dispatch_requests(data['requests'])
        if self.selective_harvest:
            self._advance_harvest()
        # The manager's thread-stop flag also closes a transport during
        # controller reattachment. Only a real user Stop (or a non-retaining
        # close) may turn that into the pilot's irreversible STOP fence.
        user_stop = self.executor._user_stop_requested() or (self.stop.is_set() and not self.detached)
        if (self.detached or self.executor.state.closed and self.retain_between_batches) and not user_stop:
            self.save()
            return
        if self._independent_demand(user_stop or self.executor.state.closed):
            self.last_error = None
            return
        # A held running node already has its full resource pool. Routine
        # exchanges do not need to rebuild admission demand or repack jobs.
        if self.split_exchange and self.hold_node and len(self.configured) == 1:
            if not user_stop and not self.executor.state.closed:
                self._connect_retained_replacement()
            live = [entry for entry in self.allocations if entry['status'] == 'R']
            held = next((entry for entry in live if not entry.get('retained_replacement')), live[0] if live else None)
            if held and held.get('pilot_token'):
                token = held['token']
                record = self.state['allocations'][token]
                if not user_stop and not self.executor.state.closed:
                    self._rollover_storage_launcher(held,record)
                idle = not any(row['host_id'] == held['host_id'] for row in self.executor._owned_attempts())
                if (not user_stop and idle and not record.get('storage_maintenance_requested')
                        and record.get('worker_storage_hold') in ('STORAGE_LIMIT_REACHED', 'FILESYSTEM_SPACE_LOW')):
                    record['storage_maintenance_requested'] = True
                    self.storage_checks.pop(held['host_id'], None)
                    self.log('PBS_STORAGE_MAINTENANCE_DRAIN', token=token, host_id=held['host_id'],
                             job_id=held['job_id'], reason=record['worker_storage_hold'])
                stopping = user_stop or self.executor.state.closed
                self.controls[token] = self.session_control(held['host_id'],record,stopping=stopping,
                                                           release_when_idle=stopping and idle)
                self.last_error = None
                self.state['heartbeat'] = stamp()
                self.save()
                return
        outstanding, pending, queued, free = self._demand()
        waiting_tasks = outstanding + pending
        shortage = len(waiting_tasks)
        active = [a for a in self.allocations if a['status'] not in TERMINAL]
        stopping = user_stop or self.executor.state.closed
        keep_for_preparation = (not self.hold_node and not (outstanding or pending) and not stopping and any(
            a['status'] == 'R' and (a.get('worker') or {}).get('accepting') for a in active)
            and self._recent_preparation_completion(self.readiness_wait_seconds()))
        owned_attempts = self.executor._owned_attempts()
        hosts_with_attempts = {row['host_id'] for row in owned_attempts}
        for allocation in self.allocations:
            if allocation['status'] not in TERMINAL:
                token, host = allocation['token'], allocation['host_id']
                record = self.state['allocations'][token]
                spec = record['spec']
                if (not stopping and self.hold_node and allocation['status'] == 'R'
                        and allocation.get('pilot_token') and host not in hosts_with_attempts
                        and record.get('worker_storage_hold') in ('STORAGE_LIMIT_REACHED', 'FILESYSTEM_SPACE_LOW')
                        and not record.get('storage_maintenance_requested')):
                    # A shared pilot keeps its physical node when a session
                    # exits. The drained worker must release its spool lock
                    # so the existing storage watcher can recover capacity.
                    record['storage_maintenance_requested'] = True
                    self.storage_checks.pop(host, None)
                    self.save()
                    self.log('PBS_STORAGE_MAINTENANCE_DRAIN', token=token, host_id=host,
                             job_id=allocation['job_id'], reason=record['worker_storage_hold'])
                current_cpus = spec['resources']['ncpus']
                requested_slots = sum(host in usable for usable, _ in waiting_tasks)
                cpu_per_task = task_cpu_footprint(spec['profile'])
                slot_memory = spec['profile'].get('pbs', {}).get('sized_task_memory_bytes')
                waiting = [request for hosts, request in waiting_tasks if host in hosts]
                estimates = [task_memory_estimate(request, self.settings, host) for request in waiting]
                memory_mismatch = slot_memory is not None and estimates and all(
                    estimate is None or estimate > slot_memory for estimate in estimates)
                if spec['profile'].get('pbs', {}).get('per_task_resources'):
                    base = self.configured[host]['profile']
                    requested_cpus, _, _ = packed_resources(waiting, base, self.settings, host)
                    ram = (int(spec['resources']['mem'][:-2]) - 1) * GIB
                    fits_current = False
                    for request in waiting:
                        try:
                            cpu, memory = resource_need(request, base, self.settings, host)
                        except (KeyError, ValueError):
                            continue
                        if cpu <= current_cpus and memory <= ram:
                            fits_current = True
                            break
                    memory_mismatch = requested_cpus > 0 and not fits_current
                else:
                    requested_cpus = min(self.settings['resources']['ncpus'], requested_slots * cpu_per_task)
                cpu_limit = self.settings['resources']['ncpus']
                growth_min = min(cpu_limit, current_cpus + max(2, (current_cpus + 7) // 8))
                # Releasing the old idle allocation is required before another
                # allocation can take ownership under this PBS host identity.
                # Give a newly ready worker a chance to round-trip its input
                # checks and claim work before declaring the allocation idle.
                first = record.get('first_usable_offer_clock') or {}
                latest = record.get('last_usable_offer_clock') or {}
                now = record['clock']
                offered_long_enough = (first.get('clock_id') == now['clock_id']
                    and latest.get('clock_id') == now['clock_id']
                    and now['monotonic'] - latest['monotonic'] <= self.readiness_wait_seconds() and
                    now['monotonic'] - first['monotonic'] >= self.readiness_wait_seconds())
                if (not stopping and not self.hold_node and self.settings.get('allocation_sizing') and allocation['status'] == 'R'
                        and (allocation.get('worker') or {}).get('accepting') and not record.get('resize_requested')
                        and offered_long_enough
                        and (memory_mismatch or
                             (requested_cpus > current_cpus and requested_cpus >= growth_min))
                        and host not in hosts_with_attempts):
                    record['resize_requested'] = True
                    self.log('PBS_IDLE_ALLOCATION_RESIZE_DRAIN', token=token, host_id=host,
                             previous_cpus=current_cpus, compatible_ready_jobs=requested_slots)
                resize_reason = None
                if (not stopping and not self.hold_node and self.settings.get('allocation_sizing') and allocation['status'] == 'Q'
                        and not record.get('resize_requested') and host not in hosts_with_attempts):
                    if 2 * requested_cpus <= current_cpus:
                        resize_reason = 'QUEUED_ALLOCATION_EXCEEDS_COMPATIBLE_DEMAND'
                    elif (requested_cpus > current_cpus and requested_cpus >= growth_min) or memory_mismatch:
                        # Unchecked preparation may need an initial bootstrap.
                        # Grow an existing reservation only for checked work.
                        checked = [request for hosts, request in outstanding if host in hosts]
                        if spec['profile'].get('pbs', {}).get('per_task_resources'):
                            checked_cpus, checked_gib, _ = packed_resources(checked, base, self.settings, host)
                            needs_memory = memory_mismatch and checked_cpus > 0 and checked_gib * GIB > ram + GIB
                        else:
                            checked_cpus = min(cpu_limit, len(checked) * cpu_per_task)
                            needs_memory = False
                        grows = checked_cpus > current_cpus and checked_cpus >= growth_min
                        if grows or needs_memory:
                            resize_reason = 'QUEUED_ALLOCATION_UNDERSIZES_COMPATIBLE_DEMAND'
                since = record.get('queued_resize_since') or {}
                if resize_reason is None:
                    record.pop('queued_resize_since', None)
                elif since.get('clock_id') != now['clock_id'] or since.get('reason') != resize_reason:
                    record['queued_resize_since'] = dict(now, reason=resize_reason)
                elif now['monotonic'] - since['monotonic'] >= self.readiness_wait_seconds():
                    # Fence late polls before cancelling an unused submission.
                    record.update(resize_requested=True, resize_reason=resize_reason)
                    self.save()
                    self.log('PBS_QUEUED_ALLOCATION_RESIZE', token=token, host_id=host,
                             previous_cpus=current_cpus, requested_cpus=requested_cpus, reason=resize_reason)
                self.controls[token] = self.session_control(host,record,stopping=stopping,
                    release_when_idle=(stopping or not self.hold_node) and not waiting
                    and not keep_for_preparation and host not in hosts_with_attempts)
                if self.offline_plan:
                    self._offline_contact(token)
                queued_stop = stopping and allocation['status'] in ('Q', 'H', 'W')
                queued_resize = record.get('resize_requested') and allocation['status'] == 'Q'
                if (queued_stop or queued_resize) and allocation.get('job_id') and host not in hosts_with_attempts:
                    result = self.gateway.call(dict(action='delete', token=allocation['token']))
                    self.log('PBS_UNUSED_ALLOCATION_CANCELLED', token=allocation['token'],
                        job_id=allocation['job_id'], result=result,
                        reason=('STOP_WITHOUT_OWNED_SCIENTIFIC_ATTEMPTS' if queued_stop
                                else record.get('resize_reason', 'QUEUED_ALLOCATION_EXCEEDS_COMPATIBLE_DEMAND')))
        if not stopping and (shortage or hosts_with_attempts):
            owned_hosts = {a['host_id'] for a in active}
            # A lost remote response may leave an intent absent from this
            # snapshot. It still occupies its prospective allocation.
            owned_hosts.update(r['spec']['profile']['host_id'] for r in self.state['allocations'].values()
                               if r['status'] not in TERMINAL)
            covered = sum(a['spec'].get('reason', {}).get('requested_task_capacity',
                          max(1, a['spec']['resources']['ncpus']//task_cpu_footprint(a['spec']['profile'])))
                          for a in self.state['allocations'].values() if a['status'] not in TERMINAL)
            for host, worker in self.configured.items():
                if host in owned_hosts: continue
                if any(record['spec']['profile']['host_id'] == host and record['status'] == 'SUBMIT_REJECTED'
                       for record in self.state['allocations'].values()): continue
                replay = self._assigned_pilot_replay(host, owned_attempts) if host in hosts_with_attempts else None
                if host in hosts_with_attempts and replay is None: continue
                if shortage <= covered and replay is None: continue
                # Reserve room for independent preparation tasks whose worker
                # preflight can only run after PBS starts. Assignment still
                # requires that task's successful preflight.
                requests = [request for usable, request in waiting_tasks if host in usable]
                demand = dict(ready_tasks=queued, existing_host_free_slots=free,
                              prospective_pbs_task_slots=covered, shortage=shortage)
                requests = requests[:max(0, shortage-covered)]
                if replay is not None:
                    pilot, original_requests = replay
                    demand['replay_pilot'] = pilot
                    requests = original_requests + requests
                if not requests: continue
                if not self._storage_accepting(worker): continue
                try:
                    spec = self._submit(host, worker, demand, requests)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    self.log('PBS_ALLOCATION_SUBMIT_HELD',host=host,reason=type(error).__name__+': '+str(error))
                    continue
                covered += spec['reason'].get('requested_task_capacity',
                    max(1, spec['resources']['ncpus']//task_cpu_footprint(spec['profile'])))
        self.last_error = None
        self.state['heartbeat'] = stamp()
        self.save()

    def start(self):
        if self.split_status:
            self.status_thread = threading.Thread(target=self._run_status, name='rpe-pbs-status', daemon=True)
            self.status_thread.start()
        if self.split_exchange:
            self.completion_thread = threading.Thread(target=self._run_completions,
                                                      name='rpe-pbs-completions', daemon=True)
            self.completion_thread.start()

        def run():
            try:
                while not self.stop.is_set():
                    try: self.cycle()
                    except Exception as error:
                        self.last_error = type(error).__name__+': '+str(error)
                        self.log('PBS_TRANSPORT_PENDING', error=self.last_error, automatic_scientific_retry=False)
                    self.stop.wait(self._poll_delay())
            finally:
                self.gateway.close()
        self.thread = threading.Thread(target=run, name='rpe-pbs-manager', daemon=True)
        self.thread.start()

    def close(self, *, retain_allocation=False):
        if self.lock.closed:
            if not self.detached:
                return
            self.lock = (self.directory/'MANAGER.lock').open('a')
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = read(self.path)
        self.detached = retain_allocation
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=self.settings.get('request_timeout_seconds',20)+4)
        if self.thread and self.thread.is_alive():
            self.gateway.close(); self.thread.join(timeout=4)
        if self.status_thread is not None:
            self.status_queue.put(None)
            self.status_thread.join()
            self.status_thread = None
        if self.completion_thread is not None:
            self.completion_thread.join()
        if self.bulk_gateway is not None:
            self.bulk_gateway.close()
        if self.evidence_pool is not None:
            self.evidence_gateway.close()
            self.evidence_pool.shutdown(wait=True, cancel_futures=True)
            self.evidence_gateway.close()
            self.evidence_pool = self.evidence_pending = None
        if self.harvest_pool is not None:
            # A read in flight may finish, but only a live manager applies its
            # result to the central ledger. Resume will recover the same owner.
            self.harvest_gateway.close()
            self.harvest_pool.shutdown(wait=True, cancel_futures=True)
            self.harvest_gateway.close()
            self.harvest_pool = self.harvest_pending = None
        if self.artifact_pool is not None:
            self.artifact_pool.shutdown(wait=True, cancel_futures=True)
            self.artifact_pool = None
            self.artifact_transfers.clear()
            self.prefetched_inputs.clear()
        try:
            if self.completion_pool is not None and self.completion_thread is None:
                self.completion_pool.shutdown(wait=True)
                self._collect_completed_requests(self.responses)
            if retain_allocation:
                if self.responses:
                    self.gateway.call(dict(action='exchange', responses=self.responses, controls={}))
                    self.responses = {}
                self.log('PBS_ALLOCATION_RETAINED_BETWEEN_BATCHES')
                return
            controls = {token:dict(stop=True, release_when_idle=True) for token,record in self.state['allocations'].items()
                        if record['status'] not in TERMINAL}
            if controls or self.responses:
                snapshot = self.gateway.call(dict(action='exchange', responses=self.responses, controls=controls))
                self.allocations = snapshot['allocations']
            for allocation in self.allocations:
                if allocation['status'] in ('Q','H','W') and allocation.get('job_id') and not any(
                        row['host_id'] == allocation['host_id'] for row in self.executor._owned_attempts()):
                    result = self.gateway.call(dict(action='delete', token=allocation['token']))
                    self.log('PBS_UNUSED_ALLOCATION_CANCELLED', token=allocation['token'],
                        job_id=allocation['job_id'], result=result, reason='CONTROLLER_CLOSE_NO_OWNED_ATTEMPTS')
        except Exception as error:
            self.log('PBS_DRAIN_CONTROL_PENDING', error=str(error), recovery='Same run resume reconciles original allocations')
        finally:
            self.gateway.close(); self.lock.close()

    def metrics(self):
        return dict(allocations=[{key:value for key,value in a.items() if key != 'scheduler'} for a in self.allocations],
            error=self.last_error, science_budget_per_task=True)
