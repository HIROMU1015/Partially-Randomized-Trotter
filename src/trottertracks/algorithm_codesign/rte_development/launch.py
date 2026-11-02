"""Source-bound authorization gate and CPU/RSS/log guards; no science import."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time

from .evidence import digest, atomic_json


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def validate_binding(auth, contract_sha, requested_source, head, parents, changed, dirty, allowed):
    s = auth.get('source_commit')
    if (not isinstance(s,str) or not re.fullmatch('[0-9a-f]{40}',s)
        or auth.get('status') != 'APPROVED_FOR_ONE_RTE_DEVELOPMENT_D1_RUN'
        or auth.get('science_execution_authorized') is not True
        or type(auth.get('runs')) is not int or auth['runs'] != 1
        or type(auth.get('retries')) is not int or auth['retries'] != 0
        or auth.get('mandatory_STOP') is not True
        or not isinstance(auth.get('explicit_execution_instruction'),str)
        or len(auth['explicit_execution_instruction'].strip()) < 10):
        raise PermissionError('separate source-bound D1 one-shot authorization pending')
    if (requested_source != s or auth.get('contract_sha256') != contract_sha
        or head == s or parents != [s] or dirty or not changed
        or not set(changed) <= set(allowed)):
        raise PermissionError('source/contract/direct-child/clean identity mismatch')


def verify_launch(root, contract_path, requested_source):
    root = Path(root); c = json.loads(Path(contract_path).read_text())
    auth = json.loads((root/c['authorization_path']).read_text())
    # Reject pending authority before runtime, registered inputs or constructors.
    if auth.get('science_execution_authorized') is not True:
        raise PermissionError('D1 preparation only; separate execution instruction required')
    head = git(root,'rev-parse','HEAD'); s = auth.get('source_commit')
    if not isinstance(s,str) or not re.fullmatch('[0-9a-f]{40}',s):
        raise PermissionError('source identity missing')
    changed = git(root,'-c','core.quotePath=false','diff','--name-only',s,head).splitlines()
    validate_binding(auth,digest(contract_path),requested_source,head,
                     git(root,'show','-s','--format=%P',head).split(), changed,
                     bool(git(root,'status','--porcelain','--untracked-files=all')),
                     [c['authorization_path'],c['optional_receipt_path']])
    if c['authorization_path'] not in changed: raise PermissionError('A must bind authorization JSON')
    if not re.fullmatch(r'git@github\.com:HIROMU1015/Partially-Randomized-Trotter\.git',
                        git(root,'remote','get-url','origin')):
        raise PermissionError('unexpected repository remote')
    remote = git(root,'ls-remote','--heads','origin','refs/heads/'+git(root,'branch','--show-current'))
    if not remote or remote.split()[0] != head: raise PermissionError('remote A mismatch')
    manifest = json.loads((root/c['source_manifest']).read_text())
    if manifest.get('focused_tests_passed') is not True: raise PermissionError('source validation incomplete')
    for path,h in manifest['critical_sha256'].items():
        if path.lower().endswith('.npz'): raise PermissionError('NPZ access forbidden')
        if digest(root/path) != h: raise PermissionError('critical identity changed: '+path)
    check_protected(root,c)
    output = root/c['result_directory']
    if output.exists(): raise FileExistsError('fresh namespace already exists; no retry')
    from .accounting import validate_contract
    validate_contract(c)
    return c,auth,head


def check_protected(root,c):
    ledger = json.loads((Path(root)/c['protected_ledger']).read_text())
    for path,h in ledger['sha256'].items():
        if path.lower().endswith('.npz'): raise PermissionError('invalid protected scope')
        if digest(Path(root)/path) != h: raise PermissionError('protected history changed: '+path)
    return True


def consume_marker(directory, receipt):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=False)
    path = directory/'one_shot_consumed.json'
    identity = atomic_json(path,receipt)
    return path,identity


def verify_job(job, inventory, marker_sha, head, source, auth_sha, contract_sha):
    if (job.get('inventory_sha256') != inventory['sha256']
        or job.get('marker_sha256') != marker_sha or job.get('execution_HEAD') != head
        or job.get('source_commit') != source or job.get('authorization_sha256') != auth_sha
        or job.get('contract_sha256') != contract_sha):
        raise PermissionError('worker provenance binding mismatch')
    found = [r for r in inventory['keys'] if r['key'] == job.get('key')]
    if len(found) != 1 or found[0]['ratio'] != job.get('ratio') or found[0]['acquisition'] != 'NEW_ONCE':
        raise PermissionError('worker key not in frozen acquisition inventory')
    return found[0]


class Guard:
    def __init__(self,caps,directory=None,worker=False):
        self.caps,self.directory,self.worker=caps,Path(directory) if directory else None,worker
        self.start=time.monotonic(); self.cpu_start=time.process_time()
        self.children=set(); self.peak_rss=0; self.reaped_cpu=0.

    @staticmethod
    def process_stats(pid):
        # /proc stat fields after command name; aggregate live children explicitly.
        text=Path(f'/proc/{pid}/stat').read_text(); fields=text[text.rfind(')')+2:].split()
        clock=os.sysconf('SC_CLK_TCK')
        cpu=(int(fields[11])+int(fields[12]))/clock
        rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
        return cpu,rss

    def check(self):
        own_cpu,own_rss=self.process_stats(os.getpid())
        aggregate_rss=own_rss; aggregate_cpu=own_cpu+self.reaped_cpu
        for pid in tuple(self.children):
            try:cpu,rss=self.process_stats(pid)
            except FileNotFoundError:continue
            aggregate_cpu+=cpu; aggregate_rss+=rss
        self.peak_rss=max(self.peak_rss,aggregate_rss)
        own_limit=self.caps['worker_RSS_MiB'] if self.worker else self.caps['parent_RSS_MiB']
        cpu_limit=self.caps['per_key_CPU_seconds'] if self.worker else self.caps['total_CPU_seconds']
        if own_rss > own_limit*1048576 or aggregate_rss > self.caps['aggregate_RSS_MiB']*1048576:
            raise RuntimeError('RSS cap; no retry')
        if aggregate_cpu > cpu_limit: raise RuntimeError('CPU cap; no retry')
        if self.directory:
            total=0
            for p in self.directory.rglob('*'):
                if p.is_file():
                    n=p.stat().st_size; total+=n
                    if p.suffix=='.log' and n>self.caps['per_key_log_bytes']:raise RuntimeError('log cap')
            if total>self.caps['output_bytes']-32768:raise RuntimeError('output cap')

    def __enter__(self):
        if signal.getitimer(signal.ITIMER_REAL)[0] or signal.getsignal(signal.SIGALRM) not in (signal.SIG_DFL,):
            raise RuntimeError('profiler/alarm conflict')
        self.old_handler=signal.getsignal(signal.SIGALRM)
        self.old_as=resource.getrlimit(resource.RLIMIT_AS)
        limit=self.caps['virtual_address_MiB']*1048576
        if self.old_as[1] != resource.RLIM_INFINITY:limit=min(limit,self.old_as[1])
        try:
            resource.setrlimit(resource.RLIMIT_AS,(limit,self.old_as[1]))
            signal.signal(signal.SIGALRM,lambda *_:self.check())
            signal.setitimer(signal.ITIMER_REAL,.1,.1)
            self.check()
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,*_):
        try: signal.setitimer(signal.ITIMER_REAL,0)
        finally:
            try: signal.signal(signal.SIGALRM,self.old_handler)
            finally: resource.setrlimit(resource.RLIMIT_AS,self.old_as)

    def snapshot(self):
        return dict(wall_seconds=time.monotonic()-self.start,
                    parent_CPU_seconds=time.process_time()-self.cpu_start,
                    completed_worker_CPU_seconds=self.reaped_cpu,
                    peak_observed_aggregate_RSS_bytes=self.peak_rss,
                    total_wall_cap=None, parallel_workers=self.caps['workers'])
