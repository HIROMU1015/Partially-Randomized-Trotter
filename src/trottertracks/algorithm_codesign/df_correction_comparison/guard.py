"""Linux external supervision; detected cap breaches kill whole process groups.

RSS is sampled every 50ms: this is containment on detection, not zero overshoot.
RLIMIT_CPU adds per-child kernel enforcement; wall and aggregate CPU/RSS are
enforced by the lightweight parent, including completed-child CPU accounting.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


class CapFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    wall: float
    cpu: float
    rss_mib: float
    worker_rss_mib: float
    per_key_wall: float = 0
    per_key_cpu: float = 0


def proc_snapshot():
    """No shell/GPU query. Return live processes and exact OS counter units."""
    ticks = os.sysconf('SC_CLK_TCK')
    page = os.sysconf('SC_PAGE_SIZE')
    out = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            s = (entry/'stat').read_text().rsplit(') ', 1)[1].split()
            out[int(entry.name)] = {'ppid': int(s[1]), 'pgrp': int(s[2]), 'state': s[0],
                'cpu': (int(s[11])+int(s[12]))/ticks,
                'rss': int(s[21])*page/(1024**2), 'start_ticks': int(s[19])}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return out


def descendants(snapshot, root):
    selected = {root}
    while True:
        more = {pid for pid, row in snapshot.items() if row['ppid'] in selected}
        if more <= selected: break
        selected |= more
    return selected


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


class Supervisor:
    def __init__(self, total, output_dir, output_cap=64*1024**2, log_cap=4*1024**2):
        self.total, self.output_dir = total, Path(output_dir)
        self.output_cap, self.log_cap = output_cap, log_cap
        self.start = time.monotonic()
        self.cpu_start = self.cpu()
        self.peak = 0
        self.receipts = []
        self.completed = []
        self.guard_poll_seconds = .05

    @staticmethod
    def cpu():
        me, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        return me.ru_utime+me.ru_stime+children.ru_utime+children.ru_stime

    def check_total(self):
        snap=proc_snapshot();pids=descendants(snap,os.getpid())
        rss=sum(snap[p]['rss'] for p in pids if p in snap and snap[p]['state']!='Z')
        live=sum(snap[p]['cpu'] for p in pids if p!=os.getpid() and p in snap and snap[p]['state']!='Z')
        self.peak=max(self.peak,rss)
        if (time.monotonic()-self.start>self.total.wall or self.cpu()+live-self.cpu_start>self.total.cpu
            or rss>self.total.rss_mib):raise CapFailure('Total cap at submission/final metadata boundary')
        sizes=[(p,p.stat().st_size) for p in self.output_dir.rglob('*') if p.is_file()]
        if sum(s for _,s in sizes)>self.output_cap or sum(s for p,s in sizes if p.suffix=='.log')>self.log_cap:
            raise CapFailure('Output/log cap at metadata boundary')

    def phase(self, jobs, limits, workers, *, per_key=False):
        """Jobs are fixed commands/result paths; submission order never adapts."""
        phase_start, cpu_start = time.monotonic(), self.cpu()
        queue = list(jobs)
        live = []
        failure = None
        phase_completed = []
        try:
            if limits.wall<=0 or limits.cpu<=0:
                raise CapFailure('Phase budget already exhausted; no launch')
            while queue or live:
                self.check_total()
                if time.monotonic()-phase_start>limits.wall or self.cpu()-cpu_start>limits.cpu:
                    raise CapFailure('Phase budget exhausted before further submissions')
                while queue and len(live) < workers:
                    job = queue.pop(0)
                    log = self.output_dir/(job['label']+'.log')
                    stream = log.open('xb')
                    env = dict(os.environ, **{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS')})
                    env.update(QISKIT_PARALLEL='FALSE', PYTHONDONTWRITEBYTECODE='1')
                    cpu_limit = limits.per_key_cpu if per_key else limits.cpu
                    def restrictions():
                        if cpu_limit:
                            limit = max(1, math.ceil(cpu_limit))
                            resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
                        resource.setrlimit(resource.RLIMIT_FSIZE, (self.output_cap, self.output_cap))
                    p = subprocess.Popen(job['command'], stdout=stream, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True, preexec_fn=restrictions)
                    live.append({'process': p, 'job': job, 'start': time.monotonic(), 'stream': stream})
                for row in list(live):
                    code = row['process'].poll()
                    if code is None: continue
                    row['stream'].close()
                    live.remove(row)
                    receipt = {'label':row['job']['label'], 'exit_status':code,
                               'wall_seconds':time.monotonic()-row['start'], 'retry':0}
                    self.receipts.append(receipt)
                    if code or not Path(row['job']['output']).is_file():
                        raise CapFailure('Child failed or output missing: '+row['job']['label'])
                    if per_key:
                        saved=json.loads(Path(row['job']['output']).read_text())
                        cpu_receipt=saved.get('_process_receipt',{}).get('CPU_seconds')
                        if cpu_receipt is None or cpu_receipt>limits.per_key_cpu or receipt['wall_seconds']>limits.per_key_wall:
                            raise CapFailure('Completed per-key wall/CPU receipt failed')
                    # Only metadata/prefix preservation; no interpretation of an incomplete stage.
                    phase_completed.append(row['job'])
                    self.completed.append(row['job']['label'])
                snap = proc_snapshot()
                pids = descendants(snap, os.getpid())
                rss = sum(snap[p]['rss'] for p in pids if p in snap and snap[p]['state'] != 'Z')
                self.peak = max(self.peak, rss)
                live_cpu = sum(snap[p]['cpu'] for p in pids if p != os.getpid() and p in snap and snap[p]['state'] != 'Z')
                cpu = self.cpu()+live_cpu
                wall = time.monotonic()-self.start
                if wall > self.total.wall or cpu-self.cpu_start > self.total.cpu or rss > self.total.rss_mib:
                    raise CapFailure('Total wall/aggregate CPU/RSS cap')
                if time.monotonic()-phase_start > limits.wall or cpu-cpu_start > limits.cpu or rss > limits.rss_mib:
                    raise CapFailure('Phase wall/aggregate CPU/RSS cap')
                for row in live:
                    group = descendants(snap, row['process'].pid)
                    worker_rss = sum(snap[p]['rss'] for p in group if p in snap and snap[p]['state'] != 'Z')
                    worker_cpu = sum(snap[p]['cpu'] for p in group if p in snap and snap[p]['state'] != 'Z')
                    if worker_rss > limits.worker_rss_mib:
                        raise CapFailure('Worker RSS cap')
                    if per_key and (time.monotonic()-row['start'] > limits.per_key_wall or worker_cpu > limits.per_key_cpu):
                        raise CapFailure('Per-key wall/CPU cap')
                sizes = [(p, p.stat().st_size) for p in self.output_dir.rglob('*') if p.is_file()]
                if sum(size for _,size in sizes) > self.output_cap or sum(size for p,size in sizes if p.suffix == '.log') > self.log_cap:
                    raise CapFailure('Output/log cap')
                time.sleep(self.guard_poll_seconds)
            return phase_completed
        except BaseException as exc:
            failure = str(exc)
            raise
        finally:
            for row in live:
                try: os.killpg(row['process'].pid, signal.SIGKILL)
                except ProcessLookupError: pass
                code = row['process'].wait()
                row['stream'].close()
                self.receipts.append({'label': row['job']['label'], 'exit_status':code,
                                      'guard_terminated':True, 'retry':0})
            self.last_phase = {'completed_labels': [j['label'] for j in phase_completed],
                               'pending_not_launched': [j['label'] for j in queue],
                               'failure':failure, 'retry':0}

    def receipt(self):
        return {'wall_seconds':time.monotonic()-self.start,
                'CPU_seconds_parent_and_reaped_children':self.cpu()-self.cpu_start,
                'peak_sampled_aggregate_RSS_MiB':self.peak,
                'RSS_poll_seconds':self.guard_poll_seconds,
                'instantaneous_memory_ceiling_claim':False,
                'overshoot_policy':'kill_process_groups_on_detection; record sampled peak',
                'child_receipts':self.receipts, 'completed_labels':self.completed,
                'last_phase':getattr(self,'last_phase',None), 'retries':0}
