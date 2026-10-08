"""Pilot-only process guard; no solver imports or registered inputs."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

PRIVATE = Path('/tmp/ra-d0-v4-exact-backend-20261009')
STATE = json.loads((PRIVATE / 'start.json').read_text())
RSS_CAP = STATE['caps']['RSS_bytes']
OUTPUT_CAP = STATE['caps']['output_bytes']

def tree_rss(pid):
    pending, seen, rss = [pid], set(), 0
    while pending:
        p = pending.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            status = Path(f'/proc/{p}/status').read_text()
            rss += int(next(s.split()[1] for s in status.splitlines() if s.startswith('VmRSS:'))) * 1024
            pending += [int(x) for x in Path(f'/proc/{p}/task/{p}/children').read_text().split()]
        except (OSError, StopIteration):
            pass
    return rss

def run(command, cwd, out, wall_cap, output_root):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    marker = out.with_suffix('.started.json')
    with marker.open('x') as f:
        json.dump({'command': command, 'cwd': str(cwd), 'started_epoch': time.time(), 'retry': 0}, f)
    remaining = STATE['start_epoch'] + 3600 - time.time()
    if remaining <= 0:
        raise RuntimeError('PILOT_WALL_CAP')
    cap = min(wall_cap, remaining)
    def limits():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (RSS_CAP, RSS_CAP))
        resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_CAP, OUTPUT_CAP))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    peak, failure = 0, None
    with out.open('xb') as stdout, out.with_suffix('.stderr.txt').open('xb') as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                                   env=env, preexec_fn=limits)
        while process.poll() is None:
            peak = max(peak, tree_rss(process.pid) + tree_rss(os.getpid()))
            size = sum(p.stat().st_size for p in Path(output_root).rglob('*') if p.is_file())
            if time.monotonic() - start >= cap:
                failure = 'WALL_CAP'
            elif peak >= RSS_CAP:
                failure = 'RSS_CAP'
            elif size >= OUTPUT_CAP:
                failure = 'OUTPUT_CAP'
            if failure:
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(.025)
        process.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {'command': command, 'cwd': str(cwd), 'returncode': process.returncode,
              'failure': failure, 'wall_seconds': time.monotonic()-start,
              'CPU_seconds': after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
              'sampled_tree_plus_supervisor_peak_RSS_bytes': peak,
              'child_high_water_RSS_KiB': after.ru_maxrss,
              'stdout_bytes': out.stat().st_size, 'stderr_bytes': out.with_suffix('.stderr.txt').stat().st_size,
              'wall_cap_seconds': cap, 'retries': 0}
    out.with_suffix('.resource.json').write_text(json.dumps(record, indent=2)+'\n')
    return record

if __name__ == '__main__':
    # Non-solver build only; LP driver uses run() with its own call ledger.
    args = sys.argv[1:]
    result = run(args[3:], args[0], args[1], float(args[2]), PRIVATE/'build_output')
    print(json.dumps(result))
    sys.exit(0 if result['returncode'] == 0 and result['failure'] is None else 1)
