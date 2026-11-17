"""Read-only PBS allocation/runtime probe; no scientific calculations."""
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time


def read(path):
    try:
        return Path(path).read_text()
    except (IOError, OSError) as error:
        return {"unavailable": str(error)}


def main():
    started = time.monotonic()
    root = Path('/home/AbeHiromu/projects/rpe-pbs-20261118/environment')
    out = dict(schema='PBS_ENVIRONMENT_PROBE_V1', scope='environment_only_no_science',
               hostname=socket.gethostname(), python=sys.version, executable=sys.executable,
               platform=platform.platform(), cwd=os.getcwd(), home=str(Path.home()),
               pbs={key: value for key, value in os.environ.items()
                    if key.startswith('PBS') and key != 'PBS_JOBCOOKIE'},
               tmpdir=os.environ.get('TMPDIR'),
               cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
               shared_input=read(root / 'shared-input.txt'),
               nodefile=read(os.environ.get('PBS_NODEFILE', '/nonexistent')),
               proc_cgroup=read('/proc/self/cgroup'),
               proc_status=read('/proc/self/status'),
               tools={name: shutil.which(name) for name in ['python3', 'python3.11', 'python3.9',
                       'gcc', 'g++', 'cmake', 'git', 'rsync', 'ssh', 'curl', 'wget', 'qstat']},
               packages={}, connections={})
    for name in ['numpy', 'scipy', 'mpmath', 'qiskit', 'numba', 'pyscf']:
        try:
            module = importlib.import_module(name)
            out['packages'][name] = dict(version=getattr(module, '__version__', 'unknown'),
                                        path=getattr(module, '__file__', None))
        except Exception as error:
            out['packages'][name] = dict(unavailable=type(error).__name__ + ': ' + str(error))
    for address, port in [('100.103.127.48', 22), ('100.103.127.48', 8763)]:
        key = '{}:{}'.format(address, port)
        try:
            connection = socket.create_connection((address, port), timeout=3)
            connection.close()
            out['connections'][key] = 'TCP_CONNECTED'
        except Exception as error:
            out['connections'][key] = type(error).__name__ + ': ' + str(error)
    for path in ['/tmp', os.environ.get('TMPDIR', '/nonexistent'), str(root)]:
        try:
            usage = shutil.disk_usage(path)
            out.setdefault('storage', {})[path] = dict(total=usage.total, free=usage.free,
                                                      readable=os.access(path, os.R_OK),
                                                      writable=os.access(path, os.W_OK))
        except OSError as error:
            out.setdefault('storage', {})[path] = dict(unavailable=str(error))
    out['elapsed_seconds'] = time.monotonic() - started
    target = root / 'PROBE.json'
    temporary = root / 'PROBE.json.tmp'
    temporary.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    temporary.replace(target)
    print(json.dumps(dict(result=str(target), hostname=out['hostname'], elapsed_seconds=out['elapsed_seconds'])))


if __name__ == '__main__':
    main()
