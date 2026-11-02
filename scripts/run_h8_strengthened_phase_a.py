"""Plan by default; H8 signal execution needs a separately approved sealed grant."""
import argparse
import importlib
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys

import h8_strengthened_contract as contract
import h8_strengthened_runtime as runtime


def execute(config_path,grant_path,fresh_path):
    config,grant,fresh=map(runtime.read,(config_path,grant_path,fresh_path))
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=contract.ROOT,text=True).strip()
    runtime.verify_grant(config,grant,fresh,config_sha256=runtime.sha(config_path),source=source)
    runtime.verify_files(config,source)
    # Full runtime hashing can outlive the 30s freshness interval. Require a
    # new fresh observation from the supervising parent, bound to this grant.
    ready=Path(config['control'])/'driver_bindings_verified.json'
    runtime.Writer(config['control']).write(ready.name,dict(source=source,config_sha256=runtime.sha(config_path)))
    deadline=__import__('time').monotonic()+30
    while True:
        refreshed=runtime.read(fresh_path)
        if refreshed.get('driver_bindings_verified') is True:
            runtime.verify_grant(config,grant,refreshed,config_sha256=runtime.sha(config_path),source=source)
            break
        if __import__('time').monotonic()>deadline:raise ValueError('FRESH_GATE_REFRESH_TIMEOUT')
        __import__('time').sleep(.1)
    for key in runtime.THREAD_KEYS:
        if os.environ.get(key)!='1':raise ValueError('THREAD_ONE_ENV:'+key)
    if Path(sys.executable).resolve()!=Path(config['python']).resolve():raise ValueError('PINNED_PYTHON')
    output=runtime.private_path(config['output']);lock=runtime.private_path(config['lock'])
    if output.exists():raise ValueError('EXCLUSIVE_NEW_OUTPUT')
    with lock.open('x') as f:f.write(source+'\n')
    output.mkdir(mode=0o700)
    scratch=output/'scratch';scratch.mkdir();mpl=scratch/'mpl';mpl.mkdir()
    font=config['font_seed']
    if runtime.sha(font['path'])!=font['sha256']:raise ValueError('FONT_SEED_CHANGED')
    (mpl/font['name']).write_bytes(Path(font['path']).read_bytes())
    for key,name in (('TMPDIR','tmp'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','cache')):
        path=scratch/name;path.mkdir();os.environ[key]=str(path)
    os.environ['MPLCONFIGDIR']=str(mpl)
    __import__('tempfile').tempdir=os.environ['TMPDIR']
    os.sched_setaffinity(0,{grant['driver_cpu']})
    r=grant['resources']
    resource.setrlimit(resource.RLIMIT_AS,(r['driver_AS_bytes'],r['driver_AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(r['CPU_seconds'],r['CPU_seconds']))
    def stop(*args):raise RuntimeError('DRIVER_STOP_SIGNAL_OR_DEADLINE')
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(s,stop)
    signal.alarm(r['wall_seconds'])
    permit=runtime.Permit(config,grant,output);writer=runtime.Writer(output,permit.guard)
    try:
        sys.path.insert(0,str(Path(config['provider_root'])/'src'))
        adapter=importlib.import_module('h8_strengthened_signal')
        adapter.evaluate(config,permit,writer)
        runtime.verify_files(config,source)
        runtime.Writer(output).write('terminal.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=source,
            scope=contract.SCOPE,call_counts=permit.used,irrevocable_reservations=permit.reserved,
            settlements=permit.settled,next_stage_authorized=False))
    except BaseException as exc:
        runtime.Writer(output).write('terminal.json',dict(status='STOP',SOURCE=source,error=str(exc),
            call_counts=permit.used,irrevocable_reservations=permit.reserved,settlements=permit.settled,
            retry=False,next_stage_authorized=False))
        raise
    finally:signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-approved',action='store_true')
    for name in ('config','grant','fresh-gate'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.execute_approved:
        if not all((args.config,args.grant,args.fresh_gate)):parser.error('sealed execution inputs required')
        execute(args.config,args.grant,args.fresh_gate)
    else:
        if any((args.config,args.grant,args.fresh_gate)):parser.error('real inputs require --execute-approved')
        import json
        print(json.dumps(contract.plan(),indent=2))


if __name__=='__main__':main()
