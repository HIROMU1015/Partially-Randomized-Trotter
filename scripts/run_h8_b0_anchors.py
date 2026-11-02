"""Plan-only default. Approved acquisition ends in mandatory STOP without a fit."""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import h8_b0_anchor_contract as c
import h8_b0_anchor_runtime as r


def execute(config_path,grant_path,fresh_path):
    config,g,f=map(r.read,(config_path,grant_path,fresh_path))
    r.verify_grant(config,g,f,config_sha=r.sha(config_path));r.verify_observer(f);r.verify_files(config)
    ready=Path(config['control'])/'driver_bindings_verified.json'
    r.Writer(config['control']).write(ready.name,dict(source=config['SOURCE'],config_sha256=r.sha(config_path)))
    deadline=time.monotonic()+30
    while True:
        fresh=r.read(fresh_path)
        if fresh.get('driver_bindings_verified') is True:
            r.verify_grant(config,g,fresh,config_sha=r.sha(config_path));r.verify_observer(fresh);break
        if time.monotonic()>deadline:raise ValueError('FRESH_REFRESH_TIMEOUT')
        time.sleep(.1)
    if Path(sys.executable).resolve()!=Path(config['python']).resolve():raise ValueError('PINNED_PYTHON')
    for key in r.THREAD_KEYS:
        if os.environ.get(key)!='1':raise ValueError('THREAD_ONE_REQUIRED')
    output=r.private_path(config['output']);lock=r.private_path(config['lock'])
    if output.exists():raise ValueError('EXCLUSIVE_OUTPUT_REQUIRED')
    with lock.open('x') as f:f.write(config['SOURCE']+'\n');f.flush();os.fsync(f.fileno())
    output.mkdir(mode=0o700);scratch=output/'scratch';scratch.mkdir()
    for key,name in [('TMPDIR','tmp'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','mpl')]:
        p=scratch/name;p.mkdir();os.environ[key]=str(p)
    font=config['font_seed'];raw=Path(font['path']).read_bytes()
    if __import__('hashlib').sha256(raw).hexdigest()!=font['sha256']:raise ValueError('FONT_SEED_CHANGED')
    (scratch/'mpl'/font['name']).write_bytes(raw)
    __import__('tempfile').tempdir=os.environ['TMPDIR']
    os.sched_setaffinity(0,{g['driver_cpu']});limits=g['resources']
    resource.setrlimit(resource.RLIMIT_AS,(limits['driver_AS_bytes'],limits['driver_AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(limits['CPU_seconds'],limits['CPU_seconds']))
    r.Writer(config['control']).write('science_affinity_bound.json',dict(driver_cpu=g['driver_cpu']))
    def stop(*args):raise RuntimeError('DRIVER_STOP_SIGNAL_OR_DEADLINE')
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(s,stop)
    signal.alarm(limits['wall_seconds']);permit=r.Permit(config,g,output)
    try:
        sys.path.insert(0,str(Path(config['provider_root'])/'src'))
        adapter=importlib.import_module('h8_b0_anchor_adapter')
        rows=adapter.evaluate(config,permit,r.Writer(output,permit.guard))
        r.verify_files(config);r.protected(config)
        terminal=dict(scope=c.SCOPE,SOURCE=config['SOURCE'],status='COMPLETE_MANDATORY_STOP',axes=rows,
            actual_calls=permit.actual(),irrevocable_reservations=permit.reserved,settlements=permit.settled,
            next_stage_authorized=False,fit_performed=False,retry=False)
        c.original.validate_terminal(c.original.proposal(c.plan()['binding']),terminal)
        r.Writer(output).write('terminal.json',terminal)
    except BaseException as exc:
        r.Writer(output).write('terminal.json',dict(status='STOP',scope=c.SCOPE,SOURCE=config['SOURCE'],
            error=str(exc),completed_calls_lower_bound=permit.actual(),
            compile_entry_markers=len(list(output.glob('wave_*/*/compile_entered.json'))),
            unknown_reserved_compile_calls=len(permit.reserved)-len(permit.settled),
            irrevocable_reservations=permit.reserved,
            settlements=permit.settled,next_stage_authorized=False,retry=False))
        raise
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-approved',action='store_true')
    for name in ('config','grant','fresh-gate'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if not a.execute_approved:
        if any((a.config,a.grant,a.fresh_gate)):p.error('real inputs require approved mode')
        print(json.dumps(c.plan(),indent=2));return
    if not all((a.config,a.grant,a.fresh_gate)):p.error('all sealed execution inputs required')
    execute(a.config,a.grant,a.fresh_gate)


if __name__=='__main__':main()
