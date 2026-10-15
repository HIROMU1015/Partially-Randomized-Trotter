"""One-shot staged runner, closed to execution without a new bound user grant.

Metadata coordinator reuses the owned, independently observing H6 supervisor.
No scientific imports, fork or affinity changes occur for --check-config.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from h6_fixed_delta_plan_v1 import KIND, PROVIDER, GIB, digest, preparation, registrations

ROOT = Path(__file__).resolve().parents[1]
THREADS = ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def private(path):
    p = Path(path)
    if not p.is_absolute() or not p.resolve().is_relative_to(Path('/home/AbeHiromu')):
        raise ValueError('HOME_ONLY_PATH')
    if any(x.is_symlink() for x in [p,*p.parents]):raise ValueError('SYMLINK_PATH')
    return p


def read(path):
    return json.loads(Path(path).read_bytes())


class Writer:
    """Exclusive files, bounded total byte use, no writable shared caches."""
    def __init__(self,directory,cap=64*2**20):
        self.directory=private(directory);self.cap=cap;self.used=0
    def write(self,name,value,**kwargs):
        if '/' in name or not name.endswith('.json'):raise ValueError('OUTPUT_NAME')
        data=json.dumps(value,allow_nan=False,separators=(',',':')).encode()+b'\n'
        if len(data)+self.used>self.cap:raise ValueError('TASK_OUTPUT_CAP')
        with (self.directory/name).open('xb') as handle:handle.write(data)
        self.used+=len(data)


def check_config(config):
    if (config.get('kind')!=KIND or config.get('plan')!=preparation() or
            config.get('approved') is not False or config.get('runtime_authorization') is not False or
            config.get('allowed_cpus')!=[] or config.get('absolute_launch_command') is not None):
        raise ValueError('PREPARATION_FLAGS_OR_PANEL')
    if config.get('provider_source')!=PROVIDER:raise ValueError('PINNED_PROVIDER')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=private(config['provider_root'])).decode().strip()!=PROVIDER:
        raise ValueError('PROVIDER_HEAD_CHANGED')
    for name, expected in config['source_hashes'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('SOURCE_CHANGED:'+name)
        blob=subprocess.check_output(['git','show',config['SOURCE']+':'+name],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError('SOURCE_BLOB_CHANGED:'+name)
    for name,expected in config['provider_hashes'].items():
        if file_hash(Path(config['provider_root'])/name)!=expected:raise ValueError('PROVIDER_CHANGED:'+name)
    for item in config['files']:
        p=private(item['path'])
        if p.stat().st_size!=item['bytes'] or file_hash(p)!=item['sha256']:
            raise ValueError('BOUND_FILE_CHANGED:'+item['logical_name'])
    # Full environment files are checked once per invocation, never once per group.
    for name,expected in config['runtime_file_hashes'].items():
        if file_hash(name)!=expected:raise ValueError('ENVIRONMENT_CHANGED')
    if config.get('carry_policy')!='NO_RESET_NO_REFUND_UNKNOWN_RETAINED':raise ValueError('CARRY_POLICY')
    private(config['output']);private(config['control'])
    if config['output']==config['control']:raise ValueError('OUTPUT_CONTROL_COLLISION')
    return True


def grant_gate(config,grant,*,check_cpus=True):
    if (grant.get('kind')!=KIND or grant.get('approved') is not True or
            grant.get('runtime_authorization') is not True or
            not isinstance(grant.get('user_approval'),str) or not grant['user_approval'].strip() or
            grant.get('config_digest')!=digest(config) or grant.get('SOURCE')!=config['SOURCE'] or
            grant.get('one_shot') is not True or grant.get('retry') is not False or
            grant.get('next_stage_authorized') is not False):
        raise ValueError('NEW_EXPLICIT_BOUND_GRANT_REQUIRED')
    r=config['resources'];cpus=[*r['worker_cpus'],r['driver_cpu']]
    if (len(r['worker_cpus'])!=8 or len(set(cpus))!=9 or
            grant.get('allowed_cpus')!=sorted(cpus) or (check_cpus and not set(cpus)<=os.sched_getaffinity(0))):
        raise ValueError('EXACT_CPU_APPROVAL_REQUIRED')
    if grant.get('stage_maximum')!=config['plan']['demand']:
        raise ValueError('NEW_STAGE_BUDGET_APPROVAL')


def provider(config):
    path=str(Path(config['provider_root'])/'src')
    if path not in sys.path:sys.path.insert(0,path)


def resources(config,mode):
    n=8 if mode=='normal' else (2 if mode=='signal' else 1)
    r=config['resources'];rss=16*GIB if mode=='heavy' else 9*GIB
    return dict(cost_workers=n,assigned_cpus=sorted(r['worker_cpus']+[r['driver_cpu']]),
        coordinator_cpu_set=[r['driver_cpu']],cost_cpu_sets=[[x] for x in r['worker_cpus'][:n]],
        cost_address_space_bytes=32*GIB,worker_rss_stop_bytes=rss,
        coordinator_address_space_bytes=2*GIB,coordinator_hard_address_space_bytes=32*GIB,
        coordinator_rss_reserve_bytes=2*GIB,aggregate_margin_bytes=2*GIB,
        aggregate_rss_bytes=(76 if mode=='normal' else 22)*GIB,
        job_ram_budget_bytes=(96 if mode=='normal' else 64)*GIB)


def observation_policy():
    return dict(interval_seconds=1.,maximum_observation_seconds=5.,journal_emit_seconds=5.,
        journal_bytes=128*2**20,fresh_host_available_bytes=int(248.25*GIB),
        fresh_effective_available_bytes=96*GIB,running_host_available_bytes=int(168.25*GIB),
        running_effective_available_bytes=16*GIB,host_full_psi_stop_percent=1.,
        cgroup_full_psi_stop_percent=0.,stop_on_new_oom=True)


def cpu_idle(cpus):
    def sample():
        return {int(v[0][3:]):list(map(int,v[1:9])) for line in Path('/proc/stat').read_text().splitlines()
            if (v:=line.split()) and v[0].startswith('cpu') and v[0][3:].isdigit()}
    a=sample();time.sleep(1);b=sample();rows=[];cores=set()
    for cpu in cpus:
        ds=[y-x for x,y in zip(a[cpu],b[cpu])];total=sum(ds)
        busy=1-(ds[3]+ds[4])/total if total else 1.
        node=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        core=((node/'physical_package_id').read_text().strip(),(node/'core_id').read_text().strip())
        if busy>.1 or core in cores:raise ValueError('CPU_BUSY_OR_PHYSICAL_SIBLING')
        cores.add(core);rows.append(dict(cpu=cpu,busy_fraction=busy,physical_core=core))
    return rows


def admit(directory,task):
    terminal=read(directory/'worker_terminal.json')
    if terminal['status']!='COMPLETE' or terminal['task']!=task:raise ValueError('TASK_TERMINAL')
    if task.get('mode')=='signal':
        x=read(directory/'panel_signals.json')
        if len(x['signals'])!=5:raise ValueError('SIGNAL_COVERAGE')
    else:
        x=read(directory/'cost_result.json')
        if x['task']!=task or [r['axis'] for r in x['rows']]!=['cosine','sine']:
            raise ValueError('PAIR_OR_TASK_MISMATCH')
        if x['calls_attempted']['compile']!=2:raise ValueError('COMPILE_COUNT')
        for row in x['rows']:
            if set(row['metrics'])!={'RZ','CX','RZ_depth','CX_depth','depth','size'} or any(
                    type(v) is not int or v<0 for v in row['metrics'].values()):raise ValueError('COST_METRICS')
        if x['rows'][0]['event_digest']!=x['rows'][1]['event_digest']:raise ValueError('PAIR_EVENTS')
    return x


def worker(config_path, directory, mode):
    config=read(config_path)
    control=private(config['control']);directory=private(directory)
    if mode not in ('signal','normal','heavy') or directory.parent!=private(config['output'])/mode:
        raise ValueError('WORKER_DIRECTORY_SCOPE')
    grant=read(control/'grant.json');grant_gate(config,grant,check_cpus=False)
    if read(control/'one_shot.json')!=dict(config_digest=digest(config),grant_digest=digest(grant),SOURCE=config['SOURCE']):
        raise ValueError('WORKER_ONE_SHOT_BINDING')
    stage=read(control/('stage_'+mode+'.json'));b=read(directory/'task_binding.json')
    if (b['manifest_digest']!=digest(stage) or
            {'index':b['index'],'task':b['task']} not in stage['registrations'] or
            b['cpus']!=resources(config,mode)['cost_cpu_sets'][b['slot']]):
        raise ValueError('WORKER_REGISTRATION_OR_CPU')
    for name,expected in config['source_hashes'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('WORKER_SOURCE_CHANGED')
    provider(config)
    from trottertracks.resource_applicability.h6_server_resources_v1 import install_parent_guard
    from trottertracks.resource_applicability.h6_matched_execution_v2 import install_limits, Progress
    from trottertracks.resource_applicability.h6_matched_execution_v3 import ResourceTraceV3
    b=read(Path(directory)/'task_binding.json');task=b['task'];r=resources(config,mode)
    install_parent_guard(read(Path(config['control'])/'coordinator.json'))
    install_limits(b['cpus'],32*GIB,64*2**20)
    resource.setrlimit(resource.RLIMIT_CPU,(86400,86400))
    # Cached font metadata avoids Matplotlib's font-scan timer thread during imports.
    cache=Path(os.environ['MPLCONFIGDIR']);cache.mkdir(exist_ok=True,parents=True)
    font=config['font_seed'];data=Path(font['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=font['sha256']:raise ValueError('FONT_SEED_CHANGED')
    (cache/font['name']).write_bytes(data)
    # Scientific inputs and archive are rechecked as bytes in each disposable
    # child. Large installed-file inventories remain a launch-only check.
    for item in config['files']:
        if item['logical_name'] in ('h6_input_snapshot.npz','snapshot_receipt.json',
                'df_receipt.json','old_preparation_archive') and file_hash(item['path'])!=item['sha256']:
            raise ValueError('WORKER_INPUT_CHANGED')
    writer=Writer(directory);progress=Progress(writer);start=time.monotonic()
    trace=ResourceTraceV3(writer.write,lambda:time.monotonic()-start)
    reason=None
    try:
        from h6_fixed_delta_science_v1 import signal, cost
        with trace.bind(),trace.stage(mode):
            (signal(config,task,writer,progress) if mode=='signal' else cost(config,task,writer,progress,trace))
    except BaseException as exc:
        reason=type(exc).__name__+':'+str(exc)[:512]
        try:trace.failure(exc)
        except Exception:pass
    writer.write('worker_terminal.json',dict(status='STOP' if reason else 'COMPLETE',reason=reason,task=task,
        SOURCE=config['SOURCE'],mandatory_stop=True))
    return 1 if reason else 0


def execute(config_path,authorization_path):
    config=read(config_path);grant=read(authorization_path)
    grant_gate(config,grant);check_config(config)
    provider(config)
    from trottertracks.resource_applicability.h6_server_resources_v1 import fresh_resources, process_identity
    from trottertracks.resource_applicability.h6_confirmation_fast_execution_v1 import supervise
    from trottertracks.resource_applicability.h6_matched_execution_v2 import install_limits
    out=private(config['output']);control=private(config['control'])
    if out.exists() or control.exists():raise ValueError('USED_OUTPUT_CONTROL')
    fresh=fresh_resources(resources(config,'normal'),observation_policy(),out)
    fresh['cpu_idle']=cpu_idle(resources(config,'normal')['assigned_cpus'])
    # The lock is consumed BEFORE spawning, and never removed after any failure.
    control.mkdir();out.mkdir();writer=Writer(control,16*2**20)
    writer.write('one_shot.json',dict(config_digest=digest(config),grant_digest=digest(grant),SOURCE=config['SOURCE']))
    writer.write('fresh.json',fresh);writer.write('coordinator.json',process_identity(os.getpid()))
    writer.write('config.json',config);writer.write('grant.json',grant)
    for k in THREADS:os.environ[k]='1'
    os.environ['QISKIT_PARALLEL']='FALSE'
    install_limits([config['resources']['driver_cpu']],32*GIB,256*2**20)
    resource.setrlimit(resource.RLIMIT_AS,(2*GIB,32*GIB))
    summaries=[];first=None;reserved=[]
    try:
        signals=[dict(index=i,task=dict(mode='signal',T=T,q=q)) for i,(T,q) in enumerate(((.4,2),(1.6,8)))]
        tasks=registrations()
        for mode,rows in [('signal',signals),('heavy',[r for r in tasks if r['task']['heavy']]),
                          ('normal',[r for r in tasks if not r['task']['heavy']])]:
            if mode!='signal':
                for row in rows:
                    task=row['task'];T=task['T'];i=0 if T==.4 else 1
                    p=out/'signal'/f'cost_confirmation_fast_{i:04d}'/'panel_signals.json'
                    record=next(x for x in read(p)['signals'] if x['cell']==task['cell'])
                    if task['cell']['method']!='B2':task['expected_signal']=record['signals']['corrected']
            directory=out/mode;directory.mkdir()
            m=dict(proposed_resources=resources(config,mode),observation_policy=observation_policy(),
                base_manifest=dict(plan=dict(caps_proposed=dict(output_bytes=2*GIB,log_bytes=8*2**20))),
                registrations=rows)
            writer.write('stage_'+mode+'.json',m)
            def command(row,child,slot):
                # Persist reservation before dispatch; failure never refunds it.
                reserved.append(deepcopy(row));writer.write(f'reserved_{len(reserved):03d}.json',dict(mode=mode,registration=row))
                return [config['python'],'-P','-B',str(Path(__file__).resolve()),'--worker',str(child),
                    '--config',str(control/'config.json'),'--mode',mode]
            summary=supervise(ROOT,directory,m,command=command,admission=admit)
            writer.write('summary_'+mode+'.json',summary);summaries.append(summary)
            if summary['first_stop_reason'] or not summary['all_dispatched_reaped']:
                first=summary['first_stop_reason'] or 'CLEANUP_UNCONFIRMED';break
    except BaseException as exc:first=first or type(exc).__name__+':'+str(exc)[:512]
    writer.write('terminal.json',dict(status='FAIL_CLOSED_STOP' if first else 'COMPLETE_MANDATORY_STOP',
        first_stop=first,SOURCE=config['SOURCE'],stage_summaries=summaries,
        reservations=reserved,carry_policy=config['carry_policy'],retry=False,next_stage_authorized=False))
    return 1 if first else 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-config',action='store_true');group.add_argument('--execute',action='store_true')
    group.add_argument('--worker');p.add_argument('--authorization');p.add_argument('--mode')
    a=p.parse_args()
    if a.check_config:check_config(read(a.config));print('METADATA_BINDING_PASS_NO_SCIENCE');return 0
    if a.worker:return worker(a.config,a.worker,a.mode)
    if not a.authorization:raise ValueError('NEW_AUTHORIZATION_REQUIRED')
    return execute(a.config,a.authorization)


if __name__=='__main__':raise SystemExit(main())
