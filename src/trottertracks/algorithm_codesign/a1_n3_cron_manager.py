"""Own-user cron submission, consume-once dispatch, no system changes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import signal

from . import a1_n3_lifecycle_observer as old
from .a1_n3_managed_runtime import durable_new,cleanup_owned,Runtime


def own_crontab():
    result=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if result.returncode==0:return result.stdout
    if result.returncode==1 and 'no crontab' in result.stderr:return None
    raise RuntimeError('own crontab unavailable: '+result.stderr)


def remove_own_entry(spec):
    current=own_crontab()
    if current is None:return
    if current!=spec['installed_crontab']:
        raise RuntimeError('own crontab changed; refusing to overwrite unrelated entries')
    subprocess.run(['crontab','-r'],check=True)
    if own_crontab() is not None:raise RuntimeError('temporary crontab removal not confirmed')


def submit_qualification(root,out):
    """Only a synthetic job; no general science command submission CLI."""
    root,out=Path(root).resolve(),Path(out).resolve()
    if own_crontab() is not None:raise RuntimeError('existing own crontab protected; STOP')
    out.mkdir(parents=True,exist_ok=False)
    entry=root/'scripts/diagnostics/run_a1_n3_manager_qualification.py'
    specpath=out/'SPEC.json'
    args=[sys.executable,'-B',str(entry),'--dispatch',str(specpath)]
    if any('%' in a or '\n' in a for a in args):raise ValueError('unsupported cron command character')
    line='* * * * * '+shlex.join(args)+' >> '+shlex.quote(str(out/'cron.log'))+' 2>&1\n'
    table='MAILTO=""\n'+line
    code=[entry,root/'src/trotterlib/parallel_validation_executor.py',
          *[root/'src/trottertracks/algorithm_codesign'/name for name in
        ('a1_n3_managed_runtime.py','a1_n3_cron_manager.py','a1_n3_lifecycle_observer.py',
         'a1_n3_interruption_diagnostics.py','shared_cpu.py')]]
    spec=dict(kind='artificial_qualification',root=str(root),output=str(out),installed_crontab=table,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        python=sys.executable,launcher_identity=old.process_identity(os.getpid()),
        job_restart_policy='no',science_authorized=False,
        source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        python_version=sys.version,python_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())
    durable_new(specpath,spec)
    if own_crontab() is not None:raise RuntimeError('own crontab concurrently changed; STOP')
    subprocess.run(['crontab','-'],input=table,text=True,check=True)
    if own_crontab()!=table:raise RuntimeError('installed crontab mismatch')
    old.event(out/'DISPATCH.jsonl','SUBMITTED',restart=False)
    return specpath


def dispatch(specpath):
    spec=json.loads(Path(specpath).read_text());out=Path(spec['output']);root=Path(spec['root'])
    if spec['kind']!='artificial_qualification' or spec['science_authorized'] is not False:
        raise ValueError('this entry point only accepts artificial qualification')
    try:durable_new(out/'CONSUMED.json',dict(identity=old.process_identity(os.getpid()),restart=False))
    except FileExistsError:
        old.event(out/'DISPATCH.jsonl','DUPLICATE_REJECTED');return 72
    # Remove the temporary trigger before starting any application process.
    remove_own_entry(spec)
    durable_new(out/'CRONTAB_RESTORED.json',dict(original_absent=True,current_absent=own_crontab() is None))
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in spec['source_sha256'].items()):
        raise ValueError('qualification source changed after submission')
    identity=old.process_identity(os.getpid())
    durable_new(out/'CRON_IDENTITY.json',identity)
    if '/system.slice/cron.service' not in identity.get('cgroup',''):
        raise RuntimeError('no confirmed session-independent cron cgroup; STOP')
    entry=root/'scripts/diagnostics/run_a1_n3_manager_qualification.py'
    env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    with (out/'qualification.log').open('xb') as log:
        child=subprocess.Popen([spec['python'],'-B',str(entry),'--suite',str(specpath)],
            cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        code=child.wait();log.flush();os.fsync(log.fileno())
    old.atomic_write_json(out/'DISPATCH_EXIT.json',old.wait_status(code))
    return code


def record_manager(argv,out,cwd):
    """Outer recorder survives application-manager failure; never restarts it."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    status=old.observe_child(argv,out/'outer',poll_seconds=.25)
    if status.get('exit_code')!=0:
        Runtime(out).request_stop('application manager abnormal exit')
        identity_file=out/'PARENT_IDENTITY.json'
        if identity_file.exists():
            identity=json.loads(identity_file.read_text())
            result=cleanup_owned(out,identity)
            old.event(out/'OUTER_CLEANUP.jsonl','OWN_GROUP_CLEANUP',result=result,
                      parent_exit='UNKNOWN if manager did not persist wait status')
        Runtime(out).classify()
    return status
