"""One adopted 752-shot pilot, using the qualified consume-once manager.

Only parent() binds scientific inputs. No resume, retry, or work on import.
"""
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time

from . import a1_n3_campaign as c
from . import a1_n3_pilot_campaign as pilot
from . import a1_n3_cron_manager as cron
from . import a1_n3_managed_runtime as managed
from . import a1_n3_lifecycle_observer as obs
from .a1_n3_B_campaign import heartbeat_storage, PYTHON

BASE='36ee9c6244eeedc875b5ddfafffb4c4cf76c8b8b'
CAMPAIGN=pilot.CAMPAIGN_CANDIDATE
SEED=pilot.SEED_CANDIDATE
OUTPUT=Path(pilot.OUTPUT_CANDIDATE)
CONTROL=OUTPUT.with_name('n3_pilot_752_manager_run01')
DOC=Path('docs/research/o2_a1_n3_pilot_science_20261107')
ENTRY=Path('scripts/tracks/algorithm_codesign/run_a1_n3_pilot_science.py')
REMOTE_BRANCH='codex/o2-a1-n3-pilot-science-20261107'


def configure_environment(root):
    os.environ['PYTHONPATH']=str(Path(root)/'src')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for name,leaf in (('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')):
        os.environ[name]='/tmp/a1-n3-pilot-science-cache-20261107/'+leaf
        Path(os.environ[name]).mkdir(parents=True,exist_ok=True)


def profile(root):
    result=c.profile(root)
    for name in (str(ENTRY),'tests/tracks/algorithm_codesign/test_a1_n3_pilot_activation.py',
                 'tests/tracks/algorithm_codesign/test_a1_n3_pilot_campaign.py',
                 'tests/tracks/algorithm_codesign/test_a1_n3_pilot_design.py'):
        result['source_sha256'][name]=hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
    return result


def verify(root,spec=None):
    root=Path(root)
    if c.kernel.n2.git(root,'diff','HEAD','--name-only'):raise ValueError('tracked SOURCE dirty')
    untracked=c.kernel.n2.git(root,'ls-files','--others','--exclude-standard').splitlines()
    if any(not (p.startswith(str(OUTPUT)+'/') or p.startswith(str(CONTROL)+'/')) for p in untracked):
        raise ValueError('unexpected untracked SOURCE files')
    if c.kernel.canonical(profile(root))!=c.kernel.canonical(c.kernel.read(root/DOC/'PROFILE_LOCK.json')):
        raise ValueError('pilot SOURCE/environment lock mismatch')
    if spec is not None:
        if spec['source_commit']!=c.kernel.n2.git(root,'rev-parse','HEAD'):raise ValueError('pilot SOURCE commit changed')
        if spec['profile_sha256']!=hashlib.sha256((root/DOC/'PROFILE_LOCK.json').read_bytes()).hexdigest():
            raise ValueError('pilot profile bytes changed')
        if (spec['campaign'],spec['seed'],spec['output'])!=(CAMPAIGN,SEED,str(root/OUTPUT)):
            raise ValueError('pilot campaign identity mismatch')
    c.n3.verify_runtime(root)
    binding=c.n3.inspect_certified_bundle(root)
    if c.kernel.canonical(binding)!=c.kernel.canonical(c.kernel.read(root/DOC/'INPUT_BINDING.json')):
        raise ValueError('pilot input binding mismatch')
    return binding


def submit(root):
    root=Path(root).resolve();configure_environment(root)
    binding=verify(root)
    if (root/OUTPUT).exists() or (root/CONTROL).exists():raise FileExistsError('pilot output collision; no retry')
    if cron.own_crontab() is not None:raise RuntimeError('existing own crontab protected')
    source=c.kernel.n2.git(root,'rev-parse','HEAD')
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+REMOTE_BRANCH],cwd=root,text=True).split()[0]
    if remote!=source:raise ValueError('published branch SHA differs from SOURCE')
    control=root/CONTROL;control.mkdir(parents=True,exist_ok=False);path=control/'SPEC.json'
    args=[PYTHON,'-B',str(root/ENTRY),'--dispatch',str(path)]
    if any('%' in a or '\n' in a for a in args):raise ValueError('unsupported cron path')
    table='MAILTO=""\n* * * * * '+shlex.join(args)+' >> '+shlex.quote(str(control/'cron.log'))+' 2>&1\n'
    spec=dict(kind='adopted_752_pilot_once',root=str(root),output=str(root/OUTPUT),campaign=CAMPAIGN,
        seed=SEED,python=PYTHON,source_commit=source,installed_crontab=table,restart=False,
        profile_sha256=hashlib.sha256((root/DOC/'PROFILE_LOCK.json').read_bytes()).hexdigest(),
        binding=binding,source_remote_verified=remote,old_measurements_reused=0,
        manager_SOURCE='5cb542ece4d208878a1dfc7011007b99297a5a55')
    managed.durable_new(path,spec)
    if cron.own_crontab() is not None:raise RuntimeError('own crontab concurrently changed')
    subprocess.run(['crontab','-'],input=table,text=True,check=True)
    if cron.own_crontab()!=table:raise RuntimeError('crontab installation mismatch')
    obs.event(control/'DISPATCH.jsonl','SUBMITTED',campaign=CAMPAIGN,restart=False)
    return path


def read_spec(path):
    spec=json.loads(Path(path).read_text())
    if spec['kind']!='adopted_752_pilot_once' or spec['restart'] is not False:raise ValueError('invalid pilot spec')
    root=Path(spec['root']);configure_environment(root)
    if Path(path).resolve()!=root/CONTROL/'SPEC.json':raise ValueError('pilot spec path mismatch')
    return root,spec


def semantic_status(out,raw):
    return 0 if raw==0 and pilot.semantic_exit_code(out)==0 else 1


def dispatch(path):
    root,spec=read_spec(path);control=root/CONTROL
    try:managed.durable_new(control/'CONSUMED.json',dict(identity=obs.process_identity(os.getpid()),restart=False))
    except FileExistsError:
        obs.event(control/'DISPATCH.jsonl','DUPLICATE_REJECTED');return 72
    try:
        cron.remove_own_entry(spec)
        managed.durable_new(control/'CRONTAB_RESTORED.json',dict(current_absent=cron.own_crontab() is None))
        identity=obs.process_identity(os.getpid());managed.durable_new(control/'CRON_IDENTITY.json',identity)
        if '/system.slice/cron.service' not in identity.get('cgroup',''):raise RuntimeError('persistent cron cgroup not confirmed')
        verify(root,spec);os.chdir(root)
        with heartbeat_storage():
            status=cron.record_manager([spec['python'],'-B',str(root/ENTRY),'--manager',str(path)],root/OUTPUT,root)
        code=semantic_status(root/OUTPUT,status.get('raw_returncode'))
        managed.durable_new(control/'DISPATCH_EXIT.json',dict(manager_status=status,semantic_exit=code,restart=False))
        return code
    except BaseException as exc:
        managed.durable_new(control/'DISPATCH_STOP.json',dict(type=type(exc).__name__,reason=str(exc),retry=False));raise


def manager(path):
    root,spec=read_spec(path)
    with heartbeat_storage():
        raw=managed.run_parent([spec['python'],'-B',str(root/ENTRY),'--parent',str(path)],root/OUTPUT,root)
    return semantic_status(root/OUTPUT,raw)


def finish_metadata(out,runtime,driver,pool,started,reason,setup):
    """Keep existing durable worker records authoritative, even on setup failure."""
    errors=[]
    if driver is None:
        runtime.request_stop(reason or 'pilot setup incomplete')
        if pool is not None:
            try:pool.shutdown(wait=True,cancel_futures=True)
            except BaseException as exc:errors.append(str(exc))
        try:c.reconcile(out)
        except BaseException as exc:errors.append(str(exc))
    try:
        runtime.classify()
        journal=out/'JOURNAL.jsonl'
        rows,tail=managed.audit.parse_records(journal.read_bytes()) if journal.exists() else ([],None)
        attempts=json.loads((out/'ATTEMPTS.json').read_text())['trials']
        c.save(out/'CAMPAIGN_LEDGER.json',dict(campaign=CAMPAIGN,retry=False,partial_parent_tail=tail,trials=[
            dict(ticket=r['ticket'],domain=r['domain'],trial=r['trial'],attempt=1,
                classification=attempts.get(r['ticket'],'PLANNED_NO_DURABLE_WORKER_START')) for r in rows if r['event']=='TRIAL_PLANNED']))
    except BaseException as exc:errors.append(str(exc));runtime.request_stop('finalization metadata failure')
    if not (out/'STOP.json').exists() or errors:
        runtime.request_stop(reason or errors)
        c.save(out/'STOP.json',dict(status='MANDATORY_STOP',purpose=pilot.PURPOSE,campaign_status='STOPPED',
            pilot_status='STOPPED',reason=reason,finalization_errors=errors,automatic_restart=False,next_stage_authorized=False))
    c.save(out/'RUN_FINALIZATION.json',dict(campaign=CAMPAIGN,setup_seconds=setup,
        elapsed_wall_seconds=time.monotonic()-started,parent_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        raw_pipeline_seconds=driver.collector.raw.classical if driver else None,errors=errors,
        automatic_restart=False,scientific_retry=False))
    return pilot.semantic_exit_code(out)


@contextmanager
def diagnostic_timing(out):
    """Observe existing fit/allocation/aggregation without changing numbers."""
    def timed(function,label):
        @wraps(function)
        def call(*args,**kwargs):
            start=time.perf_counter()
            try:return function(*args,**kwargs)
            finally:c.append(Path(out)/'DIAGNOSTIC_TIMING.jsonl',dict(function=label,seconds=time.perf_counter()-start))
        return call
    with ExitStack() as stack:
        for owner,name in ((pilot.cv,'fit_and_freeze'),(pilot.design,'allocate_evaluation'),(pilot.design,'pilot_mean')):
            stack.enter_context(patch.object(owner,name,timed(getattr(owner,name),name)))
        yield


def parent(path):
    root,spec=read_spec(path);out=root/OUTPUT
    allowed={'outer','MANAGER.jsonl','parent.log','PARENT_IDENTITY.json'}
    if not out.is_dir() or any(p.name not in allowed for p in out.iterdir()):raise FileExistsError('nonfresh pilot scientific output')
    binding=verify(root,spec)
    managed.durable_new(out/'INVOCATION.json',dict(campaign=CAMPAIGN,attempt=1,source=spec['source_commit'],retry=False))
    c._STOP_EVENT=mp.get_context('fork').Event();runtime=managed.Runtime(out,c._STOP_EVENT)
    pool=driver=None;reason=None;setup=None;started=time.monotonic()
    with ExitStack() as stack:
        stack.enter_context(runtime.signals());stack.enter_context(c.forbidden_science())
        stack.enter_context(managed.connect_legacy(c,runtime));stack.enter_context(diagnostic_timing(out))
        try:
            c.save(out/'SOURCE.json',dict(source_commit=spec['source_commit'],binding=binding,campaign=CAMPAIGN,
                purpose=pilot.PURPOSE,manager_SOURCE=spec['manager_SOURCE'],preparation_commit=BASE,
                adopted_contract=str(DOC/'EXECUTION.md'),seed=SEED,campaigns=1,old_measurements_reused=0,C1='frozen',
                conditions=dict(methods=['A1','B1-OL'],axes=['X','Y'],train_per_cell=16,probe_per_cell=8,
                    evaluation_per_method_axis=128,total_cap=752,range_guard=4,path_words=64,precision='NOT_CERTIFIED')))
            t=time.perf_counter();c._SERVICES=c.n3.load_certified_services(root);setup=time.perf_counter()-t
            c.save(out/'SETUP.json',dict(saved_bundle_decode=1,setup_seconds=setup,
                b_engine_exact=c._SERVICES[4]['total_engine_coordinate_upper_exact'],state_solve=0,engine_requalification=0))
            domains=pilot.pilot_domains()
            if len(domains)!=32 or len({d.token for d in domains})!=32:raise ValueError('32 stream domains required')
            streams=c.n3.pcg64_streams(SEED,domains,path_word_cap=64)
            c.save(out/'STREAMS.json',dict(campaign=CAMPAIGN,master_seed=SEED,attempt=1,domains=[
                dict(domain=d,path_spawn_key=[2*i],bit_spawn_key=[2*i+1]) for i,d in enumerate(domains)],
                model='independent uniform53bit assumption; PCG64 not an independence proof',children=64))
            workers=c.available_workers()
            c.save(out/'RESOURCE_START.json',dict(workers=workers,affinity_CPUs=len(os.sched_getaffinity(0)),
                load=os.getloadavg(),resources=obs.memory_snapshot(),policy='observed idle CPUs; own workers nice+10; no arbitrary cap'))
            pool=c.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'),initializer=c.worker_initialize,initargs=(out,))
            driver=pilot.connect_legacy_pipeline(out,CAMPAIGN,c._SERVICES,streams,pool,workers,runtime)
            driver.collector.raw.classical['setup']=setup
            driver.run()
        except BaseException as exc:
            reason=dict(type=type(exc).__name__,message=str(exc));runtime.request_stop(reason)
            if driver is not None:
                # Driver.run normally handles failures. If its own finalizer
                # failed, preserve STOP before the remaining worker drain.
                try:pool.shutdown(wait=True,cancel_futures=True);c.reconcile(out)
                except BaseException as final:reason['finalizer_error']=str(final)
                c.save(out/'STOP.json',dict(status='MANDATORY_STOP',purpose=pilot.PURPOSE,campaign_status='STOPPED',
                    pilot_status='STOPPED',reason=reason,automatic_restart=False,next_stage_authorized=False))
        return finish_metadata(out,runtime,driver,pool,started,reason,setup)
