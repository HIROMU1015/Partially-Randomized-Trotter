"""One adopted SCV224 activation. Reuse the fixed driver and qualified manager.

Only parent() consumes real metadata, decodes a saved bundle or constructs RNG.
No old result/cache is imported. No science, file load or job on import.
"""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import multiprocessing as mp
import os
import resource
import shlex
import subprocess
import time

from . import a1_n3_pilot_activation as routing
from . import a1_n3_campaign as c
from . import a1_n3_connection as n3
from . import a1_n3_managed_runtime as managed
from . import a1_n3_resource_admission as admission
from . import a1_n3_cron_manager as cron
from . import a1_n3_lifecycle_observer as obs
from . import scv224_pilot as pilot
from . import scv_scalar as scalar
from .a1_n3_B_campaign import PYTHON

BASE='560c6092bfb5e71af8eedc4a4be5416922764f98'
CAMPAIGN=pilot.CAMPAIGN_CANDIDATE
SEED=pilot.SEED_CANDIDATE
OUTPUT=Path('artifacts/o2_scv224/2026-11-07/campaign_run01')
CONTROL=OUTPUT.with_name('manager_run01')
DOC=Path('docs/research/o2_scv224_science_20261107')
ENTRY=Path('scripts/tracks/algorithm_codesign/run_scv224_science.py')
BRANCH='codex/o2-scv224-science-20261107'
KIND='adopted_scv224_once'


def configure_environment(root):
    os.environ['PYTHONPATH']=str(Path(root)/'src');os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for name,leaf in (('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')):
        os.environ[name]='/tmp/scv224-science-cache-20261107/'+leaf
        Path(os.environ[name]).mkdir(parents=True,exist_ok=True)


def profile(root):
    result=c.profile(root)
    for name in (str(ENTRY),'src/trottertracks/algorithm_codesign/scv224_activation.py',
                 'src/trottertracks/algorithm_codesign/scv_scalar.py',
                 'src/trottertracks/algorithm_codesign/scv224_pilot.py',
                 'tests/tracks/algorithm_codesign/test_scv224_activation.py',
                 *('src/trottertracks/algorithm_codesign/'+name+'.py' for name in
                   ('a1_n3_pilot_activation','a1_n3_B_campaign','a1_n3_managed_runtime',
                    'a1_n3_resource_admission','a1_n3_cron_manager','a1_n3_lifecycle_observer'))):
        result['source_sha256'][name]=hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
    return result


def verify(root,spec=None):
    root=Path(root)
    if c.kernel.n2.git(root,'diff','HEAD','--name-only'):raise ValueError('tracked SOURCE dirty')
    others=c.kernel.n2.git(root,'ls-files','--others','--exclude-standard').splitlines()
    if any(not(p.startswith(str(OUTPUT)+'/') or p.startswith(str(CONTROL)+'/')) for p in others):
        raise ValueError('unexpected untracked SOURCE')
    locked=c.kernel.read(root/DOC/'PROFILE_LOCK.json')
    if c.kernel.canonical(profile(root))!=c.kernel.canonical(locked):raise ValueError('SOURCE/environment lock mismatch')
    # The preparation modules and all published input bytes remain immutable.
    prepared=c.kernel.read(root/'artifacts/scv224_preparation/2026-11-07/SOURCE.json')
    for p,h in prepared['new_source_sha256'].items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h:raise ValueError('prepared SOURCE mismatch '+p)
    for row in prepared['fixed_input_byte_manifest']:
        if hashlib.sha256((root/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('fixed input mismatch '+row['path'])
    c.n3.verify_runtime(root);binding=c.n3.inspect_certified_bundle(root)
    if c.kernel.canonical(binding)!=c.kernel.canonical(c.kernel.read(root/DOC/'INPUT_BINDING.json')):
        raise ValueError('input binding mismatch')
    if spec is not None:
        if (spec['kind'],spec['campaign'],spec['seed'],spec['output'],spec['restart'])!=(KIND,CAMPAIGN,SEED,str(root/OUTPUT),False):
            raise ValueError('campaign/seed/scope mismatch')
        if spec['source_commit']!=c.kernel.n2.git(root,'rev-parse','HEAD'):raise ValueError('SOURCE commit changed')
        if spec['profile_sha256']!=hashlib.sha256((root/DOC/'PROFILE_LOCK.json').read_bytes()).hexdigest():
            raise ValueError('profile changed')
    return binding


def submit(root):
    root=Path(root).resolve();configure_environment(root);binding=verify(root)
    if (root/OUTPUT).exists() or (root/CONTROL).exists():raise FileExistsError('fresh output/control required; no retry')
    if cron.own_crontab() is not None:raise RuntimeError('existing own crontab protected')
    source=c.kernel.n2.git(root,'rev-parse','HEAD')
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=root,text=True).split()[0]
    if source!=remote:raise ValueError('published SOURCE mismatch')
    control=root/CONTROL;control.mkdir(parents=True,exist_ok=False);path=control/'SPEC.json'
    argv=[PYTHON,'-B',str(root/ENTRY),'--dispatch',str(path)]
    if any('%' in arg or '\n' in arg for arg in argv):raise ValueError('unsupported cron path')
    table='MAILTO=""\n* * * * * '+shlex.join(argv)+' >> '+shlex.quote(str(control/'cron.log'))+' 2>&1\n'
    spec=dict(kind=KIND,root=str(root),output=str(root/OUTPUT),campaign=CAMPAIGN,seed=SEED,restart=False,
        python=PYTHON,source_commit=source,installed_crontab=table,binding=binding,
        profile_sha256=hashlib.sha256((root/DOC/'PROFILE_LOCK.json').read_bytes()).hexdigest(),
        manager_SOURCE='5cb542ece4d208878a1dfc7011007b99297a5a55',source_remote_verified=remote)
    managed.durable_new(path,spec)
    if cron.own_crontab() is not None:raise RuntimeError('own crontab concurrently changed')
    subprocess.run(['crontab','-'],input=table,text=True,check=True)
    if cron.own_crontab()!=table:raise RuntimeError('crontab mismatch')
    obs.event(control/'DISPATCH.jsonl','SUBMITTED',campaign=CAMPAIGN,restart=False)
    return path


def read_spec(path):
    spec=json.loads(Path(path).read_text());root=Path(spec['root']);configure_environment(root)
    if spec['kind']!=KIND or spec['restart'] is not False or Path(path).resolve()!=root/CONTROL/'SPEC.json':
        raise ValueError('SCV spec/path/restart mismatch')
    return root,spec


def semantic_status(out,raw):
    file=Path(out)/'SCV_RESULT.json';final=Path(out)/'RUN_FINALIZATION.json'
    return 0 if raw==0 and file.exists() and json.loads(file.read_text()).get('status')=='COMPLETE' and (
        not final.exists() or not json.loads(final.read_text()).get('errors')) else 1


def _route(name,path):
    # Same consume-once, cron-cgroup, parent wait/exit/cleanup path as the
    # completed 752-shot campaign. Scoped names change only this new process.
    with patch.multiple(routing,OUTPUT=OUTPUT,CONTROL=CONTROL,DOC=DOC,ENTRY=ENTRY,
                        read_spec=read_spec,verify=verify,semantic_status=semantic_status):
        return getattr(routing,name)(path)


def dispatch(path):return _route('dispatch',path)
def manager(path):return _route('manager',path)


class OrderedDriver(pilot.Driver):
    """Get X and Y training before either fit; all other driver code unchanged.

    Y's validated batch waits in memory for its one fit. It is neither acquired
    nor priced again. Both group acquisitions use the original Driver guards.
    """
    def acquire(self,axis,stage):
        if stage!='train':return super().acquire(axis,stage)
        if axis=='X':
            if hasattr(self,'_pending_y'):raise ValueError('training already acquired')
            x=super().acquire('X','train')
            self._pending_y=super().acquire('Y','train')
            return x
        if axis!='Y' or not hasattr(self,'_pending_y'):raise ValueError('X/Y training order')
        pilot.check_source(self.out,self.campaign,self.control)
        rows=self._pending_y;del self._pending_y
        return rows


def metadata_once(root,binding,out):
    from .a1_new_target_n1 import A,B
    ids=binding['N1_identities'];one=Path(root)/c.kernel.P1
    md=scalar.metadata_from_bytes((one/'LAW.json').read_bytes(),(one/'PREPARATION.json').read_bytes(),
        law_sha256=ids['law_sha256'],catalog_sha256=ids['catalog_sha256'],
        hamiltonian_hash=ids['hamiltonian_hash'],preparation_hash=ids['preparation_hash'],a=A,b=B)
    control=scalar.select_fragment_control(md)  # the sole real selection/mean call
    c.save(Path(out)/'METADATA_CONTROL.json',dict(**control.snapshot(),N1_identities=ids,
        positive_fragments=md.positive_fragments,cells=md.sampler.cells('B1-OL'),
        bit_or_state_or_old_data_used=False))
    return control


def bind_sampler(services,control):
    """Reuse the one metadata sampler object after exact schema equality."""
    old=services[2];new=control.metadata.sampler
    for field in ('law','a','b','patterns','bank','five_bank'):
        if getattr(old,field)!=getattr(new,field):raise ValueError('saved/metadata sampler mismatch: '+field)
    return (*services[:2],new,*services[3:])


def parent(path):
    root,spec=read_spec(path);out=root/OUTPUT
    allowed={'outer','MANAGER.jsonl','parent.log','PARENT_IDENTITY.json'}
    if not out.is_dir() or any(p.name not in allowed for p in out.iterdir()):raise FileExistsError('nonfresh SCV output')
    binding=verify(root,spec)
    managed.durable_new(out/'INVOCATION.json',dict(campaign=CAMPAIGN,source=spec['source_commit'],retry=False))
    c._STOP_EVENT=mp.get_context('fork').Event();runtime=managed.Runtime(out,c._STOP_EVENT)
    pool=driver=None;driver_returned=False;reason=None;setup=None;errors=[];calls={};started=time.monotonic()
    def once(label,function):
        if label in calls:raise ValueError('no scientific retry: '+label)
        calls[label]=dict(started=1,returned=0)
        c.append(out/'ACTIVATION_CALLS.jsonl',dict(event='CALL_START',name=label))
        t=time.perf_counter()
        try:value=function()
        except BaseException as exc:
            c.append(out/'ACTIVATION_CALLS.jsonl',dict(event='CALL_EXCEPTION',name=label,type=type(exc).__name__,reason=str(exc)))
            raise
        calls[label].update(returned=1,seconds=time.perf_counter()-t)
        c.append(out/'ACTIVATION_CALLS.jsonl',dict(event='CALL_RETURN',name=label,seconds=calls[label]['seconds']))
        return value
    with ExitStack() as stack:
        stack.enter_context(runtime.signals());stack.enter_context(c.forbidden_science())
        stack.enter_context(managed.connect_legacy(c,runtime))
        try:
            c.save(out/'EXECUTION_SOURCE.json',dict(source_commit=spec['source_commit'],binding=binding,
                campaign=CAMPAIGN,purpose=pilot.PURPOSE,seed=SEED,preparation_commit=BASE,
                contract=str(DOC/'EXECUTION.md'),manager_SOURCE=spec['manager_SOURCE'],old_measurements_reused=0,C1='frozen'))
            workers=admission.wait_for_resources(out,lambda:c._STOP_EVENT.is_set())
            control=once('metadata_selection_means',lambda:metadata_once(root,binding,out))
            c._SERVICES=bind_sampler(once('saved_bundle_decode',lambda:c.n3.load_certified_services(root)),control)
            setup=calls['saved_bundle_decode']['seconds']
            c.save(out/'SETUP.json',dict(saved_bundle_decode=1,setup_seconds=setup,
                b_engine_exact=c._SERVICES[4]['total_engine_coordinate_upper_exact'],state_solve=0,engine_requalification=0))
            c.save(out/'SOURCE.json',dict(**pilot.source_fields(CAMPAIGN,control),source_commit=spec['source_commit'],
                binding=binding,seed=SEED,conditions=pilot.preparation_plan()|dict(status='ADOPTED_FOR_THIS_CAMPAIGN',
                    scientific_execution_enabled=True,pending=[]),preparation_commit=BASE,C1='frozen',old_measurements_reused=0))
            ds=pilot.domains()
            if len(ds)!=12 or len({d.token for d in ds})!=12:raise ValueError('12 domains required')
            streams=once('stream_book',lambda:c.n3.pcg64_streams(SEED,ds,path_word_cap=64))
            c.save(out/'STREAMS.json',dict(campaign=CAMPAIGN,seed=SEED,children=24,domains=[
                dict(domain=d,path_spawn_key=[2*i],bit_spawn_key=[2*i+1]) for i,d in enumerate(ds)],
                model='independent uniform53-bit assumption; PCG64 is not an independence proof'))
            c.save(out/'RESOURCE_START.json',dict(workers=workers,resources=obs.memory_snapshot(),
                affinity_CPUs=len(os.sched_getaffinity(0)),load=os.getloadavg(),new_artificial_worker_cap=False))
            pool=c.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'),initializer=c.worker_initialize,initargs=(out,))
            collector=pilot.connect_legacy_pipeline(out,CAMPAIGN,control,c._SERVICES,streams,pool,workers,runtime)
            collector.raw.classical['setup']=setup
            driver=OrderedDriver(out,CAMPAIGN,control,c._SERVICES[3],collector)
            driver.run();driver_returned=True
        except BaseException as exc:
            reason=dict(type=type(exc).__name__,message=str(exc));runtime.request_stop(reason)
        finally:
            if not driver_returned:
                runtime.request_stop(reason or 'SCV setup/driver incomplete')
                if pool is not None:
                    try:pool.shutdown(wait=True,cancel_futures=True)
                    except BaseException as exc:errors.append('drain: '+str(exc))
                try:c.reconcile(out)
                except BaseException as exc:errors.append('reconcile: '+str(exc))
            try:runtime.classify()
            except BaseException as exc:errors.append('classify: '+str(exc))
            if not (out/'STOP.json').exists():
                managed.durable_new(out/'STOP.json',dict(status='STOPPED',reason=reason,errors=errors,mandatory_STOP=True,retry=False))
            c.save(out/'SCIENCE_SETUP_CALL_COUNTS.json',dict(calls=calls,science_retry=0,extra_campaigns=0))
            c.save(out/'RUN_FINALIZATION.json',dict(campaign=CAMPAIGN,elapsed_wall_seconds=time.monotonic()-started,
                parent_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,reason=reason,errors=errors,
                legacy_pipeline_seconds=driver.collector.raw.classical if driver else None,
                metadata_seconds=calls.get('metadata_selection_means',{}).get('seconds'),setup_seconds=setup,
                unknown_classical_costs='manager/outer recorded separately; shared-page RSS is not additive',
                automatic_restart=False,scientific_retry=False))
    return semantic_status(out,0)
