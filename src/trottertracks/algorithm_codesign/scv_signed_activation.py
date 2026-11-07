"""Deferred one-shot signed pilot route through the qualified existing manager.

Preparation may inspect bytes/profile. Only explicit future submit/parent calls
can create scientific streams/services. Nothing is submitted on import.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import multiprocessing as mp
import os
import resource
import time

from . import a1_n3_campaign as c
from . import a1_n3_connection as n3
from . import a1_n3_managed_runtime as managed
from . import a1_n3_resource_admission as admission
from . import a1_n3_lifecycle_observer as obs
from . import scv224_activation as old
from . import scv_signed_pilot as pilot
from . import scv_signed_restore as restore
from . import scv_signed_runtime as runtime

BASE='138d071805430ef1e556655392e4d9f572feba40'
RESULT_COMMIT='b4512675ae1d650858256547d359210361d77f68'
META=Path('artifacts/scv_signed_support_h4_metadata/2026-11-07/run01')
DOC=Path('docs/research/scv_signed_support_448_preparation_20261107')
OUTPUT=Path('artifacts/scv_signed_support_fresh_pilot/2026-11-07/campaign_run01')
CONTROL=OUTPUT.with_name('manager_run01')
ENTRY=Path('scripts/tracks/algorithm_codesign/run_scv_signed_pilot.py')
BRANCH='codex/scv-signed-support-448-preparation-20261107'
KIND='signed_support_448_once'
CAMPAIGN,SEED=pilot.CAMPAIGN,pilot.SEED
BINDING=restore.Binding(RESULT_COMMIT,str(META/'METADATA_RESULT.json'),
    'c5895f7696711c92bf87e591a3c2264cd7672d9af70d13027db790950e305ba0',
    'c5e0c5c26b926eb4a73d1808a1c61222dc45934248c30d8aea4f8fc32c44b7e2',
    '5e252ff9fa5682850683ced97d007b59fecbf6d7617c561adb4a34d4cc7c7f89',
    'b77980d198cc9edc620daec9bf87df02b66e96575fd85327aed0793f3e86ecef',
    'ca897db7be52ad30dcfab0512d7530188fc13a9d8ca0c78430809492d01f738b',
    '56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5',
    'f02980a5cbec9478aa1f4e6dd010f117d152f89770c40389544cf876dd197299',
    'df-fragment-3:Z0Z1','df-fragment-6:Z0',8)


def configure_environment(root):
    os.environ['PYTHONPATH']=str(Path(root)/'src');os.environ['PYTHONDONTWRITEBYTECODE']='1'
    namespace=hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]
    for name,leaf in (('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')):
        os.environ[name]='/tmp/scv-signed-448-'+namespace+'/'+leaf
        Path(os.environ[name]).mkdir(parents=True,exist_ok=True)


def profile(root):
    root=Path(root);p=c.profile(root)
    # Hash all local Python library files without importing/running them. This
    # also pins transitive scientific dependencies missed by old short lists.
    paths=list(root.glob('src/**/*.py'))+[root/ENTRY,root/DOC/'SETTINGS.json']
    paths+=list((root/'tests/tracks/algorithm_codesign').glob('test_scv_signed*.py'))
    for path in paths:p['source_sha256'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return p


def input_binding(root):
    root=Path(root);binding=n3.inspect_certified_bundle(root)
    rows={}
    for path in sorted((root/META).iterdir()):
        if path.is_file():
            relative=str(path.relative_to(root));data=path.read_bytes()
            if data!=__import__('subprocess').check_output(['git','show',RESULT_COMMIT+':'+relative],cwd=root):
                raise ValueError('fixed metadata result bytes changed: '+relative)
            rows[relative]=hashlib.sha256(data).hexdigest()
    if rows.get(BINDING.result_path)!=BINDING.result_sha256:raise ValueError('saved metadata result absent')
    return dict(certified_bundle=binding,saved_metadata_commit=RESULT_COMMIT,saved_metadata_sha256=rows,
        saved_control_binding=BINDING.__dict__,scope='byte/identity only; no moment/state numerical load')


def verify(root,spec=None):
    root=Path(root)
    if c.kernel.n2.git(root,'diff','HEAD','--name-only'):raise ValueError('tracked SOURCE dirty')
    unknown=c.kernel.n2.git(root,'ls-files','--others','--exclude-standard').splitlines()
    if any(not (p.startswith(str(OUTPUT)+'/') or p.startswith(str(CONTROL)+'/')) for p in unknown):
        raise ValueError('unexpected untracked SOURCE')
    if c.kernel.canonical(profile(root))!=c.kernel.canonical(c.kernel.read(root/DOC/'PROFILE_LOCK.json')):
        raise ValueError('signed SOURCE/environment/compiler lock mismatch')
    n3.verify_runtime(root);binding=input_binding(root)
    if c.kernel.canonical(binding)!=c.kernel.canonical(c.kernel.read(root/DOC/'INPUT_BINDING.json')):
        raise ValueError('signed input binding mismatch')
    if c.kernel.canonical(pilot.settings())!=c.kernel.canonical(c.kernel.read(root/DOC/'SETTINGS.json')):
        raise ValueError('signed settings mismatch')
    if spec is not None:
        expected=(KIND,CAMPAIGN,SEED,str(root/OUTPUT),False)
        if tuple(spec.get(k) for k in ('kind','campaign','seed','output','restart'))!=expected:
            raise ValueError('signed manager/campaign/seed/output mismatch')
        if spec.get('source_commit')!=c.kernel.n2.git(root,'rev-parse','HEAD'):raise ValueError('SOURCE commit changed')
        if (spec.get('manager_SOURCE')!='5cb542ece4d208878a1dfc7011007b99297a5a55'
            or spec.get('python')!=old.PYTHON or spec.get('source_remote_verified')!=spec.get('source_commit')):
            raise ValueError('qualified manager/runtime/publication mismatch')
        if spec.get('binding')!=binding:raise ValueError('manager input binding changed')
        if spec.get('profile_sha256')!=hashlib.sha256((root/DOC/'PROFILE_LOCK.json').read_bytes()).hexdigest():
            raise ValueError('profile byte mismatch')
    return binding


def read_spec(path):
    path=Path(path);spec=json.loads(path.read_text());root=Path(spec['root']).resolve()
    if path.resolve()!=root/CONTROL/'SPEC.json' or spec.get('kind')!=KIND or spec.get('restart') is not False:
        raise ValueError('signed manager path/kind/restart mismatch')
    configure_environment(root)
    return root,spec


def semantic_status(out,raw):
    out=Path(out);result=out/'SIGNED_RESULT.json';final=out/'RUN_FINALIZATION.json'
    return 0 if raw==0 and result.exists() and json.loads(result.read_text()).get('status')=='COMPLETE' and (
        final.exists() and not json.loads(final.read_text()).get('errors')) else 1


@contextmanager
def route_binding():
    # Reuse submit/consume-once/cron cgroup/outer/run_parent without new manager.
    with patch.multiple(old,BASE=BASE,CAMPAIGN=CAMPAIGN,SEED=SEED,OUTPUT=OUTPUT,CONTROL=CONTROL,DOC=DOC,
        ENTRY=ENTRY,BRANCH=BRANCH,KIND=KIND,configure_environment=configure_environment,
        verify=verify,read_spec=read_spec,semantic_status=semantic_status):yield


def submit(root):
    with route_binding():return old.submit(root)

def dispatch(path):
    with route_binding():return old.dispatch(path)

def manager(path):
    with route_binding():return old.manager(path)


def restore_once(root):
    root=Path(root);one=root/c.kernel.P1
    return restore.restore_saved_control((root/BINDING.result_path).read_bytes(),
        (one/'LAW.json').read_bytes(),(one/'PREPARATION.json').read_bytes(),BINDING,
        commit=RESULT_COMMIT,path=BINDING.result_path)


def memory_admission(workers,snapshot,*,per_worker_bytes):
    """Observed memory headroom, not an arbitrary CPU/RAM reservation.

    Parent RSS is a starting footprint proxy, not a bound on Fraction growth.
    Actual memory pressure is checked again before each path submission.
    """
    lines=snapshot['meminfo'].splitlines()
    available=int(next(x.split()[1] for x in lines if x.startswith('MemAvailable:')))*1024
    for values in snapshot.get('ancestor_cgroups',{}).values():
        maximum,current=values.get('memory.max',''),values.get('memory.current','')
        if maximum.strip().isdigit() and current.strip().isdigit():available=min(available,int(maximum)-int(current))
    pressure=snapshot.get('pressure/memory','UNKNOWN')
    if pressure=='UNKNOWN':raise RuntimeError('memory pressure UNKNOWN; preflight unresolved')
    full=next(x for x in pressure.splitlines() if x.startswith('full '))
    if float(next(v.split('=')[1] for v in full.split() if v.startswith('avg10=')))>0:
        raise RuntimeError('observed full memory pressure; no new work')
    count=min(workers,160,available//max(1,per_worker_bytes))
    if count<1:raise RuntimeError('no observed memory headroom')
    return count


@contextmanager
def forbidden_recomputation():
    def stop(*args,**kwargs):raise RuntimeError('saved signed moments/selection must not be recomputed')
    with c.forbidden_science(),patch.object(pilot.signed.SignedControl,'__post_init__',stop),patch.object(pilot.scalar,'select_fragment_control',stop):yield


def parent(path):
    root,spec=read_spec(path);out=root/OUTPUT
    allowed={'outer','MANAGER.jsonl','parent.log','PARENT_IDENTITY.json'}
    if not out.is_dir() or any(p.name not in allowed for p in out.iterdir()):raise FileExistsError('fresh signed output required; no resume')
    binding=verify(root,spec)
    managed.durable_new(out/'INVOCATION.json',dict(campaign=CAMPAIGN,source=spec['source_commit'],retry=False))
    c._STOP_EVENT=mp.get_context('fork').Event();gate=runtime.CallGate(out,c._STOP_EVENT,CAMPAIGN)
    life=runtime.Runtime(out,gate);pool=driver=None;reason=None;errors=[];started=time.perf_counter()
    cpu0=resource.getrusage(resource.RUSAGE_SELF)
    with ExitStack() as stack:
        stack.enter_context(life.signals());stack.enter_context(forbidden_recomputation())
        stack.enter_context(runtime.connection(gate))
        try:
            workers=admission.wait_for_resources(out,lambda:c._STOP_EVENT.is_set())
            workers=memory_admission(workers,obs.memory_snapshot(),per_worker_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            with gate.call('saved_control_restore'):control,masses=restore_once(root)
            c.save(out/'SOURCE.json',dict(**pilot.source_fields(CAMPAIGN,control),source_commit=spec['source_commit'],binding=binding,
                contract_commit=BASE,C1='frozen',old_measurements_reused=0))
            c.save(out/'RESTORED_CONTROL.json',dict(snapshot=control.snapshot(),masses=masses,moment_recomputation=0))
            with gate.call('saved_bundle_decode'):
                c._SERVICES=old.bind_sampler(n3.load_certified_services(root),control)
            workers=memory_admission(workers,obs.memory_snapshot(),per_worker_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            gate.before_path=lambda:memory_admission(1,obs.memory_snapshot(),per_worker_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            with gate.call('stream_book'):streams=n3.pcg64_streams(SEED,pilot.domains(),path_word_cap=64)
            c.save(out/'STREAMS.json',dict(campaign=CAMPAIGN,seed=SEED,mapping=pilot.settings()['domain_mapping'],children=24,
                model='independent uniform53-bit assumption; PCG64 is not an independence proof'))
            c.save(out/'RESOURCE_START.json',dict(workers=workers,resources=obs.memory_snapshot(),
                affinity_CPUs=len(os.sched_getaffinity(0)),memory_growth_bound='UNKNOWN; live pressure monitoring',
                policy='observed idle/affinity/memory headroom, max useful stage tasks160, no low fixed CPU cap'))
            pool=managed.ManagedPool(life,max_workers=workers,mp_context=mp.get_context('fork'),
                initializer=c.worker_initialize,initargs=(out,))
            raw=c.Campaign(out,c._SERVICES,runtime.CountedStreams(streams,gate),pool,workers)
            collector=runtime.Collector(raw,CAMPAIGN,control,life,gate)
            driver=pilot.Driver(out,CAMPAIGN,control,c._SERVICES[3],collector,gate)
            driver.run()
        except BaseException as exc:
            reason=type(exc).__name__+': '+str(exc);life.request_stop(reason)
        finally:
            life.request_stop(reason or 'signed campaign ended')
            if driver is None or not (out/'SIGNED_RESULT.json').exists():
                if pool is not None:
                    try:pool.shutdown(wait=True,cancel_futures=True)
                    except BaseException as exc:errors.append('drain '+str(exc))
                try:c.reconcile(out);life.classify()
                except BaseException as exc:errors.append('reconcile '+str(exc))
            if not (out/'STOP.json').exists():managed.durable_new(out/'STOP.json',dict(status='STOPPED',reason=reason,retry=False,mandatory_STOP=True))
            c.save(out/'BOUNDED_CALL_COUNTS.json',gate.snapshot());cpu1=resource.getrusage(resource.RUSAGE_SELF)
            c.save(out/'RUN_FINALIZATION.json',dict(campaign=CAMPAIGN,reason=reason,errors=errors,
                elapsed_parent_wall_seconds=time.perf_counter()-started,parent_cpu_user_seconds=cpu1.ru_utime-cpu0.ru_utime,
                parent_cpu_system_seconds=cpu1.ru_stime-cpu0.ru_stime,parent_peak_RSS_bytes=cpu1.ru_maxrss*1024,
                unknown_classical_costs='unmeasured subintervals UNKNOWN; manager/outer separate; RSS/PSS not additive',
                automatic_restart=False,scientific_retry=False))
    return semantic_status(out,0)
