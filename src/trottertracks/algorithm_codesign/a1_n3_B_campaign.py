"""One independent B campaign; frozen scientific functions, qualified manager.

Import is passive. No resume, retry, parameter sweep, or old run() invocation.
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from . import a1_n3_campaign as c
from . import a1_n3_cron_manager as cron
from . import a1_n3_managed_runtime as managed
from . import a1_n3_lifecycle_observer as obs

CAMPAIGN = 'A1-N3-B-H4-m0-q1-20261106-run01'
SEED = 20261107
BASE = '2438170a7d1eada04e8760cedd6fe5f959a2c87a'
DOC = Path('docs/research/o2_a1_n3_B_science_20261106')
OUTPUT = Path('artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_run01')
CONTROL = OUTPUT.with_name('n3_campaign_B_manager_run01')
ENTRY = Path('scripts/tracks/algorithm_codesign/run_a1_n3_B_campaign.py')
PYTHON = '/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python'


def configure_environment(root):
    os.environ['PYTHONPATH'] = str(Path(root)/'src')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for name,leaf in (('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')):
        os.environ[name] = '/tmp/a1-n3-B-cache-20261106/'+leaf
        Path(os.environ[name]).mkdir(parents=True,exist_ok=True)


def profile(root):
    result = c.profile(root)
    for name in (str(ENTRY),'tests/tracks/algorithm_codesign/test_a1_n3_B_campaign.py'):
        result['source_sha256'][name] = hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
    return result


def verify(root,spec=None):
    """Static byte/environment checks, with no saved-state numerical decoding."""
    root = Path(root)
    if c.kernel.n2.git(root,'diff','HEAD','--name-only'):
        raise ValueError('tracked SOURCE dirty')
    untracked = c.kernel.n2.git(root,'ls-files','--others','--exclude-standard').splitlines()
    if any(not (p.startswith(str(OUTPUT)+'/') or p.startswith(str(CONTROL)+'/')) for p in untracked):
        raise ValueError('unexpected untracked SOURCE files')
    current = profile(root)
    if c.kernel.canonical(current) != c.kernel.canonical(c.kernel.read(root/DOC/'PROFILE_LOCK.json')):
        raise ValueError('B SOURCE/environment lock mismatch')
    if spec is not None:
        if spec['source_commit'] != c.kernel.n2.git(root,'rev-parse','HEAD'):
            raise ValueError('B source commit changed')
        if spec['profile_sha256'] != hashlib.sha256((root/DOC/'PROFILE_LOCK.json').read_bytes()).hexdigest():
            raise ValueError('B profile bytes changed')
        if (spec['campaign'],spec['seed'],spec['output']) != (CAMPAIGN,SEED,str(root/OUTPUT)):
            raise ValueError('B campaign identity mismatch')
    c.n3.verify_runtime(root)
    return c.n3.inspect_certified_bundle(root)


def submit(root):
    root = Path(root).resolve();configure_environment(root)
    binding = verify(root)
    if (root/OUTPUT).exists() or (root/CONTROL).exists():
        raise FileExistsError('B output/control collision; no retry')
    if cron.own_crontab() is not None:
        raise RuntimeError('existing own crontab protected')
    out = root/CONTROL;out.mkdir(parents=True,exist_ok=False)
    specpath = out/'SPEC.json'
    args = [PYTHON,'-B',str(root/ENTRY),'--dispatch',str(specpath)]
    if any('%' in a or '\n' in a for a in args):raise ValueError('unsupported cron path')
    table = 'MAILTO=""\n* * * * * '+shlex.join(args)+' >> '+shlex.quote(str(out/'cron.log'))+' 2>&1\n'
    source = c.kernel.n2.git(root,'rev-parse','HEAD')
    # Publication is checked by the caller before submission, and recorded here.
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/o2-a1-n3-B-science-20261106'],cwd=root,text=True).split()[0]
    if remote != source:raise ValueError('published branch SHA differs from SOURCE')
    spec = dict(kind='authorized_N3_B_once',root=str(root),output=str(root/OUTPUT),
        campaign=CAMPAIGN,seed=SEED,python=PYTHON,source_commit=source,
        profile_sha256=hashlib.sha256((root/DOC/'PROFILE_LOCK.json').read_bytes()).hexdigest(),
        installed_crontab=table,restart=False,binding=binding,source_remote_verified=remote,
        manager_SOURCE='5cb542ece4d208878a1dfc7011007b99297a5a55',old_measurements_reused=0)
    managed.durable_new(specpath,spec)
    if cron.own_crontab() is not None:raise RuntimeError('own crontab concurrently changed')
    subprocess.run(['crontab','-'],input=table,text=True,check=True)
    if cron.own_crontab()!=table:raise RuntimeError('crontab installation mismatch')
    obs.event(out/'DISPATCH.jsonl','SUBMITTED',campaign=CAMPAIGN,restart=False)
    return specpath


def read_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec['kind']!='authorized_N3_B_once' or spec['restart'] is not False:
        raise ValueError('invalid B specification')
    root=Path(spec['root']);configure_environment(root)
    if Path(path).resolve()!=root/CONTROL/'SPEC.json':raise ValueError('B specification path mismatch')
    return root,spec


def semantic_status(out,raw):
    path=Path(out)/'STOP.json'
    return 0 if raw==0 and path.exists() and json.loads(path.read_text()).get('campaign_status')=='COMPLETED' else 1


@contextmanager
def heartbeat_storage():
    """Keep manager polling, persist resource heartbeats every 15 s.

    All call/start/exit/signal/STOP events remain unsuppressed. Reduces shared
    filesystem writes for a long exact-engine job; no computation budget cap.
    """
    original=obs.event;last={}
    def event(path,kind,**fields):
        if kind=='HEARTBEAT':
            now=time.monotonic();key=str(path)
            if key in last and now-last[key]<15:return
            last[key]=now
        return original(path,kind,**fields)
    with patch.object(obs,'event',event):yield


def dispatch(path):
    root,spec=read_spec(path);control=root/CONTROL
    try:managed.durable_new(control/'CONSUMED.json',dict(identity=obs.process_identity(os.getpid()),restart=False))
    except FileExistsError:
        obs.event(control/'DISPATCH.jsonl','DUPLICATE_REJECTED');return 72
    try:
        cron.remove_own_entry(spec)
        managed.durable_new(control/'CRONTAB_RESTORED.json',dict(current_absent=cron.own_crontab() is None))
        identity=obs.process_identity(os.getpid())
        managed.durable_new(control/'CRON_IDENTITY.json',identity)
        if '/system.slice/cron.service' not in identity.get('cgroup',''):
            raise RuntimeError('persistent cron cgroup not confirmed')
        verify(root,spec);os.chdir(root)
        with heartbeat_storage():
            status=cron.record_manager([spec['python'],'-B',str(root/ENTRY),'--manager',str(path)],root/OUTPUT,root)
        code=semantic_status(root/OUTPUT,status.get('raw_returncode'))
        managed.durable_new(control/'DISPATCH_EXIT.json',dict(manager_status=status,semantic_exit=code,restart=False))
        return code
    except BaseException as exc:
        managed.durable_new(control/'DISPATCH_STOP.json',dict(type=type(exc).__name__,reason=str(exc),retry=False))
        raise


def manager(path):
    root,spec=read_spec(path)
    with heartbeat_storage():
        raw=managed.run_parent([spec['python'],'-B',str(root/ENTRY),'--parent',str(path)],root/OUTPUT,root)
    return semantic_status(root/OUTPUT,raw)


def finalize(out,runtime,campaign,pool,status,reason,started):
    failures=[];methods=None
    if status!='COMPLETED':runtime.request_stop(reason)
    for label,call in (('drain',lambda:pool.shutdown(wait=True,cancel_futures=True) if pool else None),
                       ('summary',lambda:campaign.summary() if campaign else None),
                       ('accounting',lambda:c.reconcile(out)),('attempts',runtime.classify)):
        try:
            value=call()
            if label=='accounting':methods=value
        except BaseException as exc:
            failures.append(dict(stage=label,type=type(exc).__name__,message=str(exc)))
            runtime.request_stop('finalization failure '+label)
    if status=='COMPLETED':
        try:
            if not methods or any(not v['realized_RZ_complete'] or
                any(v['results'][axis]['status']!='SUFFICIENT_CONFIDENCE_COMPLETED' for axis in ('X','Y'))
                for v in methods.values()) or set(methods)!=set(c.n3.METHODS):
                raise ValueError('incomplete confidence/price comparison gate')
        except BaseException as exc:
            failures.append(dict(stage='completion_gate',type=type(exc).__name__,message=str(exc)))
            runtime.request_stop('completion gate failed')
    try:
        journal=out/'JOURNAL.jsonl'
        rows,tail=managed.audit.parse_records(journal.read_bytes()) if journal.exists() else ([],None)
        attempts=json.loads((out/'ATTEMPTS.json').read_text())['trials']
        ledger=[dict(campaign=CAMPAIGN,attempt=1,ticket=r['ticket'],domain=r['domain'],trial=r['trial'],
            classification=attempts.get(r['ticket'],'PLANNED_NO_DURABLE_WORKER_START'))
            for r in rows if r['event']=='TRIAL_PLANNED']
        c.save(out/'CAMPAIGN_LEDGER.json',dict(trials=ledger,partial_parent_tail=tail,retry=False))
    except BaseException as exc:
        failures.append(dict(stage='ledger',type=type(exc).__name__,message=str(exc)))
        runtime.request_stop('ledger finalization failure')
    if failures:status='STOPPED'
    c.save(out/'STOP.json',dict(status='MANDATORY_STOP',campaign_status=status,reason=reason,
        finalization_errors=failures,no_retry=True,additional_campaign_authorized=False,
        elapsed_wall_seconds=time.monotonic()-started))
    text = ('# A1 N3 B campaign\n\nCampaign: '+CAMPAIGN+'\n\nStatus: '+status+
            '\n\nReason: '+json.dumps(reason,ensure_ascii=False)+
            '\n\nAll durable worker tails are authoritative; see ACCOUNTING.json, CALL_COUNTS.json and ATTEMPTS.json.\n'+
            'Incomplete comparisons carry no precision or CV improvement claim. Expected cost improvement is UNKNOWN.\n'+
            'Mandatory STOP; no automatic retry or additional campaign.\n')
    (out/'REPORT.md').write_text(text)
    return 0 if status=='COMPLETED' else 1


def parent(path):
    root,spec=read_spec(path);out=root/OUTPUT
    # Manager owns the initial directory. Science entry remains exclusive.
    allowed={'outer','MANAGER.jsonl','parent.log','PARENT_IDENTITY.json'}
    if not out.is_dir() or any(p.name not in allowed for p in out.iterdir()):
        raise FileExistsError('nonfresh B scientific output')
    binding=verify(root,spec)
    managed.durable_new(out/'INVOCATION.json',dict(campaign=CAMPAIGN,attempt=1,source=spec['source_commit'],retry=False))
    c._STOP_EVENT=mp.get_context('fork').Event()
    runtime=managed.Runtime(out,c._STOP_EVENT)
    campaign=pool=None;status='STOPPED';reason=None;started=time.monotonic()
    with ExitStack() as stack:
        stack.enter_context(runtime.signals())
        stack.enter_context(c.forbidden_science())
        stack.enter_context(managed.connect_legacy(c,runtime))
        try:
            c.save(out/'SOURCE.json',dict(source_commit=spec['source_commit'],binding=binding,
                campaign=CAMPAIGN,manager_SOURCE=spec['manager_SOURCE'],contract=BASE,
                adopted_contract=str(DOC/'EXECUTION.md'),seed=SEED,campaigns=1,
                old_measurements_reused=0,C1='frozen',conditions=dict(epsilon='1/8',alpha='1/20',
                train_per_cell=64,cal_per_cell=512,production_cap=8192,J=8,path_words=64)))
            t=time.perf_counter();c._SERVICES=c.n3.load_certified_services(root);setup=time.perf_counter()-t
            c.save(out/'SETUP.json',dict(saved_bundle_decode=1,setup_seconds=setup,
                b_engine_exact=c._SERVICES[4]['total_engine_coordinate_upper_exact'],state_solve=0,engine_requalification=0))
            for method,count in (('B0',1),('B1-O',2),('B1-OL',5),('A1',1)):
                if len(c._SERVICES[2].cells(method))!=count:raise ValueError('positive stratum mismatch')
            streams=c.n3.pcg64_streams(SEED,c.domains(),path_word_cap=64)
            c.save(out/'STREAMS.json',dict(campaign=CAMPAIGN,master_seed=SEED,
                model='independent uniform53bit assumption; PCG64 not an independence proof',
                attempt=1,domains=[dict(domain=d,path_spawn_key=[2*i],bit_spawn_key=[2*i+1]) for i,d in enumerate(c.domains())]))
            workers=c.available_workers()
            c.save(out/'RESOURCE_START.json',dict(workers=workers,affinity_CPUs=len(os.sched_getaffinity(0)),
                load=os.getloadavg(),resources=obs.memory_snapshot(),policy='observed idle CPUs; own worker nice+10; no arbitrary resource cap'))
            pool=c.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'),initializer=c.worker_initialize,initargs=(out,))
            campaign=c.Campaign(out,c._SERVICES,streams,pool,workers);campaign.classical['setup']=setup
            campaign.execute();status='COMPLETED'
        except BaseException as exc:
            reason=dict(type=type(exc).__name__,message=str(exc));runtime.request_stop(reason)
            print(c.dumps(dict(event='STOP',reason=reason)),flush=True)
        return finalize(out,runtime,campaign,pool,status,reason,started)
