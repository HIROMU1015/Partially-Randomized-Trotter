"""New-host one-shot binding. Draft flags never authorize runtime or CPU use."""
from dataclasses import dataclass,replace
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import tempfile

from . import gates
from .identity import require, fingerprint, sha, hash_id
from .prelaunch_audit import private_path, receipt_inventory, environment_profile, compiler_profile, host_readonly, static_invocations
from .resources import GiB, ROLE_CAP, HEADROOM, OUTPUT_CAP, WALL_CAP, fsync_directory
from .observer import AS_CAP, RSS_CAP, FRAME_CAP, TERMINAL_RESERVE

RUN_ID='h4-newhost-signal-compile-20261007-run01'
CARRY={'actual_invocations':20,'charged_bytes':165214360,'wall_seconds':5466.188392877579}
EXPECTED_FREEZE='75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69'
EXPECTED_JOURNAL='6b68368565cc336d283bc094f844a37ceb1966838ca5482f1f12347d0c5d669e'
EXPECTED_LOG='7108181a3b295ca92d4a73de7e7420d160898270d45a95657a34e7c6526947b1'
CONTROL_LOG_CAP=8*2**20
FILE_LIMITS={'record-':4096,'ledger-':4096,'signal-':524288,'worker-log-':8192,'map-complete':4096,'launch-stop':4096,'ledger.lock':0}

STRUCTURES={
 'plan':{'schema_version':str,'stage':str,'run_id':str,'source_commit':str,'source_root':str,
         'source_hashes':dict,'source_audit':dict,'environment_profile':dict,'compiler_profile':dict,
         'input_root':str,'stop_evidence_root':str,'stop_evidence_receipt':dict,'output_root':str,'control_root':str,
         'inputs':dict,'generation_freeze_digest':str,'templates':list,'contract_plan_fingerprint':str,
         'compiler_fingerprint':str,'environment_fingerprint':str,'requested_workers':int,
         'cpu_proposal':dict,'carry':dict,'caps':dict,'storage':dict,'sealed':bool},
 'authorization':{'schema_version':str,'stage':str,'run_id':str,'source_commit':str,'plan_fingerprint':str,
         'approved':bool,'runtime_authorization':bool,'allowed_cpus':list,'one_shot':bool,
         'permission':str,'result_prior':bool,'environment_accepted':bool,'observer_role':dict,'budget_amendment':dict},
 'review':{'schema_version':str,'stage':str,'run_id':str,'source_commit':str,'plan_fingerprint':str,
         'authorization_digest':str,'approved':bool,'runtime_authorization':bool,'reviewer':str,'mandatory_stop':bool}}
VERSIONS={k:'h4-newhost-'+k+'-v2' for k in STRUCTURES}


def structural(plan,authorization,review):
    for label,doc in [('plan',plan),('authorization',authorization),('review',review)]:
        schema=STRUCTURES[label]
        require(type(doc) is dict and set(doc)==set(schema),'newhost '+label+' schema fields')
        require(doc['schema_version']==VERSIONS[label],'newhost schema version')
        for key,kind in schema.items():require(type(doc[key]) is kind,'newhost field type: '+key)


def roles(plan):
    value=plan['cpu_proposal'];require(set(value)=={'driver','workers','observer'},'role CPU schema')
    require(len(value['workers'])==plan['requested_workers'],'worker CPU count')
    lists=[value['driver'],*value['workers'],value['observer']]
    require(all(type(x) is list and len(x)==1 and type(x[0]) is int and x[0]>=0 for x in lists),'single physical-core role CPUs')
    flat=[x[0] for x in lists];require(len(set(flat))==len(flat),'role CPUs overlap')
    return flat


def authorize(plan,authorization,review, *, explicit_launch):
    """Pure permission checks precede filesystem, affinity, output and science."""
    require(explicit_launch is True,'explicit signal/compile launch required')
    structural(plan,authorization,review)
    require(plan['sealed'] and authorization['approved'] and authorization['runtime_authorization'] and
            review['approved'] and review['runtime_authorization'],'unsealed/unapproved newhost launch')
    require(all(d['stage']=='signal_compile' and d['run_id']==RUN_ID and d['source_commit']==plan['source_commit']
                for d in (plan,authorization,review)),'newhost stage/run/source')
    require(re.fullmatch('[0-9a-f]{40}',plan['source_commit']) is not None,'actual source SHA')
    require(authorization['one_shot'] and authorization['result_prior'] and authorization['permission']=='signal_compile' and
            authorization['environment_accepted'] and review['mandatory_stop'] and bool(review['reviewer'].strip()),
            'separate one-shot environment/final review')
    p=fingerprint('h4-newhost-plan-v2',plan)
    require(authorization['plan_fingerprint']==p and review['plan_fingerprint']==p and
            review['authorization_digest']==fingerprint('h4-newhost-authorization-v2',authorization),'newhost plan/auth/review binding')
    require(plan['carry']==CARRY,'carry cannot be reset or refunded')
    require(plan['contract_plan_fingerprint']==gates.PLAN_FP,'science contract unchanged')
    require(set(plan['inputs'])==set(gates.DISTANCES),'six inputs')
    for distance,entry in plan['inputs'].items():
        require(set(entry)=={'file','bytes_sha256','input','H','DF','state'} and entry['file']=='input-'+distance+'.npz','input schema')
        for field in ('bytes_sha256','input','H','DF','state'):hash_id(entry[field])
    hash_id(plan['generation_freeze_digest']);hash_id(plan['environment_fingerprint']);hash_id(plan['compiler_fingerprint'])
    require(type(plan['requested_workers']) is int and 1<=plan['requested_workers']<=12,'up to12 workers')
    require(type(authorization['allowed_cpus']) is list and sorted(authorization['allowed_cpus'])==sorted(roles(plan)),
            'CPU proposal is not permission; exact separate role approval required')
    caps=plan['caps']
    require(set(caps)=={'actual_invocations','wall_seconds','output_bytes','driver_AS_RSS','worker_AS_RSS','headroom','monitor_seconds','observer_AS','observer_RSS'},'cap schema')
    fixed=dict(wall_seconds=WALL_CAP,output_bytes=OUTPUT_CAP,driver_AS_RSS=ROLE_CAP,worker_AS_RSS=ROLE_CAP,
               headroom=HEADROOM,monitor_seconds=5,observer_AS=AS_CAP,observer_RSS=RSS_CAP)
    require(all(type(caps[k]) is int and caps[k]==v for k,v in fixed.items()),'existing caps / observer candidates')
    role=authorization['observer_role']
    require(role==dict(approved=True,runtime_authorization=True,AS_bytes=AS_CAP,RSS_bytes=RSS_CAP),'observer extra role needs approval')
    counts=static_invocations(plan['templates'])
    cap=caps['actual_invocations'];require(type(cap) is int and cap in (74784,74804),'no arbitrary invocation cap')
    change=authorization['budget_amendment']
    require(set(change)=={'approved','from','to','authority_reference'},'budget amendment schema')
    if cap!=74784:
        require(change['approved'] is True and change['from']==74784 and change['to']==74804 and
                isinstance(change['authority_reference'],str) and bool(change['authority_reference'].strip()),'explicit +20 contract amendment')
    require(cap>=counts['cumulative_actual_worst_case'],'74764 remaining cannot guarantee all74784 logical wrappers; +20 amendment required')
    for key in ('source_root','input_root','stop_evidence_root','output_root','control_root'):
        path=Path(plan[key]);require(path.is_absolute() and path.is_relative_to('/home/AbeHiromu') and '..' not in path.parts,'home-local binding')
    require(plan['output_root']!=plan['input_root'] and RUN_ID in Path(plan['output_root']).parts and RUN_ID in Path(plan['control_root']).parts,
            'fresh dedicated run output/control')
    require(bool(plan['source_hashes']),'source closure missing')
    for path,value in plan['source_hashes'].items():
        require(type(path) is str and not Path(path).is_absolute() and '..' not in Path(path).parts,'relative source closure')
        hash_id(value)
    return BoundPermit('signal_compile',plan,plan['source_root'],fingerprint('h4-newhost-review-v2',review),authorization,review,os.getpid())


@dataclass(frozen=True)
class BoundPermit(gates.Permit):
    authorization:dict
    review:dict
    driver_pid:int
    launch_observation:object=None


def reference(root,entry):
    require(set(entry)=={'path','sha256'},'profile/source reference schema')
    rel=Path(entry['path']);require(not rel.is_absolute() and '..' not in rel.parts,'relative fixed artifact reference')
    p=private_path(root/rel);data=p.read_bytes();require(sha(data)==entry['sha256'],'profile/audit byte binding')
    return json.loads(data)


def verify_runtime(permit):
    plan=permit.plan
    authorize(plan,permit.authorization,permit.review,explicit_launch=True)
    root=private_path(permit.source_root)
    require(Path(__file__).absolute()==root/'src/trottertracks/resource_applicability/h4_geometry/launch_binding.py','loaded newhost source root')
    audit=reference(root,plan['source_audit'])
    require(audit['source_commit']==plan['source_commit'] and audit['source_hashes']==plan['source_hashes'],'source closure audit binding')
    for path,expected in plan['source_hashes'].items():
        require(sha(gates.git_blob(root,plan['source_commit'],path))==expected and sha((root/path).read_bytes())==expected,'source blob/checkout mismatch')
    live={str(p.relative_to(root)) for p in (root/'src/trottertracks/resource_applicability/h4_geometry').glob('*.py')}
    require(live<=set(plan['source_hashes']),'unlisted runtime module')
    require(sys.flags.safe_path and sys.dont_write_bytecode and all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()),'Python -P -B / process thread limits')
    env=reference(root,plan['environment_profile']);comp=reference(root,plan['compiler_profile'])
    require(sys.executable==env['python'] and environment_profile(env['dependencies'],env['installed_sources'])==env,'candidate environment changed')
    require(compiler_profile(comp['explicit_options'],comp['inherited_defaults'])==comp,'candidate compiler profile changed')
    require(env['fingerprint']==plan['environment_fingerprint'] and comp['fingerprint']==plan['compiler_fingerprint'],'plan profile fingerprints')
    contract=gates.verify_contract(root)
    require(plan['templates']==contract['templates'] and comp['explicit_options']==contract['compiler_environment_reference']['compiler']['explicit_options'],
            'scientific templates/compiler options unchanged')
    return contract,comp['explicit_options']


def verify_frozen_receipts(plan):
    evidence=reference(Path(plan['source_root']),plan['stop_evidence_receipt'])
    require(evidence['control_complete'] is True and evidence['all_old_owned_processes_ended'] is True,
            'native stop/control proof not received')
    for row in evidence['control_files']:
        require(receipt_inventory(plan['stop_evidence_root'],{row['file']:row['sha256']})['complete'],'control proof file missing')
    require(bool(evidence['control_files']),'native stop/control file inventory missing')
    expected={entry['file']:entry['bytes_sha256'] for entry in plan['inputs'].values()}
    expected['generation-freeze.json']=EXPECTED_FREEZE
    incoming=receipt_inventory(plan['input_root'],expected)
    require(incoming['complete'],'frozen inputs/freeze not received')
    stop=receipt_inventory(plan['stop_evidence_root'],{'byte-budget.journal':EXPECTED_JOURNAL,'runner.log':EXPECTED_LOG})
    require(stop['complete'],'stopped predecessor journal/log not received')
    root=private_path(plan['input_root']);freeze=json.loads((root/'generation-freeze.json').read_bytes())
    require(fingerprint('h4-generation-freeze-v1',freeze)==plan['generation_freeze_digest'] and freeze['inputs']==plan['inputs'] and
            freeze['source_commit']=='049e69919af16ad29a67a217dc7a407d6b1754a6' and freeze['stage']=='INPUTS_FROZEN_STOP' and
            freeze['mandatory_stop'] is True,'original generation lineage')
    data=(private_path(plan['stop_evidence_root'])/'byte-budget.journal').read_bytes()
    require(len(data)%128==0 and sum(int(data[i:i+128].strip()) for i in range(0,len(data),128))==CARRY['charged_bytes'],
            'predecessor byte carry proof')
    return {**freeze,'input_root':str(root),'consumed_seconds':CARRY['wall_seconds'],
            'prior_charge':CARRY['charged_bytes'],'prior_invocations':CARRY['actual_invocations']}


def fresh_gate(plan,observation, *, now=None):
    now=time.monotonic() if now is None else now
    require(0<=now-observation['observed_monotonic']<=5,'fresh launch observation')
    memory=observation['memory'];require(memory['psi_full_avg10']==0,'launch memory pressure')
    require(0<=now-memory['observed_at']<=5,'fresh memory sample')
    require(memory['available']>=(8+8*plan['requested_workers']+16)*GiB+AS_CAP,'observer-inclusive admission')
    needed=set(roles(plan));require(needed<=set(observation['scheduler_affinity']) & set(observation['online_cpus']),'role CPUs unavailable')
    topology={x['cpu']:(x['package'],x['core']) for x in observation['topology']}
    require(len({topology[c] for c in needed})==len(needed),'distinct physical role cores')
    load={x['cpu']:x['busy_fraction'] for x in observation['topology']}
    require(all(load[c] is not None and 0<=load[c]<=0.20 for c in needed),'fresh selected-core passive load exceeds20% or missing')
    fs=observation['filesystem'];storage=plan['storage']
    require(fs['block_bytes']==storage['block_bytes'],'filesystem allocation block changed')
    require(fs['available_bytes']>=storage['required_bytes'] and fs['available_inodes']>=storage['required_inodes'],'launch capacity/inodes')
    quota=observation['quota'];require(quota['status']=='KNOWN','quota not proven; no filesystem-only fallback')
    for row in quota['items']:
        if row['status']=='ACTIVE':
            require((row['available_bytes'] is None or row['available_bytes']>=storage['required_bytes']) and
                    (row['available_inodes'] is None or row['available_inodes']>=storage['required_inodes']),'launch quota capacity')
    require(storage['cumulative_charge_bound']<=OUTPUT_CAP,'cumulative output budget including observer/control')


def role_affinity(permit,role,index=None):
    """Own-process only, unreachable for false draft flags. Tests mock syscall."""
    authorize(permit.plan,permit.authorization,permit.review,explicit_launch=True)
    require(role in ('driver','worker'),'closed CPU role')
    cp=permit.plan['cpu_proposal']
    if role=='worker':
        require(type(index) is int and 0<=index<permit.plan['requested_workers'] and os.getppid()==permit.driver_pid,'worker parent/index')
        mask=set(cp['workers'][index])
    else:require(os.getpid()==permit.driver_pid,'driver ownership');mask=set(cp['driver'])
    require(mask<=set(permit.authorization['allowed_cpus']),'role permission')
    os.sched_setaffinity(0,mask)
    require(set(os.sched_getaffinity(0))==mask,'own role affinity did not bind')
    # No /tmp probing even when dill/Qiskit first query tempfile in a worker.
    tempfile.tempdir=str(private_path(Path(permit.plan['control_root'])/'private-temp'))
    if role=='worker':install_write_guard(permit.plan,worker=True)


def write_guard(plan,event,args, *, worker=False):
    """Prevent package cache/temp writes and any shared-directory mutation."""
    if event not in ('open','os.mkdir','os.remove','os.rename','os.rmdir'):return
    if not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    writing=event!='open'
    if event=='open':
        mode=args[1] or '';flags=args[2] or 0
        writing=bool((isinstance(mode,str) and any(x in mode for x in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    if not writing:return
    paths=args[:2] if event=='os.rename' else args[:1]
    for raw in paths:
        path=Path(os.fsdecode(raw))
        if not path.is_absolute():
            from .resources import MANAGED_WRITE_CONTEXT
            managed=getattr(MANAGED_WRITE_CONTEXT,'root',None)
            require(managed is not None and len(path.parts)==1 and '..' not in path.parts,'unbound relative write forbidden')
            path=Path(managed)/path
        if path==Path('/dev/null'):continue
        require(not worker,'unbudgeted worker disk/cache/temp write forbidden')
        output=Path(plan['output_root']);control=Path(plan['control_root'])
        require(path.is_relative_to(output) or path.is_relative_to(control),'shared/outside run write forbidden')
        require(not path.is_relative_to(control/'private-temp'),'unbudgeted compiler temporary file forbidden')


def install_write_guard(plan, *, worker=False):
    sys.addaudithook(lambda event,args:write_guard(plan,event,args,worker=worker))


def claim_once(permit):
    """No auto retry: an existing marker survives every failed/complete attempt."""
    authorize(permit.plan,permit.authorization,permit.review,explicit_launch=True)
    root=private_path(permit.plan['control_root'])
    require(not private_path(permit.plan['output_root']).exists(),'existing output forbids retry/resume')
    root.mkdir(parents=True,exist_ok=True,mode=0o700);fsync_directory(root.parent)
    fd=os.open(root/'one-shot.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        data=json.dumps(dict(run_id=RUN_ID,source=permit.plan['source_commit'],review=permit.review_digest,
                             driver_pid=os.getpid(),uid=os.getuid(),started_monotonic=time.monotonic())).encode()
        require(os.write(fd,data)==len(data),'one-shot write');os.fsync(fd)
    finally:os.close(fd)
    fsync_directory(root)


class CappedDriverLog(io.TextIOBase):
    def __init__(self,path,cap=CONTROL_LOG_CAP):
        self.fd=os.open(private_path(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        self.cap,self.bytes=cap,0
    def write(self,value):
        data=value.encode();require(self.bytes+len(data)<=self.cap,'driver control log cap')
        require(os.write(self.fd,data)==len(data),'driver log write');os.fsync(self.fd);self.bytes+=len(data)
        return len(value)
    def flush(self):
        if self.fd is not None:os.fsync(self.fd)
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
        super().close()


def launch(plan,authorization,review, *, explicit_launch=False):
    permit=authorize(plan,authorization,review,explicit_launch=explicit_launch)
    started=time.monotonic()
    _contract,options=verify_runtime(permit)
    verify_frozen_receipts(plan)  # bytes+metadata only before science stage
    observation=host_readonly(private_path(plan['output_root']).parent,sample_seconds=3)
    fresh_gate(plan,observation)
    permit=replace(permit,launch_observation=observation)
    claim_once(permit)
    temp=private_path(Path(plan['control_root'])/'private-temp');temp.mkdir(mode=0o700)
    os.environ['TMPDIR']=str(temp)  # this owned process and its children only
    from .resources import OutputBudget
    trace_cap=(72*3600+2)*FRAME_CAP+TERMINAL_RESERVE
    budget=OutputBudget(plan['output_root'],prior_charge=CARRY['charged_bytes'],file_limits=FILE_LIMITS)
    try:
        # Reserve all control/observer expenses before log/observer startup.
        budget.reserve(CONTROL_LOG_CAP+65536);budget.reserve(trace_cap+FRAME_CAP)
        role_affinity(permit,'driver')
        install_write_guard(plan)
        from .execution import signal_stage
        with CappedDriverLog(Path(plan['control_root'])/'runner.log') as log,redirect_stdout(log),redirect_stderr(log):
            try:
                result=signal_stage(permit,authorization,options,launch_started=started,prepared_budget=budget)
            except BaseException as exc:
                budget.write('launch-stop.json',json.dumps({'status':'FAIL_CLOSED_STOP','reason':type(exc).__name__+': '+str(exc)[:1024],
                    'consumed_seconds':CARRY['wall_seconds']+time.monotonic()-started,'automatic_retry':False}).encode())
                raise
            budget.write('launch-stop.json',json.dumps({'status':result['status'],'consumed_seconds':CARRY['wall_seconds']+time.monotonic()-started,
                'automatic_retry':False}).encode())
            return result
    finally:budget.close()
