"""Sealed PF acquisition support. Metadata only until an explicit scoped grant."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pf_direct_compile_contract as c

ROOT = Path(__file__).resolve().parents[1]
BASE = 'd510d7beda94d293432328a17fcc60d4286dc637'
PRIVATE = ROOT / '.pf_direct_execution_private'
PUBLIC = ROOT / 'artifacts/resource_applicability/track_a_pf_direct_compile_execution/2026-10-20'
SPLIT = ROOT / 'artifacts/resource_applicability/track_a_pf_direct_compile_contract_split/2026-10-20'
PLAN = ROOT / 'artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19'
GROUPS = ('current_4series_722', 'legacy_h4_df12_66')
CODE = ('scripts/pf_direct_execution.py', 'scripts/prepare_pf_direct_execution.py',
        'scripts/pf_direct_production_worker.py', 'scripts/run_pf_direct_execution.py',
        'scripts/audit_pf_direct_execution.py', 'scripts/pf_direct_compile_contract.py',
        'scripts/pf_direct_compile_adapter.py', 'scripts/h8_strengthened_export.py',
        'scripts/h8_strengthened_contract.py', 'tests/test_pf_direct_execution.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _hash(stream)


def _hash(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def resource_snapshot():
    mem = {row.split(':')[0]: int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines() if ':' in row and row.split()[1].isdigit()}
    effective = mem['MemAvailable']; levels=[]
    rel = next(row.split(':',2)[2] for row in Path('/proc/self/cgroup').read_text().splitlines() if row.startswith('0::'))
    p = Path('/sys/fs/cgroup') / rel.lstrip('/')
    cpu_slots = float(len(os.sched_getaffinity(0)))
    while p.is_relative_to('/sys/fs/cgroup'):
        row={'path':str(p)}
        for name in ('memory.max','memory.current','memory.events','cpu.max'):
            q=p/name
            if q.is_file():row[name]=q.read_text().strip()
        if row.get('memory.max','max')!='max':
            effective=min(effective,max(0,int(row['memory.max'])-int(row['memory.current'])))
        if row.get('cpu.max','max').split()[0]!='max':
            quota,period=map(int,row['cpu.max'].split());cpu_slots=min(cpu_slots,quota/period)
        levels.append(row)
        if p==Path('/sys/fs/cgroup'):break
        p=p.parent
    return dict(time_UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        monotonic=time.monotonic(),allowed_cpus=sorted(os.sched_getaffinity(0)),cpu_slots=cpu_slots,
        host_available_memory=mem['MemAvailable'],effective_available_memory=effective,
        disk_free=shutil.disk_usage(ROOT).free,levels=levels,
        PSI=Path('/proc/pressure/memory').read_text(),PSI_is_observation_only=True)


def hash_audit(files):
    def one(item):
        path,expected=item
        try:
            observed=sha(path)
            return None if observed==expected else dict(path=path,error='SHA256_CHANGED',expected=expected,actual=observed)
        except OSError as error:return dict(path=path,error=str(error))
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        errors=[e for e in pool.map(one,files.items()) if e]
    return dict(status='PASS' if not errors else 'STOP',files_checked=len(files),errors=errors)


def expected_tasks(group):
    draft=c.read_json(SPLIT/(group+'_contract_DRAFT.json'))
    proposal=PLAN/'proposed_compile_axes.csv'
    rows=c.parse_proposals(proposal.read_bytes(),draft['proposal_sha256'])
    return rows,c.select_pairs(rows,draft['proposed_task_ids'])


def authorize(spec, grant):
    c.require(spec['contract_id'] in GROUPS and grant.get('action')=='PF_DIRECT_COMPILE_SCIENTIFIC_EXECUTION'
        and grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
        'EXPLICIT_SCIENTIFIC_GRANT_REQUIRED')
    c.require(grant['spec_digest']==c.digest(spec) and grant['SOURCE']==spec['SOURCE']
        and grant['contract_id']==spec['contract_id'] and grant['retry'] is False
        and grant['next_stage_authorized'] is False,'EXACT_GRANT_BINDING')
    rows,selected=expected_tasks(spec['contract_id'])
    c.require(spec['selected_task_ids']==[r['task_id'] for r in selected]
        and spec['call_caps']==c.call_caps(selected),'FIXED_PARTITION_AND_CALLS')
    c.require(spec['excluded_conditional_axes']==48 and spec['sealed'] is True
        and spec['compiler_hash']==c.COMPILER and spec['T']==.8 and spec['P']==0
        and spec['control']=='symmetric_directional','SEALED_FIXED_CONDITIONS')
    approval=Path(grant['approval_receipt'])
    c.require(sha(approval)==grant['approval_sha256'],'USER_APPROVAL_BYTES')
    approved=c.read_json(approval)
    c.require(approved['base_commit']==BASE and approved['scientific_execution_authorized'] is True
        and approved['axes']=={'current_4series_722':722,'legacy_h4_df12_66':66}
        and approved['verbatim_user_message'],'APPROVAL_SCOPE')
    return rows,selected


def gate(spec,grant,*,full=True):
    _,selected=authorize(spec,grant)
    for rel,h in spec['source_hashes'].items():
        c.require(sha(ROOT/rel)==h,'SOURCE_CHANGED:'+rel)
        raw=subprocess.check_output(['git','show',spec['SOURCE']+':'+rel],cwd=ROOT)
        c.require(hashlib.sha256(raw).hexdigest()==h,'SOURCE_GIT_BLOB:'+rel)
    audit=hash_audit(spec['bound_files']) if full else dict(status='PASS_DEFERRED_TO_PARENT_FULL_AUDIT')
    c.require(audit['status'].startswith('PASS'),'BOUND_DEPENDENCY_BYTES_CHANGED:'+str(audit.get('errors')))
    f=resource_snapshot()
    c.require(f['cpu_slots']>0 and f['effective_available_memory']>0 and f['disk_free']>0,'NO_PERMITTED_RESOURCES')
    return dict(status='PASS',SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),resource=f,byte_audit=audit)


def candidate_pairs(selected):
    groups={}
    for row in selected: groups.setdefault((row['system'],row['candidate_id']),[]).append(row)
    return groups


def instruction_bound(native, blocks, q, formula):
    # Exact provider's conservative structural bound, not a fixed resource ceiling.
    return q*(1 if formula=='2nd' else 3)*sum(native.block_instruction_bound(b,m)
        for b in blocks for m in ('UNCONTROLLED','DIRECTIONAL'))+1


def choose_admissions(pending, active, fresh):
    """Live admission; footprint reservations exceed observed memory where needed."""
    slots=max(0,int(fresh['cpu_slots'])-len(active))
    reserve=sum(max(0,a['estimate']-a['rss']) for a in active)
    memory=max(0,fresh['effective_available_memory']-reserve)
    accepted=[]
    for task in pending:
        if slots and task['memory_estimate']<=memory:
            accepted.append(task);slots-=1;memory-=task['memory_estimate']
    return accepted
