"""Exact 216-axis retry contract. Stdlib metadata only; no scientific imports."""
import hashlib
from pathlib import Path
import re
import subprocess

import pf_direct_compile_contract as c
import pf_direct_execution as old

ROOT=Path(__file__).resolve().parents[1]
BASE='a80f89b435340898f8cc46940e9080c9bcd3b403'
OLD_SOURCE='220b9644df6e114116737fd041973973fcf86cf5'
OLD_ROOT=ROOT.parent/'pf-direct-compile-execution-20261020'
OLD_PRIVATE=OLD_ROOT/'.pf_direct_execution_private'
PRIVATE=ROOT/'.pf_direct_retry216_private'
PUBLIC=ROOT/'artifacts/resource_applicability/track_a_pf_direct_compile_retry216/2026-10-20'
OLD_PACKET='artifacts/resource_applicability/track_a_pf_direct_compile_interrupted_audit/2026-10-20'
GROUPS=('current_remaining_216',)
STATUS_PATH=OLD_PACKET+'/axis_status.csv'
SOURCE_FILES=('scripts/pf_direct_retry_contract.py','scripts/pf_direct_retry_support.py',
 'scripts/prepare_pf_direct_retry216.py','scripts/pf_direct_retry_worker.py','scripts/run_pf_direct_retry216.py',
 'scripts/observe_pf_direct_retry216.py','scripts/launch_pf_direct_retry216.py','scripts/audit_pf_direct_retry216.py',
 'scripts/run_pf_direct_execution.py','scripts/pf_direct_production_worker.py','scripts/pf_direct_execution.py',
 'scripts/pf_direct_compile_contract.py','scripts/pf_direct_compile_adapter.py',
 'scripts/h8_strengthened_export.py','scripts/h8_strengthened_contract.py',
 'tests/test_pf_direct_retry216.py')
ADMISSION_POLICY=dict(policy='LIVE_PERMITTED_RESOURCES_AND_MATCHED_CONFIRMED_FOOTPRINT',
 anchor='previous SOURCE H8 B0I p17/q8/S4 wait4 peak 5336395776 bytes',
 estimate='0.75 GiB + twice max(1, structural ratio) times confirmed peak for H8; old estimate for H2/H4',
 CPU_RAM_worker_limits=None,PSI='observation only',no_automatic_retry=True)

save=old.save
append=old.append
sha=old.sha
resource_snapshot=old.resource_snapshot
candidate_pairs=old.candidate_pairs
choose_admissions=old.choose_admissions


def pinned_blob(path):
    raw=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
    c.require((ROOT/path).read_bytes()==raw,'BASE_EVIDENCE_BYTES_CHANGED:'+path)
    return raw


def expected_tasks():
    import csv,io
    statuses=list(csv.DictReader(io.StringIO(pinned_blob(STATUS_PATH).decode())))
    ids=[r['task_id'] for r in statuses if r['status']!='SAVED_TASK_ACCEPTANCE']
    old_counts={label:sum(r['status']==label for r in statuses) for label in
        ('SAVED_TASK_ACCEPTANCE','EXPLICIT_PRECOMPILE_STOP','INTERRUPTED_NATIVE_COMPLETION_UNCONFIRMED')}
    c.require(old_counts==dict(SAVED_TASK_ACCEPTANCE=572,EXPLICIT_PRECOMPILE_STOP=200,
        INTERRUPTED_NATIVE_COMPLETION_UNCONFIRMED=16),'EXACT_PRIOR_STATE')
    proposals,allcurrent=old.expected_tasks('current_4series_722')
    selected=c.select_pairs(proposals,ids)
    c.require(len(selected)==216 and {r['system'] for r in selected}=={'H2_DF3','H4_DF10','H8_DF28'},'ONLY_REMAINING_216')
    return proposals,selected


def authorize(spec,grant):
    c.require(grant.get('action')=='PF_DIRECT_RETRY216_SCIENTIFIC_EXECUTION'
        and grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
        'EXPLICIT_RETRY216_GRANT_REQUIRED')
    c.require(grant.get('spec_digest')==c.digest(spec) and grant.get('SOURCE')==spec.get('SOURCE')
        and spec.get('contract_id')==grant.get('contract_id')==GROUPS[0], 'NEW_EXACT_CONTRACT_BINDING')
    proposals,selected=expected_tasks()
    c.require(spec.get('selected_task_ids')==[r['task_id'] for r in selected]
        and spec.get('call_caps')==c.call_caps(selected),'EXACT_216_SCOPE_AND_CALLS')
    c.require(spec.get('base_commit')==BASE and spec.get('previous_SOURCE')==OLD_SOURCE
        and spec.get('sealed') is True and isinstance(spec.get('SOURCE'),str)
        and re.fullmatch('[0-9a-f]{40}',spec['SOURCE']) is not None,'SEALED_NEW_SOURCE')
    c.require(spec.get('T')==.8 and spec.get('P')==0 and spec.get('control')=='symmetric_directional'
        and spec.get('compiler_hash')==c.COMPILER and spec.get('excluded_conditional_axes')==48,
        'UNCHANGED_SCIENTIFIC_CONDITIONS')
    c.require(grant.get('retry') is False and grant.get('next_stage_authorized') is False
        and spec.get('retry') is False and spec.get('next_stage_authorized') is False,'NO_AUTOMATIC_RETRY_OR_NEXT_STAGE')
    receipt=Path(grant['approval_receipt'])
    c.require(sha(receipt)==grant['approval_sha256'],'NEW_USER_APPROVAL_BYTES')
    approved=c.read_json(receipt)
    c.require(approved['role']=='user' and approved['authorized_axes']==216
        and approved['verbatim_user_message']=='その方針で進めて、再実行まで行って'
        and approved['scientific_execution_authorized'] is True,'NEW_216_SCIENCE_APPROVAL')
    c.require(Path(spec['output_root'])==PRIVATE/'run01'/GROUPS[0],'EXCLUSIVE_NEW_OUTPUT')
    c.require(spec.get('source_hashes') and spec.get('bound_files') and spec.get('reuse_receipt_sha256'), 'SEALED_CLOSURE_REQUIRED')
    c.require(sha(spec['reuse_receipt'])==spec['reuse_receipt_sha256'],'REUSE_RECEIPT_BYTES')
    return proposals,selected


def gate(spec,grant):
    authorize(spec,grant)
    for name,digest in spec['source_hashes'].items():
        c.require(sha(ROOT/name)==digest,'NEW_SOURCE_CHANGED:'+name)
        raw=subprocess.check_output(['git','show',spec['SOURCE']+':'+name],cwd=ROOT)
        c.require(hashlib.sha256(raw).hexdigest()==digest,'SOURCE_GIT_BLOB:'+name)
    audit=old.hash_audit(spec['bound_files'])
    c.require(audit['status']=='PASS','DEPENDENCY_CHANGED:'+str(audit['errors']))
    fresh=resource_snapshot()
    c.require(fresh['cpu_slots']>0 and fresh['effective_available_memory']>0 and fresh['disk_free']>0,'NO_PERMITTED_RESOURCES')
    return dict(status='PASS',SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),byte_audit=audit,resource=fresh)
