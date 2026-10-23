"""Metadata/hash/NPY-header preparation only. Never issue grants or run science."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import h2_accepted_compare_contract as c
import h2_accepted_diagnostics as accepted
import h2_diagnostics_binding as b

ROOT=Path(__file__).resolve().parents[1]
ORIGIN_ROOT=ROOT.parent/'h2-diagnostics-binding-20261023'
ORIGIN_ASSETS=ROOT.parent/'h2-diagnostics-binding-assets-20261023'
ASSETS=ROOT.parent/'h2-accepted-compare-assets-20261023'
OUT=ROOT/'artifacts/resource_applicability/track_a_h2_accepted_compare/2026-10-23'
RECEIVED=ROOT/'artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/received'


def evidence_audit(plan):
    previous=ROOT/'artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23'
    rows=b.read(previous/'PR_C_task_identity.json')['rows']
    runtime=b.read(RECEIVED/'runtime_provider_manifest.json')
    python=[r['saved'] for r in runtime['historical_sealed_files'] if r['saved']['role']=='fixed_python_runtime']
    result=[];newprep=plan['preparations'][1]
    for row in rows:
        cid=row['candidate'];diag=plan['records'][cid];saved_pieces=row['schedule']['pieces'];new_pieces=diag['schedule']
        same_schedule=len(saved_pieces)==len(new_pieces) and all(
          x['saved']['signed_step_time']==y['signed_time'] and x['saved']['rte_steps']==y['rte_steps']
          for x,y in zip(saved_pieces,new_pieces))
        result.append(dict(task=row['task'],axes=row['axes'],saved_projection_sha256=row['projection_sha256'],
            saved_event_digest=row['event_digest'],events_summary=row['events_summary'],
            fields=dict(hamiltonian_state_snapshot='MATCH_SAVED_METADATA' if all(row['input_binding'][k]==c.old.BINDING[k] for k in c.old.BINDING) else 'IDENTITY_MISMATCH',
              preparation='MATCH_HASH_ONLY' if row['preparation_hash']==newprep['preparation_sha256'] else 'IDENTITY_MISMATCH',
              signed_piece_schedule='MATCH_SAVED_METADATA' if same_schedule else 'IDENTITY_MISMATCH',
              event_sequence='UNKNOWN_FULL_BODY_MISSING',rotation_and_operation_stream='UNKNOWN_HISTORICAL_STREAM_MISSING',
              scalar_wrapper_lowering='UNKNOWN_COMPLETE_STREAM_JOIN_MISSING',
              compiler_settings='MATCH_SETTINGS_HASH_ONLY',
              runtime='DECLARED_INTERPRETER_BYTE_MISMATCH_ACTUAL_LOAD_UNKNOWN',
              correctness='SAVED_STATE_LIMITED_ACCEPTANCE_NOT_NEW_IDENTITY',
              cohort_and_paired_RZ='MATCH_SAVED_RECEIPT_PROJECTION'),
            new_common_preparation=dict(preparation_sha256=newprep['preparation_sha256'],operations_sha256=newprep['operations_sha256']),
            reuse_status='BLOCKED'))
    b.write(OUT/'PR_identity_67_tasks_134_axes.json',dict(status='BLOCKED',tasks=67,axes=134,
        candidate_counts=dict(Counter(r['task']['candidate']['id'] for r in result)),rows=result,
        saved_python=python,accepted_python=dict(target=plan['config']['runtime']['python_target'],sha256=plan['config']['runtime']['python_sha256']),
        runtime_note='Declared executable bytes differ. No per-worker actual historical load list; no claim of identical runtime or automatic invalidation of all comparisons.',
        new_stream_witness_available=True,historical_stream_witness_available=False,new_RZ_u_N_G_evaluated=False))
    retrieval=b.read(previous/'retrieval_audit.json')
    b.write(OUT/'locator_audit.json',dict(status='MISSING_CANONICAL_ROOT',canonical_NEW_EXECUTION=None,
        originals=retrieval['required_originals'],known_git_exporter=retrieval['exporter'],
        searches=[dict(scope='fixed received origin_inventory/acquisition_context and previous retrieval audit',result='tokenized locators only'),
          dict(scope='fixed exporter source and h2_outer32 runtime/worker Git objects',result='CLI-supplied paths; no hardcoded execution root'),
          dict(scope='known old H2 private 2026-10-19 root, immediate *map* filenames only',result='no token map found'),
          dict(scope='known metadata receipt/common-policy docs and source files',result='no canonical NEW_EXECUTION or sender map locator')],
        event_digest_rule=dict(source=c.old.ACQUISITION,path='scripts/h46_pr_pilot_native.py',
          definition='canonical JSON digest of nested ordered [piece][RTEEvent.to_dict()] list',
          source_lines='66-69',scientific_source_imported=False),
        broad_private_search=False,private_originals_published=False,
        required_owner_information='Exact canonical path to O00002 or sender private root-token map. O00001 may resolve pipeline input/storage locators.'))


def build(source):
    ASSETS.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    oldcfg=b.read(ORIGIN_ASSETS/'DIAGNOSTICS_contract.json')
    ref=dict(contract=str(ORIGIN_ASSETS/'DIAGNOSTICS_contract.json'),run_root=oldcfg['run_root'],
        completion_sha256=accepted.ORIGIN_COMPLETION,
        approval=str(ORIGIN_ASSETS/'DIAGNOSTICS_authorization01/approval.json'),
        grant=str(ORIGIN_ASSETS/'DIAGNOSTICS_authorization01/grant.json'),
        source_roots=dict(runner=str(ORIGIN_ROOT),acquisition=oldcfg['acquisition_root'],provider=oldcfg['provider_root']))
    plan=accepted.inspect_origin(ref)
    source_rows=[r for r in oldcfg['source_files'] if r['root']!='runner']
    for role,commit in (('acquisition',c.old.ACQUISITION),('provider',c.old.PROVIDER)):
        subset=[r for r in source_rows if r['root']==role]
        for row,(blob,_) in zip(subset,b.git_batch(commit,[r['path'] for r in subset])):row['git_blob_sha1']=blob
    files={r['path']:r for r in oldcfg['files']}
    for path,(blob,raw) in zip(c.RUNNER_FILES,b.git_batch(source,c.RUNNER_FILES)):
        target=(ROOT/path).resolve();c.require(target.read_bytes()==raw,'COMMITTED_NEW_RUNNER_BYTES')
        h=hashlib.sha256(raw).hexdigest()
        source_rows.append(dict(root='runner',path=path,sha256=h,git_blob_sha1=blob))
        files[str(target)]=dict(path=str(target),sha256=h,role='SOURCE:accepted_compare_runner')
    for key in ('contract','approval','grant'):
        p=Path(ref[key]);files[str(p)]=dict(path=str(p),sha256=b.sha(p),role='ORIGIN:'+key)
    run=Path(ref['run_root'])
    for p in [run/'completion.json',run/'CONSUMED.json',*sorted((run/'ledger').glob('*.json')),
              *[run/r for r in plan['files']]]:
        files[str(p)]=dict(path=str(p),sha256=b.sha(p),role='ORIGIN:immutable_diagnostics_evidence')
    runtime=dict(oldcfg['runtime']);runtime['cache_prefix']=str(ASSETS/'BYTECODE_CACHE_MUST_REMAIN_ABSENT')
    cfg=dict(schema=c.SCHEMA,series=c.SERIES,branch=c.BRANCH,source_commit=source,
        **{k:oldcfg[k] for k in ('physical','compiler','native_policy','allowance_policy','reference_policy','candidates',
                              'acquisition_root','provider_root','inputs','binding')},
        budget=c.counts(),files=sorted(files.values(),key=lambda r:r['path']),source_files=source_rows,
        runtime=runtime,run_root=str(ASSETS/'COMPARE_run01'),reuse_witnesses=[],retry=False,mandatory_STOP=True,
        accepted_diagnostics=ref)
    contract=ASSETS/'COMPARE_contract_BLOCKED.json';b.write(contract,cfg);b.write(OUT/contract.name,cfg)
    b.write(OUT/'source_index.json',source_rows);b.write(OUT/'call_budget.json',c.counts())
    evidence_audit(plan)
    b.write(OUT/'origin_acceptance.json',dict(status='ACCEPTED_METADATA_AND_NPY_HEADERS_ONLY',
        contract_sha256=accepted.ORIGIN_CONTRACT,completion_sha256=accepted.ORIGIN_COMPLETION,
        headers=plan['headers'],prior_grant_reusable=False,scientific_vectors_decoded=False,
        adaptive_input_schema_valid=True,source_and_runtime_bytes_match=True,original_results_unchanged=True))
    command=[str(b.PYTHON),'-I','-S','-B','-X','pycache_prefix='+runtime['cache_prefix'],
        str(ROOT/'scripts/launch_h2_accepted_compare.py'),'--contract',str(contract)]
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    result=subprocess.run(command,env=env,capture_output=True,text=True)
    b.write(OUT/'preflight.json',dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
        contract_sha256=b.digest(cfg),science_calls=0,scientific_imports=0,live_grants=0,
        execution_authorized=False))
    c.require(result.returncode==0,'NEW_METADATA_PREFLIGHT_ERROR:'+result.stderr[-1800:])
    report=json.loads(result.stdout);c.require(report['status']=='BLOCKED_PR_IDENTITY' and report['execution_ready'] is False,
        'MISSING_PR_WITNESS_MUST_BLOCK')
    b.write(OUT/'resources.json',dict(affinity_cpus=len(os.sched_getaffinity(0)),
        available_ram=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),
        free_disk_bytes=shutil.disk_usage(ROOT).free,arbitrary_resource_caps_added=False,shared_environment_changed=False))
    print(json.dumps(dict(status='TECHNICAL_BINDING_VERIFIED_COMPARE_BLOCKED',source=source,
        contract_sha256=b.digest(cfg),files=len(files),origin_completion=accepted.ORIGIN_COMPLETION,
        science_calls=0,live_grants=0)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True)
    build(parser.parse_args().source_commit)
