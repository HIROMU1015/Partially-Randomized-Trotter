"""Read-only H2 original evidence, alternative contract checks and preservation.

No scientific module imports, circuits, lowering, compilation, simulation,
sampling, vector decoding, cost estimation, allowances or ranking are allowed.
All output is technical metadata in a fresh namespace, never raw/event/config.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import h2_pr_identity_recovery as r
import h2_pr_saved_qpy as qpy
import audit_h2_pr_original_identity as original
import h2_pr_stream_conformance as conformance
import h2_pr_equivalence_contract as contract
import h2_pr_runtime_relevance as relevance

ROOT=r.ROOT
BASE=contract.BASE
PRIOR='artifacts/resource_applicability/track_a_h2_pr_original_identity/2026-10-23/'
DEFAULT=ROOT/'artifacts/resource_applicability/track_a_h2_pr_equivalence_contract/2026-10-23'
FILES=('scripts/h2_pr_stream_conformance.py','scripts/h2_pr_equivalence_contract.py',
 'scripts/h2_pr_runtime_relevance.py','scripts/audit_h2_pr_equivalence.py',
 'tests/test_h2_pr_equivalence_contract.py','scripts/h2_pr_saved_qpy.py',
 'scripts/h2_pr_identity_recovery.py','scripts/audit_h2_pr_original_identity.py',
 'scripts/h2_accepted_compare_contract.py','scripts/h2_common_production_contract.py',
 'scripts/h2_accepted_diagnostics.py','scripts/h2_diagnostics_binding.py',
 'src/trotterlib/h2_common_native_policy.py')


def digest(raw):return hashlib.sha256(raw).hexdigest()


def read(path):return r.loads(Path(path).read_bytes())


def write(out,name,value):
    with (out/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def watch_set(catalogue,old,root,compare):
    watched={}
    for request in catalogue.requests():watched[str(r.resolve(root,request['locator']))]=request['locator']
    for item in catalogue.origins.values():
        if any(x in ('old_consumed_marker_hash_only','old_grant_hash_only_never_published',
                     'existing_target_ledger_hash_only') for x in item['roles']):
            watched[str(r.resolve(root,item['locator']))]=item['locator']
    for item in old['sealed_files']:watched[item['path']]='ORIGINAL_SEALED:'+item['role']
    for key in ('log','exit','entered'):watched[old['supervisor'][key]]='ORIGINAL_SUPERVISOR:'+key
    origin=Path(compare['accepted_diagnostics']['run_root'])
    watched[str(origin/'results/preparation_1.json')]='ACCEPTED_DIAGNOSTICS:preparation_1'
    watched[str(origin/'completion.json')]='ACCEPTED_DIAGNOSTICS:completion'
    for item in compare['files']:watched[item['path']]='COMPARE_SEAL:'+item.get('role','input')
    return watched


def observe(paths):
    def entry(path):
        p=Path(path)
        return dict(bytes=p.stat().st_size,sha256=r.file_digest(p)) if p.is_file() else None
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        return dict(zip(paths,pool.map(entry,paths)))


def audit(config,out,source):
    r.require(out.resolve()==DEFAULT.resolve(), 'ONLY_FRESH_ALTERNATIVE_NAMESPACE')
    out.mkdir(exist_ok=True,parents=True)
    r.require(not any(p.name != 'attempts' for p in out.iterdir()), 'NEW_EVIDENCE_DIRECTORY_REQUIRED')
    source=subprocess.check_output(['git','rev-parse',source],cwd=ROOT,text=True).strip()
    r.require(len(source)==40,'IMMUTABLE_SOURCE_COMMIT')
    for rel in FILES:r.require((ROOT/rel).read_bytes()==r.git_bytes(source,rel),'AUDIT_SOURCE_BYTES')
    catalogue=r.Catalogue();old=catalogue.verify_config(config);execution=config.parents[2]
    compare=r.loads(r.git_bytes(r.BASE,r.PREVIOUS+'COMPARE_contract_BLOCKED.json'))
    r.require(r.digest(compare)==contract.OLD_DIGEST and compare['source_commit']==contract.OLD_SOURCE,
              'OLD_STRICT_CONTRACT_UNCHANGED')
    paths=sorted(watch_set(catalogue,old,execution,compare));before=observe(paths)
    previous=r.loads(r.git_bytes(BASE,PRIOR+'preservation_audit.json'))
    r.require(len(paths)==previous['checked_files']==16804,'PROTECTED_WATCH_SCOPE')
    for path,row in zip(paths,previous['files']):
        r.require(before[path]==row['after'],'PROTECTED_EVIDENCE_CHANGED_SINCE_PREVIOUS_AUDIT')
    recovery=catalogue.recover(execution)
    counts=dict(Counter(x['status'] for x in recovery['recovered']))
    r.require(counts==dict(ORIGINAL_BYTES_MATCH=2085,ARCHIVE_AND_RAW_BYTES_MATCH=402,SAVED_EVENT_CONTENT_MATCH=67),
              'ORIGINAL_RECOVERY_RECHECK_FAILED')
    prep=read(Path(compare['accepted_diagnostics']['run_root'])/'results/preparation_1.json')
    old_axes=r.loads(r.git_bytes(BASE,PRIOR+'axis_identity_134.json'))['rows']
    old_axes={(x['task'],x['axis']):x for x in old_axes}
    storage={x['task_id']:x for x in catalogue.storage}
    def one_task(anchor):
        task=anchor['task'];tid=task['id'];folder=Path(old['output'])/tid
        event=read(folder/'events_PRIVATE.json');result=read(folder/'result.json')
        r.check_events(event,anchor,[x['saved']['rte_steps'] for x in catalogue.schedule[tid]['pieces']])
        current_diag=read(Path(compare['accepted_diagnostics']['run_root'])/'results'/(task['candidate']['id']+'_diagnostics.json'))
        r.require(result['input_binding']==old['binding'] and result['scalar']==current_diag['scalar'] and
                  result['preparation_hash']==prep['preparation_sha256'],'ORIGINAL_TO_COMMON_BINDING')
        rows=[];static=[]
        for index,axis in enumerate(('cosine','sine')):
            old_axis=old_axes[(tid,axis)]
            manifests={x['manifest']['saved']['archive_file']:x['manifest']['saved']
                       for x in storage[tid]['axes'][index]['archives']}
            wrapper=original.original_qpy(folder/('quarantine_'+axis)/'wrapper.qpy.zlib',
                                           manifests['wrapper.qpy.zlib']['codec_receipt'])
            cost=read(folder/('cost_'+axis+'.json'))
            r.require(cost==result['axis_costs'][index] and cost['compiled_SHA']==old_axis['compiled_raw_sha256'],
                      'ORIGINAL_COST_RESULT_RAW_BINDING')
            compiled=original.original_qpy(folder/('quarantine_'+axis)/'compiled.qpy.zlib',
                                            manifests['compiled.qpy.zlib']['codec_receipt'])
            guard=read(folder/('compiler_phase_guard_'+str(index)+'.json'))
            r.require(guard==compiled['metadata']['H2_phase_guard'] and guard['unchanged_gate_stream'] is True and
                      guard['state_fit'] is False,'ORIGINAL_COMPILER_GUARD')
            check_path=folder/('compiled_vector_'+axis+'.json');check=read(check_path)
            r.require(check in result['checks'],'ORIGINAL_CORRECTNESS_RECEIPT_JOIN')
            correctness=dict(scope='SINGLE_SAVED_INPUT_STATE_CT01',strictly_below=check['strictly_below'],
                PASS=check['PASS'],phase_alignment=check['phase_alignment'],receipt_sha256=r.file_digest(check_path))
            contract.validate_correctness(correctness)
            try:
                structural=conformance.recognize(wrapper,event['events'],prep,axis,result['scalar'])
            except r.EvidenceError as error:
                structural=dict(status='MISMATCH',reason=str(error),conditions={x:'UNRESOLVED' for x in contract.A[2:-1]})
                structural['conditions']['complete_instruction_coverage']='MISMATCH'
            packet=dict(task=tid,candidate=task['candidate']['id'],sample=task['sample'],axis=axis,
                original_result_sha256=r.file_digest(folder/'result.json'),
                wrapper_raw_sha256=manifests['wrapper.qpy.zlib']['codec_receipt']['original']['sha256'],
                compiled_raw_sha256=cost['compiled_SHA'],events_sha256=r.file_digest(folder/'events_PRIVATE.json'),
                events_digest=event['event_digest'],cost_receipt_sha256=r.file_digest(folder/('cost_'+axis+'.json')),
                guard_receipt_sha256=r.file_digest(folder/('compiler_phase_guard_'+str(index)+'.json')),
                structural=structural,paired_evolution_sha256=original.wrapper_fields(wrapper,axis,result['scalar'])['evolution_structure_sha256'],
                historical_correctness=correctness,
                strict_contract_status='BLOCKED',alternative_policy_adopted=False,science_authorized=False)
            proof=r.digest(packet)
            statuses={name:'VERIFIED' for name in contract.A+contract.B}
            statuses.update(structural['conditions'])
            statuses['compiler_relevant_runtime_output_equivalence']='UNRESOLVED'
            statuses.update({name:'NOT_APPLICABLE' for name in contract.OUTSIDE})
            row=dict(task=tid,candidate=task['candidate']['id'],sample=task['sample'],axis=axis,
                cohort='EXPLORATION',result_sha256=packet['original_result_sha256'],correctness=correctness,
                conditions={name:dict(status=statuses[name],evidence_sha256=proof,
                    scope=(contract.condition_spec(name)['acceptance']+'; canonical JSON evidence packet digest'))
                    for name in contract.A+contract.B+contract.OUTSIDE})
            rows.append(row);static.append(packet)
        return rows,static
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        joined=list(pool.map(one_task,catalogue.tasks))
    rows=[x for part in joined for x in part[0]];packets=[x for part in joined for x in part[1]]
    for a,b in zip(packets[::2],packets[1::2]):
        r.require(a['task']==b['task'],'PAIRED_TASK')
        conformance.validate_pair(a['structural'],b['structural'],a['paired_evolution_sha256'],b['paired_evolution_sha256'])
    draft=contract.draft();preflight=contract.preflight(draft,rows)
    runtime=relevance.investigate(old,compare)
    # Reuse fixed SOURCE joins; confirm their actual files, do not assume names.
    joins=r.loads(r.git_bytes(BASE,'artifacts/resource_applicability/track_a_h2_pr_identity_recovery/2026-10-23/source_byte_joins.json'))
    for item in joins['compare_dependencies']:
        root=Path(compare[item['root']+'_root'])
        r.require(r.file_digest(root/item['path'])==item['historical_sha256']==item['current_sha256'],
                  'RELEVANT_SCIENTIFIC_SOURCE_CHANGED')
    old_command=r.loads(r.git_bytes(BASE,PRIOR+'COMPARE_preflight.json'))['command']
    env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    completed=subprocess.run(old_command,env=env,capture_output=True,text=True)
    r.require(completed.returncode==0 and r.loads(completed.stdout)['status']=='BLOCKED_PR_IDENTITY',
              'STRICT_PREFLIGHT_MUST_REMAIN_BLOCKED')
    after=observe(paths);r.require(before==after,'PROTECTED_SOURCE_INPUT_RESULT_OR_AUTHORITY_CHANGED')
    r.require(not any(k.split('.')[0] in ('numpy','qiskit','scipy','trotterlib','trottertracks') for k in sys.modules),
              'UNAUTHORIZED_SCIENTIFIC_IMPORT')
    summary={name:dict(Counter(row['conditions'][name]['status'] for row in rows)) for name in contract.A+contract.B+contract.OUTSIDE}
    write(out,'alternative_contract_DRAFT.json',draft)
    write(out,'axis_evidence_134.json',dict(rows=rows,strict_contract_accepted_axes=0,alternative_formally_accepted_axes=0,
        condition_counts=summary,technical_scope='Static source/serialized field conformance; A and B separate',new_scientific_calls=0))
    write(out,'operation_stream_evidence_134.json',dict(rows=packets,
        digest_policy='Canonical JSON sha256 via h2_pr_identity_recovery.digest; each axis condition references its packet digest',
        shape_verified_axes=sum(x['structural']['status']=='VERIFIED' for x in packets),
        unrecognized_axes=[dict(task=x['task'],axis=x['axis']) for x in packets if x['structural']['status']!='VERIFIED'],
        fresh_common_diagonal_parameters_generated=False,new_circuit_generation=0,new_native_lowering=0))
    write(out,'task_evidence_67.json',dict(rows=[dict(task=a[0]['task'],candidate=a[0]['candidate'],sample=a[0]['sample'],
        axes=2,cohort='EXPLORATION',strict_reuse='BLOCKED',alternative_adopted=False,
        remaining=['absolute deterministic common-preparation parameter binding','compiler relevant runtime output equivalence']) for a,b in joined]))
    write(out,'runtime_relevance.json',runtime)
    write(out,'SOURCE_joins.json',dict(compare_dependencies=joins['compare_dependencies'],all_16_rechecked=True,
        original_strict_SOURCE=contract.OLD_SOURCE,original_strict_contract_digest=contract.OLD_DIGEST))
    write(out,'alternative_preflight.json',preflight)
    write(out,'strict_preflight_preserved.json',dict(command=old_command,exit_code=completed.returncode,
        result=r.loads(completed.stdout),stderr=completed.stderr,new_scientific_calls=0))
    write(out,'recovery_recheck.json',dict(status='PASS',counts=counts,original_config_sha256=r.file_digest(config),
        reference_previous_full_report=PRIOR+'recovery_verified.json',full_recheck_canonical_digest=r.digest(recovery),
        new_raw_copies=0,events_or_seed_payloads_written=False,scientific_calls=0))
    write(out,'preservation_audit.json',dict(status='PASS',protected_files=len(paths),changes=0,
        previous_reference=PRIOR+'preservation_audit.json',all_match_previous_audit=True,
        before_observations_sha256=r.digest(before),after_observations_sha256=r.digest(after),
        observation_keys_scope='Only original catalogue requests/seals, accepted DIAGNOSTICS and COMPARE files',
        previous_immutable_artifacts_modified=False,source_input_runtime_result_grant_ledger_marker_STOP_changes=0,
        historical_liblzma_drift_remains_historical=True,old_consumed_grants_reused=0,new_grants=0))
    write(out,'SOURCE.json',dict(base_commit=BASE,audit_SOURCE=source,
        files=[dict(path=p,sha256=r.file_digest(ROOT/p)) for p in FILES],
        original_COMPARE_SOURCE=contract.OLD_SOURCE,original_contract_digest=contract.OLD_DIGEST,
        original_evidence_base_unchanged=True,scientific_SOURCE_modified=False))
    write(out,'STOP.json',dict(status='MANDATORY_STOP_AFTER_ALTERNATIVE_CONTRACT_TECHNICAL_AUDIT',
        alternative_adopted=False,strict_contract_changed=False,strict_reuse_accepted_axes=0,
        alternative_formal_reuse_accepted_axes=0,execution_ready=False,science_authorized=False,
        new_sampling=0,new_signal=0,new_simulation=0,new_H2_circuit_build=0,new_native_lowering=0,
        compile=0,transpile=0,new_u_N_G=0,new_cost_or_ranking=0,scientific_imports=0,live_grants=0,
        detailed_scientific_review=False,final_scientific_evidence_freeze=False,push=False,mandatory_STOP=True))
    with (out/'axis_conditions_134.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['task','candidate','sample','axis','strict','alternative_adopted',*contract.A,*contract.B,*contract.OUTSIDE])
        for row in rows:writer.writerow([row['task'],row['candidate'],row['sample'],row['axis'],'BLOCKED',False,
                                         *(row['conditions'][n]['status'] for n in contract.A+contract.B+contract.OUTSIDE)])
    print(json.dumps(dict(status=preflight['status'],axes=len(rows),condition_counts=summary,
        protected_files=len(paths),science=0,strict_accepted=0,alternative_adopted=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--source',required=True);args=parser.parse_args()
    audit(args.config.resolve(strict=True),DEFAULT,args.source)
