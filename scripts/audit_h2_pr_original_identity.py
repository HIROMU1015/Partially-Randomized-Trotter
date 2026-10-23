"""Read original H2 PR evidence and serialized circuits. Scientific calls = 0.

Only stdlib and the fixed metadata recovery helpers are imported. Never build
circuits, sample, simulate, compile, load scientific vectors, issue grants or
evaluate allowance/shots/cost/ranking. Public output contains hashes and joins,
never private configurations, event bodies, seeds or raw archives.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib
import h2_pr_identity_recovery as r
import h2_pr_saved_qpy as qpy

BASE='e7a63b2b06a124f9305483aa39bcf1ba8cafdd0d'
OUT=r.ROOT/'artifacts/resource_applicability/track_a_h2_pr_original_identity/2026-10-23'
TOOL_FILES=('scripts/audit_h2_pr_original_identity.py','scripts/h2_pr_saved_qpy.py',
            'scripts/h2_pr_identity_recovery.py','tests/test_h2_pr_identity_recovery.py',
            'tests/test_h2_pr_original_identity.py')


def write(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def original_qpy(path, expected):
    """Opaque raw stays in memory, is SHA checked, and is never QPY loaded."""
    stored=expected['stored'];rawinfo=expected['original']
    r.verify_archive(path,stored,rawinfo)
    raw=zlib.decompress(Path(path).read_bytes())
    r.require(len(raw)==rawinfo['bytes'] and hashlib.sha256(raw).hexdigest()==rawinfo['sha256'], 'RAW_QPY_SHA')
    return qpy.read_qpy(raw)


def wrapper_fields(circuit, axis, scalar):
    items=circuit['instructions'];r.require(circuit['nq']==5 and circuit['nc']==1,'FIXED_WRAPPER_REGISTERS')
    r.require(items[0]['name']=='HGate' and items[0]['wires']==[4] and
        [x['name'] for x in items[-2:]]==['HGate','Measure'] and
        [x['wires'] for x in items[-2:]]==[[4],[4]] and items[-1]['clbits']==[0], 'HADAMARD_WRAPPER_WIRES')
    end=2
    if axis=='sine':
        r.require(items[-3]['name']=='SdgGate' and items[-3]['wires']==[4],'SINE_WRAPPER')
        end=3
    body=items[1:-end]
    r.require(body[-1]['name']=='PhaseGate' and body[-1]['wires']==[4] and
        body[-1]['params']==[dict(kind='float',hex=float(-scalar*.8).hex())], 'SCALAR_ONCE_FINAL_GATE')
    return dict(status='SAVED_WRAPPER_AND_SCALAR_FIELDS_MATCH',
        evolution_structure_sha256=qpy.structural_digest(circuit,body),
        wrapper_structure_sha256=qpy.structural_digest(circuit),
        scalar_once_source_and_final_gate=True)


def run(config,source):
    catalogue=r.Catalogue();config=Path(config).resolve(strict=True)
    old=catalogue.verify_config(config);root=config.parents[2]
    r.require(config==root/r.CONFIG_SUFFIX,'EXACT_CONFIG_LAYOUT')
    for rel in TOOL_FILES:r.require((r.ROOT/rel).read_bytes()==r.git_bytes(source,rel),'COMMITTED_AUDIT_SOURCE')
    compare=r.loads(r.git_bytes(r.BASE,r.PREVIOUS+'COMPARE_contract_BLOCKED.json'))
    previous=r.loads(r.git_bytes(r.BASE,r.PREVIOUS+'preflight.json'))
    origin_result=Path(compare['accepted_diagnostics']['run_root'])/'results'
    prep_path=origin_result/'preparation_1.json';prep=r.loads(prep_path.read_bytes())
    requests=catalogue.requests();watched={}
    for request in requests:
        path=r.resolve(root,request['locator']);watched[str(path)]=request['locator']
    for item in catalogue.origins.values():
        if any(role in ('old_consumed_marker_hash_only','old_grant_hash_only_never_published','existing_target_ledger_hash_only') for role in item['roles']):
            path=r.resolve(root,item['locator']);watched[str(path)]=item['locator']
    for item in old['sealed_files']:
        watched[item['path']]='ORIGINAL_SEALED:'+item['role']
    for key in ('log','exit','entered'):watched[old['supervisor'][key]]='ORIGINAL_SUPERVISOR:'+key
    original_old_completion=Path(compare['accepted_diagnostics']['run_root'])/'completion.json'
    watched[str(prep_path)]='ACCEPTED_DIAGNOSTICS:preparation_1'
    watched[str(original_old_completion)]='ACCEPTED_DIAGNOSTICS:completion'
    for path in compare['files']:watched[path['path']]='COMPARE_SEAL:'+path.get('role','input')
    def observe(path):
        p=Path(path)
        return dict(bytes=p.stat().st_size,sha256=r.file_digest(p)) if p.is_file() else None
    paths=sorted(watched)
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        before=dict(zip(paths,pool.map(observe,paths)))
    recovery=catalogue.recover(root)
    r.require(all(x['status'] in ('ORIGINAL_BYTES_MATCH','SAVED_EVENT_CONTENT_MATCH','ARCHIVE_AND_RAW_BYTES_MATCH')
                  for x in recovery['recovered']),'ORIGINAL_EVIDENCE_NOT_COMPLETE')
    OUT.mkdir(parents=True,exist_ok=True)
    write('recovery_verified.json',recovery)
    storage={row['task_id']:row for row in catalogue.storage}
    def task_join(anchor):
        task=anchor['task'];tid=task['id'];folder=Path(old['output'])/tid
        events=r.loads((folder/'events_PRIVATE.json').read_bytes())
        result=r.loads((folder/'result.json').read_bytes())
        diag=r.loads((origin_result/(task['candidate']['id']+'_diagnostics.json')).read_bytes())
        r.require(result['scalar']==diag['scalar'] and result['input_binding']==old['binding'] and
                  result['preparation_hash']==prep['preparation_sha256'],'H2_INPUT_PREP_SCALAR_BINDING')
        basis=qpy.event_basis_join(events['events'],prep)
        eventrow=dict(task=tid,candidate=task['candidate']['id'],sample=task['sample'],
            events_file_sha256=r.file_digest(folder/'events_PRIVATE.json'),event_digest=events['event_digest'],
            piece_counts=[p['saved']['rte_steps'] for p in catalogue.schedule[tid]['pieces']],
            event_order_component_phase='CONTENT_DIGEST_AND_SCHEMA_MATCH',event_body_published=False,
            registered_basis_vs_common=basis)
        axisrows=[]
        for index,axis in enumerate(('cosine','sine')):
            public=storage[tid]['axes'][index]
            manifests={x['manifest']['saved']['archive_file']:x['manifest']['saved'] for x in public['archives']}
            folder_axis=folder/('quarantine_'+axis)
            w=original_qpy(folder_axis/'wrapper.qpy.zlib',manifests['wrapper.qpy.zlib']['codec_receipt'])
            c=original_qpy(folder_axis/'compiled.qpy.zlib',manifests['compiled.qpy.zlib']['codec_receipt'])
            cost=r.loads((folder/('cost_'+axis+'.json')).read_bytes())
            r.require(cost==result['axis_costs'][index] and
                cost['compiled_SHA']==manifests['compiled.qpy.zlib']['codec_receipt']['original']['sha256'], 'SAVED_COST_RAW_JOIN')
            guard=r.loads((folder/('compiler_phase_guard_'+str(index)+'.json')).read_bytes())
            r.require(c['metadata']['H2_phase_guard']==guard,'COMPILED_QPY_GUARD_METADATA')
            names=Counter(x['name'] for x in c['instructions'])
            r.require(names['RZGate']==cost['metrics']['rz_count'] and names['CXGate']==cost['metrics']['cx_count'] and
                len(c['instructions'])==cost['metrics']['circuit_size'],'SAVED_RZ_CX_SIZE_RECEIPT')
            r.require(set(names)<={'RZGate','SXGate','XGate','CXGate','Measure'},'SAVED_COMPILED_BASIS')
            fields=wrapper_fields(w,axis,result['scalar']);central=qpy.event_central_join(w,events['events'])
            gaussian=qpy.gaussian_matches(w,prep)
            r.require(all(row['forward_exact_occurrences']>0 for row in gaussian[:2]),'DETERMINISTIC_GAUSSIAN_BYTE_FIELDS')
            checks={x['check']:x for x in result['checks']}
            r.require(checks['compiled_vector_'+axis]==r.loads((folder/('compiled_vector_'+axis+'.json')).read_bytes()),
                      'HISTORICAL_CORRECTNESS_RECEIPT_COPY')
            axisrows.append(dict(task=tid,candidate=task['candidate']['id'],sample=task['sample'],axis=axis,
                compiled_raw_sha256=cost['compiled_SHA'],wrapper_raw_sha256=manifests['wrapper.qpy.zlib']['codec_receipt']['original']['sha256'],
                compiled_stream_sha256=qpy.stream_digest(c),event_digest=events['event_digest'],
                cost_receipt_sha256=r.file_digest(folder/('cost_'+axis+'.json')),
                cost_fingerprint=cost['fingerprint'],fingerprint_scope='Original receipt joined to original raw through accepted result/manifests; recursive fingerprint not regenerated',
                saved_RZ_CX_size_fields_match=True,compiled_guard_metadata_match=True,
                event_central_fields=central,gaussian_forward_hex_wire_matches=gaussian,wrapper_fields=fields,
                event_registered_basis_matches_common=basis,
                correctness_scope=cost['status'],historical_correctness_receipt_sha256=r.file_digest(folder/('compiled_vector_'+axis+'.json')),
                historical_gate_tolerance=checks['compiled_vector_'+axis]['strictly_below'],
                numerical_correctness_reevaluated=False,
                full_fresh_common_stream_identity='NOT_ESTABLISHED_WITHOUT_FRESH_STREAM_OR_SEPARATE_EQUIVALENCE_POLICY',
                strict_reuse='BLOCKED'))
        r.require(axisrows[0]['wrapper_fields']['evolution_structure_sha256']==axisrows[1]['wrapper_fields']['evolution_structure_sha256'],
                  'COSINE_SINE_SAME_SAVED_EVOLUTION')
        return eventrow,axisrows
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        joined=list(pool.map(task_join,catalogue.tasks))
    events=[row[0] for row in joined];axes=[axis for row in joined for axis in row[1]]
    write('event_67.json',dict(tasks=events,events=6400,pieces=336,bodies_published=False,new_sampling=0))
    write('axis_identity_134.json',dict(rows=axes,axes=134,saved_artifact_binding_pass=134,
        strict_reuse_accepted_axes=0,strict_reuse_blocked_axes=134,scientific_inequivalence_proven_axes=0,
        saved_versus_common_runtime_condition='UNSATISFIED_NOT_A_SCIENTIFIC_INEQUIVALENCE_PROOF'))
    write('task_identity_67.json',dict(rows=[dict(task=row['task'],candidate=row['candidate'],sample=row['sample'],
        event_digest=row['event_digest'],original_evidence_recovered=True,paired_axes=2,
        saved_circuit_central_join=True,gaussian_shared_forward_fields_match=True,
        fresh_complete_stream_identity=False,actual_per_worker_load_manifest=False,strict_reuse='BLOCKED') for row in events]))
    site=next(Path(p) for p in compare['runtime']['sys_path'] if 'site-packages' in p)
    runtime_rows=[]
    for item in old['sealed_files']:
        path=item['path'];current=before[path]
        row=dict(role=item['role'],saved_sha256=item['sha256'],current_original_matches_saved=current is not None and current['sha256']==item['sha256'])
        if '/site-packages/' in path:
            rel=path.split('/site-packages/',1)[1];new=site/rel
            if rel.startswith(('qiskit/','numpy/','symengine/')):
                row.update(package_relative_path=rel,adopted_same_relative_file_exists=new.is_file(),
                    adopted_same_relative_sha256=r.file_digest(new) if new.is_file() else None)
                row['adopted_bytes_match_saved']=row['adopted_same_relative_sha256']==item['sha256']
        runtime_rows.append(row)
    write('runtime_compiler_identity.json',dict(rows=runtime_rows,
        historical_inventory_scope='Pre-stage config seal inherited provider/runtime and imported modules from template; SOURCE verifies loaded origin/version, but no complete per-H2-worker observed manifest is persisted',
        current_historical_seal_status=dict(Counter('MATCH' if x['current_original_matches_saved'] else 'MISSING_OR_CHANGED' for x in runtime_rows)),
        old_python_sha256=next(x['sha256'] for x in old['sealed_files'] if x['role']=='fixed_python_runtime' and x['path']==old['runtime']['python_target']),
        adopted_python_sha256=compare['runtime']['python_sha256'],python_byte_identity=False,
        full_actual_worker_load_identity=False,version_difference_alone_proves_scientific_inequivalence=False,
        matching_source_or_names_alone_proves_identity=False,new_scientific_imports=0))
    command=previous['command'];env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    completed=subprocess.run(command,env=env,capture_output=True,text=True)
    r.require(completed.returncode==0,'PRESERVED_COMPARE_METADATA_PREFLIGHT')
    report=r.loads(completed.stdout);r.require(report['execution_ready'] is False and report['status']=='BLOCKED_PR_IDENTITY','STRICT_COMPARE_MUST_REMAIN_BLOCKED')
    write('COMPARE_preflight.json',dict(command=command,exit_code=completed.returncode,stdout=completed.stdout,
        stderr=completed.stderr,original_PR_evidence_now_recovered=True,strict_identity_witnesses=0,
        remaining_blockers=['HISTORICAL_VS_ADOPTED_RUNTIME_BYTES_AND_ACTUAL_WORKER_LOAD',
            'FULL_COMMON_OPERATION_STREAM_AND_CORRECTNESS_SCOPE_IDENTITY'],scientific_calls=0,live_grants=0,
        execution_authorized=False,execution_ready=False,existing_contract_unchanged=True))
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        after=dict(zip(paths,pool.map(observe,paths)))
    changed=[watched[p] for p in paths if before[p]!=after[p]]
    r.require(not changed,'ORIGINAL_EVIDENCE_CHANGED')
    write('preservation_audit.json',dict(status='PASS',files=[dict(locator=watched[p],before=before[p],after=after[p]) for p in paths],
        checked_files=len(paths),changes=changed,original_config_sha256=r.file_digest(config),
        DIAGNOSTICS_completion_sha256=r.file_digest(original_old_completion),
        existing_COMPARE_SOURCE_and_results_unchanged=True,grant_ledger_marker_STOP_unchanged=True,
        raw_copies_written=0,raw_archives_or_events_published=False,new_scientific_calls=0,
        new_scientific_imports=0,new_live_grants=0))
    write('SOURCE.json',dict(base_commit=BASE,audit_SOURCE=source,
        files=[dict(path=p,sha256=r.file_digest(r.ROOT/p)) for p in TOOL_FILES],
        unchanged_COMPARE_SOURCE=compare['source_commit'],unchanged_contract_sha256=r.digest(compare)))
    write('STOP.json',dict(status='MANDATORY_STOP_AFTER_ORIGINAL_EVIDENCE_AUDIT',
        strict_reuse_accepted_axes=0,strict_reuse_blocked_axes=134,PR_reuse='BLOCKED',
        execution_ready=False,execution_authorized=False,new_sampling=0,new_simulation=0,new_circuit_generation=0,
        compile=0,new_u_N_G=0,new_cost_or_ranking=0,scientific_imports=0,live_grants=0,
        detailed_scientific_review=False,final_evidence_freeze=False,push=False,mandatory_STOP=True))
    print(json.dumps(dict(status='ORIGINAL_EVIDENCE_RECOVERED_STRICT_COMPARE_BLOCKED',
        recovery_status_counts=dict(Counter(x['status'] for x in recovery['recovered'])),
        saved_axes_joined=len(axes),strict_accepted=0,blocked=134,preserved_files=len(paths),science=0,grants=0)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    args=p.parse_args();run(args.config,args.source)
