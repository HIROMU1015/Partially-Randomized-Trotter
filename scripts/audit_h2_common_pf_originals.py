"""Bounded, read-only recovery of two common Full-PF candidates (stdlib only).

Only static QPY data parsing and receipt/byte joins are allowed. Original files,
scientific routines, grants and shared environments are never modified/invoked.
The caller supplies private locators; none are embedded in this source/output.
"""
import argparse
import ast
import base64
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import zlib

import h2_pr_identity_recovery as r
import h2_pr_saved_qpy as qpy

BASE = 'ca29dd9ddc8a09cace84310f0dfdf7124652e07a'
PF = 'f36402f90bcfcbcc4d556c7a952837dc5d7dd291'
SCIENCE = '1b35f9ac2efab4c15bc00ee2d3c81643aca422f9'
GPU01 = 'e0c942fbb5dd8741dda69af0933fce88e8eaea91'
INTEGRATED = '6dafed86cc82149d55b4e6395382953d5f3631d7'
PREFIX = 'artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/'
B_PREFIX = 'artifacts/resource_applicability/track_a_h2_runtime_cost_reuse/2026-10-23/'
OUT = 'artifacts/resource_applicability/track_a_h2_common_pf_original_recovery/2026-10-23'
TARGETS = {'H2_FULL_3_4_2nd': ('2nd', 4, 1655), 'H2_FULL_3_2_4th': ('4th', 2, 2463)}
REUSED = ('scripts/h2_pr_identity_recovery.py', 'scripts/h2_pr_saved_qpy.py')
THIS = 'scripts/audit_h2_common_pf_originals.py'


def identity(raw):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                calculated_git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def verify_gzip(path, receipt):
    """Minimal gzip adaptation of r.verify_archive (which accepts zlib only).

    Use unchanged r.verified_bytes for stored SHA/size; reject excess/trailing
    payload. Never persist the uncompressed body or interpret NPZ arrays.
    """
    blob = r.verified_bytes(path, dict(bytes=receipt['compressed_bytes'], sha256=receipt['archive_sha256']))
    expected = receipt['original_bytes']
    r.require(type(expected) is int and expected >= 0, 'RAW_SIZE_TYPE')
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = decoder.decompress(blob, expected + 1)
    r.require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, 'GZIP_INCOMPLETE_OR_TRAILING')
    r.require(len(raw) == expected and hashlib.sha256(raw).hexdigest() == receipt['original_sha256'], 'RAW_SHA_OR_SIZE')
    r.require(r.file_digest(path) == receipt['archive_sha256'], 'ARCHIVE_CHANGED_DURING_READ')
    return raw


def candidate_id(value):
    return '_'.join(str(value[k]) for k in ('system', 'method', 'prefix', 'q', 'formula'))


def selected(values, cid):
    matches = [x for x in values if candidate_id(x['candidate']) == cid]
    r.require(len(matches) == 1, 'UNIQUE_TARGET_CANDIDATE')
    return matches[0]


def safe_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    r.require(not re.search(r'(?:/home/|/usr/|/tmp/|[A-Za-z]:\\)', raw), 'PRIVATE_ABSOLUTE_PATH_IN_PROJECTION')
    return raw


def run(args):
    root = Path(__file__).resolve().parents[1]
    original = args.original_root.resolve(strict=True)
    private = args.private_audit_dir.resolve(strict=True)
    output = root / OUT
    r.require(not output.exists() and not output.resolve().is_relative_to(original), 'NEW_SEPARATE_OUTPUT_ONLY')
    r.require(not private.is_relative_to(root), 'PRIVATE_LOCATORS_OUTSIDE_WORKTREE')
    for path in REUSED:
        r.require((root/path).read_bytes() == r.git_bytes(BASE, path), 'UNCHANGED_REUSED_SOURCE')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    r.require((root/THIS).read_bytes() == r.git_bytes(source, THIS), 'COMMITTED_AUDIT_SOURCE_REQUIRED')
    fixed = {}
    def git(ref, path):
        raw = r.git_bytes(ref, path)
        fixed[ref+':'+path] = dict(commit=ref, path=path, **identity(raw), acquisition='LOCAL_FIXED_GIT_OBJECT')
        return raw
    def doc(ref, name):
        return r.loads(git(ref, PREFIX+name))
    plan_path = original/'plan_APPROVED.json'
    plan = r.loads(plan_path.read_bytes())
    projection = doc(PF, 'execution_contract_APPROVED_projection.json')
    r.require(plan['SOURCE'] == SCIENCE and plan['system'] == 'H2', 'EXACT_SCIENTIFIC_SOURCE')
    r.require(r.file_digest(plan_path) == projection['private_original_plan_sha256'], 'ORIGINAL_PLAN_SHA')
    stripped = [{k:v for k,v in item.items() if k != 'path'} for item in plan['closure']]
    r.require(stripped == projection['closure'], 'PLAN_CLOSURE_PROJECTION')
    r.require(all(plan[k] == v for k,v in projection.items() if k in plan and k != 'closure'), 'PLAN_PROJECTION_FIELDS')
    r.require(sorted(plan['input_paths']) == projection['input_roles'], 'PLAN_INPUT_ROLES')
    binding = doc(PF, 'source_input_identity.json')
    r.require(binding['binding'] == plan['binding'], 'SCIENTIFIC_INPUT_BINDING')
    r.require(plan['binding']['rank'] == 3 and plan['binding']['rank_policy'] == 'tol_only_1e-8' and
              plan['T'] == .8 and plan['P'] == 0, 'FIXED_SCIENTIFIC_SCOPE')

    # Exactly this known run root + its explicit sealed dependencies. No disk crawl.
    watched = {str(p): 'ORIGINAL_RUN' for p in original.rglob('*') if p.is_file()}
    watched.update({e['path']: e['role'] for e in plan['closure']})
    watched[str(args.instructions.resolve(strict=True))] = 'OPERATING_INSTRUCTIONS_20261023'
    def observe():
        return {p: dict(sha256=r.file_digest(p), bytes=Path(p).stat().st_size) for p in sorted(watched)}
    before = observe()
    r.require(not (private/'preservation_before_PRIVATE.json').exists(), 'NEW_PRIVATE_AUDIT_ONLY')
    (private/'preservation_before_PRIVATE.json').write_text(json.dumps(before, sort_keys=True, indent=2)+'\n')
    records = []
    def original_record(path, token, role, expected=None, origin=None):
        raw = path.read_bytes(); ident = identity(raw)
        if expected is not None:
            r.require(ident['sha256'] == expected['sha256'] and ident['bytes'] == expected['bytes'], 'ORIGINAL_FILE_IDENTITY:'+role)
        row = dict(locator=token, role=role, status='原本byteまで確認済み', **ident,
                   fixed_Git_origin=origin, calculated_blob_is_not_proof_of_Git_availability=True)
        records.append(row)
        return row
    plan_rec = original_record(plan_path, '${PF_ORIGINAL}/plan_APPROVED.json', 'original_execution_plan',
        dict(sha256=projection['private_original_plan_sha256'], bytes=plan_path.stat().st_size),
        dict(commit=PF, path=PREFIX+'execution_contract_APPROVED_projection.json', json_pointer='/private_original_plan_sha256'))
    inputs = []
    for key, path in plan['input_paths'].items():
        seals = [e for e in plan['closure'] if e['role'] == 'scientific_input:'+key]
        r.require(len(seals) == 1 and seals[0]['path'] == path, 'UNIQUE_INPUT_PATH_SEAL')
        inputs.append(original_record(Path(path), '${INPUT}/'+key, 'scientific_input:'+key, seals[0]))
    preparation = doc(PF, 'input_reference_receipt.json')
    r.require(r.loads(Path(plan['input_paths']['reference_receipt']).read_bytes()) == preparation, 'ORIGINAL_PREPARATION_RECEIPT')
    regeneration = doc(PF, 'input_regeneration_comparison.json')

    # Path binding stays private. Public row IDs retain unambiguous order/role/SHA.
    closure_rows = []
    source_functions = []
    provider = Path(plan['provider']).resolve(strict=True)
    for index, entry in enumerate(plan['closure']):
        path = Path(entry['path']); current = before[str(path)]
        r.require(current == dict(sha256=entry['sha256'], bytes=entry['bytes']), 'SEALED_CURRENT_BYTE_IDENTITY:'+entry['role'])
        logical = '${SEALED}/'+str(index)
        git_binding = None
        if entry['role'].startswith('SOURCE:'):
            rel = entry['role'][7:]; logical = '${SCIENCE_SOURCE}/'+rel
            raw = git(SCIENCE, rel)
            r.require(raw == path.read_bytes(), 'SCIENTIFIC_SOURCE_GIT_BYTES')
            git_binding = dict(commit=SCIENCE, path=rel, blob_sha1=identity(raw)['calculated_git_blob_sha1'])
        elif entry['role'] == 'pinned_provider':
            logical = '${PROVIDER}/'+path.resolve().relative_to(provider).as_posix()
        elif entry['role'] in ('fixed_python_binary', 'loaded_runtime_identity', 'compiler_array_runtime_binary'):
            # basename plus ordinal is unique; real prefix mapping remains private.
            logical = '${RUNTIME}/'+str(index)+'/'+path.name
        closure_rows.append(dict(index=index, role=entry['role'], logical_path=logical, **current,
                                 status='原本byteまで確認済み', source_binding=git_binding,
                                 historical_actual_load_proven=False))
    current_roles = dict(Counter(x['role'].split(':')[0] for x in closure_rows))
    # Read AST only. Never import scientific source, compiler, guard or auditor.
    refs = [(SCIENCE, 'scripts/pf_common_run_native.py', ('install','restore','cost_one')),
            (SCIENCE, 'scripts/h2_m5_compiler_phase_guard.py', ('transpile','scalar_guard')),
            (SCIENCE, 'scripts/pf_common_run_contract.py', ('verify',)),
            (SCIENCE, 'scripts/audit_pf_common_run.py', ('audit',))]
    for ref, path, names in refs:
        raw = git(ref,path); tree = ast.parse(raw)
        for name in names:
            node = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name)
            source_functions.append(dict(commit=ref,path=path,function=name,line=node.lineno,
                file_sha256=hashlib.sha256(raw).hexdigest(),
                AST_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
                use='STATIC_SOURCE_REFERENCE_ONLY',scientific_invocations=0))
    # Compiler is from the pinned provider, not assumed to be this worktree's library.
    for rel, names in [('src/trotterlib/rte_compiled_cost.py', ('transpile_and_measure_cost','_gate_counts'))]:
        path = provider/rel; raw = path.read_bytes(); tree = ast.parse(raw)
        seals = [e for e in plan['closure'] if Path(e['path']) == path]
        r.require(seals and all(e['sha256'] == hashlib.sha256(raw).hexdigest() for e in seals), 'COMPILER_PROVIDER_SEAL')
        for name in names:
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == name)
            source_functions.append(dict(commit=None,path='${PROVIDER}/'+rel,function=name,line=node.lineno,
                file_sha256=hashlib.sha256(raw).hexdigest(),use='STATIC_SOURCE_REFERENCE_ONLY',
                scientific_invocations=0,seal_role='pinned_provider'))

    native = r.loads(gzip.decompress(git(PF,PREFIX+'native_results.json.gz')))
    inventory = doc(PF,'native_result_inventory.json')
    archive_inventory = doc(PF,'raw_archive_inventory.json')
    saved_b = r.loads(git(BASE,B_PREFIX+'saved_common_PF_cost_objects_4.json'))['rows']
    old_hashes = r.loads(git(BASE,B_PREFIX+'original_PF_result_hashes.json'))
    # Preserve B's runtime/correctness policy; this recovery does not adopt it.
    for name in ('runtime_seal_comparison.json','same_runtime_comparability.json','compiler_reuse_plan.json','REPORT.md'):
        git(BASE,B_PREFIX+name)
    endpoints = list(csv.DictReader(io.StringIO(git(PF,PREFIX+'PF_endpoint_costs.csv').decode())))
    rows = []
    archive_rows = []
    for cid, (formula,q,rz) in TARGETS.items():
        local_path = original/'COST/output'/cid/'result.json'
        packet = selected(native,cid); actual = r.loads(local_path.read_bytes())
        r.require(actual == packet, 'FIXED_NATIVE_PACKET_EQUALS_ORIGINAL_RESULT')
        target_inventory = selected(inventory,cid)
        result_rec = original_record(local_path,'${PF_ORIGINAL}/COST/output/'+cid+'/result.json',
            'native_result:'+cid,dict(bytes=local_path.stat().st_size,sha256=target_inventory['result_sha256']),
            dict(commit=PF,path=PREFIX+'native_result_inventory.json',json_pointer='/'+str(inventory.index(target_inventory))))
        r.require(packet['SOURCE'] == SCIENCE and packet['status'] == 'PASS' and packet['retry'] is False, 'ORIGINAL_NATIVE_RECEIPT')
        expected_candidate=dict(system='H2',method='FULL',prefix=3,q=q,formula=formula)
        r.require(packet['candidate'] == expected_candidate and expected_candidate in plan['rows'], 'PLAN_CANDIDATE_BINDING')
        r.require(packet['hamiltonian'] == plan['binding']['hamiltonian'] and packet['state'] == plan['binding']['state'] and
                  packet['preparation_hash'] == preparation['full_preparation'], 'H_STATE_PREPARATION_JOIN')
        parsed={}
        for rec in packet['archives']:
            counterparts=[a for a in archive_inventory if a['candidate']==cid and a['path']==rec['path']]
            r.require(len(counterparts)==1 and {k:v for k,v in counterparts[0].items() if k!='candidate'}==rec,'ARCHIVE_FIXED_INVENTORY')
            path=local_path.parent/rec['path'];raw=verify_gzip(path,rec)
            static = None
            if rec['path'].endswith('.qpy.gz'):
                circuit=qpy.read_qpy(raw);s=qpy.summary(circuit)
                static={k:s[k] for k in ('nq','nc','instruction_count','custom_definitions','stream_sha256','qpy_version','circuit_object_created','scientific_libraries_imported')}
                parsed[rec['path']]=circuit
            archive_rows.append(dict(candidate=cid,locator='${PF_ORIGINAL}/COST/output/'+cid+'/'+rec['path'],
                status='原本byteまで確認済み',archive=dict(sha256=rec['archive_sha256'],bytes=rec['compressed_bytes'],
                    calculated_git_blob_sha1=identity(path.read_bytes())['calculated_git_blob_sha1']),
                raw=dict(sha256=rec['original_sha256'],bytes=rec['original_bytes'],
                    calculated_git_blob_sha1=identity(raw)['calculated_git_blob_sha1']),
                fixed_receipt=dict(commit=PF,path=PREFIX+'raw_archive_inventory.json',json_pointer='/'+str(archive_inventory.index(counterparts[0]))),
                raw_written=False,NPZ_payload_decoded=False,QPY_static=static,
                raw_or_archive_fixed_Git_commit=None,calculated_blob_is_not_proof_of_Git_availability=True))
        for axis in ('cosine','sine'):
            b=next(x for x in saved_b if x['saved_candidate']==cid and x['axis']==axis)
            price=packet['prices'][axis];wrapper=parsed[axis+'_wrapper.qpy.gz'];compiled=parsed[axis+'_compiled.qpy.gz']
            r.require(price['metrics']['rz_count']==b['RZ']==rz,'SAVED_RZ_RECEIPT_JOIN')
            eps=[e for e in endpoints if e['id']==cid and e['epsilon']=='0.0005']
            r.require(len(eps)==1 and eps[0][axis+'_RZ']==str(rz),'FIXED_CSV_SAVED_PRICE')
            r.require(price['compiler']==plan['compiler'] and price['preparation_hash']==packet['preparation_hash'],'COMPILER_PREPARATION')
            r.require(compiled['metadata']['H2_phase_guard']==price['phase_guard']==b['guard'],'SAVED_QPY_GUARD_METADATA')
            guard=price['phase_guard']
            r.require(guard['policy']==plan['H2_phase_policy'] and guard['unchanged_gate_stream'] and not guard['state_fit'] and
                      not guard['raw_validation_threshold_changed'] and all(x['unchanged_gate_stream'] and
                      not x['input_or_output_state_fit'] and not x['postcompile_phase_correction'] for x in guard['passes']),'GUARD_SCOPE')
            r.require(wrapper['nq']==compiled['nq']==5 and wrapper['nc']==compiled['nc']==1,'WRAPPER_REGISTERS')
            ins=wrapper['instructions'];last=ins[-1]
            r.require(ins[0]['name']=='HGate' and ins[0]['wires']==[4] and ins[-2]['name']=='HGate' and
                      ins[-2]['wires']==[4] and last['name']=='Measure' and last['wires']==[4] and last['clbits']==[0], 'HADAMARD_WRAPPER_WIRES')
            if axis=='sine':r.require(ins[-3]['name']=='SdgGate' and ins[-3]['wires']==[4],'SINE_AXIS')
            check=next(x for x in packet['checks'] if x['check']=='compiled_vector_'+axis)
            r.require(check==b['correctness'] and check['PASS'] is True and check['strictly_below']==1e-7 and check['phase_alignment'] is False,'SAVED_CORRECTNESS_SCOPE')
            # All checks are historical field joins, no vector/error recomputation.
            r.require(all(x['PASS'] is True and x['phase_alignment'] is False for x in packet['checks']),'HISTORICAL_CHECK_RECEIPTS')
            archives={x['path']:x for x in packet['archives']}
            r.require(archives[axis+'_wrapper.qpy.gz']==b['wrapper'] and archives[axis+'_compiled.qpy.gz']==b['compiled'],'B_ARCHIVE_BINDING')
            rows.append(dict(candidate=cid,candidate_fields=expected_candidate,axis=axis,
                status='原本byteまで確認済み',original_plan_sha256=plan_rec['sha256'],SOURCE=SCIENCE,
                native_result_sha256=result_rec['sha256'],original_result_inventory_join=True,
                B_original_result_hash_reference=dict(commit=BASE,path=B_PREFIX+'original_PF_result_hashes.json'),
                execution_input_container_sha256=next(x['sha256'] for x in inputs if x['role']=='scientific_input:snapshot'),
                historical_binding_snapshot_sha256=plan['binding']['snapshot'],
                original_historical_snapshot_restored=False,preparation_sha256=packet['preparation_hash'],
                H=packet['hamiltonian'],state=packet['state'],scalar=packet['scalar'],
                native_policy=packet['native_policy'],compiler_hash=price['compiler'],
                compiler_settings_reference=dict(commit=BASE,path=B_PREFIX+'saved_common_PF_cost_objects_4.json',json_pointer='/rows/'+str(saved_b.index(b))+'/compiler_settings'),
                phase_guard_receipt_SHA256=r.digest(guard),phase_guard_QPY_metadata_identical=True,
                saved_RZ_per_axis=rz,cost_recomputed=False,unit='compiled_RZ_per_full_Hadamard_wrapper_axis',
                wrapper_archive=axis+'_wrapper.qpy.gz',compiled_archive=axis+'_compiled.qpy.gz',vectors_archive=axis+'_vectors.npz.gz',
                correctness=dict(receipt=check,scope='SINGLE_SAVED_INPUT_STATE',all_state_or_operator_proof=False,
                    numerical_check_reexecuted=False),
                fixed_native_receipt=dict(commit=PF,path=PREFIX+'native_results.json.gz',json_pointer='/'+str(native.index(packet))),
                wrapper_axis_wire_check=True,complete_current_common_native_stream_binding='NOT_PERFORMED_BY_THIS_RECOVERY'))

    stage_receipts=doc(PF,'stage_receipts.json');exit_saved=doc(PF,'worker_exit_receipt.json');end=doc(PF,'end_audit.json')
    original_record(original/'COST/worker_exit_receipt.json','${PF_ORIGINAL}/COST/worker_exit_receipt.json','worker_exit_receipt',
        dict(sha256=hashlib.sha256(git(PF,PREFIX+'worker_exit_receipt.json')).hexdigest(),bytes=len(git(PF,PREFIX+'worker_exit_receipt.json'))))
    r.require(r.loads((original/'COST/worker_exit_receipt.json').read_bytes())==exit_saved and exit_saved['same_parent_join'] and
              exit_saved['remaining_owned_workers']==0 and all(w['exit_code']==0 and w['reaped'] for w in exit_saved['workers']),'HISTORICAL_WORKER_END_AUDIT')
    for stage in ('SIGNAL','QUALIFY','COST','ACCOUNT'):
        r.require(r.loads((original/stage/'completion.json').read_bytes())==stage_receipts[stage],'STAGE_RECEIPT_JOIN')
        grant=r.loads((original/stage/'grant.json').read_bytes());marker=r.loads((original/stage/'CONSUMED.json').read_bytes())
        r.require(grant['SOURCE']==SCIENCE and grant['plan_sha256']==plan_rec['sha256'] and marker['grant']==grant,'PLAN_GRANT_MARKER_BINDING')
    for cid in TARGETS:
        packet=selected(native,cid)
        r.require(any(w['pid']==packet['pid'] and w['exit_code']==0 and w['reaped'] for w in exit_saved['workers']),'TARGET_PID_REAP_JOIN')
    r.require(end['status']=='PASS_MANDATORY_STOP' and end['all_native_workers_exit0_reaped'],'FIXED_END_AUDIT')
    oldstop=r.loads((original/'MANDATORY_STOP.json').read_bytes())
    r.require(oldstop['status']=='MANDATORY_STOP' and oldstop['local_result_commit']==PF and oldstop['next_stage_started'] is False,'ORIGINAL_STOP')

    remote=[]
    for item in r.loads((private/'remote_cache_PRIVATE.json').read_bytes()):
        request=item['request'];data=item['data'];encoding=data['encoding']
        raw=base64.b64decode(data['content']) if encoding=='base64' else data['content'].encode()
        local=git(request['ref'],request['path'])
        r.require(raw==local and identity(raw)['calculated_git_blob_sha1']==data['sha'],'GITHUB_FULL_BYTES_VS_LOCAL_BLOB')
        remote.append(dict(commit=request['ref'],path=request['path'],blob_sha1=data['sha'],**identity(raw),
            acquisition='GITHUB_FULL_BASE64' if encoding=='base64' else 'GITHUB_FULL_UTF8',status='固定GitHub原本まで確認済み'))
    # Verify the requested catalogues actually select the same saved PF series.
    catalogue=r.loads(git(GPU01,'docs/research/environment_evidence_gpua100_20261023/series_manifest.json'))
    series=next(x for x in catalogue['series'] if x['series_id']=='PF_COMMON_H2')
    r.require(series['commit']==PF,'GPU01_FIXED_INDEX')
    integrated=list(csv.DictReader(io.StringIO(git(INTEGRATED,'docs/data_collection/2026-10-23_three_environment_integration/research_index.csv').decode())))
    cross=next(x for x in integrated if x['series_key']=='GPU01:PF_COMMON_H2')
    r.require(PF in cross.values(),'THREE_ENVIRONMENT_FIXED_INDEX')

    after=observe();r.require(before==after,'PRESERVATION_BYTES_CHANGED')
    (private/'preservation_after_PRIVATE.json').write_text(json.dumps(after,sort_keys=True,indent=2)+'\n')
    locators=dict(original_root=str(original),provider=str(provider),audit_worktree=str(root),
                  inputs=plan['input_paths'],closure=[dict(index=i,**e) for i,e in enumerate(plan['closure'])])
    (private/'locator_map_PRIVATE.json').write_text(json.dumps(locators,sort_keys=True,indent=2)+'\n')
    r.require(not any(k.split('.')[0] in ('numpy','qiskit','scipy','symengine','trotterlib','trottertracks') for k in sys.modules),'NO_SCIENTIFIC_IMPORTS')
    reuse=[dict(commit=BASE,path=p,sha256=r.file_digest(root/p),modified=False,
                used_functions=['file_digest','verified_bytes','loads','digest','git_bytes'] if 'identity_recovery' in p else ['read_qpy','summary'],
                actual_use='READ_ONLY_BYTE_OR_STATIC_DATA_PARSER') for p in REUSED]
    reuse.extend(source_functions)
    instructions=identity(args.instructions.read_bytes())
    result={
      'SOURCE.json':dict(audit_SOURCE=source,base_commit=BASE,scientific_SOURCE=SCIENCE,PF_result_commit=PF,
          own_source=dict(path=THIS,**identity((root/THIS).read_bytes())),reused=reuse,
          operating_instructions=dict(path='docs/research/research_project_instructions_unified_2026-10-23.md',
              **instructions,version='2026-10-23',sections_applied=['8.4','10'],read_from_supplied_main_worktree=True),
          science_source_modified=False),
      'original_identity.json':dict(plan=plan_rec,inputs=inputs,other_originals=records[len(inputs)+1:],
          preparation_receipt_reference=dict(commit=PF,path=PREFIX+'input_reference_receipt.json'),
          full_preparation_sha256=preparation['full_preparation'],historical_snapshot_container_restored=False,
          regenerated_input_arrays_equivalence='HISTORICAL_RECEIPT_ONLY_NO_ARRAY_RECOMPUTATION',
          original_result_hashes_B_referenced_not_rewritten=True),
      'axis_identity_4.json':dict(rows=rows,science_calls=0,formal_adoption=0,COMPARE=0),
      'archive_identity_12.json':dict(rows=archive_rows,archives_checked=len(archive_rows),QPY_static_parsed=8,
          vector_payload_decoded=False,raw_files_written=0,private_payloads_copied=0),
      'runtime_identity.json':dict(closure=closure_rows,role_counts=current_roles,declared_paths_to_bytes_verified=True,
          all_declared_current_seals_match=True,closure_projection_matches_original_plan=True,
          fixed_python_sha256=next(x['sha256'] for x in closure_rows if x['role']=='fixed_python_binary'),
          previous_B_comparison=dict(commit=BASE,path=B_PREFIX+'runtime_seal_comparison.json'),
          historical_actual_load_complete=False,transitive_dependency_closure_complete=False,
          historical_load_receipt='未発見',current_rehash_is_not_historical_load=True,
          execution_hostname='NOT_INFERRED_FROM_CUSTODY',custody_observed=socket.gethostname()),
      'fixed_sources.json':dict(files=list(fixed.values()),remote=remote,remote_files=len(remote),
          duplicate_existing_evidence_registered=False,private_originals_not_in_fixed_Git=True),
      'preservation_audit.json':dict(status='PASS',protected_files=len(before),changes=0,
          before_sha256=r.digest(before),after_sha256=r.digest(after),
          scope='Known original H2 run root and explicit plan closure, supplied operating instructions',
          original_grants_reused=0,new_grants=0,shared_environment_changes=0,
          private_locator_map_kept_outside_worktree=True,
          resource_observation=dict(cpu_count=os.cpu_count(),affinity=len(os.sched_getaffinity(0)),
              loadavg=list(os.getloadavg()),mem_available_kB=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),
              arbitrary_caps=None,execution='Small bounded files; cached fixed blobs, static parsing; no scientific workers or GPU jobs')),
      'STOP.json':dict(status='MANDATORY_STOP',scientific_calls=0,compile=0,transpile=0,simulation=0,
          setup=0,prepare=0,wrapper_generation=0,sampling=0,new_costs=0,allowance=0,accounting=0,ranking=0,
          COMPARE=0,formal_adoption=0,new_grants=0,old_grants_reused=0,push=False,
          next_stage_started=False,private_transfer_executed=False,mandatory_STOP=True),
    }
    output.mkdir(parents=True)
    for name,value in result.items():
        with (output/name).open('x') as f:f.write(safe_json(value))
    print(json.dumps(dict(status='PASS_STATIC_ORIGINAL_RECOVERY',axes=len(rows),archives=len(archive_rows),
        closure_entries=len(closure_rows),protected_files=len(before),changes=0,scientific_calls=0,mandatory_STOP=True)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--private-audit-dir',type=Path,required=True)
    parser.add_argument('--instructions',type=Path,required=True)
    run(parser.parse_args())
