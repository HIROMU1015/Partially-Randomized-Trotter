"""Join immutable metadata and the explicitly supplied PR originals; science zero."""
import argparse
import ast
from collections import Counter
import csv
import gzip
import hashlib
import io
from pathlib import Path
import re
import socket
import subprocess
import sys

import h2_saved_cost_static_join_b2 as j
import audit_h2_pr_equivalence as preservation
import audit_h2_common_pf_originals as gpu01

r = j.r
ROOT = r.ROOT
OWN = ('scripts/h2_saved_cost_static_join_b2.py', 'scripts/audit_h2_saved_cost_static_join_b2.py',
       'tests/test_h2_saved_cost_static_join_b2.py')
PROVIDER = '217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
CURRENT_PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
STRICT = '46a3b33e3717c4fadf11846e922caf17d10156d9'
EQUIV = '65085ff7f818fd92af3424b1980f6bd9dbf1c742'
EQUIV_DIR = 'artifacts/resource_applicability/track_a_h2_pr_equivalence_contract/2026-10-23/'


def audit(config, source):
    out = ROOT/j.OUT_DIR
    r.require({p.name for p in out.iterdir()} <= {'cache','test_results.json','technical_tests.txt'}, 'FRESH_METADATA_OUTPUT')
    source = subprocess.check_output(['git','rev-parse',source], cwd=ROOT, text=True).strip()
    for path in OWN:
        r.require((ROOT/path).read_bytes() == r.git_bytes(source,path), 'COMMITTED_TECHNICAL_SOURCE')
    inventory = {}
    def git(ref, path):
        key = (ref,path)
        raw = r.git_bytes(ref,path)
        inventory[key] = dict(commit=ref,path=path,**gpu01.identity(raw),
                              acquisition='LOCAL_FIXED_GIT_OBJECT_GPU02')
        return raw
    def doc(ref,path):return r.loads(git(ref,path))
    def write(name, obj):
        # Existing sanitizer also prohibits private absolute paths in all new metadata.
        with (out/name).open('x') as stream:stream.write(gpu01.safe_json(obj))
    baseline = r.loads((out/'cache/baseline_PRIVATE.json').read_bytes())
    catalogue = r.Catalogue(); old = catalogue.verify_config(config)
    for path in OWN:git(source,path)
    bpr = doc(j.B,j.B_DIR+'saved_PR_cost_objects_134.json')['rows']
    bpf = doc(j.B,j.B_DIR+'saved_common_PF_cost_objects_4.json')['rows']
    old_runtime = doc(j.B,j.B_DIR+'runtime_seal_comparison.json')
    # Required prior evidence is pinned and read, without running either old auditor.
    for name in ('same_runtime_comparability.json','comparison_conditions.json',
                 'compiler_reuse_plan.json','runtime_policy_options_DRAFT.json'):
        doc(j.B,j.B_DIR+name)
    for name in ('REPORT.md','HANDOFF_GPUA100-gleap02.md','GAPS_AND_REQUESTS.md'):
        git(j.RECOVERY,j.REC_DIR+name)
    original = doc(j.RECOVERY,j.REC_DIR+'original_identity.json')
    recovery_axes = doc(j.RECOVERY,j.REC_DIR+'axis_identity_4.json')['rows']
    archives = doc(j.RECOVERY,j.REC_DIR+'archive_identity_12.json')['rows']
    runtime = doc(j.RECOVERY,j.REC_DIR+'runtime_identity.json')
    provider_binding = doc(j.RECOVERY,j.REC_DIR+'compiler_provider_binding.json')
    verification = doc(j.RECOVERY,j.REC_DIR+'verification.json')
    fixed = doc(j.RECOVERY,j.REC_DIR+'fixed_sources.json')
    for entry in fixed['files']:
        raw=git(entry['commit'],entry['path'])
        r.require(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],
                  'GPU01_FIXED_GIT_SOURCE_RECHECK')
    plan=doc(j.previous.PF_RESULT,j.previous.PF_PREFIX+'execution_contract_APPROVED_projection.json')
    native=r.loads(gzip.decompress(git(j.previous.PF_RESULT,j.previous.PF_PREFIX+'native_results.json.gz')))
    compiled_inventory=doc(j.previous.PF_RESULT,j.previous.PF_PREFIX+'raw_archive_inventory.json')
    original_hashes=doc(j.B,j.B_DIR+'original_PF_result_hashes.json')
    r.require(original['plan']['sha256']==j.PLAN and original['full_preparation_sha256']==j.PREP,
              'ORIGINAL_PLAN_AND_PREPARATION')
    r.require(verification['static_audit']['QPY_full_static_parses']==8 and len(archives)==12,
              'INHERITED_QPY_SCOPE')
    binding=doc(r.BASE,r.PREVIOUS+'COMPARE_contract_BLOCKED.json')
    diagnostic=Path(binding['accepted_diagnostics']['run_root'])
    prep=r.loads((diagnostic/'results/preparation_3.json').read_bytes())
    r.require(prep['preparation_sha256']==j.PREP, 'ACCEPTED_DIAGNOSTICS_FULL_PREPARATION')
    for field in ('hamiltonian','state','rank','rank_policy','basis','T','P','n_qubits','control'):
        j.same(old['binding'][field],plan['binding'][field], 'PR_PF_PHYSICAL_BINDING:'+field)
    closure_join=j.join_runtime_projection(runtime['closure'],plan['closure'])
    seals=old['sealed_files']
    delta=[]; relevant=[]
    for entry in runtime['closure']:
        role=entry['role'];rel=entry['logical_path']
        if role=='pinned_provider':
            tail=rel[len('${PROVIDER}/'):]
            hits=[s for s in seals if s['role']=='provider/runtime' and s['path'].endswith('/'+tail)]
            r.require(len(hits)==1,'PROVIDER_RELATIVE_PATH_UNIQUE')
            same=(hits[0]['sha256'],hits[0]['bytes'])==(entry['sha256'],entry['bytes'])
            raw=git(PROVIDER,tail)
            r.require(hashlib.sha256(raw).hexdigest()==entry['sha256'] and len(raw)==entry['bytes'],
                      'PF_PROVIDER_FIXED_SOURCE_BYTES')
            now=git(CURRENT_PROVIDER,tail)
            r.require(hashlib.sha256(now).hexdigest()==hits[0]['sha256'], 'PR_PROVIDER_FIXED_SOURCE_BYTES')
            delta.append(dict(closure_index=entry['index'],logical_path=rel,role=role,
                PR_original_seal_role=hits[0]['role'],same_relative_path=True,
                same_SHA_and_bytes=same,PF_byte_observer='GPUA100-gleap-01',
                PR_seal_SHA256=hits[0]['sha256'],PF_seal_SHA256=entry['sha256']))
        elif role in ('fixed_python_binary','loaded_runtime_identity','compiler_array_runtime_binary'):
            hits=[s for s in seals if (s['sha256'],s['bytes'])==(entry['sha256'],entry['bytes'])]
            r.require(len(hits)==1,'UNIQUE_RUNTIME_SEAL_MATCH')
            s=hits[0];p=Path(s['path'])
            r.require(p.name==rel.rsplit('/',1)[-1] and r.file_digest(p)==entry['sha256'] and
                      p.stat().st_size==entry['bytes'],'RUNTIME_REHASH_AND_BASENAME')
            relevant.append(dict(closure_index=entry['index'],role=role,PF_logical_path=rel,
                SHA256=entry['sha256'],bytes=entry['bytes'],PR_seal_role=s['role'],
                PR_current_bytes_rehashed_GPU02=True,PF_original_path_bytes_verified_GPU01=True,
                full_relative_runtime_origin_pair_available=False,actual_historical_load_proven=False))
    r.require(len(delta)==236 and sum(x['same_SHA_and_bytes'] for x in delta)==235 and len(relevant)==30,
              'EXACT_PROVIDER_AND_RUNTIME_SCOPE')
    role_join=j.previous.join_projection_seals(plan['closure'],seals)
    j.same(role_join['role_groups'],old_runtime['role_groups'],'PREVIOUS_RUNTIME_ROLE_GROUPS')
    # Every input is tied to the original plan entry by role, hash and byte size.
    input_rows=[]
    for entry in original['inputs']:
        hits=[(i,c) for i,c in enumerate(plan['closure']) if c['role']==entry['role']]
        r.require(len(hits)==1,'EXACT_INPUT_ROLE')
        i,c=hits[0];j.same({k:entry[k] for k in ('role','sha256','bytes')},c,'PLAN_INPUT_SEAL')
        input_rows.append(dict(original_input_pointer='/inputs/'+str(original['inputs'].index(entry)),
            original_plan_closure_index=i,role=entry['role'],sha256=entry['sha256'],bytes=entry['bytes'],
            original_byte_observer='GPUA100-gleap-01',direct_GPU02_original_read=False))
    # PR objects are checked against fixed recovery packets and original cost receipt bytes.
    packets=doc(EQUIV,EQUIV_DIR+'operation_stream_evidence_134.json')['rows']
    pmap={(p['task'],p['axis']):p for p in packets}
    taskmap={x['task']['id']:x['task'] for x in catalogue.tasks}
    rows=[]
    for i,row in enumerate(bpr):
        j.previous.join_saved_cost(row,bpr[i]);p=pmap[(row['task'],row['axis'])]
        t=taskmap[row['task']]
        r.require(t['candidate']['id']==row['candidate'] and row['candidate'] in j.previous.PR_CANDIDATES,
                  'PR_CANDIDATE_COHORT')
        folder=Path(old['output'])/row['task'];file=folder/('cost_'+row['axis']+'.json')
        cost=r.loads(file.read_bytes())
        r.require(r.file_digest(file)==row['receipt_sha256']==p['cost_receipt_sha256'] and
                  cost['compiled_SHA']==row['compiled_raw_sha256']==p['compiled_raw_sha256'] and
                  cost['metrics']['rz_count']==row['RZ'],'ORIGINAL_PR_COST_RECEIPT')
        rows.append(dict(series='PR',candidate=row['candidate'],task=row['task'],axis=row['axis'],
            cohort=row['cohort'],saved_object_ref=dict(commit=j.B,path=j.B_DIR+'saved_PR_cost_objects_134.json',
                json_pointer='/rows/'+str(i),object_sha256=r.digest(row)),
            stream_ref=dict(commit=EQUIV,path=EQUIV_DIR+'operation_stream_evidence_134.json',
                json_pointer='/rows/'+str(packets.index(p))),
            cost_receipt_SHA256=row['receipt_sha256'],compiled_raw_SHA256=row['compiled_raw_sha256'],
            original_cost_receipt_rehashed_GPU02=True,
            original_archive_observer='GPUA100-gleap02 (fixed prior recovery)',
            archive_reparse_in_B2=False,saved_cost_identity='VERIFIED',
            common_native_absolute_diagonal='UNRESOLVED_A',policy2a_usable='CONDITIONAL_NOT_ADOPTED'))
    r.require(len(rows)==134 and len({(p['task'],p['axis']) for p in rows})==134 and
              Counter(p['candidate'] for p in rows)==dict(zip(j.previous.PR_CANDIDATES,(66,68))),
              'FIXED_PR_134_AXES')
    pfjoins=[]
    for i,saved in enumerate(bpf):
        rec=next(x for x in recovery_axes if (x['candidate'],x['axis'])==(saved['saved_candidate'],saved['axis']))
        packet=gpu01.selected(native,saved['saved_candidate'])
        r.require(r.digest(packet)==saved['receipt_source']['row_digest'],'FIXED_RESULT_PACKET_DIGEST')
        join=j.join_full(saved,bpf[i],rec,packet,archives,plan)
        originals=[x for x in original['other_originals'] if x['role']=='native_result:'+saved['saved_candidate']]
        r.require(len(originals)==1 and originals[0]['sha256']==rec['native_result_sha256'], 'RESULT_FILE_SHA_JOIN')
        for arc in packet['archives']:
            found=[x for x in compiled_inventory if x['candidate']==saved['saved_candidate'] and x['path']==arc['path']]
            r.require(len(found)==1,'FIXED_RAW_INVENTORY_UNIQUE')
            j.same({k:v for k,v in found[0].items() if k!='candidate'},arc,'FIXED_ARCHIVE_RECEIPT')
        pfjoins.append(join)
        rows.append(dict(series='FULL',task=None,
            cohort=saved['cohort'],saved_object_ref=dict(commit=j.B,path=j.B_DIR+'saved_common_PF_cost_objects_4.json',
                json_pointer='/rows/'+str(i),object_sha256=r.digest(saved)),
            GPU01_original_axis_ref=dict(commit=j.RECOVERY,path=j.REC_DIR+'axis_identity_4.json',
                json_pointer='/rows/'+str(recovery_axes.index(rec))),
            compiled_raw_SHA256=saved['compiled']['original_sha256'],
            original_cost_receipt_rehashed_GPU02=False,**join,
            policy2a_usable='CONDITIONAL_NOT_ADOPTED'))
    j.validate_identity_index(rows, {(t['id'],a) for t in taskmap.values() for a in ('cosine','sine')})
    # Static scientific producer equality; no import or function invocation.
    source_rows=[]
    for path in ('scripts/h46_new2_native.py','scripts/h46_m5_native.py',
                 'scripts/h2_m5_compiler_phase_guard.py'):
        pf=git(j.previous.PF_SOURCE,path);pr=git(r.ACQUISITION,path)
        r.require(pf==pr,'SHARED_NATIVE_SOURCE_BYTES')
        actual=Path(binding['acquisition_root'])/path
        r.require(actual.read_bytes()==pr,'CURRENT_ACQUISITION_NATIVE_BYTES')
        source_rows.append(dict(path=path,PF_SOURCE=j.previous.PF_SOURCE,PR_SOURCE=r.ACQUISITION,
            SHA256=hashlib.sha256(pf).hexdigest(),exact_bytes_equal=True,
            current_acquisition_bytes_equal=True,scientific_invocations=0))
    # Read compiler/native/provider and package initializer sources at pinned commits.
    call_paths=('src/trotterlib/rte_compiled_cost.py','src/trotterlib/rte.py',
        'src/trotterlib/__init__.py','src/trottertracks/__init__.py',
        'src/trottertracks/resource_applicability/__init__.py',
        'src/trottertracks/resource_applicability/ax2b_native_df_v5.py',
        'src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py',
        'src/trottertracks/resource_applicability/h6_hash_reuse_v1.py')
    paths=[]
    for path in call_paths:
        raw=git(PROVIDER,path);r.require(raw==git(CURRENT_PROVIDER,path),'RELEVANT_PROVIDER_SOURCE_BYTES')
        tree=ast.parse(raw)
        refs=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)
              and 'h48_preflight_contract_v1' in n.value]
        imports=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and
                 'h48_preflight_contract_v1' in (n.module or '')]
        paths.append(dict(path=path,SHA256=hashlib.sha256(raw).hexdigest(),
                          explicit_h48_references=refs+imports,bytes_equal=True))
    differing=next(x for x in delta if not x['same_SHA_and_bytes'])
    h48=differing['logical_path'][len('${PROVIDER}/'):]
    r.require(h48.endswith('/h48_preflight_contract_v1.py'),'ONLY_KNOWN_PROVIDER_DIFFERENCE')
    oldsrc=git(PROVIDER,h48);newsrc=git(CURRENT_PROVIDER,h48)
    def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body
                              if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    f0,f1=functions(oldsrc),functions(newsrc)
    changed=sorted(k for k in f0 if f0[k]!=f1.get(k));added=sorted(set(f1)-set(f0))
    provider_scope=dict(path=h48,PF_SHA256=differing['PF_seal_SHA256'],
        PR_SHA256=differing['PR_seal_SHA256'],added_functions=added,changed_functions=changed,
        expected_cost_path=paths,explicit_direct_reference_found=any(p['explicit_h48_references'] for p in paths),
        function_role='H4/H8 preflight plans, resource/binding/authority checks; H8_STATE extension',
        compiler_output_noninterference='UNRESOLVED_NOT_A_COMPLETE_HISTORICAL_IMPORT_GRAPH',
        direct_cost_counter_changed=False,
        limitation='No direct reference in inspected H2 compiler/native path; dynamic/transitive historical loads not certified')
    git(j.previous.PF_SOURCE,'scripts/pf_common_run_native.py')
    git(STRICT,'scripts/h2_common_production_backend.py')
    git(j.A,'artifacts/resource_applicability/track_a_h2_pr_diagonal_reuse_a/2026-10-23/REPORT.md')
    a_stop=doc(j.A,'artifacts/resource_applicability/track_a_h2_pr_diagonal_reuse_a/2026-10-23/STOP.json')
    strict_contract=git(r.BASE,r.PREVIOUS+'COMPARE_contract_BLOCKED.json')
    alternative=git(EQUIV,EQUIV_DIR+'alternative_contract_DRAFT.json')
    runtime_limit=dict(**closure_join,role_groups=role_join['role_groups'],
        provider_relative_path_joins=delta,major_runtime_joins=relevant,
        full_runtime_origin_map='PRIVATE_GPU01_MAP_NOT_TRANSFERRED',
        declared_paths_to_bytes_verified_GPU01=runtime['declared_paths_to_bytes_verified'],
        transitive_dependency_closure_complete=False,
        missing_load_history_alone_implies_recompile=False,
        old_Py312_vs_Py311_output_equivalence='UNRESOLVED_UNCHANGED')
    def cond(name,status,scope):return dict(condition=name,status=status,evidence_scope=scope)
    conditions=[
        cond('PR134_Full4_saved_object_binding','VERIFIED','Fixed original cost/compiled raw/result/cohort joins; Full bytes inherited from GPU01'),
        cond('Hamiltonian_state_DF_preparation','VERIFIED','H/state/rank/DF fixed; Full prep equals accepted DIAGNOSTICS prefix3'),
        cond('prefix_law_schedule','VERIFIED','Full p3 q4/q2 vs PR p1 K4 outer32 intentional; no resampling or alias substitution'),
        cond('native_SOURCE_control_scalar_wrapper','VERIFIED','Exact producer bytes; symmetric_directional, scalar once, H/Sdg/measure ancilla4 in GPU01 wrapper receipt'),
        cond('Full_detailed_gate_parameter_phase_stream_vs_common','UNRESOLVED','Only full-byte parse digest + terminal checks available; no individual instruction projection in handoff'),
        cond('PR_absolute_deterministic_diagonal_A','UNRESOLVED','A fixed latest status unchanged; 6 future primitive evaluations not authorized here'),
        cond('declared_compiler_and_major_runtime','VERIFIED','Python1, loaded-runtime seal8, binary21; compiler/counter/guard source/settings match'),
        cond('provider_all_files_identical','MISMATCH','Known H4/H8 preflight file1 differs; compiler-output effect not established'),
        cond('provider_difference_cost_noninterference','UNRESOLVED','Direct inspected cost path excludes h48; complete historical dynamic import path unavailable'),
        cond('historical_actual_load_complete','UNRESOLVED','Both full history and transitive dependency closure absent; declaration/current bytes are a narrower claim'),
        cond('single_input_state_correctness','VERIFIED','Saved strict<1e-7 phase_alignment=false; native/Hadamard scope separate, no new error evaluation'),
        cond('same_declared_Py312_saved_cost_policy','CONDITIONAL','User may adopt limited saved-runtime policy, with residual stream/provider/load limits explicit; not all-state or cross-runtime proof'),
        cond('all_state_operator_guarantee','UNRESOLVED','No operator guarantee from single saved input state'),
        cond('policy2a_adoption_and_COMPARE_authority','CONDITIONAL','Not adopted; new separate SOURCE/contract/unused grant and explicit science authorization required')]
    preflight=j.policy_preflight(conditions)
    write('integrated_axis_identity_138.json',dict(rows=rows,counts=dict(PR=134,Full=4),
        saved_prices_not_remeasured=True,total_ratio_ranking_not_evaluated=True))
    write('Full_plan_input_QPY_binding.json',dict(original_plan_SHA256=j.PLAN,
        plan_ref=dict(commit=j.RECOVERY,path=j.REC_DIR+'original_identity.json',json_pointer='/plan'),
        inputs=input_rows,axes=pfjoins,archive_ref=dict(commit=j.RECOVERY,path=j.REC_DIR+'archive_identity_12.json'),
        archive_rows=12,QPY_static_parsed_GPU01=8,QPY_static_parsed_GPU02_in_B2=0,
        original_bytes_transferred=False,individual_instruction_projection_available=False,
        limitation='Digest does not expose per-gate wires, parameters, global/control phase or full operation order'))
    write('runtime_join.json',runtime_limit)
    write('provider_difference_scope.json',provider_scope)
    write('scientific_native_compiler_join.json',dict(shared_sources=source_rows,
        physical_binding_ref=dict(commit=j.previous.PF_RESULT,path=j.previous.PF_PREFIX+'execution_contract_APPROVED_projection.json',json_pointer='/binding'),
        Full_preparation_SHA256=j.PREP,accepted_DIAGNOSTICS_preparation_equal=True,
        original_input_container_difference='Original PR62c6481b vs actual PF8bb1aa35; historical array-equivalence receipt only',
        science_policy_changed=False,Full_signed_schedule_source='h46_m5_native.compose -> h46_m5_plan.schedule',
        control='symmetric_directional',scalar='original plus PR tail once after signed pieces',
        wrapper='Full Hadamard cosine/sine, system4 ancilla4 classical1 (measurement bit0)',
        full_parameter_stream_proof=False,PR_absolute_diagonal_A='UNRESOLVED',
        compiler_binding_reference=dict(commit=j.RECOVERY,path=j.REC_DIR+'compiler_provider_binding.json'),
        cost_counter='transpile_and_measure_cost -> _gate_counts -> count_ops rz',cost_unit=j.UNIT,
        old_1367_2033_costs_excluded=True))
    write('correctness_scope.json',dict(PR=dict(axes=134,scope='SINGLE_SAVED_INPUT_STATE_CT01'),
        Full=dict(axes=4,scope='SINGLE_SAVED_INPUT_STATE',
                  other_native_checks='low_control/high_leakage/independent_occupation/saved_signal/Hadamard strict<1e-9'),
        strict_compiled_threshold=1e-7,phase_alignment=False,numerical_errors_recomputed=False,
        compiled_cost_identity_distinct_from_correctness=True,all_state_operator_proven=False,
        historical_runtime_load_proven=False))
    write('policy2a_conditions.json',dict(rows=conditions,formal_adoption=False,
        evidence_complete_for_unconditional_common_native_reuse=False,
        minimum_new_scientific_calls_now=0,additional_compile_required_now=False,
        current_accepted_COMPARE='BLOCKED'))
    write('metadata_preflight.json',preflight)
    write('reuse_inventory.json',dict(reused=[
        dict(path='scripts/h2_pr_identity_recovery.py',functions=['Catalogue.verify_config','git_bytes','loads','file_digest','digest']),
        dict(path='scripts/h2_runtime_cost_reuse.py',functions=['join_saved_cost','join_projection_seals','guard_receipt']),
        dict(path='scripts/audit_h2_common_pf_originals.py',functions=['identity','selected','safe_json','verify_gzip (artificial test only)']),
        dict(path='scripts/audit_h2_pr_equivalence.py',functions=['observe','watch_set (baseline)'])],
        inherited=dict(QPY_parser='scripts/h2_pr_saved_qpy.py',PR_operation_stream='scripts/h2_pr_stream_conformance.py'),
        new_code_reason='Existing B and GPU01 auditors have separate output schemas/hosts; only fixed cross-host joins, provenance distinction and non-executable preflight added',
        scientific_runners_invoked=False,compiler_cost_correctness_algorithms_reimplemented=False))
    write('SOURCE.json',dict(technical_SOURCE=source,base=j.RECOVERY,
        scientific_SOURCE_modified=False,strict_COMPARE_SOURCE=STRICT,
        strict_contract_canonical_digest=r.digest(r.loads(strict_contract)),
        strict_contract_file_SHA256=hashlib.sha256(strict_contract).hexdigest(),
        alternative_DRAFT_file_SHA256=hashlib.sha256(alternative).hexdigest(),
        A_fixed_status_commit=j.A,A_read_only=True,
        SOURCE_files=[dict(path=p,SHA256=r.file_digest(ROOT/p)) for p in OWN]))
    r.require(not any(n.split('.')[0] in ('numpy','qiskit','scipy','symengine','trotterlib','trottertracks') for n in sys.modules),
              'NO_SCIENTIFIC_IMPORTS')
    after=preservation.observe(sorted(baseline));r.require(after==baseline,'PROTECTED_SOURCE_INPUT_RESULT_AUTHORITY_CHANGED')
    write('preservation_audit.json',dict(status='PASS',protected_files=len(baseline),changes=0,
        before_SHA256=r.digest(baseline),after_SHA256=r.digest(after),
        unrelated_worktrees_frozen=False,other_worktrees_written=False,
        old_grants_ledgers_markers_STOP_preserved=True,new_grants_issued=0,
        original_payloads_transferred=0,shared_environment_changed=False))
    write('fixed_source_inventory.json',dict(files=list(inventory.values()),
        existing_GPU01_full_Git_sources_rechecked=len(fixed['files']),
        original_PF_bytes_read_GPU02=False,new_remote_fetches=0))
    write('STOP.json',dict(status='MANDATORY_STOP_B2_METADATA_COMPLETE',actual_calls=j.previous.ZERO_CALLS,
        primitive_evaluation=0,scientific_imports=0,new_grants=0,policy2a_adopted=False,
        alternative_adopted=False,COMPARE='BLOCKED',A_remaining_condition='UNRESOLVED',
        private_transfer=0,push=0,mandatory_STOP=True))
    print(gpu01.safe_json(dict(status='PASS_STATIC_JOIN_WITH_EXPLICIT_LIMITS',PR_axes=134,Full_axes=4,
        original_PF_byte_observer='GPUA100-gleap-01',protected_files=len(baseline),
        science_calls=0,COMPARE='BLOCKED',policy_adopted=False,mandatory_STOP=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--source',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:j.forbid_execution()
    audit(args.config.resolve(strict=True),args.source)
