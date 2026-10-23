"""Read fixed receipts/seals only; never evaluate H2 circuits or cost functions."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import h2_runtime_cost_reuse as c
import audit_h2_pr_equivalence as previous

r = c.r
ROOT = r.ROOT
OUT = ROOT/'artifacts/resource_applicability/track_a_h2_runtime_cost_reuse/2026-10-23'
PRIOR = 'artifacts/resource_applicability/track_a_h2_pr_equivalence_contract/2026-10-23/'
OLD_H2 = 'artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/'
OWN = ('scripts/h2_runtime_cost_reuse.py', 'scripts/audit_h2_runtime_cost_reuse.py',
       'tests/test_h2_runtime_cost_reuse.py')


def blob_sha(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def write(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def audit(config, source):
    r.require(config.name == 'config_APPROVED_PRIVATE.json', 'ORIGINAL_CONFIG_INPUT_ONLY')
    r.require(set(p.name for p in OUT.iterdir()) <= {'cache'}, 'FRESH_OUTPUT_REQUIRED')
    source=subprocess.check_output(['git','rev-parse',source],cwd=ROOT,text=True).strip()
    for name in OWN:r.require((ROOT/name).read_bytes()==r.git_bytes(source,name),'IMMUTABLE_AUDIT_SOURCE')
    fixed={}
    def git(ref,path):
        key=(ref,path)
        if key not in fixed:
            raw=r.git_bytes(ref,path)
            fixed[key]=dict(commit=ref,path=path,sha256=hashlib.sha256(raw).hexdigest(),
                blob_sha1=blob_sha(raw),bytes=len(raw),acquisition='LOCAL_IMMUTABLE_GIT_BLOB')
        return r.git_bytes(ref,path)
    def doc(ref,path):return r.loads(git(ref,path))
    catalogue=r.Catalogue();old=catalogue.verify_config(config);execution=config.parents[2]
    strict=doc(r.BASE,r.PREVIOUS+'COMPARE_contract_BLOCKED.json')
    watched=previous.watch_set(catalogue,old,execution,strict)
    old_root=Path(doc(c.BASE,'artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23/private_metadata_audit.json')['old_spec_locator_from_public_binding']).parents[1]
    old_worktree=old_root.parents[1]
    original_specs={}
    for stage,folder in (('D','D_recovery_activation01'),('A','A_activation01'),('P','P_activation01')):
        path=old_root/folder/(stage+'_spec_EXECUTION.json')
        original_specs[stage]=r.loads(path.read_bytes());watched[str(path)]='OLD_H2_SPEC:'+stage
        for entry in original_specs[stage].get('dependencies',[]):
            if type(entry) is dict and 'path' in entry:watched[entry['path']]='OLD_H2_DEPENDENCY:'+stage
    for stage in ('D_completed_run02','A_completed_run01','FP_completed_run01','P_completed_run01'):
        for name in ('result.json','confirmation_result.json','paired_costs_projection.json',
                     'execution_binding.json','native_completion_audit.json','terminal.json'):
            path=old_worktree/OLD_H2/stage/name
            if path.is_file():watched[str(path)]='OLD_H2_RESULT:'+stage+'/'+name
    # Only explicit scientific dependencies/authority are protected, never other worktree HEAD/status.
    before=previous.observe(sorted(watched))
    r.require(all(x is not None for x in before.values()),'PROTECTED_PATH_MISSING')
    runtime_matches=c.join_projection_seals(doc(c.PF_RESULT,c.PF_PREFIX+'execution_contract_APPROVED_projection.json')['closure'],old['sealed_files'])
    runtime_matches['current_rehash_of_matched_PR_seals']=[]
    projection=doc(c.PF_RESULT,c.PF_PREFIX+'execution_contract_APPROVED_projection.json')
    for entry in projection['closure']:
        if entry['role'] not in ('fixed_python_binary','loaded_runtime_identity','compiler_array_runtime_binary'):continue
        matching=[s for s in old['sealed_files'] if (s['sha256'],s['bytes'])==(entry['sha256'],entry['bytes'])]
        r.require(len(matching)==1,'UNIQUE_ROLE_SHA_SIZE_JOIN')
        p=Path(matching[0]['path']);r.require(r.file_digest(p)==entry['sha256'],'CURRENT_MATCHED_SEAL_DRIFT')
        runtime_matches['current_rehash_of_matched_PR_seals'].append(dict(role=entry['role'],filename=p.name,
            sha256=entry['sha256'],bytes=entry['bytes'],match=True))
    r.require(all(runtime_matches['role_groups'][k]['matched']==n for k,n in
        (('fixed_python_binary',1),('loaded_runtime_identity',8),('compiler_array_runtime_binary',21))),
        'PUBLISHED_PF_VS_ORIGINAL_PR_RUNTIME_SUBSET')
    provider_missing=runtime_matches['role_groups']['pinned_provider']['missing']
    provider_diff=[]
    for name in subprocess.check_output(['git','diff','--name-only','217c971dd537b1cc7fbc13742bc0f6c263ff7f45',
        'c31ef424175c02379eafebe8fc6c7548c1d9aabe','--','src'],cwd=ROOT,text=True).splitlines():
        try:raw=git('217c971dd537b1cc7fbc13742bc0f6c263ff7f45',name)
        except subprocess.CalledProcessError:continue
        if any(hashlib.sha256(raw).hexdigest()==x['sha256'] for x in provider_missing):
            provider_diff.append(dict(path=name,old_sha256=hashlib.sha256(raw).hexdigest(),
                current_sha256=hashlib.sha256(git('c31ef424175c02379eafebe8fc6c7548c1d9aabe',name)).hexdigest(),
                scope='Different preflight contract; not evidence of scientific output mismatch'))
    runtime_matches['identified_provider_differences']=provider_diff
    costs=[];archive_checks=[];guard_passes=[]
    storage={x['task_id']:x for x in catalogue.storage}
    old_packets=doc(c.BASE,PRIOR+'operation_stream_evidence_134.json')['rows']
    old_packet_index={(x['task'],x['axis']):x for x in old_packets}
    cost_source=Path(strict['provider_root'])/'src/trotterlib/rte_compiled_cost.py'
    cost_sha=r.file_digest(cost_source)
    for anchor in catalogue.tasks:
        task=anchor['task'];folder=Path(old['output'])/task['id']
        result=r.loads((folder/'result.json').read_bytes())
        for i,axis in enumerate(('cosine','sine')):
            saved=anchor['axes'][i]['cost']['saved'];cost_path=folder/('cost_'+axis+'.json')
            cost=r.loads(cost_path.read_bytes());r.require(cost==saved==result['axis_costs'][i],'SAVED_COST_JOIN')
            packet=old_packet_index[(task['id'],axis)]
            r.require(r.file_digest(cost_path)==packet['cost_receipt_sha256'],'COST_RECEIPT_SHA')
            guard=r.loads((folder/('compiler_phase_guard_'+str(i)+'.json')).read_bytes())
            c.guard_receipt(guard);guard_passes.append(len(guard['passes']))
            archives={a['manifest']['saved']['archive_file']:a['manifest']['saved']['codec_receipt']
                for a in storage[task['id']]['axes'][i]['archives']}
            for name,receipt in archives.items():
                check=r.verify_archive(folder/('quarantine_'+axis)/name,receipt['stored'],receipt['original'])
                archive_checks.append(dict(task=task['id'],axis=axis,file=name,**check))
            row=dict(candidate=task['candidate']['id'],task=task['id'],axis=axis,cohort='EXPLORATION',
                runtime_id=c.PY312,compiler_hash=c.COMPILER,compiler_source_sha256=cost_sha,
                settings=deepcopy(c.SETTINGS),input_sha256=old['binding']['snapshot'],
                wrapper_raw_sha256=archives['wrapper.qpy.zlib']['original']['sha256'],
                compiled_raw_sha256=cost['compiled_SHA'],receipt_sha256=packet['cost_receipt_sha256'],
                RZ=cost['metrics']['rz_count'],unit='compiled_RZ_per_full_Hadamard_wrapper_axis',
                correctness=deepcopy(packet['historical_correctness']))
            r.require(row['compiled_raw_sha256']==archives['compiled.qpy.zlib']['original']['sha256'], 'COMPILED_RAW_BINDING')
            costs.append(c.join_saved_cost(row,deepcopy(row)))
    r.require(len(costs)==134 and len(archive_checks)==402,'ORIGINAL_COHORT_SCOPE')
    native=r.loads(gzip.decompress(git(c.PF_RESULT,c.PF_PREFIX+'native_results.json.gz')))
    endpoints=list(csv.DictReader(io.StringIO(git(c.PF_RESULT,c.PF_PREFIX+'PF_endpoint_costs.csv').decode())))
    pf_rows=[]
    for requested,pf_id in c.FULL_MAP.items():
        q=4 if requested.endswith('2nd') else 2;formula='2nd' if q==4 else '4th'
        target=next(x for x in native if x['candidate']==dict(system='H2',method='FULL',prefix=3,q=q,formula=formula))
        endpoint=next(x for x in endpoints if x['id']==pf_id and x['epsilon']=='0.0005')
        for axis in ('cosine','sine'):
            price=target['prices'][axis];c.guard_receipt(price['phase_guard'])
            rz=price['metrics']['rz_count'];r.require(type(rz) is int and str(rz)==endpoint[axis+'_RZ'],'SAVED_PF_CSV_RECEIPT_RZ')
            check=next(x for x in target['checks'] if x['check']=='compiled_vector_'+axis)
            r.require(check['PASS'] is True and check['strictly_below']==1e-7 and check['phase_alignment'] is False,'PF_HISTORICAL_SINGLE_STATE_SCOPE')
            archives={x['path']:x for x in target['archives']}
            pf_rows.append(dict(requested_candidate=requested,saved_candidate=pf_id,candidate_fields=target['candidate'],
                axis=axis,cohort='COMMON_PF_DETERMINISTIC_20261022',runtime_id=c.PY312,
                RZ=rz,unit='compiled_RZ_per_full_Hadamard_wrapper_axis',
                compiler_hash=price['compiler'],compiler_settings=deepcopy(c.SETTINGS),
                guard=price['phase_guard'],correctness=check,
                preparation_hash=target['preparation_hash'],native_policy=target['native_policy'],
                H=target['hamiltonian'],state=target['state'],scalar=target['scalar'],
                wrapper=archives[axis+'_wrapper.qpy.gz'],compiled=archives[axis+'_compiled.qpy.gz'],
                receipt_source=dict(commit=c.PF_RESULT,path=c.PF_PREFIX+'native_results.json.gz',
                    json_pointer='/'+str(native.index(target)),row_digest=r.digest(target)),
                raw_available_here=False,raw_identity_rehashed_here=False,
                scientific_arrays_reconstructed=True,old_snapshot_container_restored=False,
                comparable_to_current_common_PF='UNRESOLVED_PREPARATION_AND_NATIVE_STREAM_JOIN'))
    pf_end=doc(c.PF_RESULT,c.PF_PREFIX+'end_audit.json')
    r.require(pf_end['status']=='PASS_MANDATORY_STOP' and pf_end['all_native_workers_exit0_reaped'] is True,'PF_END_AUDIT')
    historical=[]
    for stage in ('D_completed_run02','A_completed_run01','FP_completed_run01'):
        data=doc('f2d1a86c431f835db700f348ce618d8604e7ee23',OLD_H2+stage+'/result.json')
        local=old_worktree/OLD_H2/stage/'result.json'
        r.require(r.file_digest(local)==hashlib.sha256(git('f2d1a86c431f835db700f348ce618d8604e7ee23',OLD_H2+stage+'/result.json')).hexdigest(),'OLD_RESULT_FIXED_GIT_JOIN')
        cells=[x['cell_id'] for x in data['measurements']]
        historical.append(dict(series=stage,SOURCE=data['SOURCE'],measurements=len(cells),
            original_outer32_candidates_found=sorted(set(cells)&set(c.PR_CANDIDATES)),
            target_Full_records=[dict(candidate=x['cell_id'],axes={a:v['RZ'] for a,v in x['axes'].items()})
                for x in data['measurements'] if x['cell_id'] in c.FULL_MAP],
            compiler=data['binding']['compiler'],H=data['binding']['hamiltonian'],state=data['binding']['state'],
            reuse_scope='D Full old-native cost object only; A/FP S2 old law and cohort are not outer32',
            raw_QPY_available_in_target_compile_workers=False if stage=='D_completed_run02' else None))
    p_result=doc('f2d1a86c431f835db700f348ce618d8604e7ee23',OLD_H2+'P_completed_run01/confirmation_result.json')
    historical.append(dict(series='P_completed_run01',SOURCE=original_specs['P']['SOURCE'],
        runtime_python_sha256=original_specs['P']['python_sha256'],
        result_commit='f2d1a86c431f835db700f348ce618d8604e7ee23',
        original_outer32_candidates_found=[],
        reuse_scope='Original H2 S2 confirmation; different law/candidate/cohort. No Y4 outer32 substitution',
        receipt_digest=r.digest(p_result)))
    runtime_matches['old_A_python_sha256']=original_specs['A']['python_sha256']
    runtime_matches['old_P_python_sha256']=original_specs['P']['python_sha256']
    runtime_matches['old_D_runtime_inherited_from_A']='Original D recovery specification preserves original runtime/provider dependencies'
    prep3=Path(strict['accepted_diagnostics']['run_root'])/'results/preparation_3.json'
    current3=r.loads(prep3.read_bytes())
    target_binding=dict(public_PF_preparation=pf_rows[0]['preparation_hash'],
        adopted_common_preparation=current3['preparation_sha256'],
        same_preparation_hash=pf_rows[0]['preparation_hash']==current3['preparation_sha256'],
        relevant_native_source_bytes_equal=git(c.PF_SOURCE,'scripts/h46_new2_native.py')==
            (Path(strict['acquisition_root'])/'scripts/h46_new2_native.py').read_bytes(),
        scientific_mismatch_established=False,exact_native_operation_stream_binding='UNRESOLVED',
        original_PF_plan_and_archive_root_available_here=False,
        missing_locator='Common-PF owner original plan_APPROVED.json and COST target raw archives; no guessed remote connection')
    # Remote files are kept only in the ignored task cache. Publish receipts and selected row references.
    external=r.loads((OUT/'cache/remote_index_content.json').read_bytes());remote=[];crosswalk=[]
    for item in external:
        raw=item['content'].encode();r.require(blob_sha(raw)==item['sha'],'GITHUB_CONTENT_BLOB_SHA')
        remote.append(dict(commit=item['ref'],path=item['path'],blob_sha1=item['sha'],
            sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),acquisition='GITHUB_FULL_UTF8_CONTENT'))
        if item['path'].endswith('research_index.csv'):
            crosswalk=[x for x in csv.DictReader(io.StringIO(item['content'])) if
                any(t in str(x) for t in ('PF_COMMON_H2','ABE:H2_','GPU02:H2','GPU02:H246_outer32'))]
    for receipt in r.loads((OUT/'cache/pf_remote_receipts.json').read_bytes()):
        raw=git(receipt['commit'],receipt['path']);r.require(blob_sha(raw)==receipt['blob_sha1'],'REMOTE_PF_BLOB_JOIN')
        remote.append(dict(receipt,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    for ref,prefix in [('b8bd64339169ecdf61b90fb7b575a3937fdc0c8c','docs/research/environment_evidence/gpua100-gleap02_20261023/'),
                       ('e0c942fbb5dd8741dda69af0933fce88e8eaea91','docs/research/environment_evidence_gpua100_20261023/')]:
        doc(ref,prefix+'series_manifest.json');git(ref,prefix+'README.md')
    reusable=[]
    for ref,path,functions in [(r.ACQUISITION,'scripts/h2_m5_compiler_phase_guard.py',('transpile','scalar_guard')),
        (r.ACQUISITION,'scripts/h46_pr_pilot_native.py',('gate',)),
        ('217c971dd537b1cc7fbc13742bc0f6c263ff7f45','src/trotterlib/rte_compiled_cost.py',('transpile_and_measure_cost','_gate_counts')),
        ('217c971dd537b1cc7fbc13742bc0f6c263ff7f45','src/trotterlib/df_trotter/circuit.py',('simulate_statevector',)),
        (c.PF_SOURCE,'scripts/pf_common_run_native.py',('cost_one',))]:
        raw=git(ref,path);tree=ast.parse(raw)
        for name in functions:
            node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
            reusable.append(dict(commit=ref,path=path,function=name,line=node.lineno,
                source_sha256=hashlib.sha256(raw).hexdigest(),
                function_AST_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
                actual_scientific_invocations=0,
                use='Reference unchanged helper; full run_task/cost_one rebuilds circuits and must not be used as saved-QPY adapter'))
    for name in ('REPORT.md','alternative_contract_DRAFT.json','runtime_conditions.json','runtime_relevance.json',
        'axis_evidence_134.json','correctness_scope.json','minimal_additional_validation_DRAFT.json','execution_readiness_conditions.json'):
        git(c.BASE,PRIOR+name)
    oldplan=doc(c.BASE,PRIOR+'minimal_additional_validation_DRAFT.json')['B_archived_wrapper_sensitivity_probe']
    probe=c.probe_draft(costs,oldplan)
    after=previous.observe(sorted(watched));r.require(before==after,'PROTECTED_DEPENDENCY_OR_AUTHORITY_CHANGED')
    r.require(not any(x.split('.')[0] in ('numpy','qiskit','scipy','symengine','trotterlib','trottertracks') for x in sys.modules),'NO_SCIENTIFIC_IMPORT')
    write('saved_PR_cost_objects_134.json',dict(rows=costs,statistics_recomputed=False))
    write('saved_common_PF_cost_objects_4.json',dict(rows=pf_rows,original_costs_unchanged=True))
    write('historical_cost_inventory.json',dict(rows=historical,older_A_P_not_outer32=True))
    write('same_runtime_comparability.json',dict(case='B1_SAVED_PY312_COSTS_EXIST_WITH_BINDING_LIMITS',
        same_runtime_cost_coverage=dict(PR=134,Full=4),fresh_Py311_Full_costs_found=0,
        full_historical_load_proven=False,policy_adopted=False,COMPARE='BLOCKED',
        target_native_binding=target_binding,additional_compile_required_now=False,
        first_next_step='Metadata/source/raw receipt recovery and binding; zero compile',
        original_B_cross_runtime_condition_unchanged='UNRESOLVED',
        original_A_absolute_diagonal_condition_unchanged='UNRESOLVED'))
    write('runtime_seal_comparison.json',runtime_matches)
    write('PR_archive_identity_recheck.json',dict(archives=402,checks=archive_checks,
        vector_payload_decoded=False,QPY_loaded_as_circuit=False,guard_pass_count_histogram=dict(Counter(guard_passes))))
    write('compiler_reuse_plan.json',dict(functions=reusable,reimplemented_compiler=False,
        QPY_priority='Original saved wrapper; reuse qiskit.qpy.load only after separate approval',
        existing_cost_counter='rte_compiled_cost.transpile_and_measure_cost -> _gate_counts -> count_ops rz',
        current_source_scope='Metadata adapter and artificial call rehearsal only',
        production_runner_required_now=False,scientific_calls=0))
    write('preserved_four_axis_probe_DRAFT.json',probe)
    write('metadata_preflight.json',c.preflight())
    write('remote_acquisition_receipts.json',dict(files=remote,
        interpretation='Full UTF8 versus first-line/blob-ID acquisition distinguished; gzip evidence read from fixed local Git only'))
    write('three_environment_crosswalk.json',dict(rows=crosswalk,
        input_commits=['b8bd64339169ecdf61b90fb7b575a3937fdc0c8c','e0c942fbb5dd8741dda69af0933fce88e8eaea91',
                       'f284c501bd8332fa5e8fc098b5424e34c666174f'],
        integration_commit='6dafed86cc82149d55b4e6395382953d5f3631d7',
        no_additional_matching_original_cohort_from_ABE=True,custody_is_not_execution_host=True))
    write('preservation_audit.json',dict(status='PASS',protected_files=len(watched),changes=0,
        before_sha256=r.digest(before),after_sha256=r.digest(after),
        scope='Explicit original PR catalogue/runtime/input/compiler/authority; old H2 stage metadata; accepted DIAGNOSTICS and strict COMPARE dependencies',
        unrelated_worktrees_observation_only=True,old_grants_reused=0,new_grants=0,
        historical_liblzma_drift_not_repaired=True))
    write('SOURCE.json',dict(audit_SOURCE=source,base_commit=c.BASE,files=[dict(path=n,sha256=r.file_digest(ROOT/n)) for n in OWN],
        strict_COMPARE_SOURCE=c.alternative.OLD_SOURCE,strict_contract_digest=c.alternative.OLD_DIGEST,
        science_source_modified=False))
    write('fixed_source_inventory.json',dict(files=list(fixed.values())))
    write('STOP.json',dict(status='MANDATORY_STOP_AFTER_RUNTIME_COST_REUSE_INVESTIGATION',
        actual_scientific_calls=c.ZERO_CALLS,actual_H2_compile=0,new_u_N_G=0,scientific_imports=0,
        alternative_adopted=False,COMPARE='BLOCKED',live_grants=0,execution_ready=False,
        original_134_axis_B_condition='UNRESOLVED',mandatory_STOP=True))
    print(json.dumps(dict(status='PASS_METADATA_AUDIT_MANDATORY_STOP',protected=len(watched),
        PR_saved_axes=len(costs),Full_saved_axes=len(pf_rows),new_scientific_calls=0,
        case='B1_WITH_BINDING_LIMITS',COMPARE='BLOCKED')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source',required=True);args=parser.parse_args()
    audit(args.config.resolve(strict=True),args.source)
