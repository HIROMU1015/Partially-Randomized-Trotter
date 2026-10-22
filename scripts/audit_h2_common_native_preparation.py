"""Read pinned Git/received metadata and write only the new technical namespace.

No saved signal arithmetic, allowance evaluation, scientific imports, circuit
IO, sampling or compile. This is a metadata/AST audit, not a science runner.
"""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/resource_applicability/track_a_h2_common_native_allowance_preparation/2026-10-23'
BASE='cc7404ed189f8c768ecc50a2066485975c9c68f4'
EXPORT='e3531956af6a6d359e2f1fb320e9a5179f41eb98'
AUDIT='366b2fd24f7e86aa82b5b518b6e3774520f65596'
P='artifacts/resource_applicability/track_a_pf_order_reversal_h2_common_policy_preparation/2026-10-23/'
R='artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/'
E='artifacts/resource_applicability/track_a_h2_outer32_metadata_export/2026-10-23/'
A='artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23/'
SOURCES={}


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])


def fixed(commit,path):
    b=git('show',commit+':'+path)
    blob=git('rev-parse',commit+':'+path).decode().strip()
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==blob
    SOURCES[commit,path]=dict(commit=commit,path=path,git_blob_sha1=blob,
        sha256=hashlib.sha256(b).hexdigest(),retrieval='LOCAL_FIXED_GIT_BLOB',
        url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+commit+'/'+path)
    return b


def data(commit,path):return json.loads(fixed(commit,path))
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('h2_technical_metadata_audit',ROOT/'src/trotterlib/h2_common_native_policy.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    implementation_paths=['src/trotterlib/h2_common_native_policy.py',
        'scripts/prepare_h2_common_native.py','scripts/audit_h2_common_native_preparation.py',
        'tests/test_h2_common_native_policy.py']
    implementation_files=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
                          for p in implementation_paths]
    implementation_commit=git('log','-1','--format=%H','--',*implementation_paths).decode().strip()
    if not implementation_commit or any(
            subprocess.run(['git','-C',str(ROOT),'cat-file','-e',implementation_commit+':'+p],
                           capture_output=True).returncode or
            git('show',implementation_commit+':'+p)!=(ROOT/p).read_bytes()
            for p in implementation_paths):
        implementation_commit=None
    for name in ['REPORT.md','comparison_ABC_15.csv','evidence_rows_75.csv','SOURCE_INDEX.md','shot_static_checks.json']:
        fixed(AUDIT,A+name)
    for name in ['REPORT.md','ALLOWANCE_OPTIONS.md','EXECUTION_PLAN_DRAFT.md','missing_evidence.json','SOURCE_INDEX.md']:
        fixed(BASE,P+name)
    for name in ['REPORT.md','evidence_status_update.json','provider_scope_update.json',
                 'received/runtime_provider_manifest.json','received/task_acceptance_67.json',
                 'received/signal_records_2.json','received/old_pf_receipts.json']:
        fixed(BASE,R+name)
    manifest=data(BASE,R+'acquisition_manifest.json')
    received=[]
    for rec in manifest['files']:
        b=fixed(BASE,R+'received/'+rec['path'])
        original=fixed(EXPORT,E+rec['path'])
        assert b==original
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==rec['expected_git_blob_sha1']
        received.append(dict(path=rec['path'],same_export_and_received_bytes=True))
    assert len(received)==150
    refs={}
    for key,(commit,path,symbol) in module.BINDINGS.items():
        b=fixed(commit,path);tree=ast.parse(b)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==symbol)
        refs[key]=dict(SOURCES[commit,path],symbol=symbol,line=fn.lineno,
            function_AST_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest())
    for commit,path in [
        ('217c971dd537b1cc7fbc13742bc0f6c263ff7f45','src/trotterlib/df_trotter/ops.py'),
        (module.PROVIDER_SOURCE,'src/trotterlib/df_trotter/ops.py'),
        (module.PROVIDER_SOURCE,'src/trotterlib/pr2_matched_accuracy_m1_execution.py'),
        (module.ACQUISITION_SOURCE,'scripts/h2_outer32_runtime.py'),
        (module.ACQUISITION_SOURCE,'scripts/h46_m5_signal_contract.py'),
        ('217c971dd537b1cc7fbc13742bc0f6c263ff7f45','src/trottertracks/resource_applicability/h6_matched_accounting_v1.py'),
        ('217c971dd537b1cc7fbc13742bc0f6c263ff7f45','src/trottertracks/resource_applicability/h6_matched_port_v1.py'),
        ('cf99e4452df96291d7e9d37d76d2e91be5cbc413','scripts/h2_panels_science.py')]:
        fixed(commit,path)
    runtime=data(BASE,R+'received/runtime_provider_manifest.json')
    provider=[]
    for rec in runtime['historical_sealed_files']:
        saved=rec['saved']
        if not saved['path'].startswith('${NEW_PROVIDER}/'):
            continue
        path=saved['path'].split('${NEW_PROVIDER}/',1)[1]
        b=fixed(module.PROVIDER_SOURCE,path)
        assert hashlib.sha256(b).hexdigest()==saved['sha256']
        provider.append(dict(path=path,sealed_git_sha256_equal=True))
    assert len(provider)==278
    mismatch=[dict(path=r['saved']['path'],saved_sha256=r['saved']['sha256'],
                   sender_observed_sha256=r['current_sha256'])
              for r in runtime['historical_sealed_files'] if r['matches_saved'] is False]
    assert len(mismatch)==1 and 'lzma' in mismatch[0]['path']
    tasks=data(BASE,R+'received/task_acceptance_67.json')
    assert tasks['task_count']==67 and tasks['axis_count']==134 and tasks['other_cohorts_pooled'] is False
    assert tasks['audit_original_sha256']=='38ad4d594a3be9289bdf1697d9a37a6d677384899a3320666c75761a304f7723'
    wanted={module.CANDIDATES[2].id:33,module.CANDIDATES[3].id:34}
    for cid,n in wanted.items():
        selected=[t for t in tasks['tasks'] if t['task']['candidate']['id']==cid]
        assert len(selected)==n and {t['task']['sample'] for t in selected}==set(range(n))
        for t in selected:
            result=t['result']['saved']
            assert result['SOURCE']==module.ACQUISITION_SOURCE
            assert result['conditional_sampling'] is False and result['cohort_pooling'] is False
            assert {r['axis'] for r in result['axis_costs']}=={'cosine','sine'}
            assert all(g['PASS'] and g['phase_alignment'] is False for g in result['checks'])
    excerpts=data(BASE,P+'saved_input_excerpts.json')
    new=data(BASE,R+'received/signal_records_2.json')
    old=data(BASE,R+'received/old_pf_receipts.json')
    coverage=[]
    for rec in excerpts['old_signals']:
        mapped=module.map_old_diagnostics(rec,excerpts['old_reference'],BASE+':'+P+'saved_input_excerpts.json')
        coverage.append(dict(candidate=mapped.candidate_id,series='OLD_FULL_PF',
                             mapped_inputs=asdict(mapped),missing_for_common_series=mapped.missing(),
                             numeric_generator_executed=False))
    for rec in new['records']:
        mapped=module.map_pr_diagnostics(rec,BASE+':'+R+'received/signal_records_2.json')
        coverage.append(dict(candidate=mapped.candidate_id,series='SAVED_OUTER32_PR',
                             mapped_inputs=asdict(mapped),missing_for_common_series=mapped.missing(),
                             excluded_observations=['raw vector closure is not corrected signal closure',
                               'piece-end maximum norm is not both-path full-stage coverage',
                               'stage proxy is not measured action count'],numeric_generator_executed=False))
    old_compiled=[]
    for row in old['rows']:
        old_compiled.append(dict(candidate=row['candidate'],
            compiled_action_phase_receipt=row['compiled_action_phase_receipt'],
            compiled_circuit_saved_locator=row['compiled_circuit_saved_locator'],
            compiled_vector_saved_locator=row['compiled_vector_saved_locator']))
    # Exact known original H2 metadata only. Never read a grant or any array.
    old_missing=data(BASE,P+'missing_evidence.json')
    locators=next(r['exact_known_paths'] for r in old_missing['files']
                  if r['id']=='OLD_ORIGINAL_METADATA')
    known_checks=[]
    for i,locator in enumerate(locators):
        path=Path(locator)
        check=dict(locator_id='OLD_ORIGINAL_METADATA_'+str(i),exists=path.is_file())
        if path.is_file():
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            check.update(sha256=before,read_only=True,
                         post_read_unchanged=hashlib.sha256(path.read_bytes()).hexdigest()==before)
        known_checks.append(check)
    private_proof=dict(exact_metadata_checks=known_checks,private_bodies_exported=False,
                       scientific_arrays_loaded=False,grants_read=False,worker_checks=[])
    recovery=Path(locators[0]).parents[1]/'D_recovery_activation01/D_spec_EXECUTION.json'
    public=data('829d697ae7905b3ea90a61da9d2d4da15d8b757a',
        'artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/D_completed_run02/execution_binding.json')
    if recovery.is_file():
        b=recovery.read_bytes();assert hashlib.sha256(b).hexdigest()==public['spec_file_sha256']
        original_spec=json.loads(b)
        assert original_spec['SOURCE']==old['compile_SOURCE']['saved']
        private_proof.update(recovery_spec_hash_matches_public_binding=True,
                             recovery_spec_sha256=public['spec_file_sha256'])
        for row in old['rows']:
            path=Path(original_spec['attempt'])/'output/compile_workers'/row['candidate']
            if not path.is_dir():
                private_proof['worker_checks'].append(dict(candidate=row['candidate'],exists=False))
                continue
            files=[q for q in path.iterdir() if q.is_file()]
            checked=[]
            for name,key in [('control.json','control'),('task_result.json','task_result')]:
                raw=(path/name).read_bytes();value=json.loads(raw)
                assert value==row[key]['saved']
                checked.append(dict(name=name,sha256=hashlib.sha256(raw).hexdigest(),
                                    same_as_received_projection=True,
                                    post_read_unchanged=(path/name).read_bytes()==raw))
            private_proof['worker_checks'].append(dict(candidate=row['candidate'],exists=True,
                immediate_file_names=[q.name for q in files],checked=checked,
                QPY_or_vector_archives_in_examined_folder=any(q.suffix in {'.qpy','.npz','.npy'} for q in files),
                scope='ONLY_THIS_KNOWN_CANDIDATE_WORKER; NO_GLOBAL_ABSENCE_CLAIM'))
        assert recovery.read_bytes()==b
    write('private_recovery_metadata_check.json',private_proof)
    missing=['shared_reference_policy_and_candidate_rebinding',
       'PR_independent_path_vector_discrepancies','PR_both_path_work_action_counts',
       'PR_corrected_signal_unit_closure','PR_both_path_full_stage_norm_coverage',
       'common_Gaussian_preparation_operation_stream_acceptance',
       'old_Full_PF_compiled_vector_or_new_common_native_correctness',
       'new_common_runtime_loaded_closure_seal_and_liblzma_resolution',
       'identity_witness_for_saved_PR_C_reuse','exact_backend_scientific_call_recipe',
       'separate_policy_adoption_and_one_shot_execution_approval']
    plan=dict(schema='h2_common_preflight_v1',status='DRAFT_NOT_EXECUTABLE',
       science_authorized=False,native_adopted=False,allowance_adopted=False,
       physical=module.PHYSICAL,compiler=module.COMPILER,candidates=[asdict(c) for c in module.CANDIDATES],
       native_policy=module.NATIVE_POLICY,allowance_policy=module.ALLOWANCE_POLICY,
       production_dispatch_bound=False,grant=None,ledger=None,marker=None,
       retry=False,mandatory_STOP=True,new_call_counts=None,missing_inputs=missing)
    outcome=module.validate_draft(plan)
    write('preflight_config_DRAFT.json',plan)
    write('preflight_result.json',outcome)
    write('diagnostic_coverage_4.json',coverage)
    write('kernel_bindings.json',refs)
    write('deferred_native_recipes_4.json',[module.deferred_kernel_recipe(asdict(c))
                                           for c in module.CANDIDATES])
    write('source_manifest.json',list(SOURCES.values()))
    write('metadata_verification.json',dict(status='STATIC_METADATA_PASS_WITH_RETAINED_GAPS',
      received_export_byte_checks=received,provider_sealed_checks=provider,
      accepted_PR_tasks=67,accepted_PR_axes=134,correctness_reperformed=False,
      historical_current_runtime_mismatches=mismatch,
      current_sender_observations_not_rechecked_as_local_runtime=True,
      old_compiled_evidence=old_compiled,science_GO=False,scientific_calls=0,
      new_u_N_G_evaluations=0,scientific_source_imports=0))
    write('execution_contract_DRAFT.json',dict(status='DRAFT_NOT_EXECUTABLE',
      implementation_source_commit=implementation_commit,
      implementation_source_binding='FIXED_TECHNICAL_CODE_ONLY_NOT_A_SCIENCE_RUNNER' if implementation_commit else 'TO_BE_PINNED_AFTER_TECHNICAL_TESTS',
      implementation_files=implementation_files,
      kernel_bindings='kernel_bindings.json',preflight_config='preflight_config_DRAFT.json',
      scientific_execution_authorized=False,policies_formally_adopted=False,
      exact_scientific_call_counts=None,grant=None,ledger=None,marker=None,
      gates=[dict(id=0,name='PINNED_EVIDENCE',status='STATIC_PARTIAL_WITH_GAPS'),
             dict(id=1,name='COMMON_NATIVE_ALLOWANCE_PROPOSAL',status='IMPLEMENTED_MOCK_ONLY_NOT_ADOPTED'),
             dict(id=2,name='MISSING_SCIENTIFIC_DIAGNOSTICS',status='NOT_AUTHORIZED'),
             dict(id=3,name='FOUR_FIXED_POINT_ACCOUNTING',status='NOT_AUTHORIZED'),
             dict(id=4,name='SAVE_AUDIT_MANDATORY_STOP',status='REQUIRED')],
      new_science_scope_candidates=[
        dict(kind='PR_DIAGNOSTICS',candidates=list(wanted),need='only absent independent-vector/work/norm/closure diagnostics',sampling=0,compile=0,count=None),
        dict(kind='COMMON_REFERENCE',need='prefer accepted saved old FP reference; new reference only if reuse is scientifically rejected',count=None),
        dict(kind='FULL_COMMON_NATIVE_VALIDATION',candidates=[c.id for c in module.CANDIDATES[:2]],axes=['cosine','sine'],need='new series if no identity-equivalent accepted existing artifacts',count=None),
        dict(kind='PR_C_REUSE',saved_tasks=67,need='common preparation/wrapper/kernel/runtime/event-law witness; otherwise separately propose only affected axes',automatic_sampling=False,automatic_compile=False),
        dict(kind='COMMON_POLICY_ACCOUNTING',candidates=[c.id for c in module.CANDIDATES],need='u/margin, paired shots and G only after explicit approval',quantum_shot_execution=0,count=None)],
      correctness_proposal=dict(native_occupation_wrapper_strictly_below=1e-9,
          compiled_saved_state_vector_strictly_below=1e-7,phase_alignment=False,
          thresholds_are_not_shot_allowances=True,all_state_operator_certificate=False),
      one_shot_requirements=['bind final SOURCE/input/compiler/runtime and exact per-kind call budget',
          'new unused run namespace; external approval receipt and non-reusable grant digest',
          'write consumed marker atomically before first scientific call',
          'append per-kind reservation to own ledger before each scientific kernel',
          'all children exit/reap; nonzero/missing/numeric mismatch stops; no scientific retry',
          'only used SOURCE/input/compiler/runtime/results/grant/ledger/marker are immutable',
          'unrelated worktree HEAD/status and transient PSI/competition do not kill workers'],
      missing_inputs=missing,retry=False,mandatory_STOP=True))
    lines=['# 固定SOURCE・入力索引','','取得は固定Git objectのread-only読取。科学codeはASTとして解析し、import/実行していない。既存公開projectionは親commitから再利用し、再コピー・再pushしていない。','','| commit | path | blob SHA1 | SHA256 |','|---|---|---|---|']
    for r in SOURCES.values():lines.append('| `'+r['commit']+'` | ['+r['path']+']('+r['url']+') | `'+r['git_blob_sha1']+'` | `'+r['sha256']+'` |')
    (OUT/'SOURCE_INDEX.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(received_files=150,provider_files=278,PR_tasks=67,PR_axes=134,
                         source_records=len(SOURCES),execution_ready=False,science_calls=0)))


if __name__=='__main__':run()
